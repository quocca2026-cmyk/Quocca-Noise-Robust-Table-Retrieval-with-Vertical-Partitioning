import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ==========================================
# 1. System Prompts
# ==========================================
SYS_PROMPT_1_QUERY_GEN = """You are an expert in 'Discriminative Table Retrieval'.
Your goal is to generate 5 distinct questions that act as a **unique signature** for this table, ensuring it can be found among thousands of other similar tables.

=====================
### STRATEGY: DISCRIMINATIVE ANCHORING
1. **Identify Primary Entities**: Find the main subjects (e.g., specific player names, specific city names, specific years, unique award titles).
2. **Target Specific Rows**: Don't just summarize. Create questions that anchor to the most 'unique' or 'extreme' rows in the table.
3. **Keyword Density**: Ensure every generated question contains the **exact proper nouns** (Names, Titles, Dates) found in the table.

=====================
### RULES
1. Output ONLY ONE valid JSON object with the following schema:

{{
  "feta_id": {table_id},
  "table_name": "{table_name}",
  "questions": [
    {{
        "question": "<Discriminative, self-contained question containing specific proper nouns>",
        "thought": "<Step-by-step reasoning: Explain the logic of how you combined specific data points and columns to make this question unique>",
        "relevant_columns": ["col1", "col2"]
    }},
    ...
  ]
}}

2. Generate **exactly 5** questions with different levels of granularity:
   - **Q1 (Entity-Focus):** Targets the main person/subject and their most notable record or identity.
   - **Q2 (Temporal-Focus):** Links a specific year/season/date to a unique result in that period.
   - **Q3 (Comparison/Superlative):** Focuses on "highest", "lowest", "first", or "total" values while mentioning the subject name.
   - **Q4 (Multi-Column Integration): Requires combining data from several different columns to form a complete answer.
   - **Q5 (High-Complexity Reasoning): A discriminative question that links multiple attributes to uniquely identify a specific data point.

3. **Question Constraints**:
   - **NEVER** use generic phrases like "in this table" or "the provided data".
   - **ALWAYS** include specific subject names.
   - **BE SPECIFIC.**
   - **IMPORTANT:** Ensure valid JSON format. Escape backslashes (e.g., use \\\\ instead of \\).

4. **Column Selection Guidelines (Split-Specific)**:
   - **Target Standard is {target_c} columns**: Most questions should aim for this number to maintain optimal signal density.
   - Use fewer columns for simple facts and more columns (up to {max_c}) only when complex reasoning requires it.
   - Do Not list more than {max_c} columns to prevent information noise.

5. Output ONLY the JSON object. No markdown code fences, no extra text."""

SYS_PROMPT_2_REASONING = """You are an expert in 'Table Reasoning and Evidence Extraction'. 
Your goal is to analyze a given question against a table and provide a structured logical path to the answer.

=====================
### REASONING PROTOCOL: EVIDENCE CHAINING
1. **Analyze Question Intent**: Identify what the question is specifically asking for (e.g., a single value, a comparison, a trend).
2. **Column Filtering**: Locate all columns that contain necessary information. Distinguish between 'Filtering Columns' and 'Target Columns'.
3. **Logic Formulation**: Describe the step-by-step process:
   - Step 1: Identify the anchor entity.
   - Step 2: Cross-reference across the identified row.
   - Step 3: Extract the specific cell value requested.
=====================

### RULES
1. **Thought Structure**: The 'thought' must be concise but explain the 'linkage' (e.g., "Linking the artist to the 'Sales' column to find the maximum value").
2. **Relevant Columns Guidelines**:
   - List only the columns strictly necessary for answering.
   - **Limit**: Maximum 7 columns.
3. **No Generalization**: Do not say "I looked at the table." Say "I identified the row where 'Name' is 'Bobby Rahal'."
4. Output should strictly follow this format:
Thought: [Your logical reasoning here]
Relevant Columns: ["Col1", "Col2", ...]"""


# ==========================================
# 2. Dataset Formatting (Token-Aware Truncation)
# ==========================================
def safe_format_chatml(sys_prompt, user_content, assistant_content, tokenizer, max_input_len=2048):
    """
    Safely truncates ONLY the input (System + User) to max_input_len tokens,
    preventing ChatML formatting breakage and ensuring the output doesn't get truncated by SFTTrainer.
    """
    # 1. Construct the raw input prompt without closing the user tag yet
    raw_input_text = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_content}"
    
    # 2. Tokenize the input text
    input_ids = tokenizer.encode(raw_input_text, add_special_tokens=False)
    
    # 3. Truncate if it exceeds max_input_len
    # Reserve some buffer (e.g., 10 tokens) for the closing tags that we will append next
    if len(input_ids) > (max_input_len - 10):
        input_ids = input_ids[:(max_input_len - 10)]
    
    # 4. Decode back to text
    truncated_input_text = tokenizer.decode(input_ids)
    
    # 5. Safely append the closing user tag and the assistant output
    full_text = f"{truncated_input_text}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"
    
    return full_text


def load_and_format_dataset(jsonl_file_path, target_c, max_c, tokenizer, max_input_len=2048):
    formatted_data = []
    
    formatted_sys_prompt_1 = SYS_PROMPT_1_QUERY_GEN.replace("{target_c}", str(target_c)).replace("{max_c}", str(max_c))
    
    table_groups = {}
    task2_raw_data = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            feta_id = item["feta_id"]
            table_name = item["table_name"]
            T_array = item["input"]["table_array"]
            T_str = json.dumps(T_array, ensure_ascii=False)
            
            q_ins = item["input"]["question"]
            r = item["output"]["thought"]
            c_list = item["output"]["relevant_columns"]
            c_str = json.dumps(c_list, ensure_ascii=False)
            
            if feta_id not in table_groups:
                table_groups[feta_id] = {
                    "table_name": table_name,
                    "T_str": T_str,
                    "questions": []
                }
            
            table_groups[feta_id]["questions"].append({
                "question": q_ins,
                "thought": r,
                "relevant_columns": c_list
            })
            
            task2_raw_data.append({
                "T_str": T_str,
                "q_ins": q_ins,
                "r": r,
                "c_str": c_str
            })

    # ---------------------------------------------------------
    # Task 1: Generate prompt 
    # ---------------------------------------------------------
    for feta_id, group_data in table_groups.items():
        expected_json_output = json.dumps({
            "feta_id": feta_id,
            "table_name": group_data["table_name"],
            "questions": group_data["questions"]
        }, ensure_ascii=False, indent=2)

        user_content = f"Metadata: feta_id={feta_id}, table_name={group_data['table_name']}\nTable:\n{group_data['T_str']}"
        
        # Apply token-aware truncation
        text_task1 = safe_format_chatml(
            sys_prompt=formatted_sys_prompt_1,
            user_content=user_content,
            assistant_content=expected_json_output,
            tokenizer=tokenizer,
            max_input_len=max_input_len
        )
        formatted_data.append({"text": text_task1})

    # ---------------------------------------------------------
    # Task 2: Reasoning prompt
    # ---------------------------------------------------------
    for t2_item in task2_raw_data:
        user_content = f"Table:\n{t2_item['T_str']}\n\nQuestion:\n{t2_item['q_ins']}"
        assistant_content = f"Thought: {t2_item['r']}\nRelevant Columns: {t2_item['c_str']}"
        
        # Apply token-aware truncation
        text_task2 = safe_format_chatml(
            sys_prompt=SYS_PROMPT_2_REASONING,
            user_content=user_content,
            assistant_content=assistant_content,
            tokenizer=tokenizer,
            max_input_len=max_input_len
        )
        formatted_data.append({"text": text_task2})
            
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=42) 
    
    return dataset

# ==========================================
# 3. Fine-Tuning Script (LoRA)
# ==========================================
def train_qwen(dataset, tokenizer, model_id):
    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("Loading Qwen 2.5 7B Model...")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={'': local_rank},
    )
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    
    # Data Collator 
    response_template_text = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_text, tokenizer=tokenizer)
    
    # Training Hyperparameters
    training_args = TrainingArguments(
        output_dir="./qwen-table-distill-checkpoint",
        per_device_train_batch_size=2,   
        gradient_accumulation_steps=8,   
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,              
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        bf16=True,                       
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",                
        gradient_checkpointing=True,                               
        gradient_checkpointing_kwargs={"use_reentrant": False}     
    )
    
    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=4096,  # Total sequence length (Input 2048 + Output 2048)
        tokenizer=tokenizer,
        data_collator=collator,
        args=training_args
    )
    
    print("Starting dual-objective distillation training...")
    trainer.train()
    
    # Please update to your actual relative path 
    save_path = "/qwen-table-distill-final"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Training Complete! Model and tokenizer saved to {save_path}")


if __name__ == "__main__":
    # Please update to your actual relative path (generate dataset previously before running this script)
    data_path = "/Quocca/dataset/FeTaQA/.jsonl" 
    model_id = "/Qwen2.5-7B-Instruct" 
    
    TARGET_C = 3
    MAX_C = 7
    MAX_INPUT_LEN = 2048 # limit input tokens to save space for Output
    
    # Load Tokenizer
    print("Loading tokenizer for precise input truncation...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  
    
    # Format dataset with token-aware truncation
    print(f"Formatting dataset with max_input_len={MAX_INPUT_LEN}...")
    train_dataset = load_and_format_dataset(
        jsonl_file_path=data_path, 
        target_c=TARGET_C, 
        max_c=MAX_C, 
        tokenizer=tokenizer, 
        max_input_len=MAX_INPUT_LEN
    )
    print(f"Total training samples (Task 1 + Task 2 combined): {len(train_dataset)}")
    
    # Start Training
    train_qwen(train_dataset, tokenizer, model_id)