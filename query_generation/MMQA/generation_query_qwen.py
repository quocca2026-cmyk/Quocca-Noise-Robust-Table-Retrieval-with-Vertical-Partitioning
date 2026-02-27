import json
import os
import re
from typing import Dict, List, Any
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
from transformers import AutoTokenizer

# CUDA Memory Optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

class TableQuestionGeneratorVLLM:
    def __init__(self, base_model_path: str, lora_path: str, tensor_parallel_size=2,
                 gpu_memory_utilization=0.9, max_model_len=4096):
        print(f"Loading Base Model from: {base_model_path}")
        print(f"Loading LoRA Adapter from: {lora_path}")
        
        # Load tokenizer for precise token truncation
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.lora_path = lora_path
        self.llm = LLM(
            model=base_model_path,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=64,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16"
        )
        
        # Sampling config for generation
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

    # Serialize table data into a text prompt representation
    def format_table_for_prompt(self, table_name: str, columns: List[str],
                                content: List[List[Any]]) -> str:
        table_str = f"### [TABLE: {table_name}]\n"
        table_str += f"- Columns: {', '.join(columns)}\n"
        table_str += "- Key Data Rows:\n"
        # Increased to ensure enough content before 2048-token truncation
        for row in content[:30]:
            table_str += f"  * {row}\n"
        return table_str

    def create_prompt(self, table_str: str, table_id: int, table_name: str, split_config: dict) -> str:
        max_c = split_config['max']
        target_c = split_config['target']

        system_prompt = f"""You are an expert in 'Discriminative Table Retrieval'.
Your goal is to generate 5 distinct questions that act as a **unique signature** for this table.

=====================
### STRATEGY: DISCRIMINATIVE ANCHORING
1. **Identify Primary Entities**: Find the main subjects (e.g., specific player names, specific city names).
2. **Target Specific Rows**: Don't just summarize. Create questions that anchor to the most 'unique' or 'extreme' rows.
3. **Keyword Density**: Ensure every generated question contains the **exact proper nouns**.

=====================
### RULES
1. Output ONLY ONE valid JSON object with the following schema:

{{
  "mmqa_id": {table_id},
  "table_name": "{table_name}",
  "questions": [
    {{
        "question": "<Discriminative, self-contained question containing specific proper nouns>",
        "relevant_columns": ["col1", "col2"]
    }},
    ...
  ]
}}

2. Generate **exactly 5** questions with different levels of granularity:
   - **Q1 (Entity-Focus):** Targets the main person/subject and their most notable record or identity.
   - **Q2 (Temporal-Focus):** Links a specific year/season/date to a unique result in that period.
   - **Q3 (Comparison/Superlative):** Focuses on "highest", "lowest", "first", or "total" values while mentioning the subject name.
   - **Q4 (Multi-Column Integration):** Requires combining data from several different columns to form a complete answer.
   - **Q5 (High-Complexity Reasoning):** A discriminative question that links multiple attributes to uniquely identify a specific data point.

3. **Question Constraints**:
   - **ALWAYS** include specific subject names (Proper Nouns).
   - **NEVER** use generic phrases like "in this table".
   - **IMPORTANT:** Ensure valid JSON format.

4. **Column Selection Guidelines (Split-Specific)**:
   - **Target Standard is {target_c} columns**: Most questions should aim for this number to maintain optimal signal density.
   - Use fewer columns for simple facts and more columns (up to {max_c}) only when complex reasoning requires it.
   - Do Not list more than {max_c} columns to prevent information noise.

5. Output ONLY the JSON object. No markdown code fences, no extra text.
"""
        user_prompt = f"### INPUT TABLE DATA\n{table_str}\n\nGenerate the JSON object with 5 discriminative questions as specified."
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 2048 token truncation
        tokens = self.tokenizer.encode(full_prompt)
        if len(tokens) > 2048:
            full_prompt = self.tokenizer.decode(tokens[:2048], skip_special_tokens=False)
            if not full_prompt.endswith("<|im_start|>assistant\n"):
                if "<|im_start|>assistant\n" not in full_prompt[-25:]:
                    full_prompt += "\n<|im_start|>assistant\n"
        return full_prompt

    # Parser LLM output and handles common JSON formatting errors
    def parse_and_flatten_json(self, raw_text: str) -> List[Dict]:
        try:
            # Strip markdown and whitespace
            text = raw_text.replace("```json", "").replace("```", "").strip()
            
            # Extract main json block
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx == -1 or end_idx == -1: return []
            
            json_str = text[start_idx:end_idx+1]
            
            # Fix common escape character issues in generated text
            json_str = re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', json_str)
            # Removing commas
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)

            try:
                data = json.loads(json_str, strict=False)
            except:
                return []
            
            m_id = data.get("mmqa_id")
            questions = data.get("questions", [])
            
            flattened = []
            if isinstance(questions, list):
                for q_item in questions:
                    if isinstance(q_item, dict):
                        flattened.append({
                            "mmqa_id": m_id,
                            "question": q_item.get("question"),
                            "relevant_columns": q_item.get("relevant_columns")
                        })
            return flattened
        except Exception:
            return []

    def run_research_pipeline(self, input_path: str, output_path: str, split_config: dict, batch_size: int = 16):
        prompts = []
        with open(input_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                t_id = data.get('table_id', i)
                t_name = data.get('table_name', f'Table_{t_id}')
                t_cols = data.get('table_columns', [])
                content = data.get('table_content', [])
                
                t_str = self.format_table_for_prompt(t_name, t_cols, content)
                prompts.append(self.create_prompt(t_str, t_id, t_name, split_config))

        print(f"Generating questions for {len(prompts)} tables...")
        
        # Define LoRA request with explicit argument names
        lora_req = LoRARequest(lora_name="adapter", lora_int_id=1, lora_path=self.lora_path)

        with open(output_path, "w", encoding="utf-8") as out_f:
            for batch_idx in tqdm(range(0, len(prompts), batch_size)):
                batch_prompts = prompts[batch_idx:batch_idx + batch_size]
                outputs = self.llm.generate(batch_prompts, self.sampling_params, lora_request=lora_req)

                for output in outputs:
                    raw_text = output.outputs[0].text.strip()
                    flat_questions = self.parse_and_flatten_json(raw_text)
                    for q in flat_questions:
                        out_f.write(json.dumps(q, ensure_ascii=False) + "\n")

# --- Execution ---
if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct" 
    LORA_PATH = "Quocca/model/checkpoint-7600"
    
    generator = TableQuestionGeneratorVLLM(BASE_MODEL, LORA_PATH)

    # Split settings
    # Methodology-specific hyperparameters (max, target): 6/2 for 'two', 8/3 for 'three'
    split_settings = {
        "two":   {"max": None, "target": None},
        "three": {"max": None, "target": None}
    }

    # Iterate over splits
    for split in ["two", "three"]: 
        cur_input = f"Quocca/dataset/MMQA/all_tables_unique_{split}.jsonl" 
        cur_output = f"/{split}_qwen.jsonl" 
        
        if os.path.exists(cur_input):
            print(f"[Process] Processing {split} split -> {cur_output}")
            config = split_settings[split]
            generator.run_research_pipeline(cur_input, cur_output, split_config=config, batch_size=16)