import json
import os
import re
from typing import Dict, List, Any
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
from transformers import AutoTokenizer

# CUDA memory optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


class TableQuestionGeneratorVLLM:
    def __init__(self, model_path: str, lora_path: str = None, tensor_parallel_size=2,
                 gpu_memory_utilization=0.9, max_model_len=4096):
        print(f"[Info] Loading Base Model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.lora_path = lora_path
        
        self.llm = LLM(
            model=model_path,
            enable_lora=True if lora_path else False,
            max_loras=1,
            max_lora_rank=64,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16"
        )
        
        if lora_path:
            print(f"[Info] LoRA Adapter Enabled: {lora_path}")
        
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
        for row in content[:30]:
            table_str += f"  * {row}\n"
        return table_str

    # Constructs the system and user prompt for Qwen
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
  "feta_id": {table_id},
  "table_name": "{table_name}",
  "questions": [
    {{
        "question": "<Discriminative question...>",
        "thought": "<Brief reasoning: How you combined the context with specific data points>",
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
   - **NEVER** use generic phrases like "in this table".
   - **ALWAYS** include specific subject names.
   - **BE SPECIFIC.**
   - **IMPORTANT:** Ensure valid JSON format. Escape backslashes (e.g., use \\\\ instead of \\).

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
            
            if start_idx == -1 or end_idx == -1:
                return []
            
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
            
            f_id = data.get("feta_id")
            questions = data.get("questions", [])
            
            flattened = []
            if isinstance(questions, list):
                for q_item in questions:
                    if isinstance(q_item, dict):
                        flattened.append({
                            "feta_id": f_id,
                            "question": q_item.get("question"),
                            "thought": q_item.get("thought"),
                            "relevant_columns": q_item.get("relevant_columns")
                        })
            return flattened

        except Exception:
            return []

    def run_research_pipeline(self, input_path: str, output_path: str, split_config: dict, batch_size: int = 16):
        prompts = []
        with open(input_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                t_id = data.get('feta_id') or data.get('table_id', i)
                t_name = data.get('table_name', f'Table_{t_id}')
                t_input = data.get('input', {})
                t_cols = data.get('table_columns') or t_input.get('table_columns') or (t_input.get('table_array')[0] if t_input.get('table_array') else [])
                content = data.get('table_content') or t_input.get('table_content') or (t_input.get('table_array')[1:] if t_input.get('table_array') else [])
                
                t_str = self.format_table_for_prompt(t_name, t_cols, content)
                prompts.append(self.create_prompt(t_str, t_id, t_name, split_config))

        print(f"[Process] Generating {len(prompts)} tables...")
        
        lora_req = LoRARequest(lora_name="table_adapter", lora_int_id=1, lora_path=self.lora_path) if self.lora_path else None

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
    BASE_MODEL = "Qwen2.5-7B-Instruct" 
    LORA_PATH = "Quocca/model/checkpoint-7600"
    generator = TableQuestionGeneratorVLLM(model_path=BASE_MODEL, lora_path=LORA_PATH, tensor_parallel_size=2)

    # Methodology-specific hyperparameters (max, target): 7/3
    split_settings = {
        "train": {"max": None, "target":  None},
        "dev":   {"max": None, "target":  None},
        "test":  {"max": None, "target":  None}
    }
    
    for split in ["train", "dev", "test"]: 
        cur_input = f"Quocca/dataset/FeTaQA/{split}.jsonl" 
        cur_output = f"/{split}_qwen.jsonl" 
        
        if os.path.exists(cur_input):
            print(f"[Process] Processing {split} split -> {cur_output}")
            config = split_settings[split]
            generator.run_research_pipeline(cur_input, cur_output, split_config=config, batch_size=16)