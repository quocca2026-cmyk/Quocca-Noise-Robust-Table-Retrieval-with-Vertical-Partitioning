import json
import os
import re
from typing import Dict, List, Any
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer

# CUDA Memory Optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

class TableQuestionGeneratorVLLM:
    def __init__(self, model_path: str, tensor_parallel_size=2,
                 gpu_memory_utilization=0.9, max_model_len=4096):
        print(f"Loading Local Model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
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

    # Constructs the system and user prompt for Llama-3
    def create_prompt(self, table_str: str, table_id: int, table_name: str, split_config: dict) -> str:
        max_c = split_config['max']
        target_c = split_config['target']

        system_prompt = f"""You are an expert in 'Discriminative Table Retrieval'.
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
   - **Q4 (Multi-Column Integration): Requires combining data from several different columns to form a complete answer.
   - **Q5 (High-Complexity Reasoning): A discriminative question that links multiple attributes to uniquely identify a specific data point.

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
        user_prompt = f"""### INPUT TABLE DATA
{table_str}

Generate the JSON object with 5 discriminative questions as specified. 
"""
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # 2048 token truncation
        tokens = self.tokenizer.encode(full_prompt)
        if len(tokens) > 2048:
            full_prompt = self.tokenizer.decode(tokens[:2048], skip_special_tokens=False)
            if not full_prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"):
                if "<|start_header_id|>assistant<|end_header_id|>" not in full_prompt[-50:]:
                    full_prompt += "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
        return full_prompt

    
    # Parser LLM output and handles common JSON formatting errors
    def parse_and_flatten_json(self, raw_text: str) -> List[Dict]:
        """Flatten nested JSON to JSONL structure"""
        try:
            # Strip markdown and whitespace
            text = raw_text.replace("```json", "").replace("```", "").strip()
            
            # Extract main json block
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx == -1 or end_idx == -1: return []
            
            # Fix common escape character issues in generated text
            json_str = text[start_idx:end_idx+1]
            
            # Removing commas
            json_str = re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', json_str)
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
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            for batch_idx in tqdm(range(0, len(prompts), batch_size)):
                batch_prompts = prompts[batch_idx:batch_idx + batch_size]
                outputs = self.llm.generate(batch_prompts, self.sampling_params)

                for output in outputs:
                    raw_text = output.outputs[0].text.strip()
                    flat_questions = self.parse_and_flatten_json(raw_text)
                    for q in flat_questions:
                        out_f.write(json.dumps(q, ensure_ascii=False) + "\n")

# --- Execution ---
if __name__ == "__main__":
    MODEL_PATH = "Llama-3.3-70B-Instruct" 
    generator = TableQuestionGeneratorVLLM(model_path=MODEL_PATH, tensor_parallel_size=2)

    # Methodology-specific hyperparameters (max, target): 6/2 for 'two', 8/3 for 'three'
    split_settings = {
        "two":   {"max": None, "target": None},
        "three": {"max": None, "target": None}
    }

    # Iterate over splits
    for split in ["two", "three"]: 
        cur_input = f"Quocca/dataset/MMQA/all_tables_unique_{split}.jsonl" 
        cur_output = f"{split}_llama.jsonl" 
        
        if not os.path.exists(cur_input):
            print(f"Skip: {cur_input} not found.")
            continue
            
        print(f"Processing {split} split -> {cur_output}")
        config = split_settings[split]
        generator.run_research_pipeline(cur_input, cur_output, split_config=config, batch_size=16)