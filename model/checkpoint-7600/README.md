---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- knowledge-distillation
- table-retrieval
- student-model
---

# Quocca: Distilled Student Model (Section 3.3)

This model card contains the LoRA adapter weights for the **lightweight student model** described in **Section 3.3 (Instructional Distillation for Scalability)** of our double-anonymous submission.

## Model Details
- **Developed by:** Anonymous Authors
- **Model Type:** PEFT LoRA Adapter (Student Model)
- **Base Model (Student):** `Qwen/Qwen2.5-7B-Instruct`
- **Teacher Model:** `meta-llama/Llama-3.3-70B-Instruct` (Used to generate distillation triplets)
- **Task:** Instructional Distillation (Query Generation & Reasoning Partitioning)
- **LoRA Hyperparameters:** `r=64`, `lora_alpha=128`, `target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`

## Training Data

The model was fine-tuned using the FeTaQA dataset. The specific training data utilized can be found in this repository at:
* `dataset/FeTaQA/train.jsonl`

## How to Load

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load Base Model
base_model_id = "Qwen/Qwen2.5-7B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", torch_dtype=torch.bfloat16)

# 2. Apply Distilled LoRA Adapter
adapter_path = "./model/checkpoint-7600" 
student_model = PeftModel.from_pretrained(base_model, adapter_path)

print("Section 3.3 Student Model Loaded!")