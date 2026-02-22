# Quocca: Dynamic Column Partitioning for Table Retrieval using LLMs

This repository contains the anonymous implementation for the paper:  
📄 **[Quocca: Noise-Robust Table Retrieval with Vertical Partitioning]** (Under Review)

> We propose **Quocca** (**Qu**ery-**o**riented **c**olumn-wise **c**ontextual semantic **a**lignment), a simple yet effective table retrieval framework that shifts the paradigm from holistic table encoding to vertical partitioning-based sub-table encoding.
---

## 🗂️ Repository Contents

Our codebase is organized into three main modules:

- 📁 `Dataset/`  
Contains the table corpora (FeTaQA, MMQA) used in our experiments.

- 📁 `Evaluation/`  
The text-based dense retriever pipeline and evaluation scripts to calculate Recall@k and MRR metrics. It supports the text embedding model (e.g., `stella`) to compute semantic similarities between queries and text-serialized (Markdown) partitioned sub-tables.

- 📁 `Model/`  
Contains the LoRA checkpoints obtained via knowledge distillation. It also includes the specific prompt templates used during the fine-tuning process to train these models.

- 📁 `Query_Generation/`  
LLM prompt templates used for dynamic column partitioning and query generation to filter semantic noise from wide tables.
---