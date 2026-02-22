import sys
import os
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------
# Argument Parsing & Configuration
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate Dense Retriever Models on MMQA (Mixed Corpus)")

parser.add_argument("--row_limit", type=int, default=0, help="Number of rows to include in markdown")
args = parser.parse_args()

MODEL_CONFIG = {
    "stella": {
        "path": "../models/stella_v5", # Please update to your actual relative path
        "query_kwargs": {"prompt_name": "s2p_query"},
        "batch_size": 32,
        "trust_remote_code": True
    }
}

selected_config = MODEL_CONFIG["stella"]

# -----------------------------------------------------------
# Setup
# -----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Info] Loading Model: STELLA")

model = SentenceTransformer(
    selected_config['path'], 
    trust_remote_code=selected_config.get('trust_remote_code', True), 
    device=DEVICE,
    model_kwargs=selected_config.get('model_kwargs', {})
)

model.max_seq_length = 512

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
def create_markdown(columns, content, row_limit=0):
    if not columns: return ""
    
    if row_limit > 0:
        content = content[:row_limit]
        
    header = "| " + " | ".join(map(str, columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in content])
    return f"{header}\n{sep}\n{body}"

# Strict Recall requires ALL gold tables to be found
def calculate_strict_metrics(q_embs, c_embs, queries, gold_map, corpus_ids, top_k_list=[1, 5, 10]):

    if c_embs.dtype in [torch.float16, torch.bfloat16]:
        sims = torch.mm(q_embs.float(), c_embs.float().T).cpu().numpy()
    else:
        sims = torch.mm(q_embs, c_embs.T).cpu().numpy()
        
    num_queries = len(queries)
    hits = {k: 0 for k in top_k_list}
    mrr_sum = 0
    max_k = max(top_k_list)
    
    k_part = min(200, sims.shape[1])
    part_indices = np.argpartition(-sims, k_part, axis=1)[:, :k_part]
    
    for i in range(num_queries):
        q_text = queries[i]
        row_indices = part_indices[i]
        row_scores = sims[i, row_indices]
        
        sorted_local_idx = np.argsort(-row_scores)
        top_indices = row_indices[sorted_local_idx]
        
        retrieved_real_ids = []
        seen = set()
        
        for idx in top_indices:
            full_id = str(corpus_ids[idx])
            real_id = full_id.split(':::')[0] # Extract ID_suffix
            
            if real_id not in seen:
                retrieved_real_ids.append(real_id)
                seen.add(real_id)
            if len(retrieved_real_ids) >= max_k:
                break
                
        targets = gold_map.get(q_text, set())
        if not targets: continue
        
        # Strict Recall
        for k in top_k_list:
            top_k_retrieved = set(retrieved_real_ids[:k])
            if targets.issubset(top_k_retrieved):
                hits[k] += 1
                
        # MRR
        first_hit_rank = float('inf')
        for target in targets:
            if target in retrieved_real_ids:
                rank = retrieved_real_ids.index(target) + 1
                if rank < first_hit_rank:
                    first_hit_rank = rank
        
        if first_hit_rank != float('inf'):
            mrr_sum += (1.0 / first_hit_rank)

    results = {f"S-Recall@{k}": (hits[k] / num_queries) * 100 for k in top_k_list}
    results["MRR"] = (mrr_sum / num_queries)
    return results

def load_mmqa_gold(path_tuples: List[Tuple[str, str]]):
    gold_map = defaultdict(set)
    queries = []
    print(f"[Process] Loading MMQA Gold Mappings...")
    for f_path, suffix in path_tuples:
        if not os.path.exists(f_path): 
            print(f"   [Warning] {f_path} not found.")
            continue
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    raw_id = str(d.get('mmqa_id') or d.get('table_id'))
                    unique_table_id = f"{raw_id}_{suffix}"
                    for q_item in d.get('related_gold_questions', []):
                        q_text = q_item.get('gold_question')
                        if q_text:
                            gold_map[q_text].add(unique_table_id)
                            # Removing duplicates
                            if q_text not in queries:
                                queries.append(q_text)
                except: continue
    return queries, gold_map

def load_mmqa_corpus_files(path_tuples: List[Tuple[str, str]], is_subset=False, row_limit=0):
    markdowns, ids = [], []
    label = "Subset" if is_subset else "Original"
    print(f"   [Process] Loading {label} Tables...")
    for f_path, suffix in path_tuples:
        if not os.path.exists(f_path):
            print(f"   [Warning] {f_path} not found.")
            continue
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if is_subset:
                        raw_id = str(d.get('mmqa_id'))
                        inp = d.get('input', {})
                        t_cols = inp.get('table_array', [[]])[0]
                        t_cont = inp.get('table_array', [[], []])[1:]
                        q_text = inp.get('question') or d.get('question')
                    else:
                        raw_id = str(d.get('table_id') or d.get('mmqa_id'))
                        t_cols = d.get('table_columns', [])
                        t_cont = d.get('table_content', [])
                        q_text = None

                    base_id = f"{raw_id}_{suffix}"
                    source_tag = "sub" if is_subset else "orig"
                    final_unique_id = f"{base_id}:::{source_tag}"
                    
                    if t_cols:
                        md = create_markdown(t_cols, t_cont, row_limit=row_limit)
                        if q_text: md = f"Question: {q_text}\nTable Content:\n{md}"
                        
                        markdowns.append(md)
                        ids.append(final_unique_id)
                except: continue
    return markdowns, ids

# -----------------------------------------------------------
# Main Execution
# -----------------------------------------------------------
if __name__ == "__main__":
    
    # Original Data Directory 
    BASE_DIR = "Quocca/dataset/MMQA" 
    
    GOLD_PATHS = [
        (f"{BASE_DIR}/gold_mapping/mmqa_id_to_gold_mapping_three.jsonl", "3h"),
        (f"{BASE_DIR}/gold_mapping/mmqa_id_to_gold_mapping_two.jsonl", "2h")
    ]
    
    ORIG_PATHS = [
        (f"{BASE_DIR}/original_dataset/all_tables_unique_three.jsonl", "3h"),
        (f"{BASE_DIR}/original_dataset/all_tables_unique_two.jsonl", "2h")
    ]
    
    # Please update to your actual relative path 
    SUBSET_PATHS = [
        (f"{BASE_DIR}/[YOUR_3H_SUBSET_FILE].jsonl", "3h"),
        (f"{BASE_DIR}/[YOUR_2H_SUBSET_FILE].jsonl", "2h")
    ]

    # Load and Encode Queries
    queries, gold_map = load_mmqa_gold(GOLD_PATHS)
    q_kwargs = selected_config['query_kwargs']
    q_embs = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True, **q_kwargs)
    print(f"   [Info] Query encoding complete: {len(queries)} queries")
    
    batch_size = selected_config['batch_size']

    # Load Original and Subset Corpora
    print("\n" + "="*95)
    print(f"[Evaluation] Mixed Corpus (Original + Subset) | Row limit: {args.row_limit}")
    print("="*95)
    
    orig_mds, orig_ids = load_mmqa_corpus_files(ORIG_PATHS, is_subset=False, row_limit=args.row_limit)
    subset_mds, subset_ids = load_mmqa_corpus_files(SUBSET_PATHS, is_subset=True, row_limit=args.row_limit)
    
    # Combine and Encode Target Corpus
    combined_mds = orig_mds + subset_mds
    combined_ids = orig_ids + subset_ids
    
    corpus_embs = model.encode(combined_mds, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)
    
    metrics = calculate_strict_metrics(q_embs, corpus_embs, queries, gold_map, combined_ids)
    
    # Final Evaluation Report
    print("\n" + "#"*90)
    print(f"[FINAL REPORT] MMQA Strict Recall | Mixed Corpus | Rows: {args.row_limit}")
    print(f"{'Dataset':<18} | {'S-R@1':<8} | {'S-R@5':<8} | {'S-R@10':<8} | {'MRR':<10}")
    print("-" * 90)
    print(f"{'Mixed Dataset':<18} | {metrics['S-Recall@1']:>6.2f}% | {metrics['S-Recall@5']:>6.2f}% | {metrics['S-Recall@10']:>6.2f}% | {metrics['MRR']:>10.4f}")
    print("#"*90 + "\n")