import sys
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------
# Argument Parsing & Configuration
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate Dense Retriever Models on FeTaQA (Mixed Corpus)")

parser.add_argument("--row_limit", type=int, default=5, help="Number of rows to include in markdown")
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
def create_markdown(columns, content, row_limit=5):
    if not columns: return ""
    
    if row_limit > 0:
        content = content[:row_limit]
        
    header = "| " + " | ".join(map(str, columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in content])
    return f"{header}\n{sep}\n{body}"

def calculate_metrics(q_embs, c_embs, gold_ids, corpus_ids, top_k_list=[1, 5, 10]):
    if c_embs.dtype in [torch.float16, torch.bfloat16]:
        sims = torch.mm(q_embs.float(), c_embs.float().T).cpu().numpy()
    else:
        sims = torch.mm(q_embs, c_embs.T).cpu().numpy()
    
    num_queries = len(gold_ids)
    hits = {k: 0 for k in top_k_list}
    mrr_sum = 0
    max_k = max(top_k_list)
    
    k_part = min(50, sims.shape[1])
    part_indices = np.argpartition(-sims, k_part, axis=1)[:, :k_part]
    
    for i in range(num_queries):
        row_indices = part_indices[i]
        row_scores = sims[i, row_indices]
        sorted_local_idx = np.argsort(-row_scores)
        top_indices = row_indices[sorted_local_idx]
        
        seen = set()
        final_top_ids = []
        for idx in top_indices:
            real_id = str(corpus_ids[idx]).split('_')[0]
            if real_id not in seen:
                final_top_ids.append(real_id)
                seen.add(real_id)
            if len(final_top_ids) >= max_k:
                break
        
        target_id = str(gold_ids[i])
        for k in top_k_list:
            if target_id in final_top_ids[:k]:
                hits[k] += 1
        
        if target_id in final_top_ids:
            rank = final_top_ids.index(target_id) + 1
            mrr_sum += (1.0 / rank)
            
    results = {f"Recall@{k}": (hits[k] / num_queries) * 100 for k in top_k_list}
    results["MRR"] = (mrr_sum / num_queries)
    return results

def load_all_queries_and_gold(base_dir, splits=["test", "dev", "train"]):
    all_queries, all_gold_ids = [], []
    
    print(f"[Process] Loading unified queries (Splits: {splits}) ...")
    for s in splits:
        path = os.path.join(base_dir, f"{s}.jsonl")
        
        if not os.path.exists(path):
            print(f"   [Warning] {path} not found. Skipping.")
            continue
            
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    q = d.get('input', {}).get('question') or d.get('question')
                    f_id = str(d.get('feta_id'))
                    if q:
                        all_queries.append(q)
                        all_gold_ids.append(f_id)
                        count += 1
                except: continue
        print(f"   [Success] {s.upper()}: {count} queries loaded.")
        
    return all_queries, all_gold_ids

def load_corpus_from_files(file_paths, is_subset=False, row_limit=5):
    markdowns, ids = [], []
    for fp in file_paths:
        if not os.path.exists(fp): continue
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    f_id = str(d.get('feta_id'))
                
                    t_arr = d.get('input', {}).get('table_array') or d.get('table_array')
                    t_cols = d.get('table_columns')
                    t_cont = d.get('table_content')
                    if not t_cols and t_arr: t_cols, t_cont = t_arr[0], t_arr[1:]
                    
                    md = create_markdown(t_cols, t_cont, row_limit=row_limit)

                    if is_subset:
                        anchor = d.get('input', {}).get('question') or d.get('question')
                        if anchor:
                            md = f"Question: {anchor}\nTable Content:\n{md}"
                    
                    markdowns.append(md)
                    ids.append(f_id)
                except: continue
    return markdowns, ids

# -----------------------------------------------------------
# Main Execution
# -----------------------------------------------------------
if __name__ == "__main__":
    
    # Original Data Directory 
    ORIG_DIR = "Quocca/dataset/FeTaQA" 
    
    # Please update to your actual relative path 
    SUBSET_DIR = "/[YOUR_SUBSET_FILE].jsonl" 
    
    SPLITS = ["test", "dev", "train"]
    
    # Please update to your actual relative path 
    SUBSET_SUFFIX = ".jsonl"

    # Load and Encode Queries
    queries, gold_ids = load_all_queries_and_gold(ORIG_DIR, SPLITS)
    q_kwargs = selected_config['query_kwargs']
    q_embs = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True, **q_kwargs)
    print(f"   [Info] Query encoding complete: {len(queries)} queries")
    
    batch_size = selected_config['batch_size']

    # Load Original and Subset Corpora
    print("\n" + "="*95)
    print(f"[Evaluation] Mixed Corpus (Original + Subset) | Row limit: {args.row_limit}")
    print("="*95)
    
    orig_files = [os.path.join(ORIG_DIR, f"{s}.jsonl") for s in SPLITS]
    sub_files = [os.path.join(SUBSET_DIR, f"{s}_{SUBSET_SUFFIX}") for s in SPLITS]
    
    orig_mds, orig_ids = load_corpus_from_files(orig_files, is_subset=False, row_limit=args.row_limit)
    subset_mds, subset_ids = load_corpus_from_files(sub_files, is_subset=True, row_limit=args.row_limit)
    
    # Combine and Encode Target Corpus
    combined_mds = orig_mds + subset_mds
    combined_ids = orig_ids + subset_ids
    
    print(f"   [Info] Encoding {len(combined_mds)} total tables...")
    corpus_embs = model.encode(combined_mds, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)
    
    metrics = calculate_metrics(q_embs, corpus_embs, gold_ids, combined_ids)
    
    # Final Evaluation Report
    print("\n" + "#"*90)
    print(f"[FINAL REPORT] FeTaQA Evaluation | Mixed Corpus | Rows: {args.row_limit}")
    print(f"{'Method':<18} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8} | {'MRR':<10}")
    print("-" * 90)
    print(f"{'Mixed Dataset':<18} | {metrics['Recall@1']:>6.2f}% | {metrics['Recall@5']:>6.2f}% | {metrics['Recall@10']:>6.2f}% | {metrics['MRR']:>10.4f}")
    print("#"*90 + "\n")