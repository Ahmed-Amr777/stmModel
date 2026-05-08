"""
Step 5 — Baseline evaluation using pretrained CLAP-asm (no fine-tuning).

Evaluation protocol:
  - Pool  : all test-split function instances
  - Query : each instance in the pool
  - Match : instances with the SAME function name (any opt level)
  - For each query, rank the entire pool by cosine similarity (excluding self)
  - Compute Top-1, Top-5, MRR, same-func avg sim, cross-func avg sim, gap

Output: results/baseline_metrics.json

Usage:
    python scripts/05_baseline_eval.py
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

ROOT       = Path(__file__).parent.parent
TRAIN_DIR  = ROOT / "data/training"
RESULTS    = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "hustcw/clap-asm"


# ──────────────────────────────────────────────
# Load test instances
# ──────────────────────────────────────────────

def load_test_instances() -> list[dict]:
    records = [
        json.loads(line)
        for line in (TRAIN_DIR / "functions.jsonl").read_text().splitlines()
        if json.loads(line)["split"] == "test"
    ]
    print(f"  Test instances : {len(records)}")
    unique_names = len(set(r["name"] for r in records))
    print(f"  Unique functions : {unique_names}")
    return records


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)
    print(f"  Device : {device}\n")
    return tok, mdl, device


@torch.no_grad()
def embed_batch(instructions_list: list[list[str]], tok, mdl, device) -> torch.Tensor:
    """Embed a batch of instruction lists. Returns (B, H) L2-normalised tensor."""
    dicts = [{str(i): instr for i, instr in enumerate(instrs)}
             for instrs in instructions_list]
    inputs = tok(dicts, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return mdl(**inputs).cpu()


def embed_all(records: list[dict], tok, mdl, device, batch_size=32) -> torch.Tensor:
    """Embed all records in batches. Returns (N, H) tensor."""
    all_embs = []
    n = len(records)
    for i in range(0, n, batch_size):
        batch = records[i: i + batch_size]
        embs  = embed_batch([r["instructions"] for r in batch], tok, mdl, device)
        all_embs.append(embs)
        print(f"  Embedded {min(i + batch_size, n)}/{n}", end="\r")
    print()
    return torch.cat(all_embs, dim=0)   # (N, H)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(records: list[dict], embeddings: torch.Tensor) -> dict:
    """
    For every instance as query, rank the pool (excluding self).
    Correct = same function name, any opt level.
    """
    names  = [r["name"] for r in records]
    n      = len(records)

    # Full cosine similarity matrix (already L2-normalised)
    sim_matrix = (embeddings @ embeddings.T).numpy()   # (N, N)

    top1_hits  = 0
    top5_hits  = 0
    mrr_sum    = 0.0
    same_sims  = []
    cross_sims = []

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -2.0          # exclude self

        ranked = np.argsort(sims)[::-1]   # descending

        # same / cross avg similarity (excluding self)
        for j in range(n):
            if j == i:
                continue
            if names[j] == names[i]:
                same_sims.append(sim_matrix[i, j])
            else:
                cross_sims.append(sim_matrix[i, j])

        # retrieval metrics: first correct match
        first_rank = None
        for rank, j in enumerate(ranked, start=1):
            if names[j] == names[i]:
                first_rank = rank
                break

        if first_rank is None:
            continue   # singleton in pool (shouldn't happen after filtering)

        if first_rank == 1:
            top1_hits += 1
        if first_rank <= 5:
            top5_hits += 1
        mrr_sum += 1.0 / first_rank

    total = n
    same_avg  = float(np.mean(same_sims))
    cross_avg = float(np.mean(cross_sims))

    return {
        "top1_accuracy":      round(top1_hits / total, 4),
        "top5_accuracy":      round(top5_hits / total, 4),
        "mrr":                round(mrr_sum   / total, 4),
        "same_func_avg_sim":  round(same_avg,  4),
        "cross_func_avg_sim": round(cross_avg, 4),
        "gap":                round(same_avg - cross_avg, 4),
        "total_queries":      total,
        "top1_hits":          top1_hits,
        "top5_hits":          top5_hits,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    t0 = time.time()

    print("Loading test instances...")
    records = load_test_instances()

    tok, mdl, device = load_model()

    print("Embedding test instances...")
    embeddings = embed_all(records, tok, mdl, device)
    print(f"  Embedding matrix : {embeddings.shape}\n")

    print("Computing metrics...")
    metrics = compute_metrics(records, embeddings)

    elapsed = time.time() - t0
    metrics["eval_time_seconds"] = round(elapsed, 1)
    metrics["model"] = MODEL_NAME
    metrics["split"] = "test"

    out = RESULTS / "baseline_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))

    print("\n" + "=" * 45)
    print("BASELINE RESULTS (pretrained CLAP-asm)")
    print("=" * 45)
    print(f"  Top-1 accuracy    : {metrics['top1_accuracy']:.4f}  ({metrics['top1_hits']}/{metrics['total_queries']})")
    print(f"  Top-5 accuracy    : {metrics['top5_accuracy']:.4f}  ({metrics['top5_hits']}/{metrics['total_queries']})")
    print(f"  MRR               : {metrics['mrr']:.4f}")
    print(f"  Same-func avg sim : {metrics['same_func_avg_sim']:.4f}")
    print(f"  Cross-func avg sim: {metrics['cross_func_avg_sim']:.4f}")
    print(f"  Gap               : {metrics['gap']:.4f}")
    print(f"\n  Saved -> {out.relative_to(ROOT)}")
    print(f"  Time   : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
