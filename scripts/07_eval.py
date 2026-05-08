"""
Step 7 — Evaluate the fine-tuned CLAP-asm model.

Same protocol as 05_baseline_eval.py but loads LoRA adapters on top of the
pretrained model instead of the vanilla pretrained weights.

Output: results/finetuned_metrics.json

Usage:
    python scripts/07_eval.py
    python scripts/07_eval.py --adapters models/finetuned/lora_adapters
"""

import sys
import json
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

ROOT       = Path(__file__).parent.parent
TRAIN_DIR  = ROOT / "data/training"
RESULTS    = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME   = "hustcw/clap-asm"
DEFAULT_ADAPTERS = ROOT / "models/finetuned/lora_adapters"


# ──────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────

def load_test_instances() -> list[dict]:
    records = [
        json.loads(line)
        for line in (TRAIN_DIR / "functions.jsonl").read_text().splitlines()
        if json.loads(line)["split"] == "test"
    ]
    print(f"  Test instances   : {len(records)}")
    print(f"  Unique functions : {len(set(r['name'] for r in records))}")
    return records


def load_model(adapters_path: Path):
    print(f"Loading base model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"Loading LoRA adapters: {adapters_path.relative_to(ROOT)}")
    mdl = PeftModel.from_pretrained(mdl, str(adapters_path))
    mdl = mdl.merge_and_unload()   # merge LoRA into weights for faster inference
    mdl.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)
    print(f"  Device : {device}\n")
    return tok, mdl, device


# ──────────────────────────────────────────────
# Embed
# ──────────────────────────────────────────────

@torch.no_grad()
def embed_batch(instructions_list, tok, mdl, device):
    dicts  = [{str(i): instr for i, instr in enumerate(instrs)}
              for instrs in instructions_list]
    inputs = tok(dicts, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return mdl(**inputs).cpu()


def embed_all(records, tok, mdl, device, batch_size=32):
    all_embs = []
    n = len(records)
    for i in range(0, n, batch_size):
        batch = records[i: i + batch_size]
        embs  = embed_batch([r["instructions"] for r in batch], tok, mdl, device)
        all_embs.append(embs)
        print(f"  Embedded {min(i + batch_size, n)}/{n}", end="\r")
    print()
    return torch.cat(all_embs, dim=0)


# ──────────────────────────────────────────────
# Metrics  (identical logic to baseline eval)
# ──────────────────────────────────────────────

def compute_metrics(records, embeddings) -> dict:
    names      = [r["name"] for r in records]
    n          = len(records)
    sim_matrix = (embeddings @ embeddings.T).numpy()

    top1_hits = 0
    top5_hits = 0
    mrr_sum   = 0.0
    same_sims  = []
    cross_sims = []

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -2.0
        ranked = np.argsort(sims)[::-1]

        for j in range(n):
            if j == i:
                continue
            if names[j] == names[i]:
                same_sims.append(sim_matrix[i, j])
            else:
                cross_sims.append(sim_matrix[i, j])

        first_rank = None
        for rank, j in enumerate(ranked, start=1):
            if names[j] == names[i]:
                first_rank = rank
                break

        if first_rank is None:
            continue

        if first_rank == 1:
            top1_hits += 1
        if first_rank <= 5:
            top5_hits += 1
        mrr_sum += 1.0 / first_rank

    same_avg  = float(np.mean(same_sims))
    cross_avg = float(np.mean(cross_sims))

    return {
        "top1_accuracy":      round(top1_hits / n, 4),
        "top5_accuracy":      round(top5_hits / n, 4),
        "mrr":                round(mrr_sum   / n, 4),
        "same_func_avg_sim":  round(same_avg,  4),
        "cross_func_avg_sim": round(cross_avg, 4),
        "gap":                round(same_avg - cross_avg, 4),
        "total_queries":      n,
        "top1_hits":          top1_hits,
        "top5_hits":          top5_hits,
    }


# ──────────────────────────────────────────────
# Compare vs baseline
# ──────────────────────────────────────────────

def print_comparison(baseline: dict, finetuned: dict):
    print("\n" + "=" * 55)
    print(f"{'Metric':<22} {'Baseline':>10} {'Fine-tuned':>10} {'Delta':>10}")
    print("-" * 55)
    keys = [
        ("top1_accuracy",      "Top-1 Accuracy"),
        ("top5_accuracy",      "Top-5 Accuracy"),
        ("mrr",                "MRR"),
        ("same_func_avg_sim",  "Same-func sim"),
        ("cross_func_avg_sim", "Cross-func sim"),
        ("gap",                "Gap"),
    ]
    for key, label in keys:
        b = baseline.get(key, 0)
        f = finetuned.get(key, 0)
        delta = f - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<20} {b:>10.4f} {f:>10.4f} {sign}{delta:>9.4f}")
    print("=" * 55)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", default=str(DEFAULT_ADAPTERS))
    args = parser.parse_args()

    adapters_path = Path(args.adapters)
    if not adapters_path.exists():
        print(f"ERROR: adapter path not found: {adapters_path}")
        print("Run scripts/06_finetune.py first.")
        sys.exit(1)

    t0 = time.time()

    print("Loading test instances...")
    records = load_test_instances()

    tok, mdl, device = load_model(adapters_path)

    print("Embedding test instances...")
    embeddings = embed_all(records, tok, mdl, device)
    print(f"  Embedding matrix : {embeddings.shape}\n")

    print("Computing metrics...")
    metrics = compute_metrics(records, embeddings)

    elapsed = time.time() - t0
    metrics["eval_time_seconds"] = round(elapsed, 1)
    metrics["model"]    = MODEL_NAME
    metrics["adapters"] = str(adapters_path.relative_to(ROOT))
    metrics["split"]    = "test"

    out = RESULTS / "finetuned_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))

    print("\n" + "=" * 45)
    print("FINE-TUNED RESULTS")
    print("=" * 45)
    print(f"  Top-1 accuracy    : {metrics['top1_accuracy']:.4f}  ({metrics['top1_hits']}/{metrics['total_queries']})")
    print(f"  Top-5 accuracy    : {metrics['top5_accuracy']:.4f}  ({metrics['top5_hits']}/{metrics['total_queries']})")
    print(f"  MRR               : {metrics['mrr']:.4f}")
    print(f"  Same-func avg sim : {metrics['same_func_avg_sim']:.4f}")
    print(f"  Cross-func avg sim: {metrics['cross_func_avg_sim']:.4f}")
    print(f"  Gap               : {metrics['gap']:.4f}")

    baseline_path = RESULTS / "baseline_metrics.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        print_comparison(baseline, metrics)

    print(f"\n  Saved -> {out.relative_to(ROOT)}")
    print(f"  Time   : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
