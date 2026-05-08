"""
CLAP-asm Test Suite — Visual Similarity Tests
==============================================
Tests:
  1. Self-similarity sanity check (all functions vs themselves → should be 1.0)
  2. O0 vs O2 per-function similarity (same function, different optimization)
  3. Full N×N similarity heatmap (all CRC O0 + GPIO O0 functions)
  4. O0 vs O2 bar chart (same-func vs cross-func baseline)
  5. Nearest-neighbor retrieval: O2 query → find match in O0 pool

Usage:
    python testing/clap.py
"""

import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "model"))

MODEL_NAME = "hustcw/clap-asm"

# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────

def load_functions(json_path: str) -> dict:
    data = json.loads((ROOT / json_path).read_text())
    return {fn["name"]: fn for fn in data["functions"]}


def fn_to_clap_input(fn: dict) -> dict:
    return {str(i): instr for i, instr in enumerate(fn["instructions"])}


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl.eval()
    print("Model loaded.\n")
    return tok, mdl


@torch.no_grad()
def embed(fn: dict, tok, mdl) -> torch.Tensor:
    inputs = tok([fn_to_clap_input(fn)], padding=True, return_tensors="pt")
    return mdl(**inputs)  # already L2-normalized, shape (1, H)


def similarity(fn_a, fn_b, tok, mdl) -> float:
    return (embed(fn_a, tok, mdl) @ embed(fn_b, tok, mdl).T).item()


# ──────────────────────────────────────────────
# Test 1 — Self-similarity sanity check
# ──────────────────────────────────────────────

def test_self_similarity(all_fns: dict, tok, mdl):
    print("=" * 60)
    print("TEST 1: Self-similarity (expect 1.0000 for all)")
    print("=" * 60)
    scores = {}
    for name, fn in all_fns.items():
        s = similarity(fn, fn, tok, mdl)
        scores[name] = s
        status = "PASS" if s > 0.9999 else "FAIL"
        print(f"  [{status}] {name:<35s} {s:.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    names = [n.replace("HAL_", "") for n in scores]
    vals  = list(scores.values())
    colors = ["green" if v > 0.9999 else "red" for v in vals]
    bars = ax.bar(names, vals, color=colors, edgecolor="black")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Expected: 1.0")
    ax.set_ylim(0.95, 1.01)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Test 1: Self-Similarity (fn vs itself)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "testing" / "test1_self_similarity.png", dpi=120)
    plt.show()
    plt.close(fig)
    print()


# ──────────────────────────────────────────────
# Test 2 — O0 vs O2 per-function comparison
# ──────────────────────────────────────────────

def test_o0_vs_o2(crc_o0: dict, crc_o2: dict, tok, mdl):
    print("=" * 60)
    print("TEST 2: O0 vs O2 — same function, different optimization")
    print("=" * 60)
    shared = [n for n in crc_o0 if n in crc_o2]
    scores = {}
    for name in shared:
        s = similarity(crc_o0[name], crc_o2[name], tok, mdl)
        scores[name] = s
        o0_len = len(crc_o0[name]["instructions"])
        o2_len = len(crc_o2[name]["instructions"])
        print(f"  {name:<30s}  O0={o0_len:>3} instr  O2={o2_len:>3} instr  sim={s:.4f}")

    # Bar chart with instruction-count annotation
    fig, ax1 = plt.subplots(figsize=(10, 5))
    names  = [n.replace("HAL_CRC_", "") for n in scores]
    vals   = list(scores.values())
    x      = np.arange(len(names))
    bars   = ax1.bar(x, vals, color="steelblue", edgecolor="black", width=0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Cosine Similarity (O0 vs O2)", color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_title("Test 2: O0 vs O2 Similarity per Function")

    ax2 = ax1.twinx()
    o0_lens = [len(crc_o0[n]["instructions"]) for n in scores]
    o2_lens = [len(crc_o2[n]["instructions"]) for n in scores]
    ax2.plot(x, o0_lens, "o--", color="orange", label="O0 instr count")
    ax2.plot(x, o2_lens, "s--", color="red",    label="O2 instr count")
    ax2.set_ylabel("Instruction Count", color="gray")
    ax2.legend(loc="upper right")

    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="steelblue")

    plt.tight_layout()
    plt.savefig(ROOT / "testing" / "test2_o0_vs_o2.png", dpi=120)
    plt.show()
    plt.close(fig)
    print()


# ──────────────────────────────────────────────
# Test 3 — Full N×N similarity heatmap
# ──────────────────────────────────────────────

def test_similarity_matrix(all_fns: dict, tok, mdl):
    print("=" * 60)
    print("TEST 3: Full N×N similarity matrix (heatmap)")
    print("=" * 60)
    names = list(all_fns.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    embeddings = {}
    for name, fn in all_fns.items():
        embeddings[name] = embed(fn, tok, mdl)

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix[i, j] = (embeddings[a] @ embeddings[b].T).item()

    short = [n.replace("HAL_CRC_", "CRC_").replace("HAL_GPIO_", "GPIO_") for n in names]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, vmin=0.3, vmax=1.0, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=9)
    ax.set_title("Test 3: Pairwise Cosine Similarity — CRC O0 + GPIO O0")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "black" if 0.5 < val < 0.9 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    # Draw boundary between CRC and GPIO blocks
    boundary = sum(1 for name in names if "CRC" in name)
    for line in [boundary - 0.5]:
        ax.axhline(line, color="white", linewidth=2)
        ax.axvline(line, color="white", linewidth=2)

    plt.tight_layout()
    plt.savefig(ROOT / "testing" / "test3_similarity_matrix.png", dpi=120)
    plt.show()
    plt.close(fig)
    print()


# ──────────────────────────────────────────────
# Test 4 — Same-func vs cross-func comparison
# ──────────────────────────────────────────────

def test_same_vs_cross(crc_o0: dict, crc_o2: dict, gpio_o0: dict, tok, mdl):
    print("=" * 60)
    print("TEST 4: Same-func (O0 vs O2) vs cross-func baseline")
    print("=" * 60)
    shared = [n for n in crc_o0 if n in crc_o2]

    same_scores  = {}
    cross_scores = {}
    cross_fn     = gpio_o0["HAL_GPIO_EXTI_IRQHandler"]  # small GPIO func, fair baseline

    for name in shared:
        same_scores[name]  = similarity(crc_o0[name], crc_o2[name], tok, mdl)
        cross_scores[name] = similarity(crc_o0[name], cross_fn,     tok, mdl)
        verdict = "PASS" if same_scores[name] > cross_scores[name] else "FAIL"
        print(f"  [{verdict}] {name:<30s}  same={same_scores[name]:.4f}  cross={cross_scores[name]:.4f}")

    names  = [n.replace("HAL_CRC_", "") for n in shared]
    x      = np.arange(len(names))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, [same_scores[n]  for n in shared], width, label="Same func O0↔O2", color="steelblue",  edgecolor="black")
    b2 = ax.bar(x + width/2, [cross_scores[n] for n in shared], width, label="Cross func vs GPIO_Init", color="salmon", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Test 4: Same-Function (O0↔O2) vs Cross-Function Similarity")
    ax.legend()

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(ROOT / "testing" / "test4_same_vs_cross.png", dpi=120)
    plt.show()
    plt.close(fig)
    print()


# ──────────────────────────────────────────────
# Test 5 — Nearest-neighbor retrieval
# ──────────────────────────────────────────────

def test_nearest_neighbor(crc_o0: dict, crc_o2: dict, gpio_o0: dict, tok, mdl):
    print("=" * 60)
    print("TEST 5: Nearest-neighbor retrieval (O2 query -> O0 pool)")
    print("         Correct if top-1 = same function name")
    print("=" * 60)

    pool = {**crc_o0, **gpio_o0}
    pool_names = list(pool.keys())
    pool_embs  = torch.cat([embed(fn, tok, mdl) for fn in pool.values()], dim=0)

    results = {}
    correct = 0
    for name, fn_o2 in crc_o2.items():
        q_emb = embed(fn_o2, tok, mdl)
        sims  = (q_emb @ pool_embs.T).squeeze()
        ranked = torch.argsort(sims, descending=True)
        top3   = [(pool_names[i], sims[i].item()) for i in ranked[:3]]
        hit    = top3[0][0] == name
        if hit: correct += 1
        results[name] = (top3, hit)
        status = "HIT " if hit else "MISS"
        print(f"  [{status}] Query: {name}")
        for rank, (match, score) in enumerate(top3, 1):
            marker = " <-- CORRECT" if match == name else ""
            print(f"         #{rank}: {match:<35s} {score:.4f}{marker}")

    print(f"\n  Retrieval accuracy: {correct}/{len(crc_o2)} = {100*correct/len(crc_o2):.0f}%")

    # Heatmap: O2 queries × O0 pool
    n_q = len(crc_o2)
    n_p = len(pool_names)
    mat = np.zeros((n_q, n_p))
    q_names = list(crc_o2.keys())
    for i, name in enumerate(q_names):
        q_emb = embed(crc_o2[name], tok, mdl)
        sims  = (q_emb @ pool_embs.T).squeeze().numpy()
        mat[i] = sims

    short_q = [n.replace("HAL_CRC_", "") for n in q_names]
    short_p = [n.replace("HAL_CRC_", "CRC_").replace("HAL_GPIO_", "GPIO_") for n in pool_names]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mat, vmin=0.3, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_xticks(range(n_p)); ax.set_xticklabels(short_p, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_q)); ax.set_yticklabels([f"{n} (O2)" for n in short_q], fontsize=8)
    ax.set_title("Test 5: Nearest-Neighbor Retrieval — O2 queries vs O0 pool")

    for i in range(n_q):
        for j in range(n_p):
            val = mat[i, j]
            color = "black" if 0.4 < val < 0.85 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    # Box correct matches
    for i, name in enumerate(q_names):
        if name in pool_names:
            j = pool_names.index(name)
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                linewidth=2.5, edgecolor="blue", facecolor="none"
            ))

    plt.tight_layout()
    plt.savefig(ROOT / "testing" / "test5_nearest_neighbor.png", dpi=120)
    plt.show()
    plt.close(fig)
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tok, mdl = load_model()

    crc_o0  = load_functions("data/json/normalized/hal_crc_O0_normalized.json")
    crc_o2  = load_functions("data/json/normalized/stm32f1xx_hal_crc_O2_normalized.json")
    gpio_o0 = load_functions("data/json/normalized/stm32f1xx_hal_gpio_O0_normalized.json")

    all_fns = {**crc_o0, **gpio_o0}  # full pool for heatmap

    test_self_similarity(all_fns, tok, mdl)
    test_o0_vs_o2(crc_o0, crc_o2, tok, mdl)
    test_similarity_matrix(all_fns, tok, mdl)
    test_same_vs_cross(crc_o0, crc_o2, gpio_o0, tok, mdl)
    test_nearest_neighbor(crc_o0, crc_o2, gpio_o0, tok, mdl)

    print("All test plots saved to testing/")
