"""
CLAP-asm Heatmap Test Suite — Optimization-Level Comparison
============================================================
Tests (all using data/json/*_clap.json files):
  1. Per-function optimization heatmaps — CRC
     For each CRC function: 5×5 cosine-similarity matrix across O0/O1/O2/O3/Os
  2. Full CRC all-functions × all-optimizations heatmap
     Big matrix: every (function, opt) pair vs every other
  3. Per-function optimization heatmaps — GPIO
     Same as test 1 but for GPIO functions
  4. Full GPIO all-functions × all-optimizations heatmap
  5. Cross-library heatmap — CRC O0 vs GPIO O0
     Rows = CRC functions, Cols = GPIO functions

Output PNGs saved to: testing/heatmaps/

Usage:
    python testing/clap_heatmaps.py
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
OUT_DIR = Path(__file__).parent / "heatmaps"
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "hustcw/clap-asm"
OPT_LEVELS = ["O0", "O1", "O2", "O3", "Os"]


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────

def load_functions(json_path) -> dict:
    data = json.loads((ROOT / json_path).read_text())
    return {fn["name"]: fn for fn in data["functions"]}


def load_all_opts(library: str) -> dict[str, dict]:
    """Returns {opt_level: {fn_name: fn_dict}} for the given library (crc or gpio)."""
    result = {}
    for opt in OPT_LEVELS:
        path = f"data/json/stm32f1xx_hal_{library}_{opt}_clap.json"
        result[opt] = load_functions(path)
    return result


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
    instr_dict = {str(i): instr for i, instr in enumerate(fn["instructions"])}
    inputs = tok([instr_dict], padding=True, return_tensors="pt")
    return mdl(**inputs)  # L2-normalized, shape (1, H)


def embed_all(fn_dict: dict, tok, mdl) -> dict[str, torch.Tensor]:
    return {name: embed(fn, tok, mdl) for name, fn in fn_dict.items()}


# ──────────────────────────────────────────────
# Heatmap helper
# ──────────────────────────────────────────────

def _annotate_heatmap(ax, matrix, fontsize=8):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "black" if 0.35 < val < 0.88 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=fontsize, color=color)


def save_heatmap(matrix, row_labels, col_labels, title, save_path,
                 vmin=0.0, vmax=1.0, figsize=(8, 6), fontsize=8):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=fontsize)
    ax.set_title(title)
    _annotate_heatmap(ax, matrix, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")


# ──────────────────────────────────────────────
# Test 1 & 3 — Per-function 5×5 optimization heatmaps
# ──────────────────────────────────────────────

def test_per_function_opt_heatmaps(library: str, all_opts: dict, tok, mdl):
    label = library.upper()
    print("=" * 60)
    print(f"TEST: {label} — Per-function 5×5 optimization heatmaps")
    print("=" * 60)

    # Collect common function names across all opt levels
    common_fns = set(all_opts["O0"].keys())
    for opt in OPT_LEVELS[1:]:
        common_fns &= set(all_opts[opt].keys())
    common_fns = sorted(common_fns)
    print(f"  Functions in all {len(OPT_LEVELS)} opt levels: {common_fns}")

    # Pre-embed each (opt, fn) pair
    embeddings: dict[tuple, torch.Tensor] = {}
    for opt in OPT_LEVELS:
        for name in common_fns:
            if name in all_opts[opt]:
                embeddings[(opt, name)] = embed(all_opts[opt][name], tok, mdl)

    # One 5×5 heatmap per function
    for fn_name in common_fns:
        matrix = np.zeros((len(OPT_LEVELS), len(OPT_LEVELS)))
        for i, opt_i in enumerate(OPT_LEVELS):
            for j, opt_j in enumerate(OPT_LEVELS):
                if (opt_i, fn_name) in embeddings and (opt_j, fn_name) in embeddings:
                    matrix[i, j] = (embeddings[(opt_i, fn_name)] @
                                    embeddings[(opt_j, fn_name)].T).item()
        short = fn_name.replace(f"HAL_{label}_", "")
        title = f"{label} '{short}' — Similarity across optimization levels"
        fname = f"{library}_func_{short.lower()}.png"
        save_heatmap(matrix, OPT_LEVELS, OPT_LEVELS, title,
                     OUT_DIR / fname, vmin=0.5, vmax=1.0, figsize=(6, 5))
    print()


# ──────────────────────────────────────────────
# Test 2 & 4 — Full library all-functions × all-opts heatmap
# ──────────────────────────────────────────────

def test_full_library_heatmap(library: str, all_opts: dict, tok, mdl):
    label = library.upper()
    print("=" * 60)
    print(f"TEST: {label} — Full (functions × opts) heatmap")
    print("=" * 60)

    # Build ordered list of (opt, fn_name) entries
    entries = []
    for opt in OPT_LEVELS:
        for fn_name in sorted(all_opts[opt].keys()):
            entries.append((opt, fn_name))

    # Embed everything (de-duplicate: only embed once per unique (opt, fn) pair)
    embs: dict[tuple, torch.Tensor] = {}
    for opt, fn_name in entries:
        key = (opt, fn_name)
        if key not in embs:
            embs[key] = embed(all_opts[opt][fn_name], tok, mdl)

    n = len(entries)
    matrix = np.zeros((n, n))
    for i, ki in enumerate(entries):
        for j, kj in enumerate(entries):
            matrix[i, j] = (embs[ki] @ embs[kj].T).item()

    labels = [f"{opt}/{fn.replace(f'HAL_{label}_','')}" for opt, fn in entries]

    title = f"{label} — All functions × all optimization levels"
    fname = f"{library}_full_matrix.png"
    figsize = (max(10, n * 0.55), max(8, n * 0.45))
    save_heatmap(matrix, labels, labels, title,
                 OUT_DIR / fname, vmin=0.3, vmax=1.0,
                 figsize=figsize, fontsize=7)

    # Draw dividers between opt-level blocks
    block_sizes = [len(all_opts[opt]) for opt in OPT_LEVELS]
    fig2, ax2 = plt.subplots(figsize=figsize)
    im = ax2.imshow(matrix, vmin=0.3, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax2, label="Cosine Similarity")
    ax2.set_xticks(range(n)); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_title(title + " (with opt-level borders)")
    _annotate_heatmap(ax2, matrix, fontsize=6)
    pos = 0
    for bs in block_sizes[:-1]:
        pos += bs
        ax2.axhline(pos - 0.5, color="white", linewidth=1.5)
        ax2.axvline(pos - 0.5, color="white", linewidth=1.5)
    plt.tight_layout()
    bordered_path = OUT_DIR / f"{library}_full_matrix_bordered.png"
    plt.savefig(bordered_path, dpi=120)
    plt.close(fig2)
    print(f"  Saved: {bordered_path.relative_to(ROOT)}")
    print()


# ──────────────────────────────────────────────
# Test 5 — Cross-library heatmap CRC O0 vs GPIO O0
# ──────────────────────────────────────────────

def test_cross_library_heatmap(crc_all: dict, gpio_all: dict, tok, mdl):
    print("=" * 60)
    print("TEST: Cross-library heatmap — CRC O0 vs GPIO O0")
    print("=" * 60)

    crc_fns  = crc_all["O0"]
    gpio_fns = gpio_all["O0"]

    crc_names  = sorted(crc_fns.keys())
    gpio_names = sorted(gpio_fns.keys())

    crc_embs  = {n: embed(crc_fns[n],  tok, mdl) for n in crc_names}
    gpio_embs = {n: embed(gpio_fns[n], tok, mdl) for n in gpio_names}

    matrix = np.zeros((len(crc_names), len(gpio_names)))
    for i, cn in enumerate(crc_names):
        for j, gn in enumerate(gpio_names):
            matrix[i, j] = (crc_embs[cn] @ gpio_embs[gn].T).item()

    row_labels = [n.replace("HAL_CRC_", "CRC_") for n in crc_names]
    col_labels = [n.replace("HAL_GPIO_", "GPIO_") for n in gpio_names]

    figsize = (max(10, len(gpio_names) * 1.2), max(6, len(crc_names) * 0.9))
    save_heatmap(matrix, row_labels, col_labels,
                 "Cross-library: CRC O0 vs GPIO O0",
                 OUT_DIR / "cross_crc_vs_gpio_O0.png",
                 vmin=0.0, vmax=1.0, figsize=figsize, fontsize=8)
    print()


# ──────────────────────────────────────────────
# Test 6 — Same-function similarity across opt levels (CRC + GPIO combined)
# ──────────────────────────────────────────────

def test_same_fn_across_opts(crc_all: dict, gpio_all: dict, tok, mdl):
    """
    For each function present across all opt levels, compute:
      - diagonal: same function, all pairs of opt levels
    Produces one combined heatmap per library showing all functions × all (opt_i vs opt_j) pairs.
    """
    print("=" * 60)
    print("TEST: Same-function similarity across all opt pairs")
    print("=" * 60)

    for library, all_opts in [("crc", crc_all), ("gpio", gpio_all)]:
        label = library.upper()
        common = sorted(set(all_opts["O0"].keys()))
        for opt in OPT_LEVELS[1:]:
            common = [n for n in common if n in all_opts[opt]]

        opt_pairs = [(oi, oj) for oi in OPT_LEVELS for oj in OPT_LEVELS if oi <= oj]
        pair_labels = [f"{a}↔{b}" for a, b in opt_pairs]

        matrix = np.zeros((len(common), len(opt_pairs)))
        for i, fn_name in enumerate(common):
            for j, (oi, oj) in enumerate(opt_pairs):
                ei = embed(all_opts[oi][fn_name], tok, mdl)
                ej = embed(all_opts[oj][fn_name], tok, mdl)
                matrix[i, j] = (ei @ ej.T).item()

        fn_labels = [n.replace(f"HAL_{label}_", "") for n in common]
        title = f"{label} — Same function similarity across optimization pairs"
        save_heatmap(matrix, fn_labels, pair_labels, title,
                     OUT_DIR / f"{library}_same_fn_across_opts.png",
                     vmin=0.5, vmax=1.0, figsize=(max(10, len(opt_pairs)*0.8),
                                                   max(5, len(common)*0.6)),
                     fontsize=8)
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tok, mdl = load_model()

    print("Loading clap JSON data...\n")
    crc_all  = load_all_opts("crc")
    gpio_all = load_all_opts("gpio")

    # Test 1: CRC per-function 5×5 opt heatmaps
    test_per_function_opt_heatmaps("crc", crc_all, tok, mdl)

    # Test 2: CRC full matrix (all functions × all opts)
    test_full_library_heatmap("crc", crc_all, tok, mdl)

    # Test 3: GPIO per-function 5×5 opt heatmaps
    test_per_function_opt_heatmaps("gpio", gpio_all, tok, mdl)

    # Test 4: GPIO full matrix (all functions × all opts)
    test_full_library_heatmap("gpio", gpio_all, tok, mdl)

    # Test 5: Cross-library CRC O0 vs GPIO O0
    test_cross_library_heatmap(crc_all, gpio_all, tok, mdl)

    # Test 6: Same-function similarity across all opt pairs
    test_same_fn_across_opts(crc_all, gpio_all, tok, mdl)

    print(f"\nAll heatmaps saved to testing/heatmaps/")
