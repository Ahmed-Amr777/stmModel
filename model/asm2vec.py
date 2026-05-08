"""
asm2vec.py — Assembly function similarity using SVD token embeddings (numpy only).

This implements a proper ML-based embedding approach:

  1. Build a token co-occurrence matrix across all assembly instructions
     (which tokens appear together in the same instruction)
  2. Apply SVD (Singular Value Decomposition) to factorize the matrix
     → each token gets a dense learned embedding vector
  3. Represent each function as the AVERAGE of its token embeddings
  4. Cosine similarity between function vectors

Why this is ML (not just counting):
  - SVD finds the latent structure in the data
  - Semantically similar tokens (e.g. ldr/str, bne/beq) end up close in embedding space
  - This is Latent Semantic Analysis — the mathematical foundation of Word2Vec

Requires: numpy only (no external ML libraries)

Two tests:
  Test 1 — Same function vs itself           → expected similarity ≈ 1.0
  Test 2 — Two completely different functions → expected similarity < 1.0
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_functions(json_path: str) -> dict:
    """Load a normalized JSON file → {function_name: function_dict}."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return {fn["name"]: fn for fn in data["functions"]}


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_function(instructions: list) -> list:
    """Convert a list of instruction strings into a flat list of tokens."""
    tokens = []
    for instr in instructions:
        tokens.extend(instr.strip().split())
    return tokens


def tokenize_instructions(instructions: list) -> list:
    """
    Convert instructions into a list of token lists (one list per instruction).
    Each instruction = one context window for co-occurrence.
    """
    return [instr.strip().split() for instr in instructions if instr.strip()]


# ---------------------------------------------------------------------------
# Co-occurrence matrix + SVD embeddings
# ---------------------------------------------------------------------------

def build_vocab(all_functions: dict) -> dict:
    """Build a {token: index} vocabulary from all functions."""
    vocab = {}
    for fn in all_functions.values():
        for token in tokenize_function(fn["instructions"]):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def build_cooccurrence_matrix(all_functions: dict, vocab: dict) -> np.ndarray:
    """
    Build a symmetric token co-occurrence matrix.

    For each assembly instruction, every pair of tokens in that instruction
    is counted as co-occurring. This captures which operands appear with
    which mnemonics (e.g. 'ldr' co-occurs with 'r3', '[r7', '#4]').

    Shape: (vocab_size, vocab_size)
    """
    V = len(vocab)
    matrix = np.zeros((V, V), dtype=np.float32)

    for fn in all_functions.values():
        for instruction_tokens in tokenize_instructions(fn["instructions"]):
            # All pairs within the same instruction co-occur
            for i, t1 in enumerate(instruction_tokens):
                for t2 in instruction_tokens:
                    if t1 in vocab and t2 in vocab:
                        matrix[vocab[t1], vocab[t2]] += 1.0

    return matrix


def learn_embeddings(cooccurrence: np.ndarray, embedding_dim: int = 64) -> np.ndarray:
    """
    Factorize the co-occurrence matrix using SVD to get token embeddings.

    SVD: M ≈ U · Σ · Vᵀ
    Token embeddings = U · sqrt(Σ)  (left singular vectors scaled by singular values)

    This is exactly how GloVe and LSA derive embeddings — the matrix factorization
    finds the latent dimensions that best explain the co-occurrence patterns.

    Returns: embedding matrix of shape (vocab_size, embedding_dim)
    """
    # Apply PPMI (Positive Pointwise Mutual Information) weighting first
    # PPMI downweights common co-occurrences and upweights informative ones
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    col_sums = cooccurrence.sum(axis=0, keepdims=True)
    total    = cooccurrence.sum()

    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1e-9, row_sums)
    col_sums = np.where(col_sums == 0, 1e-9, col_sums)

    # PMI = log( P(i,j) / (P(i) * P(j)) )
    pmi = np.log((cooccurrence * total) / (row_sums * col_sums) + 1e-9)

    # PPMI = max(PMI, 0)
    ppmi = np.maximum(pmi, 0)

    # SVD on the PPMI matrix
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)

    # Keep only top `embedding_dim` dimensions
    dim = min(embedding_dim, len(S))
    embeddings = U[:, :dim] * np.sqrt(S[:dim])   # shape: (vocab_size, dim)

    # L2-normalize each token embedding
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    return embeddings


# ---------------------------------------------------------------------------
# Function vector
# ---------------------------------------------------------------------------

def function_vector(instructions: list, vocab: dict, embeddings: np.ndarray) -> np.ndarray:
    """
    Represent a function as the average of its token embeddings.

    OOV tokens (not in vocab) are skipped.
    Returns zero vector if no known tokens found.
    """
    tokens = tokenize_function(instructions)
    vecs = [embeddings[vocab[t]] for t in tokens if t in vocab]

    if not vecs:
        return np.zeros(embeddings.shape[1])

    vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity (vectors are already L2-normalized)."""
    return float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    # Enable ANSI colors on Windows
    os.system("")

    # ── ANSI color codes
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED    = "\033[41m"
    BG_BLUE   = "\033[44m"
    BG_CYAN   = "\033[46m"
    BG_MAG    = "\033[45m"
    BG_WHITE  = "\033[47m"
    BG_DARK   = "\033[100m"

    # ── Visual helpers
    def sim_bar(score, width=30):
        """Create a colored progress bar for a similarity score."""
        filled = int(score * width)
        empty  = width - filled
        if score >= 0.95:
            color = GREEN
        elif score >= 0.85:
            color = CYAN
        elif score >= 0.70:
            color = YELLOW
        else:
            color = RED
        bar = color + "█" * filled + DIM + "░" * empty + RESET
        return bar

    def heatmap_cell(score):
        """Return a colored block character for heatmap visualization."""
        if score >= 0.98:
            return f"\033[48;5;46m  {score:.2f}  {RESET}"   # bright green
        elif score >= 0.95:
            return f"\033[48;5;34m  {score:.2f}  {RESET}"   # green
        elif score >= 0.90:
            return f"\033[48;5;28m  {score:.2f}  {RESET}"   # dark green
        elif score >= 0.85:
            return f"\033[48;5;226m\033[30m  {score:.2f}  {RESET}"  # yellow
        elif score >= 0.80:
            return f"\033[48;5;214m\033[30m  {score:.2f}  {RESET}"  # orange
        elif score >= 0.70:
            return f"\033[48;5;208m\033[30m  {score:.2f}  {RESET}"  # dark orange
        else:
            return f"\033[48;5;196m  {score:.2f}  {RESET}"  # red

    def box_print(lines, color=CYAN):
        """Print text inside a Unicode box."""
        max_len = max(len(line) for line in lines)
        print(f"{color}╔{'═' * (max_len + 2)}╗{RESET}")
        for line in lines:
            print(f"{color}║{RESET} {line:<{max_len}} {color}║{RESET}")
        print(f"{color}╚{'═' * (max_len + 2)}╝{RESET}")

    # ── Paths (resolved relative to this file)
    base     = Path(__file__).parent.parent
    json_dir = base / "data" / "json" / "normalized"

    path_crc_O0  = json_dir / "hal_crc_O0_normalized.json"
    path_crc_O2  = json_dir / "stm32f1xx_hal_crc_O2_normalized.json"
    path_gpio_O0 = json_dir / "stm32f1xx_hal_gpio_O0_normalized.json"

    for p in (path_crc_O0, path_crc_O2, path_gpio_O0):
        if not p.exists():
            raise FileNotFoundError(f"Missing data file: {p}")

    # ── Header
    print()
    box_print([
        f"{BOLD}ASM2VEC — ARM Binary Code Similarity Analysis{RESET}",
        f"{DIM}SVD Token Embeddings + Cosine Similarity{RESET}",
    ], MAGENTA)

    # ── Load all three datasets
    print(f"\n{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD} LOADING DATA{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")
    data_crc_O0  = load_functions(str(path_crc_O0))
    data_crc_O2  = load_functions(str(path_crc_O2))
    data_gpio_O0 = load_functions(str(path_gpio_O0))

    print(f"  {GREEN}✓{RESET} CRC  (O0) : {BOLD}{len(data_crc_O0)}{RESET} functions  {DIM}({path_crc_O0.name}){RESET}")
    print(f"  {GREEN}✓{RESET} CRC  (O2) : {BOLD}{len(data_crc_O2)}{RESET} functions  {DIM}({path_crc_O2.name}){RESET}")
    print(f"  {GREEN}✓{RESET} GPIO (O0) : {BOLD}{len(data_gpio_O0)}{RESET} functions {DIM}({path_gpio_O0.name}){RESET}")

    # ── Build corpus from ALL functions across all three files
    all_funcs = {}
    for name, fn in data_crc_O0.items():
        all_funcs[f"CRC_O0::{name}"] = fn
    for name, fn in data_crc_O2.items():
        all_funcs[f"CRC_O2::{name}"] = fn
    for name, fn in data_gpio_O0.items():
        all_funcs[f"GPIO_O0::{name}"] = fn

    # ── Train
    print(f"\n{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD} TRAINING{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")

    vocab        = build_vocab(all_funcs)
    cooccurrence = build_cooccurrence_matrix(all_funcs, vocab)
    print(f"  {GREEN}✓{RESET} Vocabulary   : {BOLD}{len(vocab)}{RESET} unique tokens")
    print(f"  {GREEN}✓{RESET} Co-occurrence : {BOLD}{cooccurrence.shape[0]}x{cooccurrence.shape[1]}{RESET} matrix")

    EMBEDDING_DIM = 32
    embeddings = learn_embeddings(cooccurrence, embedding_dim=EMBEDDING_DIM)
    print(f"  {GREEN}✓{RESET} Embeddings   : {BOLD}{embeddings.shape}{RESET} (PPMI + SVD)")
    print(f"  {GREEN}✓{RESET} Training complete!")

    # ── Test runner with visual output
    def run_test(test_num, title, instructions_a, label_a, instructions_b, label_b, expected):
        va = function_vector(instructions_a, vocab, embeddings)
        vb = function_vector(instructions_b, vocab, embeddings)
        sim = cosine_similarity(va, vb)

        print(f"\n{BLUE}┌{'─' * 58}┐{RESET}")
        print(f"{BLUE}│{RESET} {BOLD}TEST {test_num}{RESET} {DIM}│{RESET} {title}")
        print(f"{BLUE}├{'─' * 58}┤{RESET}")
        print(f"{BLUE}│{RESET}  {CYAN}A:{RESET} {label_a}  {DIM}({len(instructions_a)} instr){RESET}")
        print(f"{BLUE}│{RESET}  {CYAN}B:{RESET} {label_b}  {DIM}({len(instructions_b)} instr){RESET}")
        print(f"{BLUE}│{RESET}")
        print(f"{BLUE}│{RESET}  Score: {BOLD}{sim:.4f}{RESET}  {sim_bar(sim)}  {DIM}{expected}{RESET}")
        print(f"{BLUE}└{'─' * 58}┘{RESET}")
        return sim

    results = {}

    # ── Run all 6 tests
    print(f"\n{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD} SIMILARITY TESTS{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")

    results["T1"] = run_test(
        1, "Identity (same function vs itself)",
        data_crc_O0["HAL_CRC_Init"]["instructions"], "HAL_CRC_Init (CRC, O0)",
        data_crc_O0["HAL_CRC_Init"]["instructions"], "HAL_CRC_Init (CRC, O0)",
        "expected: 1.0"
    )

    results["T2"] = run_test(
        2, "Structurally similar (Accumulate vs Calculate)",
        data_crc_O0["HAL_CRC_Accumulate"]["instructions"], "HAL_CRC_Accumulate (CRC, O0)",
        data_crc_O0["HAL_CRC_Calculate"]["instructions"],  "HAL_CRC_Calculate (CRC, O0)",
        "expected: very high"
    )

    results["T3"] = run_test(
        3, "Cross-optimization (O0 vs O2, same function)",
        data_crc_O0["HAL_CRC_Init"]["instructions"],  "HAL_CRC_Init (CRC, O0)",
        data_crc_O2["HAL_CRC_Init"]["instructions"],  "HAL_CRC_Init (CRC, O2)",
        "expected: moderate-high"
    )

    results["T4"] = run_test(
        4, "Different peripheral (CRC Init vs GPIO Init)",
        data_crc_O0["HAL_CRC_Init"]["instructions"],   "HAL_CRC_Init  (CRC, O0)",
        data_gpio_O0["HAL_GPIO_Init"]["instructions"],  "HAL_GPIO_Init (GPIO, O0)",
        "expected: moderate"
    )

    results["T5"] = run_test(
        5, "Init vs DeInit (same peripheral)",
        data_crc_O0["HAL_CRC_Init"]["instructions"],   "HAL_CRC_Init   (CRC, O0)",
        data_crc_O0["HAL_CRC_DeInit"]["instructions"], "HAL_CRC_DeInit (CRC, O0)",
        "expected: moderate"
    )

    results["T6"] = run_test(
        6, "ReadPin vs WritePin (GPIO)",
        data_gpio_O0["HAL_GPIO_ReadPin"]["instructions"],  "HAL_GPIO_ReadPin  (GPIO, O0)",
        data_gpio_O0["HAL_GPIO_WritePin"]["instructions"], "HAL_GPIO_WritePin (GPIO, O0)",
        "expected: moderately high"
    )

    # ====================================================================
    # RANKED SUMMARY with visual bars
    # ====================================================================
    print(f"\n{MAGENTA}╔{'═' * 58}╗{RESET}")
    print(f"{MAGENTA}║{RESET}{BOLD}  RANKED SUMMARY                                          {MAGENTA}║{RESET}")
    print(f"{MAGENTA}╠{'═' * 58}╣{RESET}")

    test_descriptions = {
        "T1": "Identity (same vs same)        ",
        "T2": "Accumulate vs Calculate        ",
        "T3": "Cross-opt (O0 vs O2)           ",
        "T4": "CRC Init vs GPIO Init          ",
        "T5": "Init vs DeInit (CRC)           ",
        "T6": "ReadPin vs WritePin (GPIO)     ",
    }

    # Sort by similarity descending
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (key, score) in enumerate(ranked, 1):
        medal = ["🥇", "🥈", "🥉", "  ", "  ", "  "][rank - 1]
        desc = test_descriptions[key]
        bar = sim_bar(score, width=20)
        print(f"{MAGENTA}║{RESET} {medal} {key} {desc} {BOLD}{score:.4f}{RESET} {bar} {MAGENTA}║{RESET}")

    print(f"{MAGENTA}╚{'═' * 58}╝{RESET}")

    # ====================================================================
    # COLOR LEGEND
    # ====================================================================
    print(f"\n  {BOLD}Legend:{RESET}  {GREEN}██{RESET} >= 0.95  {CYAN}██{RESET} >= 0.85  {YELLOW}██{RESET} >= 0.70  {RED}██{RESET} < 0.70")

    # ====================================================================
    # ALL-vs-ALL HEATMAP
    # ====================================================================
    print(f"\n{MAGENTA}╔{'═' * 58}╗{RESET}")
    print(f"{MAGENTA}║{RESET}{BOLD}  ALL-vs-ALL SIMILARITY HEATMAP                            {MAGENTA}║{RESET}")
    print(f"{MAGENTA}╚{'═' * 58}╝{RESET}")

    # Collect all functions with short labels
    labeled_funcs = []
    for name in sorted(data_crc_O0.keys()):
        labeled_funcs.append((f"CRC0:{name}", data_crc_O0[name]["instructions"]))
    for name in sorted(data_crc_O2.keys()):
        labeled_funcs.append((f"CRC2:{name}", data_crc_O2[name]["instructions"]))
    for name in sorted(data_gpio_O0.keys()):
        labeled_funcs.append((f"GPIO:{name}", data_gpio_O0[name]["instructions"]))

    # Compute all vectors
    func_vecs = [(label, function_vector(instr, vocab, embeddings))
                 for label, instr in labeled_funcs]

    # Create short numeric labels for column headers
    N = len(func_vecs)
    col_width = 8

    # Print column index header
    print(f"\n{'':>28s}", end="")
    for j in range(N):
        print(f" {DIM}[{j:>2d}]{RESET}   ", end="")
    print()

    # Print each row
    for i, (label_i, vec_i) in enumerate(func_vecs):
        # Truncate label for display
        short_lbl = label_i[:25]
        print(f"  {DIM}[{i:>2d}]{RESET} {short_lbl:<22s}", end="")
        for j, (label_j, vec_j) in enumerate(func_vecs):
            sim = cosine_similarity(vec_i, vec_j)
            print(f" {heatmap_cell(sim)}", end="")
        print()

    # Print the index legend below
    print(f"\n  {BOLD}Index Legend:{RESET}")
    for i, (label, _) in enumerate(func_vecs):
        print(f"    {DIM}[{i:>2d}]{RESET} {label}")

    print(f"\n{DIM}{'─' * 60}{RESET}")
    print(f"  {GREEN}■{RESET} >= 0.98 (identical)   {GREEN}■{RESET} >= 0.95 (very similar)")
    print(f"  {GREEN}■{RESET} >= 0.90 (similar)     {YELLOW}■{RESET} >= 0.85 (moderate)")
    print(f"  {YELLOW}■{RESET} >= 0.80 (low-mod)     {RED}■{RESET} >= 0.70 (low)")
    print(f"  {RED}■{RESET} <  0.70 (dissimilar)")
    print(f"{DIM}{'─' * 60}{RESET}")
    print()
