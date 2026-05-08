"""
asm2vec.py — Self-contained assembly function similarity using TF-IDF + Cosine Similarity.

Replaces the broken external asm2vec library dependency with a pure-NumPy
TF-IDF representation of ARM assembly instructions. This approach:
  - Tokenises each instruction into mnemonic + operand tokens
  - Builds a corpus-wide TF-IDF vocabulary
  - Represents each function as a weighted token vector
  - Computes cosine similarity between any two function vectors

No external ML library required beyond NumPy.
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_functions(json_path: str) -> dict:
    """Load a normalized JSON file and return a dict keyed by function name."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return {fn["name"]: fn for fn in data["functions"]}


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(instructions: list) -> list:
    """
    Split ARM assembly instructions into individual tokens.

    Each instruction is split on whitespace so that mnemonic and operands
    are treated as separate vocabulary entries.
    """
    tokens = []
    for instr in instructions:
        tokens.extend(instr.strip().split())
    return tokens


# ---------------------------------------------------------------------------
# TF-IDF corpus helpers
# ---------------------------------------------------------------------------

def build_vocab(all_token_lists: list) -> dict:
    """Return a {token: index} mapping over all token lists."""
    vocab = {}
    for token_list in all_token_lists:
        for token in token_list:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def compute_idf(all_token_lists: list, vocab: dict) -> np.ndarray:
    """
    Compute smoothed IDF scores for every token in *vocab*.

    idf(t) = log((N + 1) / (df(t) + 1)) + 1   (sklearn-style smoothing)
    """
    N = len(all_token_lists)
    doc_freq = Counter()
    for token_list in all_token_lists:
        for token in set(token_list):
            doc_freq[token] += 1

    idf = np.zeros(len(vocab))
    for token, idx in vocab.items():
        df = doc_freq.get(token, 0)
        idf[idx] = math.log((N + 1) / (df + 1)) + 1
    return idf


def tfidf_vector(tokens: list, vocab: dict, idf: np.ndarray) -> np.ndarray:
    """
    Compute the L2-normalised TF-IDF vector for a single function's token list.
    """
    if not tokens:
        return np.zeros(len(vocab))

    tf_counts = Counter(tokens)
    total = len(tokens)
    vec = np.zeros(len(vocab))

    for token, count in tf_counts.items():
        if token in vocab:
            idx = vocab[token]
            vec[idx] = (count / total) * idf[idx]   # TF * IDF

    # L2 normalise so cosine similarity == dot product
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    dot = float(np.dot(vec_a, vec_b))
    # vecs are already L2-normalised; clamp for floating-point safety
    return max(-1.0, min(1.0, dot))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Paths (resolved relative to this file so the script works from anywhere)
    base      = Path(__file__).parent.parent
    json_dir  = base / "data" / "json" / "normalized"
    path_a    = json_dir / "hal_crc_O0_normalized.json"
    path_b    = json_dir / "hal_crc_O0_normalized.json"

    for p in (path_a, path_b):
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")

    # ── Load functions
    print("Loading ARM assembly functions...")
    data_a = load_functions(str(path_a))
    data_b = load_functions(str(path_b))

    fn_name_a = "HAL_CRC_Init"
    fn_name_b = "HAL_CRC_Init"

    if fn_name_a not in data_a:
        raise KeyError(f"Function '{fn_name_a}' not found in {path_a.name}. "
                       f"Available: {list(data_a.keys())}")
    if fn_name_b not in data_b:
        raise KeyError(f"Function '{fn_name_b}' not found in {path_b.name}. "
                       f"Available: {list(data_b.keys())}")

    fn_a = data_a[fn_name_a]
    fn_b = data_b[fn_name_b]

    # ── Build TF-IDF representation over the entire combined corpus
    print("Building TF-IDF representation over corpus...")
    all_functions   = {**data_a, **data_b}
    all_token_lists = [tokenize(fn["instructions"]) for fn in all_functions.values()]

    vocab = build_vocab(all_token_lists)
    idf   = compute_idf(all_token_lists, vocab)

    print(f"  Corpus : {len(all_functions)} functions, {len(vocab)} unique tokens")

    # ── Vectorise the two target functions
    vec_a = tfidf_vector(tokenize(fn_a["instructions"]), vocab, idf)
    vec_b = tfidf_vector(tokenize(fn_b["instructions"]), vocab, idf)

    similarity = cosine_similarity(vec_a, vec_b)

    # ── Report
    print("\n" + "=" * 40)
    print(f"Function A : {fn_name_a}  ({len(fn_a['instructions'])} instructions)")
    print(f"Function B : {fn_name_b}  ({len(fn_b['instructions'])} instructions)")
    print("-" * 40)
    print(f"Similarity Score (TF-IDF Cosine): {similarity:.4f}")
    print("=" * 40)