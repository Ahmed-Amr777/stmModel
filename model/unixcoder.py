"""
UniXcoder Embedding Pipeline for STM32F1 HAL Functions
=======================================================
Replaces jTrans. Uses subword tokenization (BPE) so it
won't produce [UNK] for ARM assembly tokens.

UniXcoder was trained on code in 6 programming languages.
Assembly is NOT one of them, but BPE tokenization means
it can still tokenize ARM instructions at the subword level
instead of mapping everything to [UNK].

Usage:
    python unixcoder_embed.py

Requirements:
    pip install torch transformers
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "microsoft/unixcoder-base"
MAX_LENGTH = 512  # UniXcoder max token length


# ─────────────────────────────────────────────
# Step 1: Load normalized JSON data
# ─────────────────────────────────────────────

def load_functions(json_path: str) -> dict:
    """Load all functions from a normalized JSON. Returns {name: fn_dict}."""
    data = json.loads(Path(json_path).read_text())
    return {fn["name"]: fn for fn in data["functions"]}


def get_function(json_path: str, name: str) -> dict:
    """Get a single function by name from a normalized JSON."""
    funcs = load_functions(json_path)
    if name not in funcs:
        available = list(funcs.keys())
        raise KeyError(f"Function '{name}' not found. Available: {available}")
    return funcs[name]


# ─────────────────────────────────────────────
# Step 2: Load UniXcoder model
# ─────────────────────────────────────────────

def load_model():
    """Load UniXcoder tokenizer and model."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    return tokenizer, model


# ─────────────────────────────────────────────
# Step 3: Encode a function into an embedding
# ─────────────────────────────────────────────

def encode_function(fn: dict, tokenizer, model) -> torch.Tensor:
    """
    Encode a function dict into a normalized embedding vector.

    Takes the normalized instruction list, joins into a single string,
    tokenizes with BPE (no [UNK] problem), and extracts embedding.
    """
    # Join instructions with newline (preserves structure better than space)
    text = "\n".join(fn["instructions"])

    # Tokenize with UniXcoder format: <s> code </s>
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    # Debug: show tokenization (first call only)
    if not hasattr(encode_function, "_debug_shown"):
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        unk_count = tokens.count("<unk>")
        total = len(tokens)
        print(f"  Tokens: {total}, UNK tokens: {unk_count} ({100*unk_count/total:.1f}%)")
        if unk_count > total * 0.3:
            print("  WARNING: High UNK rate! Tokenization may be poor.")
        else:
            print("  OK: Low UNK rate. Tokenization is working.")
        # Show first 30 tokens
        safe = [t.encode("ascii", errors="replace").decode() for t in tokens[:30]]
        print(f"  First 30 tokens: {safe}")
        encode_function._debug_shown = True

    with torch.no_grad():
        outputs = model(**inputs)

    # Use CLS token embedding (first token)
    embedding = outputs.last_hidden_state[:, 0, :]

    # Normalize to unit vector (for cosine similarity)
    return F.normalize(embedding, dim=-1)


# ─────────────────────────────────────────────
# Step 4: Compute similarity between two functions
# ─────────────────────────────────────────────

def compute_similarity(fn_a: dict, fn_b: dict, tokenizer, model) -> float:
    """Compute cosine similarity between two function dicts."""
    emb_a = encode_function(fn_a, tokenizer, model)
    emb_b = encode_function(fn_b, tokenizer, model)
    return (emb_a @ emb_b.T).item()


# ─────────────────────────────────────────────
# Step 5: Embed all functions and build index
# ─────────────────────────────────────────────

def embed_all_functions(json_paths: list, tokenizer, model) -> tuple:
    """
    Embed all functions from multiple JSON files.

    Returns:
        (embeddings_tensor, labels_list, functions_dict)
    """
    all_embeddings = []
    all_labels = []
    all_functions = {}

    for json_path in json_paths:
        print(f"\nProcessing {json_path}...")
        funcs = load_functions(json_path)

        for name, fn_dict in funcs.items():
            try:
                emb = encode_function(fn_dict, tokenizer, model)
                all_embeddings.append(emb)
                all_labels.append(name)
                all_functions[name] = fn_dict
                print(f"  Embedded: {name} ({len(fn_dict['instructions'])} instructions)")
            except Exception as e:
                print(f"  Error embedding {name}: {e}")

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, all_labels, all_functions


# ─────────────────────────────────────────────
# Step 6: Search for nearest functions
# ─────────────────────────────────────────────

def search_nearest(query_embedding, all_embeddings, labels, k=5):
    """
    Find k nearest functions to a query embedding using cosine similarity.

    Returns list of (label, similarity_score) tuples.
    """
    # Cosine similarity (embeddings are already normalized)
    similarities = (query_embedding @ all_embeddings.T).squeeze()

    # Get top-k
    top_k = torch.topk(similarities, k=min(k, len(labels)))

    results = []
    for score, idx in zip(top_k.values, top_k.indices):
        results.append((labels[idx], score.item()))

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # --- CONFIGURATION ---
    json_a = "data/json/normalized/hal_crc_O0_normalized.json"
    json_b = "data/json/normalized/stm32f1xx_hal_gpio_O0_normalized.json"
    json_a_o2 = "data/json/normalized/stm32f1xx_hal_crc_O2_normalized.json"

    func_name_a = "HAL_CRC_Accumulate"
    func_name_b = "HAL_GPIO_EXTI_IRQHandler"

    # --- STEP 1: Load model ---
    tokenizer, model = load_model()

    # --- STEP 2: Load functions ---
    fn_a = get_function(json_a, func_name_a)
    fn_b = get_function(json_b, func_name_b)

    print(f"\nFunction A: {func_name_a} ({len(fn_a['instructions'])} instructions)")
    print(f"Function B: {func_name_b} ({len(fn_b['instructions'])} instructions)")

    # --- STEP 3: Check tokenization quality ---
    print(f"\nTokenization check for {func_name_a}:")
    text_a = "\n".join(fn_a["instructions"])
    tokens_a = tokenizer.tokenize(text_a)
    unk_count = sum(1 for t in tokens_a if t == "<unk>")
    print(f"  Total tokens: {len(tokens_a)}")
    print(f"  UNK tokens: {unk_count}")
    safe_tokens = [t.encode("ascii", errors="replace").decode() for t in tokens_a[:40]]
    print(f"  First 40 tokens: {safe_tokens}")

    # --- STEP 4: Compute similarity ---
    print(f"\nComputing similarity...")
    score = compute_similarity(fn_a, fn_b, tokenizer, model)
    print(f"Similarity between {func_name_a} and {func_name_b}: {score:.4f}")

    # --- STEP 5: Compare same function to itself ---
    print(f"\nSelf-similarity check:")
    self_score = compute_similarity(fn_a, fn_a, tokenizer, model)
    print(f"  {func_name_a} vs itself: {self_score:.4f} (should be 1.0)")

    # --- STEP 5b: Compare O0 vs O2 optimization ---
    print(f"\nO0 vs O2 optimization similarity check:")
    fn_a_o2 = get_function(json_a_o2, func_name_a)
    print(f"  {func_name_a} O0: {len(fn_a['instructions'])} instructions")
    print(f"  {func_name_a} O2: {len(fn_a_o2['instructions'])} instructions")
    o2_score = compute_similarity(fn_a, fn_a_o2, tokenizer, model)
    print(f"  {func_name_a} O0 vs O2: {o2_score:.4f}")

    # --- STEP 6: Embed all and search ---
    print(f"\nEmbedding all functions...")
    embeddings, labels, functions = embed_all_functions(
        [json_a, json_b], tokenizer, model
    )
    print(f"Total embeddings: {embeddings.shape}")

    # Search for nearest to func_a
    query_emb = encode_function(fn_a, tokenizer, model)
    results = search_nearest(query_emb, embeddings, labels, k=5)

    print(f"\nNearest functions to {func_name_a}:")
    for label, sim in results:
        print(f"  {label:40s} similarity: {sim:.4f}")