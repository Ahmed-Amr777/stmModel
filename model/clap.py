"""
CLAP-asm Embedding Pipeline for STM32F1 HAL Functions
======================================================
CLAP-asm (hustcw/clap-asm) was trained on x86 assembly from IDA Pro.
ARM is NOT a supported architecture — but we test it empirically.

Input format required by AsmTokenizer:
    List of dicts: [{"0": "push {r7}", "1": "sub sp, #28", ...}]
    Keys are string instruction indices; values are raw instruction strings.

The model's forward() already returns L2-normalized embeddings directly
(mean pooling + linear projection + F.normalize) — no extra normalization needed.

Usage:
    python model/clap.py

Requirements:
    pip install torch transformers
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "hustcw/clap-asm"
ROOT = Path(__file__).parent.parent  # preprocessing/


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_functions(json_path: str) -> dict:
    data = json.loads((ROOT / json_path).read_text())
    return {fn["name"]: fn for fn in data["functions"]}


def get_function(json_path: str, name: str) -> dict:
    funcs = load_functions(json_path)
    if name not in funcs:
        raise KeyError(f"Function '{name}' not found. Available: {list(funcs.keys())}")
    return funcs[name]


def fn_to_clap_input(fn: dict) -> dict:
    """
    Convert a normalized function dict to CLAP-asm's expected input format.
    AsmTokenizer expects: {"0": "instr0", "1": "instr1", ...}
    """
    return {str(i): instr for i, instr in enumerate(fn["instructions"])}


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    print("Model loaded.")
    return tokenizer, model


# ─────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────

def encode_function(fn: dict, tokenizer, model) -> torch.Tensor:
    """Encode a function dict into a normalized embedding vector."""
    clap_input = fn_to_clap_input(fn)

    # Debug tokenization on first call
    if not getattr(encode_function, "_debug_shown", False):
        sample_tokens = []
        unk_count = 0
        total_tokens = 0
        for instr in list(clap_input.values())[:5]:
            toks = tokenizer.tokenize(instr.replace(',', ''), max_length=20, truncation=True, add_special_tokens=False)
            safe = [t.encode("ascii", errors="replace").decode() for t in toks]
            sample_tokens.extend(safe)
            unk_count += sum(1 for t in toks if "unk" in t.lower())
            total_tokens += len(toks)
        print(f"  Sample tokens (first 5 instrs): {sample_tokens[:30]}")
        pct = 100 * unk_count / total_tokens if total_tokens else 0
        print(f"  UNK in sample: {unk_count}/{total_tokens} ({pct:.1f}%)")
        if pct > 30:
            print("  WARNING: High UNK rate — CLAP tokenizer likely doesn't handle ARM well.")
        else:
            print("  OK: Low UNK rate.")
        encode_function._debug_shown = True

    inputs = tokenizer([clap_input], padding=True, return_tensors="pt")

    with torch.no_grad():
        # AsmEncoder.forward() returns normalized embedding directly
        embedding = model(**inputs)

    return embedding  # shape: (1, hidden_size), already L2-normalized


def compute_similarity(fn_a: dict, fn_b: dict, tokenizer, model) -> float:
    emb_a = encode_function(fn_a, tokenizer, model)
    emb_b = encode_function(fn_b, tokenizer, model)
    return (emb_a @ emb_b.T).item()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    json_o0   = "data/json/normalized/hal_crc_O0_normalized.json"
    json_o2   = "data/json/normalized/stm32f1xx_hal_crc_O2_normalized.json"
    json_gpio = "data/json/normalized/stm32f1xx_hal_gpio_O0_normalized.json"

    func_name = "HAL_CRC_Accumulate"

    tokenizer, model = load_model()

    fn_o0   = get_function(json_o0,   func_name)
    fn_o2   = get_function(json_o2,   func_name)
    fn_gpio = get_function(json_gpio, "HAL_GPIO_EXTI_IRQHandler")

    print(f"\n{func_name}: O0={len(fn_o0['instructions'])} instr, O2={len(fn_o2['instructions'])} instr")

    print(f"\nTokenization check (CLAP-asm on ARM):")
    encode_function._debug_shown = False
    _ = encode_function(fn_o0, tokenizer, model)
    encode_function._debug_shown = True

    print(f"\n--- Similarity Results ---")

    self_score  = compute_similarity(fn_o0, fn_o0,   tokenizer, model)
    o2_score    = compute_similarity(fn_o0, fn_o2,   tokenizer, model)
    cross_score = compute_similarity(fn_o0, fn_gpio, tokenizer, model)

    print(f"Self (O0 vs O0):          {self_score:.4f}  (expected: 1.0000)")
    print(f"Same func O0 vs O2:       {o2_score:.4f}  (higher = model links same func across opts)")
    print(f"Cross func (CRC vs GPIO): {cross_score:.4f}  (lower = model discriminates functions)")

    print(f"\n--- Verdict ---")
    if self_score > 0.99 and o2_score > cross_score:
        print("  PASS: CLAP-asm generalizes to ARM. O0/O2 more similar than cross-function. USABLE.")
    elif self_score > 0.99 and o2_score <= cross_score:
        print("  FAIL: CLAP-asm does NOT discriminate on ARM. O0/O2 <= cross-function similarity.")
        print("  Recommendation: fall back to UniXcoder or Asm2Vec.")
    else:
        print("  FAIL: Self-similarity != 1.0. CLAP-asm broken on ARM input. Fall back immediately.")
