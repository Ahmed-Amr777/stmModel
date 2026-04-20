import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn.functional as F
# will not work as the tokenizer is build on x86 code but we use arm instructions, but we can still test the loading and embedding process 
# you will got to many unk tokens but at least you can see the difference between two different functions instead of getting 0.9999 similarity for everything due to the collapse of the embedding space when all instructions are [UNK].

MODEL_NAME = "PurCL/jtrans-mfc"
JTRANS_MAX_POS = 2902  # checkpoint position embedding size


def load_functions(json_path: str) -> dict[str, dict]:
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


def encode_function(fn: dict, tokenizer, model) -> torch.Tensor:
    """Encode a function dict into a normalized embedding vector."""
    text = " ".join(fn["instructions"])
    print(text)
    print("---"*20)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=JTRANS_MAX_POS)

    print("INPUT IDS:")
    print(inputs["input_ids"])

    print("\nDECODED TOKENS:")
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

    print("\nDECODED TEXT (check lossless?):")
    print(tokenizer.decode(inputs["input_ids"][0]))
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(embedding, dim=-1)


def compute_similarity(fn_a: dict, fn_b: dict, tokenizer, model) -> float:
    """Compute cosine similarity between two function dicts."""
    emb_a = encode_function(fn_a, tokenizer, model)
    emb_b = encode_function(fn_b, tokenizer, model)
    return (emb_a @ emb_b.T).item()


def load_model():
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.max_position_embeddings = JTRANS_MAX_POS
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=JTRANS_MAX_POS)
    model = AutoModel.from_pretrained(MODEL_NAME, config=config)
    model.eval()
    return tokenizer, model


if __name__ == "__main__":
    json_a = "data/json/normalized/hal_crc_O0_normalized.json"
    json_b = "data/json/normalized/stm32f1xx_hal_gpio_O0_normalized.json"

    # Pick which function to compare
    fn_a = get_function(json_a, "HAL_CRC_Accumulate")
    fn_b = get_function(json_b, "HAL_GPIO_EXTI_IRQHandler")

    tokenizer, model = load_model()
    score = compute_similarity(fn_a, fn_b, tokenizer, model)
    print(f"Similarity: {score:.4f}")
