import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig

MODEL_NAME = "PurCL/jtrans-mfc"
JTRANS_MAX_POS = 2902 

class SimilarityProjector(nn.Module):
    """
    A small projection head to 'stretch' the jTrans embeddings.
    This helps prevent the 0.9999 similarity collapse.
    """
    def __init__(self, input_dim=768, output_dim=256):
        super().__init__()
        # Using a fixed seed ensures your 'projection' is consistent across runs
        torch.manual_seed(42)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)

def encode_function_improved(fn: dict, tokenizer, model, projector) -> torch.Tensor:
    """Extracts the [CLS] token and passes it through a projection head."""
    text = " ".join(fn["instructions"])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=JTRANS_MAX_POS)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # FIX 1: Use the [CLS] token (index 0) instead of .mean()
    # This represents the 'intent' of the function rather than the 'average' word
    cls_embedding = outputs.last_hidden_state[:, 0, :] 
    
    # FIX 2: Project the embedding to a different space to increase variance
    projected_emb = projector(cls_embedding)
    return projected_emb

def compute_metrics(emb_a, emb_b):
    """Compute multiple metrics to show the difference clearly."""
    cosine_sim = F.cosine_similarity(emb_a, emb_b).item()
    # L1 Distance (Manhattan) is often better at showing gaps when cosine is high
    l1_dist = torch.norm(emb_a - emb_b, p=1).item()
    return cosine_sim, l1_dist

def load_resources():
    print("Loading jTrans model...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.max_position_embeddings = JTRANS_MAX_POS
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=JTRANS_MAX_POS)
    model = AutoModel.from_pretrained(MODEL_NAME, config=config)
    model.eval()
    
    # Initialize the projector
    projector = SimilarityProjector(input_dim=config.hidden_size)
    projector.eval()
    
    return tokenizer, model, projector

if __name__ == "__main__":
    # Update these paths to your ARM JSON files
    json_a = "data/json/normalized/hal_crc_O0_normalized.json"
    #json_b = "data/json/normalized/stm32f1xx_hal_crc_O2_normalized.json"
    json_b = "data/json/normalized/stm32f1xx_hal_gpio_O0_normalized.json"
    tokenizer, model, projector = load_resources()

    try:
        # Get two different functions to test contrast
        fn_a = json.loads(Path(json_a).read_text())["functions"][0]
        fn_b = json.loads(Path(json_b).read_text())["functions"][1]

        emb_a = encode_function_improved(fn_a, tokenizer, model, projector)
        emb_b = encode_function_improved(fn_b, tokenizer, model, projector)

        score, dist = compute_metrics(emb_a, emb_b)

        print("\n" + "="*30)
        print(f"Function A: {fn_a['name']}")
        print(f"Function B: {fn_b['name']}")
        print("-" * 30)
        print(f"Cosine Similarity: {score:.6f}")
        print(f"L1 Distance:      {dist:.6f} (Higher = More Different)")
        print("="*30)

    except Exception as e:
        print(f"Error: {e}")