"""
Step 6 — Fine-tune CLAP-asm with LoRA + InfoNCE contrastive loss.

Setup:
  - LoRA adapters on query/value attention layers (rank 8)
  - InfoNCE loss with in-batch negatives (temperature 0.07)
  - Validate on val pairs each epoch, save best model by val loss
  - fp16 training for RTX 3060 6GB

Outputs:
  models/finetuned/lora_adapters/   (best checkpoint)
  results/training_log.csv

Usage:
    python scripts/06_finetune.py
    python scripts/06_finetune.py --config configs/finetune_config.yaml
    python scripts/06_finetune.py --epochs 5 --batch-size 8
"""

import sys
import csv
import json
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

ROOT      = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "data/training"
MODEL_DIR = ROOT / "models/finetuned"
RESULTS   = ROOT / "results"


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

def load_config(path: str, overrides: dict) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in overrides.items():
        if v is not None:
            # support dotted keys like training.batch_size
            keys = k.split(".")
            d = cfg
            for key in keys[:-1]:
                d = d[key]
            d[keys[-1]] = v
    return cfg


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

def load_functions_index() -> dict[str, list[str]]:
    """Returns {id: instructions_list}."""
    index = {}
    for line in (TRAIN_DIR / "functions.jsonl").read_text().splitlines():
        r = json.loads(line)
        index[r["id"]] = r["instructions"]
    return index


class PairDataset(Dataset):
    def __init__(self, pairs_file: Path, fn_index: dict):
        self.pairs = [json.loads(l) for l in pairs_file.read_text().splitlines()]
        self.fn_index = fn_index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        anchor   = self.fn_index[pair["anchor_id"]]
        positive = self.fn_index[pair["positive_id"]]
        return anchor, positive


def collate_fn(batch):
    anchors   = [b[0] for b in batch]
    positives = [b[1] for b in batch]
    return anchors, positives


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

def build_model(cfg: dict):
    model_name = cfg["model_name"]
    print(f"Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    lora_cfg = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
    )
    mdl = get_peft_model(mdl, lora_cfg)
    mdl.print_trainable_parameters()
    return tok, mdl


# ──────────────────────────────────────────────
# Embedding + Loss
# ──────────────────────────────────────────────

def encode(instructions_list: list[list[str]], tok, mdl, device, max_length: int):
    dicts  = [{str(i): instr for i, instr in enumerate(instrs)}
              for instrs in instructions_list]
    inputs = tok(dicts, padding=True, return_tensors="pt")
    # manually truncate token ids to max_length
    for k in inputs:
        if inputs[k].dim() == 2 and inputs[k].shape[1] > max_length:
            inputs[k] = inputs[k][:, :max_length]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return mdl(**inputs)   # L2-normalised (B, H)


def infonce_loss(anchor_emb: torch.Tensor,
                 positive_emb: torch.Tensor,
                 temperature: float) -> torch.Tensor:
    """
    Symmetric InfoNCE with in-batch negatives.
    anchor_emb, positive_emb: (B, H) L2-normalised.
    Diagonal of the (B,B) similarity matrix = correct pairs.
    """
    logits = (anchor_emb @ positive_emb.T) / temperature   # (B, B)
    labels = torch.arange(len(anchor_emb), device=anchor_emb.device)
    loss_a = F.cross_entropy(logits,   labels)   # anchor -> positive
    loss_p = F.cross_entropy(logits.T, labels)   # positive -> anchor
    return (loss_a + loss_p) / 2


# ──────────────────────────────────────────────
# Train / Val loops
# ──────────────────────────────────────────────

def run_epoch(loader, tok, mdl, device, cfg, optimizer=None,
              scheduler=None, scaler=None, train=True):
    mdl.train(train)
    total_loss = 0.0
    temperature = cfg["training"]["temperature"]
    max_length  = cfg["max_length"]

    for anchors, positives in loader:
        if train:
            optimizer.zero_grad()

        with autocast(enabled=(cfg["hardware"]["fp16"] and device == "cuda")):
            a_emb = encode(anchors,   tok, mdl, device, max_length)
            p_emb = encode(positives, tok, mdl, device, max_length)
            loss  = infonce_loss(a_emb, p_emb, temperature)

        if train:
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                optimizer.step()
            if scheduler:
                scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/finetune_config.yaml")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(
        ROOT / args.config,
        {"training.epochs":     args.epochs,
         "training.batch_size": args.batch_size,
         "training.learning_rate": args.lr}
    )

    device = cfg["hardware"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        cfg["hardware"]["fp16"] = False
    print(f"Device: {device}\n")

    # ── Data ──
    fn_index = load_functions_index()
    train_ds = PairDataset(TRAIN_DIR / "pairs_train.jsonl", fn_index)
    val_ds   = PairDataset(TRAIN_DIR / "pairs_val.jsonl",   fn_index)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"Train pairs: {len(train_ds)}  |  Val pairs: {len(val_ds)}")
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}\n")

    # ── Model ──
    tok, mdl = build_model(cfg)
    mdl = mdl.to(device)

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, mdl.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=0.01,
    )
    total_steps   = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps  = cfg["training"]["warmup_steps"]
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = GradScaler() if (cfg["hardware"]["fp16"] and device == "cuda") else None

    # ── Training loop ──
    log_path  = RESULTS / "training_log.csv"
    save_path = MODEL_DIR / "lora_adapters"
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    log_rows = []

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Time':>7}  {'Best':>5}")
    print("-" * 48)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_loss = run_epoch(train_loader, tok, mdl, device, cfg,
                               optimizer, scheduler, scaler, train=True)

        with torch.no_grad():
            val_loss = run_epoch(val_loader, tok, mdl, device, cfg, train=False)

        elapsed = time.time() - t0
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            mdl.save_pretrained(str(save_path))
            tok.save_pretrained(str(save_path))

        marker = " <--" if is_best else ""
        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  {elapsed:>6.0f}s{marker}")

        log_rows.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "best": is_best,
        })

    # ── Save log ──
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","best"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Adapters saved: {save_path.relative_to(ROOT)}")
    print(f"Training log  : {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
