"""
Step 4 — Build training/val/test dataset from CLAP JSONs.

Reads all data/json/*_clap.json files, groups function instances by name,
splits function NAMES 70/15/15, then generates all (anchor, positive) pairs
within the same function name across different optimization levels.

Outputs:
    data/training/functions.jsonl   — one function instance per line
    data/training/pairs_train.jsonl
    data/training/pairs_val.jsonl
    data/training/pairs_test.jsonl
    data/training/splits.json       — summary stats

Usage:
    python scripts/04_build_dataset.py
    python scripts/04_build_dataset.py --min-instances 2  # default
    python scripts/04_build_dataset.py --seed 42
"""

import json
import random
import argparse
import itertools
from pathlib import Path
from collections import defaultdict

ROOT      = Path(__file__).parent.parent
CLAP_DIR  = ROOT / "data/json"
OUT_DIR   = ROOT / "data/training"


# ──────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────

def load_all_clap_jsons(clap_dir: Path) -> list[dict]:
    """
    Walk all *_clap.json files and return a flat list of function records:
      {id, name, source, opt, instructions}
    """
    records = []
    for json_file in sorted(clap_dir.glob("*_clap.json")):
        # Parse source name and opt level from filename: <source>_<opt>_clap.json
        stem = json_file.stem                     # e.g. stm32f1xx_hal_crc_O0_clap
        stem = stem[: -len("_clap")]              # stm32f1xx_hal_crc_O0
        # opt is the last underscore-separated token
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            print(f"  [SKIP] unrecognized filename: {json_file.name}")
            continue
        source, opt = parts[0], parts[1]
        if opt not in {"O0", "O1", "O2", "O3", "Os"}:
            print(f"  [SKIP] unrecognized opt '{opt}' in {json_file.name}")
            continue

        data = json.loads(json_file.read_text())
        for fn in data["functions"]:
            if not fn.get("instructions"):
                continue
            rec_id = f"{fn['name']}|{source}|{opt}"
            records.append({
                "id":           rec_id,
                "name":         fn["name"],
                "source":       source,
                "opt":          opt,
                "instructions": fn["instructions"],
            })

    return records


# ──────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────

def split_names(names: list[str], train=0.70, val=0.15, seed=42) -> tuple[set, set, set]:
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val   = int(n * val)
    return (
        set(shuffled[:n_train]),
        set(shuffled[n_train: n_train + n_val]),
        set(shuffled[n_train + n_val:]),
    )


# ──────────────────────────────────────────────
# Pairs
# ──────────────────────────────────────────────

def make_pairs(instances: list[dict]) -> list[dict]:
    """All ordered (anchor, positive) pairs where anchor opt != positive opt."""
    pairs = []
    for a, b in itertools.combinations(instances, 2):
        if a["opt"] != b["opt"]:
            pairs.append({"anchor_id": a["id"], "positive_id": b["id"]})
    return pairs


# ──────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────

def write_jsonl(path: Path, records: list[dict]):
    path.write_text("\n".join(json.dumps(r) for r in records))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-instances", type=int, default=2,
                        help="Minimum opt-level instances required to keep a function")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──
    print("Loading CLAP JSONs...")
    records = load_all_clap_jsons(CLAP_DIR)
    print(f"  {len(records)} function instances from {CLAP_DIR}")

    # ── Group by name ──
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_name[r["name"]].append(r)

    # ── Filter: keep only names with enough instances ──
    qualified = {
        name: insts
        for name, insts in by_name.items()
        if len(insts) >= args.min_instances
    }
    dropped = len(by_name) - len(qualified)
    print(f"  {len(by_name)} unique function names -> "
          f"{len(qualified)} kept (>={args.min_instances} instances), "
          f"{dropped} dropped")

    # ── Split function names ──
    all_names   = sorted(qualified.keys())
    train_names, val_names, test_names = split_names(
        all_names, train=args.train, val=args.val, seed=args.seed
    )
    print(f"\n  Split: train={len(train_names)}  val={len(val_names)}  test={len(test_names)}")

    # ── Collect all instances (preserve split label for reference) ──
    def label(name):
        if name in train_names: return "train"
        if name in val_names:   return "val"
        return "test"

    all_instances = []
    for name, insts in qualified.items():
        for inst in insts:
            inst["split"] = label(name)
            all_instances.append(inst)

    # ── Generate pairs ──
    pairs_train, pairs_val, pairs_test = [], [], []
    for name in train_names:
        pairs_train.extend(make_pairs(qualified[name]))
    for name in val_names:
        pairs_val.extend(make_pairs(qualified[name]))
    for name in test_names:
        pairs_test.extend(make_pairs(qualified[name]))

    total_pairs = len(pairs_train) + len(pairs_val) + len(pairs_test)
    print(f"\n  Pairs: train={len(pairs_train)}  val={len(pairs_val)}  test={len(pairs_test)}  total={total_pairs}")

    # ── Write ──
    print("\nWriting files...")

    fn_path = OUT_DIR / "functions.jsonl"
    write_jsonl(fn_path, all_instances)
    print(f"  {fn_path.relative_to(ROOT)}  ({len(all_instances)} records)")

    for split, pairs in [("train", pairs_train), ("val", pairs_val), ("test", pairs_test)]:
        p = OUT_DIR / f"pairs_{split}.jsonl"
        write_jsonl(p, pairs)
        print(f"  {p.relative_to(ROOT)}  ({len(pairs)} pairs)")

    splits_summary = {
        "seed":          args.seed,
        "min_instances": args.min_instances,
        "total_functions":  len(qualified),
        "total_instances":  len(all_instances),
        "total_pairs":      total_pairs,
        "train": {"functions": len(train_names), "instances": sum(len(qualified[n]) for n in train_names), "pairs": len(pairs_train)},
        "val":   {"functions": len(val_names),   "instances": sum(len(qualified[n]) for n in val_names),   "pairs": len(pairs_val)},
        "test":  {"functions": len(test_names),  "instances": sum(len(qualified[n]) for n in test_names),  "pairs": len(pairs_test)},
        "function_names": {
            "train": sorted(train_names),
            "val":   sorted(val_names),
            "test":  sorted(test_names),
        }
    }
    splits_path = OUT_DIR / "splits.json"
    splits_path.write_text(json.dumps(splits_summary, indent=2))
    print(f"  {splits_path.relative_to(ROOT)}")

    print("\nDone.")
    print(f"\nSummary:")
    print(f"  Functions : {len(qualified)}")
    print(f"  Instances : {len(all_instances)}")
    print(f"  Pairs     : {total_pairs}")
    for split in ("train", "val", "test"):
        s = splits_summary[split]
        print(f"    {split:5s} — {s['functions']:3d} funcs  {s['instances']:4d} instances  {s['pairs']:5d} pairs")


if __name__ == "__main__":
    main()
