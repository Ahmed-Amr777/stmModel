"""
Step 1 — Compile all STM32F1 HAL .c files and run extract + normalize.

For every .c file in the HAL Src directory:
  - Compile at O0/O1/O2/O3/Os → data/objects/<name>_<opt>.o
  - Extract instructions        → data/json/<name>_<opt>.json
  - Normalize to CLAP format   → data/json/<name>_<opt>_clap.json

Usage:
    python scripts/01_compile.py            # all files
    python scripts/01_compile.py --dry-run  # show what would run
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dataprocessing.buildingobj import build, CUBE_ROOT
from dataprocessing.extractFunctionsFromObj import obj_to_json
from dataprocessing.normalization import normalize_json

SRC_DIR   = ROOT / CUBE_ROOT / "Drivers/STM32F1xx_HAL_Driver/Src"
OBJ_DIR   = ROOT / "data/objects"
JSON_DIR  = ROOT / "data/json"
CLAP_DIR  = ROOT / "data/json"

SKIP = {
    "stm32f1xx_hal_msp_template.c",
    "stm32f1xx_hal_timebase_rtc_alarm_template.c",
    "stm32f1xx_hal_timebase_tim_template.c",
}

OPT_LEVELS = ["O0", "O1", "O2", "O3", "Os"]


def process_file(c_file: Path, opt: str, dry_run: bool) -> bool:
    obj_out  = OBJ_DIR  / f"{c_file.stem}_{opt}.o"
    json_out = JSON_DIR / f"{c_file.stem}_{opt}.json"
    clap_out = CLAP_DIR / f"{c_file.stem}_{opt}_clap.json"

    if dry_run:
        print(f"  [DRY] {c_file.name} -{opt} → {obj_out.name}")
        return True

    # ── compile ──
    if not obj_out.exists():
        try:
            build(str(c_file), opt, str(OBJ_DIR))
        except Exception as e:
            print(f"  [COMPILE FAIL] {c_file.name} -{opt}: {e}")
            return False
    else:
        print(f"  [SKIP compile] {obj_out.name} already exists")

    # ── extract ──
    if not json_out.exists():
        try:
            obj_to_json(str(obj_out), str(JSON_DIR))
        except Exception as e:
            print(f"  [EXTRACT FAIL] {obj_out.name}: {e}")
            return False
    else:
        print(f"  [SKIP extract] {json_out.name} already exists")

    # ── normalize ──
    if not clap_out.exists():
        try:
            normalize_json(str(json_out), str(clap_out))
        except Exception as e:
            print(f"  [NORM FAIL] {json_out.name}: {e}")
            return False
    else:
        print(f"  [SKIP norm] {clap_out.name} already exists")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--file", help="Process a single .c file name only")
    args = parser.parse_args()

    c_files = sorted(f for f in SRC_DIR.glob("*.c") if f.name not in SKIP)

    if args.file:
        c_files = [f for f in c_files if args.file in f.name]
        if not c_files:
            print(f"No matching file for: {args.file}")
            sys.exit(1)

    total   = len(c_files) * len(OPT_LEVELS)
    done    = 0
    failed  = []
    t0      = time.time()

    print(f"Files: {len(c_files)}  ×  opts: {len(OPT_LEVELS)}  =  {total} jobs\n")

    for c_file in c_files:
        print(f"[{c_file.name}]")
        for opt in OPT_LEVELS:
            ok = process_file(c_file, opt, args.dry_run)
            done += 1
            if not ok:
                failed.append(f"{c_file.name} -{opt}")
            pct = 100 * done / total
            print(f"  ({done}/{total}  {pct:.0f}%)  -{opt}  {'OK' if ok else 'FAIL'}")
        print()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s  —  {done - len(failed)}/{total} succeeded")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
