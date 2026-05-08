"""
Extract Ghidra P-code operations from ARM object files and save as JSON.

Output mirrors extractFunctionsFromObj.py but replaces the "instructions" list
with a "pcode" list of P-code operation strings.

Requires:
    pip install pypcode pyelftools
    arm-none-eabi-nm  (on PATH)

Usage:
    python dataprocessing/pcode.py                      # all .o in data/objects/
    python dataprocessing/pcode.py data/objects/foo.o   # single file
"""

import json
import subprocess
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile
from pypcode import Context, PcodePrettyPrinter

# Ghidra SLEIGH language ID for ARM Cortex-M (Thumb/Thumb-2, little-endian, 32-bit)
SLEIGH_LANG = "ARM:LE:32:Cortex"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_function_symbols(obj_path: Path) -> list[dict]:
    """Use arm-none-eabi-nm to enumerate .text (T) symbols with offsets and sizes."""
    out = subprocess.check_output(
        ["arm-none-eabi-nm", "--print-size", "--numeric-sort", str(obj_path)],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    functions = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2].upper() == "T":
            offset, size, _, name = parts
            functions.append({
                "name": name,
                "offset": int(offset, 16),
                "size_bytes": int(size, 16),
            })
    return functions


def _load_text_section(obj_path: Path) -> tuple[bytes, int]:
    """
    Return (raw_bytes, section_vma) for the .text section using pyelftools.
    VMA is the base address used when generating absolute P-code addresses.
    """
    with obj_path.open("rb") as f:
        elf = ELFFile(f)
        section = elf.get_section_by_name(".text")
        if section is None:
            raise ValueError(f"No .text section in {obj_path}")
        return section.data(), section["sh_addr"]


def _ops_to_strings(translation) -> list[str]:
    """Convert all P-code ops from a pypcode Translation into printable strings."""
    return [PcodePrettyPrinter.fmt_op(op) for op in translation.ops]


# ──────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────

def extract_pcode_functions(obj_path: Path) -> list[dict]:
    functions = _get_function_symbols(obj_path)
    if not functions:
        return []

    text_bytes, text_vma = _load_text_section(obj_path)
    ctx = Context(SLEIGH_LANG)

    for fn in functions:
        start = fn["offset"] - text_vma  # byte index into text_bytes
        fn_bytes = text_bytes[start: start + fn["size_bytes"]]

        if not fn_bytes:
            fn["pcode"] = []
            fn["num_pcode_ops"] = 0
            continue

        # translate() lifts the bytes to P-code; base_address keeps addresses meaningful
        tx = ctx.translate(fn_bytes, fn["offset"], 0, 0, 4096)
        pcode_ops = _ops_to_strings(tx)
        fn["pcode"] = pcode_ops
        fn["num_pcode_ops"] = len(pcode_ops)

    return functions


# ──────────────────────────────────────────────
# JSON output
# ──────────────────────────────────────────────

def obj_to_pcode_json(obj_path: str, out_dir: str = "data/json/pcode") -> Path:
    obj_path = Path(obj_path)
    if not obj_path.exists():
        raise FileNotFoundError(f"Object file not found: {obj_path}")

    functions = extract_pcode_functions(obj_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / (obj_path.stem + "_pcode.json")
    payload = {
        "source_object": str(obj_path),
        "sleigh_lang": SLEIGH_LANG,
        "functions": functions,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    return out_file


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = sorted(Path("data/objects").glob("*.o"))

    for obj_file in targets:
        try:
            result = obj_to_pcode_json(str(obj_file))
            fns = json.loads(result.read_text())["functions"]
            total_ops = sum(f["num_pcode_ops"] for f in fns)
            print(f"Saved: {result}  ({len(fns)} functions, {total_ops} P-code ops)")
        except Exception as e:
            print(f"ERROR {obj_file}: {e}", file=sys.stderr)
