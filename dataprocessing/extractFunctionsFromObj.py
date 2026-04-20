import subprocess
import re
import json
import sys
from pathlib import Path


def extract_functions(obj_path: Path) -> list[dict]:
    nm_out = subprocess.check_output(
        ["arm-none-eabi-nm", "--print-size", "--numeric-sort", str(obj_path)],
        stderr=subprocess.DEVNULL,
        text=True,
    )

    functions = []
    for line in nm_out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2].upper() == "T":
            offset, size, _, name = parts
            functions.append({
                "name": name,
                "offset": int(offset, 16),
                "size_bytes": int(size, 16),
                "instructions": [],
            })

    if not functions:
        return functions

    objdump_out = subprocess.check_output(
        ["arm-none-eabi-objdump", "-d", "--no-show-raw-insn", str(obj_path)],
        stderr=subprocess.DEVNULL,
        text=True,
    )

    func_header = re.compile(r"^[0-9a-f]+ <(.+?)>:$")
    insn_line = re.compile(r"^\s+([0-9a-f]+):\s+(.+)$")
    current_func = None
    blocks: dict[str, list[str]] = {}

    for line in objdump_out.splitlines():
        m = func_header.match(line)
        if m:
            current_func = m.group(1)
            blocks[current_func] = []
        elif current_func is not None:
            m2 = insn_line.match(line)
            if m2:
                blocks[current_func].append(m2.group(2).strip())

    for fn in functions:
        fn["instructions"] = blocks.get(fn["name"], [])

    return functions


def obj_to_json(obj_path: str, out_dir: str = "data/json") -> Path:
    obj_path = Path(obj_path)
    if not obj_path.exists():
        raise FileNotFoundError(f"Object file not found: {obj_path}")

    functions = extract_functions(obj_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / (obj_path.stem + ".json")
    payload = {
        "source_object": str(obj_path),
        "functions": functions,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    return out_file


if __name__ == "__main__":

    result = obj_to_json("data/objects/stm32f1xx_hal_crc_O2.o", "data/json")
    print(f"Saved: {result}")
