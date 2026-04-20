import re
import json
from pathlib import Path

# Any b* mnemonic that is NOT bl/blx/bx
BRANCH_RE = re.compile(r"^b(?!l|x)\S*$")

# Matches objdump targets:  2e2 <Func+0x2e2>  or bare hex  2e2
TARGET_WITH_LABEL = re.compile(r"([0-9a-f]+)\s+<[^>]+>")
TARGET_BARE       = re.compile(r"^([0-9a-f]+)$")

# Matches call targets: bl/blx with address
CALL_RE = re.compile(r"^(bl|blx)$")


def normalize_function(fn: dict) -> dict:
    jump_map: dict[str, int] = {}

    def get_jump_id(addr: str) -> str:
        if addr not in jump_map:
            jump_map[addr] = len(jump_map) + 1
        return f"[JUMP_{jump_map[addr]}]"

    def normalize_instruction(raw: str) -> str:
        # Remove tabs and commas
        instr = re.sub(r"\t+", " ", raw).strip().replace(",", "")
        # Remove ARM inline comments
        instr = re.sub(r"\s*@.*$", "", instr).strip()

        # Rule 1: delete .word lines
        if instr.startswith(".word"):
            return None

        # Rule 2: delete nop
        if instr == "nop":
            return None

        parts = instr.split(" ", 1)
        mnemonic = parts[0].lower()
        operands = parts[1] if len(parts) > 1 else ""

        # Rule 3: bx lr → keep as return
        if mnemonic == "bx" and operands.strip() == "lr":
            return instr

        # Rule 4: bl/blx → [CALL]
        if CALL_RE.match(mnemonic):
            operands = TARGET_WITH_LABEL.sub("[CALL]", operands)
            operands = TARGET_BARE.sub("[CALL]", operands.strip())
            return f"{mnemonic} {operands}".strip()

        # Rule 5: any branch → [JUMP_N] (same target addr = same ID)
        if BRANCH_RE.match(mnemonic):
            m = TARGET_WITH_LABEL.search(operands)
            if m:
                token = get_jump_id(m.group(1))
                operands = TARGET_WITH_LABEL.sub(token, operands)
            else:
                m2 = TARGET_BARE.match(operands.strip())
                if m2:
                    operands = get_jump_id(m2.group(1))
            return f"{mnemonic} {operands}".strip()

        # Rule 6: everything else — keep
        return instr

    normalized_instructions = [
        n for i in fn["instructions"]
        if (n := normalize_instruction(i)) is not None
    ]

    return {
        "name": fn["name"],
        "offset": fn["offset"],
        "size_bytes": fn["size_bytes"],
        "instructions": normalized_instructions,
    }


def normalize_json(input_path: str, output_dir: str = None) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSON not found: {input_path}")

    data = json.loads(input_path.read_text())

    normalized = {
        "source_object": data["source_object"],
        "functions": [normalize_function(fn) for fn in data["functions"]],
    }

    out_dir = Path(output_dir) if output_dir else Path("data/json/normalized")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (input_path.stem + "_normalized.json")
    out_file.write_text(json.dumps(normalized, indent=2))
    return out_file


if __name__ == "__main__":
    result = normalize_json("data/json/stm32f1xx_hal_crc_O2.json")
    print(f"Saved: {result}")
