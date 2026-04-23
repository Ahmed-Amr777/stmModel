from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)

print(f"Vocab size: {tokenizer.vocab_size}")
print()

# === TEST 1: Which ARM tokens are in the vocabulary? ===
print("=== ARM tokens: known vs split ===")
arm_tokens = [
    # Mnemonics
    'push', 'pop', 'str', 'ldr', 'add', 'sub', 'mov', 'movs',
    'cmp', 'and', 'orr', 'eor', 'mvn', 'bic',
    'lsl', 'lsr', 'asr',
    'ldrb', 'strb', 'ldrh', 'strh',
    'uxtb', 'uxth', 'sxtb',
    'mul', 'mla', 'nop',
    # Branches
    'b', 'bl', 'bx', 'blx',
    'beq', 'bne', 'bgt', 'bge', 'blt', 'ble',
    'bhi', 'bcc', 'bcs', 'bls',
    'cbz', 'cbnz',
    # Registers
    'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7',
    'r8', 'r9', 'r10', 'r11', 'r12',
    'sp', 'lr', 'pc',
    # Our special tokens
    'INSTR', 'INSTR1', 'INSTR10', 'INSTR20',
    'EXTFUNC',
    # Symbols
    '{', '}', '[', ']', '#', ',',
]

known = []
split = []
for token in arm_tokens:
    pieces = tokenizer.tokenize(token)
    if len(pieces) == 1 and not pieces[0].startswith('##'):
        known.append(token)
        status = "✅ KNOWN"
    else:
        split.append(token)
        status = "⚠️ SPLIT"
    print(f"  {token:15s} → {str(pieces):40s} {status}")

print(f"\nKnown as single token: {len(known)}/{len(arm_tokens)}")
print(f"Split into subwords:   {len(split)}/{len(arm_tokens)}")

# === TEST 2: How does CLAP tokenize full instructions? ===
print("\n=== Full instruction tokenization ===")
test_instructions = [
    "push {r7, lr}",
    "sub sp, #28",
    "str r0, [r7, #12]",
    "ldr r3, [r7, #4]",
    "cmp r3, #0",
    "bne INSTR10",
    "cbz r0, INSTR17",
    "bl EXTFUNC",
    "movs r3, #1",
    "strb r2, [r3, #5]",
    "ldr.w r3, [r2, r3, lsl #2]",
    "and.w r2, r3, #255",
    "bx lr",
    "pop {r4, pc}",
]

for instr in test_instructions:
    tokens = tokenizer.tokenize(instr)
    unk = sum(1 for t in tokens if t == '[UNK]')
    print(f"  {instr:35s} → {tokens}")
    if unk > 0:
        print(f"    ⚠️ {unk} UNK tokens!")

# === TEST 3: Does INSTR format work? ===
print("\n=== INSTR token check ===")
for i in [1, 5, 10, 15, 20, 30, 50]:
    token = f"INSTR{i}"
    pieces = tokenizer.tokenize(token)
    print(f"  {token:15s} → {pieces}")