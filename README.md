# STM Model — Preprocessing Pipeline

Binary similarity detection for STM32F1 HAL functions using assembly embeddings.

## Project Structure

```
preprocessing/
├── dataprocessing/
│   ├── buildingobj.py          # Compile .c → .o at multiple optimization levels
│   ├── extractFunctionsFromObj.py  # Extract function disassembly from .o → JSON
│   └── normalization.py        # Normalize ARM assembly instructions
├── model/
│   ├── clap.py                 # CLAP-asm embedding model
│   ├── asm2vec.py              # Asm2Vec embedding model
│   ├── jtrans.py / jtransplus.py   # jTrans embedding models
│   ├── tfidf.py                # TF-IDF baseline
│   └── unixcoder.py            # UniXcoder embedding model
├── testing/
│   └── clap.py                 # Similarity tests and plots
└── data/
    ├── STM32CubeF1/            # STM32 HAL source (not tracked in git)
    ├── compiled_data/          # stm32f1xx_hal_conf.h
    ├── objects/                # Compiled .o files
    └── json/                   # Extracted + normalized function JSON
```

---

## How We Build the `.o` Files

The core of the pipeline is compiling STM32F1 HAL `.c` source files into ARM object files at multiple GCC optimization levels. This is done in [`dataprocessing/buildingobj.py`](dataprocessing/buildingobj.py).

### Toolchain

You need the ARM bare-metal GCC cross-compiler:

```bash
# Ubuntu / Debian
sudo apt install gcc-arm-none-eabi

# macOS
brew install --cask gcc-arm-embedded
```

Verify with:

```bash
arm-none-eabi-gcc --version
```

### What the compiler command looks like

For each `.c` file and each optimization level, the build function runs:

```bash
arm-none-eabi-gcc -c \
  -mcpu=cortex-m3 -mthumb \
  -DSTM32F103xB -DUSE_HAL_DRIVER \
  -g \
  -O0 \                                              # or O1, O2, O3, Os
  -IDrivers/STM32F1xx_HAL_Driver/Inc \
  -IDrivers/CMSIS/Device/ST/STM32F1xx/Include \
  -IDrivers/CMSIS/Include \
  -Idata/compiled_data \
  stm32f1xx_hal_crc.c \
  -o data/objects/stm32f1xx_hal_crc_O0.o
```

Key flags explained:

| Flag | Purpose |
|------|---------|
| `-c` | Compile only, do not link |
| `-mcpu=cortex-m3 -mthumb` | Target the STM32F1 Cortex-M3 in Thumb mode |
| `-DSTM32F103xB` | Select the exact MCU variant |
| `-DUSE_HAL_DRIVER` | Enable the HAL layer |
| `-g` | Include debug symbols (needed for `nm` symbol extraction) |
| `-O0` .. `-Os` | Optimization level — each produces a different `.o` |

### Output naming

Each `.c` file is compiled at all 5 levels, producing:

```
data/objects/
├── stm32f1xx_hal_crc_O0.o
├── stm32f1xx_hal_crc_O1.o
├── stm32f1xx_hal_crc_O2.o
├── stm32f1xx_hal_crc_O3.o
└── stm32f1xx_hal_crc_Os.o
```

### Running the build

```python
from dataprocessing.buildingobj import build_all_opts

results = build_all_opts("data/STM32CubeF1/Drivers/STM32F1xx_HAL_Driver/Src/stm32f1xx_hal_crc.c")
# returns {"O0": Path(...), "O1": Path(...), ...}
```

Or directly:

```bash
python dataprocessing/buildingobj.py
```

---

## Full Pipeline

### 1. Compile → `.o`

```bash
python dataprocessing/buildingobj.py
```

### 2. Extract functions → JSON

Uses `arm-none-eabi-nm` to find function symbols and `arm-none-eabi-objdump` to disassemble them:

```bash
python dataprocessing/extractFunctionsFromObj.py
```

Output (`data/json/stm32f1xx_hal_crc_O2.json`):

```json
{
  "source_object": "data/objects/stm32f1xx_hal_crc_O2.o",
  "functions": [
    {
      "name": "HAL_CRC_Accumulate",
      "offset": 0,
      "size_bytes": 72,
      "instructions": ["push {r4 r5 r7 lr}", "sub sp #8", ...]
    }
  ]
}
```

### 3. Normalize assembly

Strips noise from instructions so the model sees semantically stable tokens:

```bash
python dataprocessing/normalization.py
```

Normalization rules applied:
- Remove `.word` padding and `nop` instructions
- Replace call targets (`bl`, `blx`) with `[CALL]`
- Replace branch targets with `[JUMP_N]` (same address → same ID)
- Strip ARM inline comments and tabs

### 4. Embed and compare

```bash
python model/clap.py      # CLAP-asm
python model/unixcoder.py # UniXcoder
python model/asm2vec.py   # Asm2Vec
```

---

## Requirements

```bash
pip install torch transformers
```

ARM toolchain binaries required on PATH:
- `arm-none-eabi-gcc`
- `arm-none-eabi-nm`
- `arm-none-eabi-objdump`
