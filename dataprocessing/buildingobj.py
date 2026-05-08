import subprocess
from pathlib import Path

CUBE_ROOT = "data/STM32CubeF1"
CONF_DIR  = "data/compiled_data"   # contains stm32f1xx_hal_conf.h
OUT_DIR   = "data/objects"

INCLUDE_FLAGS = [
    f"-I{CUBE_ROOT}/Drivers/STM32F1xx_HAL_Driver/Inc",
    f"-I{CUBE_ROOT}/Drivers/CMSIS/Device/ST/STM32F1xx/Include",
    f"-I{CUBE_ROOT}/Drivers/CMSIS/Include",
    f"-I{CONF_DIR}",
]

BASE_FLAGS = [
    "arm-none-eabi-gcc", "-c",
    "-mcpu=cortex-m3", "-mthumb",
    "-DSTM32F103xB", "-DUSE_HAL_DRIVER",
    "-g",
]

OPTIMIZATION_LEVELS = ["O0", "O1", "O2", "O3", "Os"]


def build(c_file: str, opt: str = "O0", out_dir: str = OUT_DIR) -> Path:
    """
    Compile a single .c file to a .o object file.

    Args:
        c_file: path to the .c source file
        opt:    optimization level — O0, O1, O2, O3, or Os
        out_dir: directory to write the .o file into

    Returns:
        Path to the produced .o file
    """
    c_path = Path(c_file)
    out_path = Path(out_dir) / f"{c_path.stem}_{opt}.o"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = BASE_FLAGS + [f"-{opt}"] + INCLUDE_FLAGS + [str(c_path), "-o", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def build_all_opts(c_file: str, out_dir: str = OUT_DIR) -> dict[str, Path]:
    """Compile a .c file at every optimization level. Returns {opt: path}."""
    return {opt: build(c_file, opt, out_dir) for opt in OPTIMIZATION_LEVELS}


if __name__ == "__main__":
    src = f"{CUBE_ROOT}/Drivers/STM32F1xx_HAL_Driver/Src/stm32f1xx_hal_crc.c"

    results = build_all_opts(src)
    for opt, path in results.items():
        print(f"{opt}: {path}")
