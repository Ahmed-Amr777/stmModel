"""
Step 1b — Compile FreeRTOS source files at all optimization levels.

Compiles:
  FreeRTOS core : tasks, queue, list, event_groups, timers, stream_buffer, croutine
  Port layer    : portable/GCC/ARM_CM3/port.c  (Cortex-M3)
  CMSIS wrappers: CMSIS_RTOS/cmsis_os.c, CMSIS_RTOS_V2/cmsis_os2.c

Then runs extract + normalize on each .o → data/json/<name>_<opt>_clap.json

Usage:
    python scripts/01b_compile_rtos.py
"""

import sys
import subprocess
import time
from pathlib import Path

ROOT      = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dataprocessing.extractFunctionsFromObj import obj_to_json
from dataprocessing.normalization import normalize_json

CUBE_ROOT  = ROOT / "data/STM32CubeF1"
CONF_DIR   = ROOT / "data/compiled_data"
OBJ_DIR    = ROOT / "data/objects"
JSON_DIR   = ROOT / "data/json"

FREERTOS   = CUBE_ROOT / "Middlewares/Third_Party/FreeRTOS/Source"
FREERTOS_CONFIG = CUBE_ROOT / "Projects/STM3210C_EVAL/Applications/FreeRTOS/FreeRTOS_ThreadCreation/Inc"

OPT_LEVELS = ["O0", "O1", "O2", "O3", "Os"]

BASE_FLAGS = [
    "arm-none-eabi-gcc", "-c",
    "-mcpu=cortex-m3", "-mthumb",
    "-DSTM32F103xB", "-DUSE_HAL_DRIVER",
    "-g",
]

INCLUDE_FLAGS = [
    f"-I{FREERTOS}/include",
    f"-I{FREERTOS}/portable/GCC/ARM_CM3",
    f"-I{FREERTOS_CONFIG}",
    f"-I{CUBE_ROOT}/Drivers/STM32F1xx_HAL_Driver/Inc",
    f"-I{CUBE_ROOT}/Drivers/CMSIS/Device/ST/STM32F1xx/Include",
    f"-I{CUBE_ROOT}/Drivers/CMSIS/Include",
    f"-I{CONF_DIR}",
]

# Files to compile: (label_name, relative_path_from_FREERTOS)
RTOS_FILES = [
    ("freertos_tasks",        FREERTOS / "tasks.c"),
    ("freertos_queue",        FREERTOS / "queue.c"),
    ("freertos_list",         FREERTOS / "list.c"),
    ("freertos_event_groups", FREERTOS / "event_groups.c"),
    ("freertos_timers",       FREERTOS / "timers.c"),
    ("freertos_stream_buffer",FREERTOS / "stream_buffer.c"),
    ("freertos_croutine",     FREERTOS / "croutine.c"),
    ("freertos_port",         FREERTOS / "portable/GCC/ARM_CM3/port.c"),
    ("freertos_cmsis_os",     FREERTOS / "CMSIS_RTOS/cmsis_os.c"),
]


def process(label: str, src: Path, opt: str) -> bool:
    obj_out  = OBJ_DIR  / f"{label}_{opt}.o"
    json_out = JSON_DIR / f"{label}_{opt}.json"
    clap_out = JSON_DIR / f"{label}_{opt}_clap.json"

    # ── compile ──
    if not obj_out.exists():
        cmd = BASE_FLAGS + [f"-{opt}"] + INCLUDE_FLAGS + [str(src), "-o", str(obj_out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [FAIL] {label} -{opt}: {result.stderr.strip()[:120]}")
            return False
    else:
        print(f"  [SKIP compile] {obj_out.name}")

    # ── extract ──
    if not json_out.exists():
        try:
            obj_to_json(str(obj_out), str(JSON_DIR))
        except Exception as e:
            print(f"  [FAIL extract] {label} -{opt}: {e}")
            return False
    else:
        print(f"  [SKIP extract] {json_out.name}")

    # ── normalize ──
    if not clap_out.exists():
        try:
            normalize_json(str(json_out), str(clap_out))
        except Exception as e:
            print(f"  [FAIL norm] {label} -{opt}: {e}")
            return False
    else:
        print(f"  [SKIP norm] {clap_out.name}")

    return True


def main():
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    total  = len(RTOS_FILES) * len(OPT_LEVELS)
    done   = 0
    failed = []
    t0     = time.time()

    print(f"RTOS files: {len(RTOS_FILES)}  x  opts: {len(OPT_LEVELS)}  =  {total} jobs\n")

    for label, src in RTOS_FILES:
        print(f"[{label}]")
        for opt in OPT_LEVELS:
            ok = process(label, src, opt)
            done += 1
            if not ok:
                failed.append(f"{label} -{opt}")
            print(f"  ({done}/{total}  {100*done//total}%)  -{opt}  {'OK' if ok else 'FAIL'}")
        print()

    elapsed = time.time() - t0
    print("=" * 55)
    print(f"Done in {elapsed:.1f}s  —  {done - len(failed)}/{total} succeeded")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
