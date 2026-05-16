"""
Chainlit-side bridge to the audio denoise pipeline.

Spawns audio_denoise_runner.py inside denoise_venv. Loud-logs every step.
Renamed from `denoise.py` to avoid any conflict with older Denoise.py files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Optional


_HERE = os.path.dirname(os.path.abspath(__file__))

DENOISE_PYTHON = os.environ.get(
    "DENOISE_PYTHON",
    os.path.join(_HERE, "denoise_venv", "bin", "python"),
)
DENOISE_RUNNER = os.environ.get(
    "DENOISE_RUNNER",
    os.path.join(_HERE, "audio_denoise_runner.py"),
)
DENOISE_TIMEOUT = float(os.environ.get("DENOISE_TIMEOUT", "60"))


def _log(msg: str) -> None:
    print(f"[denoise] {msg}", flush=True)


def preprocess_audio(
    input_path: str,
    output_path: Optional[str] = None,
    timeout: Optional[float] = None,
    fallback_to_raw: bool = True,
) -> str:
    """
    Run the denoise pipeline (DFN + silcut) inside denoise_venv.

    Returns path to the cleaned wav file. On failure, falls back to copying
    the raw audio to `output_path` if `fallback_to_raw=True` (default).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"preprocess_audio: input not found: {input_path}")
    if output_path is None:
        output_path = input_path + ".cleaned.wav"
    if timeout is None:
        timeout = DENOISE_TIMEOUT

    _log(f"called.  in={input_path}")
    _log(f"        out={output_path}")
    _log(f"        runner_python={DENOISE_PYTHON}")
    _log(f"        runner_script={DENOISE_RUNNER}")

    missing = []
    if not os.path.exists(DENOISE_PYTHON):
        missing.append(f"DENOISE_PYTHON not found at {DENOISE_PYTHON}")
    if not os.path.exists(DENOISE_RUNNER):
        missing.append(f"DENOISE_RUNNER not found at {DENOISE_RUNNER}")
    if missing:
        for m in missing:
            _log(f"MISSING: {m}")
        if fallback_to_raw:
            _log("falling back to RAW audio")
            shutil.copyfile(input_path, output_path)
            return output_path
        raise RuntimeError("denoise runner missing: " + "; ".join(missing))

    t0 = time.perf_counter()
    try:
        rc = subprocess.run(
            [DENOISE_PYTHON, DENOISE_RUNNER, input_path, output_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        _log(f"TIMEOUT after {elapsed:.1f}s")
        if fallback_to_raw:
            shutil.copyfile(input_path, output_path)
            return output_path
        raise

    elapsed = time.perf_counter() - t0
    _log(f"runner finished in {elapsed:.2f}s rc={rc.returncode}")

    if rc.stderr.strip():
        _log("runner stderr (last 20 lines):")
        for line in rc.stderr.strip().splitlines()[-20:]:
            print(f"  {line}", flush=True)
    if rc.stdout.strip():
        _log("runner stdout (last 5 lines):")
        for line in rc.stdout.strip().splitlines()[-5:]:
            print(f"  {line}", flush=True)

    if rc.returncode != 0 or not os.path.exists(output_path):
        _log(f"runner FAILED (rc={rc.returncode}, output_exists={os.path.exists(output_path)})")
        if fallback_to_raw:
            _log("falling back to RAW audio")
            shutil.copyfile(input_path, output_path)
            return output_path
        raise RuntimeError(f"denoise runner failed: rc={rc.returncode}")

    in_size = os.path.getsize(input_path)
    out_size = os.path.getsize(output_path)
    _log(f"sizes: input={in_size}B  cleaned={out_size}B")
    if in_size == out_size:
        _log("WARNING: cleaned file size matches input. Likely fallback fired inside runner.")

    return output_path


def smoke_test(input_path: str) -> None:
    out = preprocess_audio(input_path, fallback_to_raw=False)
    print(f"smoke_test ok -> {out}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python audio_denoise.py <input_audio_file>")
        sys.exit(2)
    smoke_test(sys.argv[1])
