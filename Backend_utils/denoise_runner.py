"""
Standalone CLI denoiser. Runs INSIDE its own venv (denoise_venv) so its
torch / torchaudio / deepfilternet versions do not have to match whatever
the chainlit env has.

Usage:
    python denoise_runner.py <input_path> <output_path>

Pipeline: ffmpeg-to-wav 16k mono -> DeepFilterNet -> ffmpeg silcut+loudnorm.
If DeepFilterNet fails to load or run, the pipeline still produces a wav
(just without the DFN step), so the caller never gets stuck without output.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from typing import Optional


DEFAULT_SR = 16000


def to_wav(in_path: str, out_path: str, sr: int = DEFAULT_SR) -> None:
    cmd = (
        f'ffmpeg -y -i {shlex.quote(in_path)} '
        f'-ac 1 -c:a pcm_s16le -ar {sr} {shlex.quote(out_path)}'
    )
    rc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if rc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"to_wav failed: {rc.stderr[-500:]}")


def silcut_loudnorm(in_path: str, out_path: str, sr: int = DEFAULT_SR) -> None:
    af = (
        "silenceremove="
        "start_periods=1:stop_periods=-1:"
        "start_threshold=-50dB:stop_threshold=-50dB:"
        "start_silence=0.2:stop_silence=0.2,"
        "loudnorm"
    )
    cmd = (
        f'ffmpeg -y -i {shlex.quote(in_path)} '
        f'-af "{af}" -c:a pcm_s16le -ar {sr} {shlex.quote(out_path)}'
    )
    rc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if rc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"silcut_loudnorm failed: {rc.stderr[-500:]}")


def deepfilternet_denoise(in_path: str, out_path: str) -> None:
    """Imports happen here so a missing package is reported per-call,
    not at module import time."""
    from df.enhance import enhance, init_df, load_audio, save_audio  # type: ignore

    model, df_state, _ = init_df()
    audio, _ = load_audio(in_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(out_path, enhanced, df_state.sr())


def run_pipeline(in_path: str, out_path: str) -> None:
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"input not found: {in_path}")

    wav_path = out_path + ".step1.wav"
    dfn_path = out_path + ".step2.wav"
    intermediates = [wav_path, dfn_path]

    try:
        # Step 1: normalise to wav 16k mono.
        to_wav(in_path, wav_path)
        current = wav_path

        # Step 2: DeepFilterNet (skipped on failure).
        try:
            deepfilternet_denoise(current, dfn_path)
            current = dfn_path
        except Exception as e:
            print(f"[denoise_runner] DFN failed, skipping: {e}", file=sys.stderr)

        # Step 3: silcut + loudnorm.
        try:
            silcut_loudnorm(current, out_path)
        except Exception as e:
            print(
                f"[denoise_runner] silcut failed, using stage output as final: {e}",
                file=sys.stderr,
            )
            if current != out_path:
                shutil.copyfile(current, out_path)

    finally:
        for p in intermediates:
            if p != out_path and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: denoise_runner.py <input> <output>", file=sys.stderr)
        return 2
    in_path, out_path = argv
    try:
        run_pipeline(in_path, out_path)
    except Exception as e:
        print(f"[denoise_runner] fatal: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
