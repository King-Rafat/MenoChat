"""
Standalone CLI denoiser. Runs INSIDE denoise_venv.

Usage:
    python audio_denoise_runner.py <input_path> <output_path>

Pipeline: ffmpeg-to-wav 16k mono -> DeepFilterNet -> ffmpeg silcut+loudnorm.
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

    print(f"[runner] step1 to_wav: {in_path} -> {wav_path}", file=sys.stderr)
    try:
        to_wav(in_path, wav_path)
        current = wav_path

        print(f"[runner] step2 DFN: {current} -> {dfn_path}", file=sys.stderr)
        try:
            deepfilternet_denoise(current, dfn_path)
            current = dfn_path
            print(f"[runner] step2 ok", file=sys.stderr)
        except Exception as e:
            print(f"[runner] DFN FAILED, skipping: {e}", file=sys.stderr)

        print(f"[runner] step3 silcut: {current} -> {out_path}", file=sys.stderr)
        try:
            silcut_loudnorm(current, out_path)
            print(f"[runner] step3 ok", file=sys.stderr)
        except Exception as e:
            print(f"[runner] silcut FAILED, copying stage output: {e}", file=sys.stderr)
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
        print("usage: audio_denoise_runner.py <input> <output>", file=sys.stderr)
        return 2
    in_path, out_path = argv
    try:
        run_pipeline(in_path, out_path)
    except Exception as e:
        print(f"[runner] FATAL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
