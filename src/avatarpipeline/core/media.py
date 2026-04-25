"""Shared ffmpeg/ffprobe helpers used across pipelines."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(cmd: list[str], error_message: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{error_message}:\n{result.stderr[-1000:]}")
    return result


def audio_duration(path: str | Path) -> float:
    """Return audio duration in seconds, or 0.0 if probing fails."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def video_info(
    path: str | Path,
    *,
    default_width: int = 1920,
    default_height: int = 1080,
    default_fps: float = 25.0,
) -> dict:
    """Return width, height, duration, and fps for a video."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        stream = json.loads(result.stdout)["streams"][0]
        num, den = stream.get("r_frame_rate", f"{int(default_fps)}/1").split("/")
        fps = float(num) / float(den)
        return {
            "width": int(stream.get("width", default_width)),
            "height": int(stream.get("height", default_height)),
            "duration": float(stream.get("duration", 0.0)),
            "fps": fps,
        }
    except Exception:
        return {
            "width": default_width,
            "height": default_height,
            "duration": 0.0,
            "fps": default_fps,
        }


def concat_audio(inputs: list[str | Path], output_path: str | Path) -> None:
    """Concatenate audio files in order into a PCM WAV."""
    if not inputs:
        raise ValueError("concat_audio requires at least one input file.")
    ffmpeg_inputs: list[str] = []
    for path in inputs:
        ffmpeg_inputs.extend(["-i", str(path)])
    _run(
        [
            "ffmpeg", "-y",
            *ffmpeg_inputs,
            "-filter_complex", f"concat=n={len(inputs)}:v=0:a=1[aout]",
            "-map", "[aout]",
            "-c:a", "pcm_s16le",
            str(output_path),
        ],
        "Audio concat failed",
    )


def generate_silence(duration: float, output_path: str | Path) -> None:
    """Write a silent 16 kHz mono PCM WAV."""
    duration = max(0.05, float(duration))
    _run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", str(duration),
            "-c:a", "pcm_s16le",
            str(output_path),
        ],
        "Silence generation failed",
    )


def resample_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int,
    channels: int,
) -> None:
    """Resample audio to the requested PCM WAV shape."""
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-c:a", "pcm_s16le",
            str(output_path),
        ],
        "Audio resample failed",
    )


def normalize_to_16k_mono(input_path: str | Path, output_path: str | Path) -> None:
    """Convert audio to 16 kHz mono PCM WAV."""
    resample_audio(input_path, output_path, sample_rate=16_000, channels=1)
