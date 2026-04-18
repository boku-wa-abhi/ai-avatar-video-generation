"""
avatarpipeline.narration.composer — Compose a narrated video from PPTX + JSON.

For each slide (in order):
  1. Display the slide image.
  2. Play its narration audio (generated via Kokoro TTS).
  3. Hold the slide for a configurable pause before advancing.

The final output is a single MP4 file where every slide is time-locked to its
narration.

Public API
----------
``compose_narrated_video(pptx_path, json_data, output_path, voice, pause_seconds)``
    Generator that yields ``(status_message: str, result_path: str | None)``
    tuples.  The final successful yield has ``result_path`` set to the output
    MP4.  All preceding yields have ``result_path = None``.
"""
from __future__ import annotations

import json as _json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Generator

from loguru import logger

from avatarpipeline import OUTPUT_DIR
from .validator import validate_sync
from .slide_renderer import render_slides

DEFAULT_PAUSE: float = 1.5


# ── Low-level ffmpeg helpers ─────────────────────────────────────────────────

def _audio_duration(path: str | Path) -> float:
    """Return the duration of an audio/video file in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return 0.0
    data = _json.loads(r.stdout)
    for stream in data.get("streams", []):
        dur = stream.get("duration")
        if dur:
            return float(dur)
    return 0.0


def _gen_silence(duration: float, output_path: str) -> None:
    """Write a silent WAV of the requested length."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", str(duration),
        "-c:a", "pcm_s16le",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Silence generation failed:\n{r.stderr[-500:]}")


def _concat_audio(paths: list[str], output_path: str) -> None:
    """Concatenate audio files in order using ffmpeg filter_complex."""
    inputs: list[str] = []
    for p in paths:
        inputs += ["-i", p]
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", f"concat=n={len(paths)}:v=0:a=1[aout]",
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Audio concat failed:\n{r.stderr[-500:]}")


# ── Public composer ──────────────────────────────────────────────────────────

def compose_narrated_video(
    pptx_path: str | Path,
    json_data: dict,
    output_path: str | Path,
    voice: str = "af_heart",
    pause_seconds: float = DEFAULT_PAUSE,
) -> Generator[tuple[str, str | None], None, None]:
    """Compose a narrated presentation video.

    This is a **generator**.  Iterate it to drive the pipeline and receive
    real-time status messages suitable for displaying in a progress UI.

    Yields:
        ``(message, result_path)`` where ``result_path`` is ``None`` for all
        progress messages and the final output MP4 path on completion.

    Raises:
        :class:`ValueError`  — when validation fails (errors surfaced via message
                               before raising).
        :class:`RuntimeError` — when an ffmpeg step fails.
    """
    pptx_path = Path(pptx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = output_path.parent / f"_narr_{run_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step 1: Validate sync ────────────────────────────────────────────
        yield "Validating sync…", None
        result = validate_sync(pptx_path, json_data)
        if not result.ok:
            raise ValueError("Validation failed:\n" + "\n".join(result.errors))
        yield f"Validation passed — {result.slide_count} slides ✓", None

        # ── Step 2: Render slides ────────────────────────────────────────────
        yield "Rendering slides…", None
        slides_dir = work_dir / "slides"
        slide_images = render_slides(pptx_path, slides_dir)
        if len(slide_images) != result.slide_count:
            raise RuntimeError(
                f"Slide render produced {len(slide_images)} images "
                f"but expected {result.slide_count}."
            )
        yield f"Slides rendered ({len(slide_images)} images)", None

        # ── Step 3: TTS per slide ────────────────────────────────────────────
        yield "Loading TTS engine…", None
        from avatarpipeline.voice.kokoro import VoiceGenerator
        vg = VoiceGenerator()

        entries = sorted(json_data["slides"], key=lambda e: int(e["slide_number"]))
        audio_dir = work_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        audio_paths: list[str] = []
        n = len(entries)
        for idx, entry in enumerate(entries):
            slide_num = int(entry["slide_number"])
            narration = (entry.get("narration") or "").strip()
            out_wav = str(audio_dir / f"narr_{slide_num:03d}.wav")
            if narration:
                vg.generate(narration, voice=voice, out_path=out_wav)
            else:
                _gen_silence(0.5, out_wav)
            audio_paths.append(out_wav)
            yield f"TTS {idx + 1}/{n}: slide {slide_num}", None

        # ── Step 4: Build per-slide video clips ──────────────────────────────
        clips_dir = work_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        clip_paths: list[str] = []

        for idx, (slide_img, audio_wav) in enumerate(zip(slide_images, audio_paths)):
            slide_num = idx + 1
            audio_dur = _audio_duration(audio_wav)
            total_dur = audio_dur + pause_seconds

            # Append pause silence to the narration audio
            pad_wav = str(clips_dir / f"pad_{slide_num:03d}.wav")
            padded_wav = str(clips_dir / f"padded_{slide_num:03d}.wav")
            _gen_silence(pause_seconds, pad_wav)
            _concat_audio([audio_wav, pad_wav], padded_wav)

            clip_path = str(clips_dir / f"clip_{slide_num:03d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(slide_img),
                "-i", padded_wav,
                "-c:v", "libx264", "-tune", "stillimage",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p",
                "-t", str(total_dur),
                "-vf", (
                    "scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"
                ),
                "-shortest",
                clip_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"FFmpeg clip {slide_num} failed:\n{r.stderr[-600:]}")
            clip_paths.append(clip_path)
            yield f"Clip {idx + 1}/{n}: slide {slide_num} encoded", None

        # ── Step 5: Concatenate ──────────────────────────────────────────────
        yield "Concatenating clips…", None
        concat_txt = work_dir / "concat.txt"
        with open(concat_txt, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "libx264",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg concatenation failed:\n{r.stderr[-600:]}")

        logger.info(f"Narrated video saved: {output_path}")
        yield "Done!", str(output_path)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
