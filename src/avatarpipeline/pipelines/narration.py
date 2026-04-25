"""
avatarpipeline.pipelines.narration — Compose a narrated video from PDF + JSON.

Pipeline (in order):
  1. Validate sync (PDF page count vs JSON, sequence, range checks).
  2. Generate TTS narration audio for every slide.
  3. Build a master audio track: each narration followed by a silence pad,
     all concatenated into a single WAV file.  Also compute per-slide display
     durations from the generated audio plus any JSON timing overrides.
  4. Render PDF pages → PNG images (one per slide).
  5. Encode a slideshow video in **one ffmpeg pass**: the concat-demuxer feeds
     slide images with explicit per-image durations while the master audio is
     muxed in — no per-clip intermediates.

Public API
----------
``compose_narrated_video(pdf_path, json_data, output_path, voice, pause_seconds, ...)``
    Generator that yields ``(status_message: str, result_path: str | None)``.
    The final successful yield has ``result_path`` set to the output MP4.
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Generator

from loguru import logger

from avatarpipeline.core.media import (
    audio_duration as _audio_duration,
    concat_audio as _concat_audio,
    generate_silence as _gen_silence,
    normalize_to_16k_mono as _normalize_audio,
)
from avatarpipeline.pipelines._slide_pdf import render_slides
from avatarpipeline.pipelines._validate import validate_sync

DEFAULT_PAUSE: float = 1.5


# ── Public composer ──────────────────────────────────────────────────────────

def compose_narrated_video(
    pdf_path: str | Path,
    json_data: dict | list,
    output_path: str | Path,
    voice: str = "af_heart",
    pause_seconds: float = DEFAULT_PAUSE,
    tts_engine: str = "kokoro",
    mlx_voice_choice: str | None = None,
    mlx_preset_voice: str | None = None,
    mlx_model_id: str | None = None,
    mlx_language: str | None = None,
) -> Generator[tuple[str, str | None], None, None]:
    """Compose a narrated presentation video.

    This is a **generator** — iterate it to drive the pipeline and receive
    real-time status messages for display in a progress UI.

    Yields:
        ``(message, result_path)`` tuples.  ``result_path`` is ``None`` for all
        progress messages and the final output MP4 path on success.

    Raises:
        :class:`ValueError`   — when sync validation fails.
        :class:`RuntimeError` — when an ffmpeg step fails.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = output_path.parent / f"_narr_{run_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step 1: Validate sync ────────────────────────────────────────────
        yield "Validating sync…", None
        result = validate_sync(pdf_path, json_data)
        if not result.ok:
            raise ValueError("Validation failed:\n" + "\n".join(result.errors))
        yield f"Validation passed — {result.slide_count} pages ✓", None

        # ── Step 2: TTS — generate narration audio for every slide ───────────
        yield "Loading TTS engine…", None
        tts_engine = (tts_engine or "kokoro").strip().lower()
        vg = None
        studio = None
        if tts_engine == "mlx":
            from avatarpipeline.engines.tts.mlx import MlxVoiceStudio
            studio = MlxVoiceStudio()
        else:
            from avatarpipeline.engines.tts.kokoro import VoiceGenerator
            vg = VoiceGenerator()

        normalized_json = result.json_data
        default_pause = float(normalized_json.get("default_pause_seconds", pause_seconds))
        default_display = float(normalized_json.get("default_display_seconds", 0.0) or 0.0)
        entries = sorted(normalized_json["slides"], key=lambda e: int(e["slide_number"]))
        audio_dir = work_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        narr_paths: list[str] = []
        narr_audio_durations: list[float] = []
        n = len(entries)
        for idx, entry in enumerate(entries):
            slide_num = int(entry["slide_number"])
            narration = (entry.get("narration") or "").strip()
            out_wav = str(audio_dir / f"narr_{slide_num:03d}.wav")
            if narration:
                if tts_engine == "mlx":
                    if studio is None:
                        raise RuntimeError("MLX voice studio is not available.")
                    raw_wav = audio_dir / f"narr_{slide_num:03d}_raw.wav"
                    if mlx_preset_voice:
                        generated = studio.synthesize_with_preset(
                            text=narration,
                            preset_voice=mlx_preset_voice,
                            model_id=mlx_model_id,
                            lang_code=mlx_language or "ja",
                            output_path=str(raw_wav),
                        )
                    else:
                        if not mlx_voice_choice:
                            raise ValueError("Select a saved MLX voice or a Qwen preset voice for Slide Narrator Japanese mode.")
                        generated = studio.synthesize_with_voice(
                            text=narration,
                            voice_choice=mlx_voice_choice,
                            model_id=mlx_model_id,
                            lang_code=mlx_language or "ja",
                            output_path=str(raw_wav),
                        )
                    _normalize_audio(generated, out_wav)
                    Path(generated).unlink(missing_ok=True)
                else:
                    if vg is None:
                        raise RuntimeError("Kokoro voice generator is not available.")
                    vg.generate(narration, voice=voice, out_path=out_wav)
            else:
                _gen_silence(0.1, out_wav)
            narr_paths.append(out_wav)
            narr_audio_durations.append(max(0.1, _audio_duration(out_wav)))
            yield f"TTS {idx + 1}/{n}: slide {slide_num}", None

        # ── Step 3: Build master audio track ─────────────────────────────────
        # Each narration is extended to at least the requested slide display
        # duration, then an optional post-slide pause is appended.  The image
        # durations mirror the final audio segment durations.
        yield "Building master audio…", None

        padded_dir = work_dir / "padded"
        padded_dir.mkdir(exist_ok=True)
        padded_paths: list[str] = []
        slide_durations: list[float] = []

        for entry, narr_wav, audio_dur in zip(entries, narr_paths, narr_audio_durations):
            slide_num = int(entry["slide_number"])
            min_display = float(entry.get("display_seconds", default_display) or 0.0)
            slide_pause = float(entry.get("pause_seconds", default_pause) or 0.0)
            display_dur = max(audio_dur, min_display)
            hold_dur = max(0.0, display_dur - audio_dur)
            total_dur = display_dur + slide_pause
            slide_durations.append(total_dur)

            segment_inputs = [narr_wav]
            hold_wav = str(padded_dir / f"hold_{slide_num:03d}.wav")
            pause_wav = str(padded_dir / f"pause_{slide_num:03d}.wav")
            padded_wav = str(padded_dir / f"padded_{slide_num:03d}.wav")
            if hold_dur > 0:
                _gen_silence(hold_dur, hold_wav)
                segment_inputs.append(hold_wav)
            if slide_pause > 0:
                _gen_silence(slide_pause, pause_wav)
                segment_inputs.append(pause_wav)
            if len(segment_inputs) == 1:
                shutil.copy2(narr_wav, padded_wav)
            else:
                _concat_audio(segment_inputs, padded_wav)
            padded_paths.append(padded_wav)

        master_audio = str(work_dir / "master.wav")
        _concat_audio(padded_paths, master_audio)
        logger.debug(
            f"Master audio built: {sum(slide_durations):.1f}s total, "
            f"{len(slide_durations)} slides"
        )

        # ── Step 4: Render slides → PNG ──────────────────────────────────────
        yield "Rendering slides…", None
        slides_dir = work_dir / "slides"
        slide_images = render_slides(pdf_path, slides_dir)
        if len(slide_images) != result.slide_count:
            raise RuntimeError(
                f"Slide render produced {len(slide_images)} images "
                f"but expected {result.slide_count}."
            )
        yield f"Slides rendered ({len(slide_images)} images)", None

        # ── Step 5: Encode slideshow video — one ffmpeg pass ─────────────────
        # ffmpeg concat demuxer: each slide PNG shown for its computed duration,
        # master audio muxed in.  No per-clip intermediates.
        yield "Encoding slideshow video…", None

        images_txt = work_dir / "images.txt"
        with open(images_txt, "w") as f:
            for img, dur in zip(slide_images, slide_durations):
                f.write(f"file '{img}'\n")
                f.write(f"duration {dur:.4f}\n")
            # ffmpeg image-concat quirk: list the last frame a second time
            # so the duration of the penultimate entry is honoured correctly.
            f.write(f"file '{slide_images[-1]}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(images_txt),
            "-i", master_audio,
            "-vf", (
                "scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"
            ),
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(output_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg slideshow encode failed:\n{r.stderr[-600:]}")

        logger.info(f"Narrated video saved: {output_path}")
        yield "Done!", str(output_path)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
