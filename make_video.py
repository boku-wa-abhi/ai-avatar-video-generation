#!/usr/bin/env python3
"""
make_video.py — One-command AI avatar video pipeline.

Usage:
    python make_video.py --script "Hello! I'm your AI assistant..." \\
                         --orientation 9:16 \\
                         --out output/final.mp4

Full pipeline (7 steps):
    1. TTS: text → audio/speech.wav  (Kokoro)
    2. Audio prep: resample to 16 kHz mono
    3. Lip-sync: avatar + audio → lip-synced MP4  (LatentSync, or MuseTalk)
    4. Face enhancement: CodeFormer / GFPGAN
    5. Background: composite onto canvas
    6. Captions: faster-whisper transcription → SRT → burn
    7. Final encode: H.264 / AAC delivery, +faststart

CLI flags:
    --script TEXT           The spoken script.  [required]
    --orientation TEXT      9:16 | 16:9 | 1:1   [default: 9:16]
    --voice TEXT            Kokoro voice id      [default: af_heart]
    --out TEXT              Output MP4 path      [default: output/final.mp4]
    --background TEXT       black|white|blur|path-to-image  [default: black]
    --music TEXT            Path to background music file    [default: none]
    --preview               Open result in default player
    --no-captions           Skip subtitle generation
    --no-enhance            Skip face enhancement step
    --musetalk-only         Use MuseTalk instead of LatentSync
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load .env (HF_TOKEN etc.) before anything else
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    load_dotenv(str(_env_file))

# MPS tuning
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# Put the pipeline root on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Return a compact timestamp string like '20240101_153045'."""
    return time.strftime("%Y%m%d_%H%M%S")


def _step(n: int, total: int, label: str) -> None:
    logger.info(f"[{n}/{total}] ── {label}")


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    script: str,
    orientation: str = "9:16",
    voice: str = "af_heart",
    output_path: str = "output/final.mp4",
    background: str = "black",
    music_path: str | None = None,
    include_captions: bool = True,
    include_enhance: bool = True,
    use_musetalk: bool = False,
    preview: bool = False,
) -> str:
    """Execute the full video generation pipeline.

    Args:
        script:           The spoken text.
        orientation:      Canvas orientation string ("9:16", "16:9", "1:1").
        voice:            Kokoro voice ID.
        output_path:      Final deliverable path.
        background:       "black", "white", "blur", or image path.
        music_path:       Optional background music file.
        include_captions: Burn subtitles into the output.
        include_enhance:  Run face restoration.
        use_musetalk:     Use MuseTalk; default is LatentSync.
        preview:          Open output in default media player.

    Returns:
        Absolute path to the final MP4.
    """
    from caption_gen import CaptionGenerator
    from face_enhancer import FaceEnhancer
    from latentsync_infer import LatentSyncInference
    from musetalk_infer import MuseTalkInference
    from video_assembler import VideoAssembler
    from voice_gen import VoiceGenerator

    pipeline_dir = Path(__file__).resolve().parent
    run_id = _ts()
    TOTAL = 7
    wall_start = time.time()

    # Expected avatar PNG
    avatar_png = pipeline_dir / "avatar" / "avatar.png"
    if not avatar_png.exists():
        raise FileNotFoundError(
            f"Avatar image not found: {avatar_png}\n"
            "Place a portrait PNG at avatar/avatar.png"
        )

    # Working directories
    audio_dir = pipeline_dir / "audio"
    output_dir = Path(output_path).resolve().parent
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intermediate file paths
    speech_wav = str(audio_dir / f"speech_{run_id}.wav")
    speech_16k = str(audio_dir / f"speech_{run_id}_16k.wav")
    lipsync_mp4 = str(pipeline_dir / "output" / f"lipsync_{run_id}.mp4")
    enhanced_mp4 = str(pipeline_dir / "output" / f"enhanced_{run_id}.mp4")
    bg_mp4 = str(pipeline_dir / "output" / f"composed_{run_id}.mp4")
    srt_path = str(pipeline_dir / "captions" / f"captions_{run_id}.srt")
    Path(srt_path).parent.mkdir(parents=True, exist_ok=True)

    final_path = str(Path(output_path).resolve())

    # ── Step 1: TTS ─────────────────────────────────────────────────────────
    _step(1, TOTAL, f"TTS — Kokoro voice: {voice}")
    t = time.time()
    vg = VoiceGenerator()
    vg.generate(script, voice=voice, out_path=speech_wav)
    logger.info(f"   TTS done in {_elapsed(t)}")

    # ── Step 2: Resample to 16 kHz ──────────────────────────────────────────
    _step(2, TOTAL, "Audio prep — 16 kHz mono")
    t = time.time()
    vg.convert_to_16k(speech_wav, speech_16k)
    logger.info(f"   Resample done in {_elapsed(t)}")

    # ── Step 3: Lip-sync ────────────────────────────────────────────────────
    if use_musetalk:
        _step(3, TOTAL, "Lip-sync — MuseTalk 1.5")
        t = time.time()
        ms = MuseTalkInference()
        ms.prepare_avatar(str(avatar_png))
        # Note: MuseTalk returns the first MP4 found in output_dir
        lipsync_mp4 = ms.run(str(avatar_png), speech_16k, output_dir="output")
    else:
        _step(3, TOTAL, "Lip-sync — LatentSync 1.6")
        t = time.time()
        ls = LatentSyncInference()
        lipsync_mp4 = ls.run(str(avatar_png), speech_16k, output_path=lipsync_mp4)

    logger.info(f"   Lip-sync done in {_elapsed(t)}")

    # ── Step 4: Face enhancement ────────────────────────────────────────────
    if include_enhance:
        _step(4, TOTAL, "Face enhancement — CodeFormer / GFPGAN")
        t = time.time()
        fe = FaceEnhancer()
        enhanced_mp4 = fe.enhance(lipsync_mp4, enhanced_mp4)
        logger.info(f"   Enhancement done in {_elapsed(t)}")
    else:
        logger.info(f"[4/{TOTAL}] ── Face enhancement SKIPPED")
        import shutil
        shutil.copy(lipsync_mp4, enhanced_mp4)

    # ── Step 5: Background composite ────────────────────────────────────────
    _step(5, TOTAL, f"Background composite — {background}")
    t = time.time()
    va = VideoAssembler()
    bg_mp4 = va.add_background(enhanced_mp4, orientation=orientation, background=background, output_path=bg_mp4)

    if music_path:
        logger.info("   Mixing background music...")
        music_out = bg_mp4.replace(".mp4", "_music.mp4")
        bg_mp4 = va.add_music(bg_mp4, music_path, output_path=music_out)

    logger.info(f"   Composite done in {_elapsed(t)}")

    # ── Step 6: Captions ────────────────────────────────────────────────────
    if include_captions:
        _step(6, TOTAL, "Captions — faster-whisper transcription")
        t = time.time()
        cg = CaptionGenerator()
        srt_path = cg.transcribe(speech_16k, srt_path)
        logger.info(f"   Transcription done in {_elapsed(t)}")
    else:
        logger.info(f"[6/{TOTAL}] ── Captions SKIPPED")
        srt_path = None

    # ── Step 7: Final encode ────────────────────────────────────────────────
    _step(7, TOTAL, "Final encode — H.264 / AAC +faststart")
    t = time.time()
    va.finalize(
        bg_mp4,
        final_path,
        srt_path=srt_path,
        include_captions=include_captions,
    )
    logger.info(f"   Encode done in {_elapsed(t)}")

    # ── Done ────────────────────────────────────────────────────────────────
    logger.success(
        f"\n{'='*55}\n"
        f"  Pipeline complete in {_elapsed(wall_start)}\n"
        f"  Output: {final_path}\n"
        f"{'='*55}"
    )

    if preview:
        import subprocess
        subprocess.Popen(["open", final_path])

    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI avatar video generation — one-command pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--script", required=True, help="Spoken script text")
    p.add_argument("--orientation", default="9:16", choices=["9:16", "16:9", "1:1"],
                   help="Canvas aspect ratio  [default: 9:16]")
    p.add_argument("--voice", default="af_heart",
                   help="Kokoro voice ID  [default: af_heart]")
    p.add_argument("--out", default="output/final.mp4", help="Output MP4 path")
    p.add_argument("--background", default="black",
                   help="Background: black | white | blur | /path/to/image.jpg")
    p.add_argument("--music", default=None, help="Background music file path")
    p.add_argument("--preview", action="store_true",
                   help="Open result in default media player when done")
    p.add_argument("--no-captions", action="store_true", help="Skip subtitle generation")
    p.add_argument("--no-enhance", action="store_true", help="Skip face enhancement")
    p.add_argument("--musetalk-only", action="store_true",
                   help="Use MuseTalk instead of LatentSync for lip-sync")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run_pipeline(
            script=args.script,
            orientation=args.orientation,
            voice=args.voice,
            output_path=args.out,
            background=args.background,
            music_path=args.music,
            include_captions=not args.no_captions,
            include_enhance=not args.no_enhance,
            use_musetalk=args.musetalk_only,
            preview=args.preview,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise
