"""
avatarpipeline.pipelines.avatar — 7-step AI avatar video generation pipeline.

Orchestrates: TTS → audio prep → lip-sync → face enhancement →
              background composite → captions → final encode.

Usage:
    from avatarpipeline.pipelines.avatar import run_pipeline

    run_pipeline(
        script="Hello, I'm your AI avatar!",
        orientation="9:16",
        voice="af_heart",
        output_path="data/output/final.mp4",
    )
"""

import shutil
import time
from pathlib import Path

from loguru import logger

from avatarpipeline import AUDIO_DIR, AVATARS_DIR, CAPTIONS_DIR, OUTPUT_DIR, ROOT


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


def _step(n: int, total: int, label: str) -> None:
    logger.info(f"[{n}/{total}] ── {label}")


def run_pipeline(
    script: str,
    orientation: str = "9:16",
    voice: str = "af_heart",
    output_path: str | None = None,
    background: str = "black",
    music_path: str | None = None,
    include_captions: bool = True,
    include_enhance: bool = True,
    lipsync_engine: str = "musetalk",
    preview: bool = False,
) -> str:
    """Execute the full video generation pipeline.

    Args:
        script:           The spoken text.
        orientation:      Canvas aspect ratio ("9:16", "16:9", "1:1").
        voice:            Kokoro voice ID (e.g. "af_heart", "bm_george").
        output_path:      Final deliverable MP4 path. Defaults to data/output/final.mp4.
        background:       "black" | "white" | "blur" | path-to-image.
        music_path:       Optional background music file.
        include_captions: Burn auto-generated subtitles into the output.
        include_enhance:  Run face restoration (CodeFormer / GFPGAN).
        lipsync_engine:   "musetalk" | "sadtalker" | "sadtalker_hd".
        preview:          Open the output in the default media player when done.

    Returns:
        Absolute path to the final MP4.
    """
    from avatarpipeline.engines import get_lipsync_engine, get_tts_engine
    from avatarpipeline.postprocess.assembler import VideoAssembler
    from avatarpipeline.postprocess.captions import CaptionGenerator
    from avatarpipeline.postprocess.enhancer import FaceEnhancer

    run_id = _ts()
    TOTAL = 7
    wall_start = time.time()

    # Canonical paths
    final_path = str(Path(output_path).resolve()) if output_path else str(OUTPUT_DIR / "final.mp4")
    avatar_png = AVATARS_DIR / "avatar.png"

    if not avatar_png.exists():
        raise FileNotFoundError(
            f"Avatar image not found: {avatar_png}\n"
            "Place a portrait PNG at data/avatars/avatar.png, or upload one via the dashboard."
        )

    Path(final_path).parent.mkdir(parents=True, exist_ok=True)

    speech_wav  = str(AUDIO_DIR / f"speech_{run_id}.wav")
    speech_16k  = str(AUDIO_DIR / f"speech_{run_id}_16k.wav")
    lipsync_mp4 = str(OUTPUT_DIR / f"lipsync_{run_id}.mp4")
    enhanced_mp4 = str(OUTPUT_DIR / f"enhanced_{run_id}.mp4")
    bg_mp4      = str(OUTPUT_DIR / f"composed_{run_id}.mp4")
    srt_path    = str(CAPTIONS_DIR / f"captions_{run_id}.srt")
    CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: TTS ──────────────────────────────────────────────────────────
    _step(1, TOTAL, f"TTS — Kokoro voice: {voice}")
    t = time.time()
    vg = get_tts_engine("kokoro")
    vg.generate(script, voice=voice, out_path=speech_wav)
    logger.info(f"   TTS done in {_elapsed(t)}")

    # ── Step 2: Resample to 16 kHz ──────────────────────────────────────────
    _step(2, TOTAL, "Audio prep — 16 kHz mono")
    t = time.time()
    vg.convert_to_16k(speech_wav, speech_16k)
    logger.info(f"   Resample done in {_elapsed(t)}")

    # ── Step 3: Lip-sync ────────────────────────────────────────────────────
    _step(3, TOTAL, f"Lip-sync — {lipsync_engine}")
    t = time.time()
    lipsync = get_lipsync_engine(lipsync_engine)
    if lipsync_engine in ("sadtalker", "sadtalker_hd"):
        lipsync_mp4 = lipsync.run(str(avatar_png), speech_16k, output_path=lipsync_mp4)
    else:
        if hasattr(lipsync, "prepare_avatar"):
            avatar_png = Path(lipsync.prepare_avatar(str(avatar_png)))
        lipsync_mp4 = lipsync.run(str(avatar_png), speech_16k)

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
        shutil.copy(lipsync_mp4, enhanced_mp4)

    # ── Step 5: Background composite ────────────────────────────────────────
    _step(5, TOTAL, f"Background composite — {background}")
    t = time.time()
    va = VideoAssembler()
    composed = va.add_background(
        enhanced_mp4, orientation=orientation,
        background=background, output_path=bg_mp4,
    )

    if music_path:
        logger.info("   Mixing background music...")
        music_out = composed.replace(".mp4", "_music.mp4")
        composed = va.add_music(composed, music_path, output_path=music_out)

    logger.info(f"   Composite done in {_elapsed(t)}")

    # ── Step 6: Captions ────────────────────────────────────────────────────
    srt_result = None
    if include_captions:
        _step(6, TOTAL, "Captions — faster-whisper transcription")
        t = time.time()
        cg = CaptionGenerator()
        srt_result = cg.transcribe(speech_16k, srt_path)
        logger.info(f"   Captions done in {_elapsed(t)}")
    else:
        logger.info(f"[6/{TOTAL}] ── Captions SKIPPED")

    # ── Step 7: Final encode ────────────────────────────────────────────────
    _step(7, TOTAL, "Final encode — H.264 / AAC +faststart")
    t = time.time()
    va.finalize(composed, final_path, srt_path=srt_result, include_captions=include_captions)
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
