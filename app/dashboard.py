#!/usr/bin/env python3
"""
app.dashboard — Gradio web dashboard for the AI Avatar Video Pipeline.

Start with:
    python scripts/run_dashboard.py
    — or —
    python -m app.dashboard

Opens automatically at http://localhost:7860
"""

import base64
import gc
import glob
import json as _json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import yaml
from loguru import logger
from PIL import Image

# ── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from avatarpipeline import (
    ASSETS_DIR,
    AUDIO_DIR,
    AVATARS_DIR,
    CAPTIONS_DIR,
    OUTPUT_DIR,
    PRESENTATIONS_DIR,
)
from avatarpipeline.core.media import audio_duration
from avatarpipeline.engines.tts.mlx import MlxVoiceStudio
from avatarpipeline.pipelines.podcast import (
    LAYOUT_CHOICES as PODCAST_LAYOUTS,
    OVERLAY_CHOICES as PODCAST_OVERLAYS,
    build_timeline_from_tracks,
    compose_podcast_sequential,
    compose_podcast_video,
    generate_per_speaker_audio,
    get_unique_speakers,
    mix_audio_tracks,
    parse_podcast_script,
    resample_16k,
)
from avatarpipeline.pipelines._validate import validate_sync as _narration_validate
from avatarpipeline.pipelines.narration import compose_narrated_video, DEFAULT_PAUSE as NARRATION_DEFAULT_PAUSE
from avatarpipeline.pipelines.presenter import (
    OUTPUT_MODE_ALL as PRESENTER_OUTPUT_MODE_ALL,
    OUTPUT_MODE_ONE_BY_ONE as PRESENTER_OUTPUT_MODE_ONE_BY_ONE,
    compose_slide_presenter_video,
)

CONFIG_PATH = ROOT / "configs" / "settings.yaml"
ENV_PATH    = ROOT / ".env"

# MPS tuning
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ── Voice mapping ────────────────────────────────────────────────────────────
VOICE_CHOICES = {
    "Heart — Warm Female (default)": "af_heart",
    "Bella — Confident Female":      "af_bella",
    "Sarah — Natural Female":        "af_sarah",
    "Nicole — Soft Female":          "af_nicole",
    "Adam — Deep Male":              "am_adam",
    "Michael — Conversational Male": "am_michael",
    "Emma — British Female":         "bf_emma",
    "Isabella — British Female":     "bf_isabella",
    "George — British Male":         "bm_george",
    "Lewis — British Male":          "bm_lewis",
    "Kumo — Japanese Male":          "jm_kumo",
    "Alpha — Japanese Female":       "jf_alpha",
    "Gongitsune — Japanese Female":  "jf_gongitsune",
    "Nezumi — Soft Narrative Japanese Female":   "jf_nezumi",
    "Tebukuro — Gentle Narrative Japanese Female": "jf_tebukuro",
}

ORIENTATION_MAP = {
    "Portrait 9:16":  "9:16",
    "Landscape 16:9": "16:9",
    "Square 1:1":     "1:1",
}

_cancel_event = threading.Event()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ── Avatar helpers ───────────────────────────────────────────────────────────

def save_uploaded_avatar(file_path: str) -> tuple[str | None, str, list]:
    if file_path is None:
        return None, "No file selected.", get_avatar_gallery()
    try:
        AVATARS_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(file_path)
        original_name = src.stem
        img = Image.open(file_path).convert("RGBA")
        width, height = img.width, img.height
        gallery_copy = AVATARS_DIR / f"{original_name}.png"
        if gallery_copy.name != "avatar.png":
            img.save(str(gallery_copy), "PNG")
        dest = AVATARS_DIR / "avatar.png"
        img.save(str(dest), "PNG")
        return str(dest), f"Avatar saved ({width}×{height}, aspect preserved)", get_avatar_gallery()
    except Exception as e:
        return None, f"Upload failed: {e}", get_avatar_gallery()


def get_avatar_gallery() -> list[str]:
    files = []
    for p in ["*.png", "*.jpg", "*.jpeg"]:
        files.extend(glob.glob(str(AVATARS_DIR / p)))
    return sorted(files, key=os.path.getmtime, reverse=True)


def get_avatar_choices() -> list[str]:
    return [Path(path).name for path in get_avatar_gallery()]


def _resolve_avatar_choice(choice: str | None) -> str | None:
    if not choice:
        return None
    for path in get_avatar_gallery():
        if Path(path).name == choice:
            return path
    candidate = AVATARS_DIR / choice
    return str(candidate) if candidate.exists() else None


def preview_saved_avatar(choice: str | None) -> tuple[str | None, str]:
    resolved = _resolve_avatar_choice(choice)
    if not resolved:
        return None, "No avatar selected."
    width, height = _probe_image_size(resolved)
    return resolved, f"Selected: {Path(resolved).name} ({width}×{height})"


def refresh_avatar_dropdown(selected_choice: str | None = None):
    choices = get_avatar_choices()
    value = selected_choice if selected_choice in choices else (choices[0] if choices else None)
    preview, status = preview_saved_avatar(value)
    return gr.update(choices=choices, value=value), preview, status


def save_presenter_avatar(file_path: str | None, selected_choice: str | None = None):
    if not file_path:
        return refresh_avatar_dropdown(selected_choice)
    try:
        AVATARS_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(file_path)
        img = Image.open(file_path).convert("RGBA")
        ext = src.suffix.lower() if src.suffix.lower() in (".png", ".jpg", ".jpeg") else ".png"
        dest = AVATARS_DIR / f"{src.stem}{ext}"
        if ext == ".png":
            img.save(str(dest), "PNG")
        else:
            img.convert("RGB").save(str(dest), "JPEG")
        choices = get_avatar_choices()
        value = dest.name if dest.name in choices else (choices[0] if choices else None)
        preview, _ = preview_saved_avatar(value)
        status = f"Avatar saved: {dest.name} ({img.width}×{img.height})"
        return gr.update(choices=choices, value=value), preview, status
    except Exception as exc:
        preview, status = preview_saved_avatar(selected_choice)
        return gr.update(choices=get_avatar_choices(), value=selected_choice), preview, f"Avatar save failed: {exc}"


def select_avatar_from_gallery(evt: gr.SelectData) -> tuple[str | None, str]:
    gallery = get_avatar_gallery()
    if evt.index < len(gallery):
        selected = gallery[evt.index]
        dest = AVATARS_DIR / "avatar.png"
        if Path(selected).resolve() != dest.resolve():
            shutil.copy(selected, str(dest))
        width, height = _probe_image_size(dest)
        aspect = _aspect_ratio_code_for_image(dest) or "custom"
        return str(dest), f"Selected: {Path(selected).name} ({width}×{height}, {aspect})"
    return None, "Selection failed."


def get_video_history() -> list[str]:
    files = glob.glob(str(OUTPUT_DIR / "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:10]


def _probe_image_size(path: str | Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return 0, 0


def _aspect_ratio_code_for_image(path: str | Path) -> str | None:
    width, height = _probe_image_size(path)
    if width <= 0 or height <= 0:
        return None
    ratio = width / height
    candidates = {
        "9:16": 9 / 16,
        "16:9": 16 / 9,
        "1:1": 1.0,
    }
    best = min(candidates.items(), key=lambda item: abs(ratio - item[1]))
    return best[0] if abs(ratio - best[1]) <= 0.18 else None


def open_output_folder() -> str:
    subprocess.Popen(["open", str(OUTPUT_DIR)])
    return f"Opened {OUTPUT_DIR}"


def _list_output_mp4s() -> list[str]:
    """Return all mp4s in OUTPUT_DIR sorted oldest-first (natural sequence order)."""
    files = glob.glob(str(OUTPUT_DIR / "*.mp4"))
    # Exclude files that start with "merged_" to avoid re-merging merged videos
    files = [f for f in files if not Path(f).name.startswith("merged_")]
    files.sort(key=os.path.getmtime)
    return files


def get_merge_choices() -> list[str]:
    """Return display names for the checkbox group (filename only)."""
    return [Path(f).name for f in _list_output_mp4s()]


def refresh_merge_list() -> dict:
    choices = get_merge_choices()
    return gr.update(choices=choices, value=choices)


def merge_output_videos(selected_names: list[str], custom_name: str) -> tuple[str | None, str]:
    """Concatenate selected videos (in the order listed) into one MP4."""
    if not selected_names:
        return None, "Select at least one video."
    if len(selected_names) < 2:
        return None, "Select at least 2 videos to merge."

    all_files = {Path(f).name: f for f in _list_output_mp4s()}
    valid = [all_files[n] for n in selected_names if n in all_files]
    missing = [n for n in selected_names if n not in all_files]
    if missing:
        return None, f"Files not found in output folder: {', '.join(missing)}"
    if len(valid) < 2:
        return None, "Need at least 2 valid video files."

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = (custom_name or "").strip().rstrip(".mp4") or f"merged_{run_id}"
    if not stem.endswith(".mp4"):
        stem += ".mp4"
    out_path = OUTPUT_DIR / stem

    concat_txt = OUTPUT_DIR / f"_concat_{run_id}.txt"
    try:
        with open(concat_txt, "w") as f:
            for p in valid:
                f.write(f"file '{p}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return None, f"ffmpeg merge failed:\n{r.stderr[-400:]}"
        size_mb = out_path.stat().st_size / 1_048_576
        return str(out_path), (
            f"Merged {len(valid)} videos → {out_path.name} ({size_mb:.1f} MB)\n"
            + "\n".join(f"  {i+1}. {Path(p).name}" for i, p in enumerate(valid))
        )
    except Exception as exc:
        logger.error(f"Video merge failed: {exc}")
        return None, f"Error: {exc}"
    finally:
        concat_txt.unlink(missing_ok=True)


def open_presentations_folder() -> str:
    subprocess.Popen(["open", str(PRESENTATIONS_DIR)])
    return f"Opened {PRESENTATIONS_DIR}"


def generate_voice_preview(voice_display: str) -> str | None:
    voice_id = VOICE_CHOICES.get(voice_display)
    if not voice_id:
        return None
    try:
        from avatarpipeline.engines.tts.kokoro import VoiceGenerator
        vg = VoiceGenerator()
        name = voice_display.split("—")[0].strip()
        out = str(AUDIO_DIR / f"preview_{voice_id}.wav")
        preview_text = (
            f"こんにちは、{name}です。よろしくお願いします。"
            if voice_id.startswith(("jf_", "jm_"))
            else f"Hello, I'm {name}. Nice to meet you!"
        )
        vg.generate(preview_text, voice=voice_id, out_path=out)
        return out
    except Exception as e:
        logger.warning(f"Voice preview failed: {e}")
        return None


def load_settings() -> tuple[str, str]:
    cfg = _load_config()
    key = cfg.get("elevenlabs_key", "") or ""
    status = "Key configured" if key else "No API key needed — using local Kokoro TTS (free)"
    return key, status


def save_settings(api_key: str) -> str:
    cfg = _load_config()
    cfg["elevenlabs_key"] = api_key
    _save_config(cfg)
    return "Settings saved"


def update_char_count(text: str) -> str:
    n = len(text) if text else 0
    return f"{n:,} characters · Kokoro TTS (local, free)"


# ── Time estimation ──────────────────────────────────────────────────────────

_LIPSYNC_REALTIME_FACTOR = {
    "MuseTalk 1.5":     14,
    "SadTalker 256px":  15,
    "SadTalker HD":     90,
}


def _audio_duration_from_file(path: str) -> float:
    return audio_duration(path)


def estimate_generation_time(script: str, audio_file: str | None, engine: str) -> str:
    has_audio = bool(audio_file and Path(audio_file).exists())
    if has_audio:
        dur = _audio_duration_from_file(audio_file)
        if dur <= 0:
            return ""
        source_label = f"Uploaded audio — {dur:.1f}s"
        tts_secs = 0
    elif script and script.strip():
        words = len(script.split())
        if words == 0:
            return ""
        dur = words / 150 * 60
        source_label = f"{words:,} words → ~{dur:.0f}s of speech"
        tts_secs = max(8, dur * 0.15)
    else:
        return ""

    factor = _LIPSYNC_REALTIME_FACTOR.get(engine, 14)
    lipsync_secs = dur * factor
    overhead_secs = 12
    total_secs = tts_secs + lipsync_secs + overhead_secs

    def _fmt(s: float) -> str:
        m = int(s) // 60
        sec = int(s) % 60
        return f"{m}m {sec:02d}s" if m else f"{sec}s"

    return (
        f"<div class='estimate-card'>"
        f"<span class='estimate-icon'>schedule</span>"
        f"<div><b>Estimated time:</b> {_fmt(total_secs)}<br>"
        f"<span class='estimate-detail'>{source_label} · Video ≈ {_fmt(dur)} · "
        f"Lip-sync: {_fmt(lipsync_secs)}</span></div></div>"
    )


# ── Video metadata ───────────────────────────────────────────────────────────

def get_video_metadata(video_path: str, gen_time_secs: float) -> str:
    if not video_path or not Path(video_path).exists():
        return ""
    try:
        size_mb = Path(video_path).stat().st_size / 1_048_576
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "default=noprint_wrappers=1", video_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        width = height = duration = "?"
        for line in r.stdout.strip().split("\n"):
            if line.startswith("width="):
                width = line.split("=")[1]
            elif line.startswith("height="):
                height = line.split("=")[1]
            elif line.startswith("duration="):
                duration = f"{float(line.split('=')[1]):.1f}s"
        gm = int(gen_time_secs) // 60
        gs = int(gen_time_secs) % 60
        return (
            "<div class='meta-grid'>"
            f"<div class='meta-item'><span class='material-symbols-outlined meta-icon'>aspect_ratio</span>"
            f"<div><span class='meta-label'>Resolution</span><span class='meta-value'>{width}×{height}</span></div></div>"
            f"<div class='meta-item'><span class='material-symbols-outlined meta-icon'>timer</span>"
            f"<div><span class='meta-label'>Duration</span><span class='meta-value'>{duration}</span></div></div>"
            f"<div class='meta-item'><span class='material-symbols-outlined meta-icon'>save</span>"
            f"<div><span class='meta-label'>File Size</span><span class='meta-value'>{size_mb:.1f} MB</span></div></div>"
            f"<div class='meta-item'><span class='material-symbols-outlined meta-icon'>bolt</span>"
            f"<div><span class='meta-label'>Generated in</span><span class='meta-value'>{gm}m {gs}s</span></div></div>"
            "</div>"
        )
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline generation (streaming progress via yield)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_video(
    script, audio_file, voice_choice, orientation, music_volume,
    background_file, lipsync_engine, enhance_face, add_captions,
    preview_mode, caption_font_size, caption_position,
    mt_batch_size=8, mt_bbox_shift=0,
    st_expression_scale=1.0, st_pose_style=0, st_still=True, st_preprocess="full",
    progress=gr.Progress(track_tqdm=False),
):
    _cancel_event.clear()
    wall_start = time.time()
    TOTAL = 7
    states = ["waiting"] * TOTAL
    times = [""] * TOTAL
    engine_label = lipsync_engine

    def elapsed_str():
        s = int(time.time() - wall_start)
        return f"{s // 60}m {s % 60:02d}s"

    def step_time(t0):
        d = time.time() - t0
        return f"{d:.1f}s" if d < 60 else f"{int(d)//60}m {int(d)%60}s"

    def render(pct, msg=""):
        return _build_progress_html(states, times, pct, elapsed_str(), engine_label, msg)

    has_audio = bool(audio_file and Path(audio_file).exists())
    if not has_audio and (not script or not script.strip()):
        states[0] = "error"
        yield None, render(0, "Enter a script or upload an audio file."), "", get_video_history()
        return

    avatar_png = AVATARS_DIR / "avatar.png"
    if not avatar_png.exists():
        states[0] = "error"
        yield None, render(0, "No avatar found — upload a PNG/JPG in the Avatar section."), "", get_video_history()
        return

    voice_id = VOICE_CHOICES.get(voice_choice, "af_heart")
    orient_code = ORIENTATION_MAP.get(orientation, "9:16")
    background = str(background_file) if background_file and Path(background_file).exists() else "black"
    avatar_aspect = _aspect_ratio_code_for_image(avatar_png)
    if background == "black" and avatar_aspect == orient_code:
        background = str(avatar_png)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(OUTPUT_DIR / f"studio_{run_id}.mp4")
    engine_map = {"MuseTalk 1.5": "musetalk", "SadTalker 256px": "sadtalker", "SadTalker HD": "sadtalker_hd"}
    engine_key = engine_map.get(lipsync_engine, "musetalk")

    progress(0, desc="Starting pipeline...")
    yield None, render(0), "", get_video_history()

    try:
        from avatarpipeline.engines.tts.kokoro import VoiceGenerator
        from avatarpipeline.postprocess.assembler import VideoAssembler
    except ImportError as e:
        states[0] = "error"
        yield None, render(0, f"Import failed: {e}"), "", get_video_history()
        return

    try:
        speech_wav = audio_file if has_audio else None
        if has_audio:
            states[0] = "skipped"; times[0] = "—"
            progress(1 / TOTAL, desc="Step 1/7: Voice synthesis (skipped)")
            yield None, render(1 / TOTAL), "", get_video_history()
        else:
            if _cancel_event.is_set():
                yield None, render(0, "Cancelled."), "", get_video_history(); return
            states[0] = "active"
            progress(1 / TOTAL, desc="Step 1/7: Voice synthesis")
            yield None, render(1 / TOTAL), "", get_video_history()
            t0 = time.time()
            vg = VoiceGenerator()
            speech_wav = str(AUDIO_DIR / f"speech_{run_id}.wav")
            vg.generate(script, voice=voice_id, out_path=speech_wav)
            states[0] = "done"; times[0] = step_time(t0)
            yield None, render(1 / TOTAL), "", get_video_history()

        if _cancel_event.is_set():
            yield None, render(1 / TOTAL, "Cancelled."), "", get_video_history(); return
        states[1] = "active"
        progress(2 / TOTAL, desc="Step 2/7: Audio prep")
        yield None, render(2 / TOTAL), "", get_video_history()
        t0 = time.time()
        speech_16k = str(AUDIO_DIR / f"speech_{run_id}_16k.wav")
        if has_audio:
            r16 = subprocess.run(
                ["ffmpeg", "-y", "-i", speech_wav,
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", speech_16k],
                capture_output=True, text=True,
            )
            if r16.returncode != 0:
                raise RuntimeError(f"Audio conversion failed:\n{r16.stderr[-500:]}")
        else:
            vg.convert_to_16k(speech_wav, speech_16k)
        states[1] = "done"; times[1] = step_time(t0)
        yield None, render(2 / TOTAL), "", get_video_history()

        if _cancel_event.is_set():
            yield None, render(2 / TOTAL, "Cancelled."), "", get_video_history(); return
        states[2] = "active"
        progress(3 / TOTAL, desc=f"Step 3/7: Lip-sync ({lipsync_engine})")
        yield None, render(3 / TOTAL), "", get_video_history()
        t0 = time.time()
        lipsync_mp4 = str(OUTPUT_DIR / f"lipsync_{run_id}.mp4")
        if engine_key in ("sadtalker", "sadtalker_hd"):
            from avatarpipeline.engines.lipsync.sadtalker import SadTalkerInference
            st = SadTalkerInference(preset=engine_key)
            lipsync_mp4 = st.run(
                str(avatar_png), speech_16k, output_path=lipsync_mp4,
                expression_scale=st_expression_scale, pose_style=st_pose_style,
                still=st_still, preprocess=st_preprocess,
            )
        else:
            from avatarpipeline.engines.lipsync.musetalk import MuseTalkInference
            ms = MuseTalkInference()
            ms.prepare_avatar(str(avatar_png))
            lipsync_mp4 = ms.run(
                str(avatar_png), speech_16k,
                batch_size=mt_batch_size, bbox_shift=mt_bbox_shift,
            )
        states[2] = "done"; times[2] = step_time(t0)
        yield None, render(3 / TOTAL), "", get_video_history()

        if enhance_face and not preview_mode:
            if _cancel_event.is_set():
                yield None, render(3 / TOTAL, "Cancelled."), "", get_video_history(); return
            states[3] = "active"
            progress(4 / TOTAL, desc="Step 4/7: Face enhancement")
            yield None, render(4 / TOTAL), "", get_video_history()
            t0 = time.time()
            enhanced_mp4 = str(OUTPUT_DIR / f"enhanced_{run_id}.mp4")
            from avatarpipeline.postprocess.enhancer import FaceEnhancer
            fe = FaceEnhancer()
            enhanced_mp4 = fe.enhance(lipsync_mp4, enhanced_mp4)
            states[3] = "done"; times[3] = step_time(t0)
        else:
            enhanced_mp4 = str(OUTPUT_DIR / f"enhanced_{run_id}.mp4")
            shutil.copy(lipsync_mp4, enhanced_mp4)
            states[3] = "skipped"; times[3] = "—"
        yield None, render(4 / TOTAL), "", get_video_history()

        if _cancel_event.is_set():
            yield None, render(4 / TOTAL, "Cancelled."), "", get_video_history(); return
        states[4] = "active"
        progress(5 / TOTAL, desc="Step 5/7: Composite")
        yield None, render(5 / TOTAL), "", get_video_history()
        t0 = time.time()
        va = VideoAssembler()
        composed_mp4 = str(OUTPUT_DIR / f"composed_{run_id}.mp4")
        composed_mp4 = va.add_background(enhanced_mp4, orientation=orient_code, background=background, output_path=composed_mp4)
        if music_volume > 0 and background_file and Path(background_file).suffix.lower() in (".mp3", ".wav", ".m4a"):
            music_out = composed_mp4.replace(".mp4", "_music.mp4")
            composed_mp4 = va.add_music(composed_mp4, str(background_file), music_volume=music_volume, output_path=music_out)
        states[4] = "done"; times[4] = step_time(t0)
        yield None, render(5 / TOTAL), "", get_video_history()

        srt_path = None
        if add_captions and not preview_mode:
            if _cancel_event.is_set():
                yield None, render(5 / TOTAL, "Cancelled."), "", get_video_history(); return
            states[5] = "active"
            progress(6 / TOTAL, desc="Step 6/7: Captions")
            yield None, render(6 / TOTAL), "", get_video_history()
            t0 = time.time()
            from avatarpipeline.postprocess.captions import CaptionGenerator
            cg = CaptionGenerator()
            srt_path = str(CAPTIONS_DIR / f"captions_{run_id}.srt")
            srt_path = cg.transcribe(speech_16k, srt_path)
            states[5] = "done"; times[5] = step_time(t0)
        else:
            states[5] = "skipped"; times[5] = "—"
        yield None, render(6 / TOTAL), "", get_video_history()

        if _cancel_event.is_set():
            yield None, render(6 / TOTAL, "Cancelled."), "", get_video_history(); return
        states[6] = "active"
        progress(7 / TOTAL, desc="Step 7/7: Final encode")
        yield None, render(7 / TOTAL), "", get_video_history()
        t0 = time.time()
        va.finalize(composed_mp4, output_path, srt_path=srt_path, include_captions=(add_captions and not preview_mode))
        states[6] = "done"; times[6] = step_time(t0)

        wall_secs = time.time() - wall_start
        progress(1.0, desc="Done!")
        m, s = int(wall_secs) // 60, int(wall_secs) % 60
        yield (
            output_path,
            render(1.0, f"Pipeline complete in {m}m {s:02d}s"),
            get_video_metadata(output_path, wall_secs),
            get_video_history(),
        )
    except Exception as exc:
        for i, s in enumerate(states):
            if s == "active":
                states[i] = "error"; break
        logger.error(f"Dashboard pipeline error: {exc}")
        for p in OUTPUT_DIR.glob(f"*{run_id}*"):
            if "studio_" not in p.name:
                p.unlink(missing_ok=True)
        yield None, render(0, f"Error: {exc}"), "", get_video_history()


def cancel_generation():
    _cancel_event.set()
    return f"[{_ts()}] Cancel requested — stopping after current step..."


def _toggle_lipsync_params(engine: str):
    is_musetalk = "MuseTalk" in engine
    return gr.update(visible=is_musetalk), gr.update(visible=not is_musetalk)


# ── Progress panel renderer ──────────────────────────────────────────────────

_STEP_ICONS = {
    "done":    ("check_circle", "step-done"),
    "active":  ("sync",         "step-active"),
    "waiting": ("radio_button_unchecked", "step-waiting"),
    "skipped": ("remove_circle_outline",  "step-skipped"),
    "error":   ("error",        "step-error"),
}

STEP_NAMES = [
    "Voice Synthesis", "Audio Resampling", "Lip-sync Generation",
    "Face Enhancement", "Background Composite", "Caption Generation", "Final Encode",
]

def _build_progress_html(step_states, step_times, pct, elapsed, engine="", message=""):
    rows = []
    for i, (name, state, t) in enumerate(zip(STEP_NAMES, step_states, step_times)):
        icon_name, css_cls = _STEP_ICONS[state]
        label = f"{name} — {engine}" if i == 2 and engine else name
        rows.append(
            f'<div class="step-row {css_cls}">'
            f'<span class="material-symbols-outlined step-icon-m">{icon_name}</span>'
            f'<span class="step-name">{label}</span>'
            f'<span class="step-time-val">{t}</span>'
            f'</div>'
        )
    bar_pct = max(0, min(100, int(pct * 100)))
    footer = f'<div class="progress-message">{message}</div>' if message else ""
    return (
        f'<div class="progress-panel">'
        f'<div class="progress-header"><span class="material-symbols-outlined">manufacturing</span> Pipeline Progress</div>'
        f'{"".join(rows)}'
        f'<div class="progress-track"><div class="progress-fill" style="width:{bar_pct}%"></div></div>'
        f'<div class="progress-elapsed">{elapsed}</div>'
        f'{footer}'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Text-to-Audio Only
# ═══════════════════════════════════════════════════════════════════════════════

def generate_audio_only(script, voice_choice, progress=gr.Progress()):
    if not script or not script.strip():
        return None, "Enter a script to generate audio."
    voice_id = VOICE_CHOICES.get(voice_choice, "af_heart")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(AUDIO_DIR / f"tts_{run_id}.wav")
    try:
        progress(0.2, desc="Generating speech with Kokoro TTS...")
        from avatarpipeline.engines.tts.kokoro import VoiceGenerator
        vg = VoiceGenerator()
        vg.generate(script, voice=voice_id, out_path=output_path)
        progress(1.0, desc="Done!")
        return output_path, f"Audio saved — {Path(output_path).name}"
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return None, f"Generation failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# MLX Voice Studio
# ═══════════════════════════════════════════════════════════════════════════════

def _mlx_voice_summary(profile: dict | None) -> str:
    if not profile:
        return (
            "No saved voices yet.\n\n"
            "Upload a short reference clip above, save it, then select it here for "
            "text-to-voice or voice-to-voice conversion."
        )

    transcript = (profile.get("reference_text", "") or "").strip()
    if len(transcript) > 180:
        transcript = f"{transcript[:177]}..."

    created_at = (profile.get("created_at", "") or "").replace("T", " ").replace("+00:00", " UTC")
    duration = profile.get("duration_seconds", 0.0) or 0.0
    return (
        f"**{profile.get('name', profile.get('slug', 'Saved Voice'))}**\n\n"
        f"- Voice ID: `{profile.get('slug', 'unknown')}`\n"
        f"- Reference length: {duration:.1f}s\n"
        f"- Model hint: `{profile.get('model_hint', MlxVoiceStudio.DEFAULT_TTS_MODEL)}`\n"
        f"- Saved: {created_at or 'Unknown'}\n"
        f"- Transcript: {transcript or 'Not stored'}"
    )


def _mlx_voice_dropdown_update(selected_choice: str | None = None):
    studio = MlxVoiceStudio()
    choices = studio.list_voice_choices()
    value = selected_choice if selected_choice in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def _mlx_voice_library_state(selected_choice: str | None = None):
    studio = MlxVoiceStudio()
    choices = studio.list_voice_choices()
    value = selected_choice if selected_choice in choices else (choices[0] if choices else None)
    preview, summary = get_mlx_voice_profile_details(value)
    return gr.update(choices=choices, value=value), preview, summary


def get_mlx_voice_profile_details(choice: str | None) -> tuple[str | None, str]:
    studio = MlxVoiceStudio()
    profile = studio.get_voice_profile(choice)
    if not profile:
        return None, _mlx_voice_summary(None)
    return profile.get("reference_audio_path"), _mlx_voice_summary(profile)


def refresh_mlx_voice_library(selected_choice: str | None = None):
    return _mlx_voice_library_state(selected_choice)


def save_mlx_voice_profile(name, reference_audio, reference_text, model_choice, progress=gr.Progress()):
    if not reference_audio:
        return "Upload a reference audio clip first.", _mlx_voice_dropdown_update(None), None, _mlx_voice_summary(None)

    try:
        progress(0.2, desc="Saving reference voice locally...")
        studio = MlxVoiceStudio()
        profile = studio.save_voice_profile(
            name=name,
            audio_path=reference_audio,
            transcript=reference_text,
            model_id=model_choice,
        )
        choice = f"{profile['name']} [{profile['slug']}]"
        dropdown, preview, summary = _mlx_voice_library_state(choice)
        progress(1.0, desc="Saved")
        return (
            f"Saved voice profile — `{profile['slug']}`",
            dropdown,
            preview,
            summary,
        )
    except Exception as e:
        logger.error(f"MLX voice save failed: {e}")
        return f"Save failed: {e}", _mlx_voice_dropdown_update(None), None, _mlx_voice_summary(None)


def generate_mlx_voice_audio(script, voice_choice, model_choice, language_choice, speed, pitch_shift, progress=gr.Progress()):
    if not script or not script.strip():
        return None, "Enter text to generate audio."

    try:
        progress(0.15, desc="Loading selected voice...")
        studio = MlxVoiceStudio()
        progress(0.45, desc="Generating speech with MLX voice cloning...")
        output = studio.synthesize_with_voice(
            text=script,
            voice_choice=voice_choice,
            model_id=model_choice,
            lang_code=language_choice,
            speed=float(speed),
            pitch_shift=float(pitch_shift),
        )
        progress(1.0, desc="Done")
        return output, f"Audio saved — {Path(output).name}"
    except Exception as e:
        logger.error(f"MLX cloned TTS failed: {e}")
        return None, f"Generation failed: {e}"


def convert_mlx_voice_audio(source_audio, voice_choice, transcript_override, model_choice, language_choice, speed, pitch_shift, progress=gr.Progress()):
    if not source_audio:
        return None, "", "Upload audio to convert first."

    try:
        progress(0.2, desc="Transcribing uploaded audio locally...")
        studio = MlxVoiceStudio()
        output, transcript = studio.convert_voice(
            source_audio=source_audio,
            voice_choice=voice_choice,
            transcript_override=transcript_override,
            model_id=model_choice,
            lang_code=language_choice,
            speed=float(speed),
            pitch_shift=float(pitch_shift),
        )
        progress(1.0, desc="Done")
        return output, transcript, f"Converted audio saved — {Path(output).name}"
    except Exception as e:
        logger.error(f"MLX voice conversion failed: {e}")
        return None, "", f"Conversion failed: {e}"


_MLX_MODEL_LABELS = MlxVoiceStudio.model_labels()
_MLX_LANGUAGE_LABELS = MlxVoiceStudio.language_labels()
_MLX_PRESET_VOICE_LABELS = MlxVoiceStudio.preset_voice_labels()
_MLX_INITIAL_CHOICES = MlxVoiceStudio().list_voice_choices()
_MLX_INITIAL_VOICE = _MLX_INITIAL_CHOICES[0] if _MLX_INITIAL_CHOICES else None
_MLX_INITIAL_PREVIEW, _MLX_INITIAL_SUMMARY = get_mlx_voice_profile_details(_MLX_INITIAL_VOICE)


def _label_for_model_id(model_id: str) -> str:
    for label, value in MlxVoiceStudio.MODEL_CHOICES.items():
        if value == model_id:
            return label
    return model_id


_MLX_DEFAULT_CLONE_MODEL_LABEL = _label_for_model_id(MlxVoiceStudio.DEFAULT_TTS_MODEL)
_MLX_DEFAULT_PRESET_MODEL_LABEL = _label_for_model_id(MlxVoiceStudio.DEFAULT_PRESET_TTS_MODEL)


# ═══════════════════════════════════════════════════════════════════════════════
# Podcast helpers
# ═══════════════════════════════════════════════════════════════════════════════

PODCAST_STEP_NAMES = [
    "Prepare Audio Tracks",
    "Lip-sync Speaker A",
    "Lip-sync Speaker B",
    "Compose Podcast Video",
    "Finalize",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Slide Narrator helpers
# ═══════════════════════════════════════════════════════════════════════════════

NARRATION_STEP_NAMES = [
    "Validate Sync",
    "Generate Narration Audio",
    "Build Master Audio",
    "Render Slides",
    "Encode Slideshow Video",
]

NARRATION_TTS_KOKORO = "English — Kokoro TTS"
NARRATION_TTS_MLX_JA = "Japanese — MLX Voice"
NARRATION_TTS_CHOICES = [
    NARRATION_TTS_KOKORO,
    NARRATION_TTS_MLX_JA,
]
PRESENTER_OUTPUT_MODE_CHOICES = [
    PRESENTER_OUTPUT_MODE_ALL,
    PRESENTER_OUTPUT_MODE_ONE_BY_ONE,
]
NARRATION_JA_SOURCE_KOKORO = "Kokoro Japanese Preset"
NARRATION_JA_SOURCE_SAVED = "Saved Voice Clone"
NARRATION_JA_SOURCE_PRESET = "Qwen Preset Voice"
NARRATION_JA_KOKORO_CHOICES = {
    "Kumo — Japanese Male (default)": "jm_kumo",
    "Alpha — Japanese Female": "jf_alpha",
    "Gongitsune — Japanese Female": "jf_gongitsune",
}
PRESENTER_STEP_NAMES = [
    "Validate Sync",
    "Narration Audio",
    "Presenter Lip-sync",
    "Save Master Audio",
    "Render Slides",
    "Compose Selected Slides",
    "Export Outputs",
]


def _build_narration_progress_html(
    step_states: list[str],
    step_times: list[str],
    pct: float,
    elapsed: str,
    detail: str = "",
    message: str = "",
) -> str:
    """Build progress-panel HTML for the Slide Narrator pipeline."""
    rows = []
    for i, (name, state, t) in enumerate(zip(NARRATION_STEP_NAMES, step_states, step_times)):
        icon_name, css_cls = _STEP_ICONS[state]
        label = f"{name}: {detail}" if i == 1 and detail else name
        rows.append(
            f'<div class="step-row {css_cls}">'
            f'<span class="material-symbols-outlined step-icon-m">{icon_name}</span>'
            f'<span class="step-name">{label}</span>'
            f'<span class="step-time-val">{t}</span>'
            f'</div>'
        )
    bar_pct = max(0, min(100, int(pct * 100)))
    footer = f'<div class="progress-message">{message}</div>' if message else ""
    return (
        f'<div class="progress-panel">'
        f'<div class="progress-header">'
        f'<span class="material-symbols-outlined">slideshow</span> Narration Pipeline</div>'
        f'{"".join(rows)}'
        f'<div class="progress-track"><div class="progress-fill" style="width:{bar_pct}%"></div></div>'
        f'<div class="progress-elapsed">{elapsed}</div>'
        f'{footer}'
        f'</div>'
    )


def _build_presenter_progress_html(
    step_states: list[str],
    step_times: list[str],
    pct: float,
    elapsed: str,
    detail: str = "",
    message: str = "",
) -> str:
    rows = []
    for i, (name, state, t) in enumerate(zip(PRESENTER_STEP_NAMES, step_states, step_times)):
        icon_name, css_cls = _STEP_ICONS[state]
        label = f"{name}: {detail}" if i in (1, 2, 5) and detail else name
        rows.append(
            f'<div class="step-row {css_cls}">'
            f'<span class="material-symbols-outlined step-icon-m">{icon_name}</span>'
            f'<span class="step-name">{label}</span>'
            f'<span class="step-time-val">{t}</span>'
            f'</div>'
        )
    bar_pct = max(0, min(100, int(pct * 100)))
    footer = f'<div class="progress-message">{message}</div>' if message else ""
    return (
        f'<div class="progress-panel">'
        f'<div class="progress-header">'
        f'<span class="material-symbols-outlined">present_to_all</span> Slide Presenter Pipeline</div>'
        f'{"".join(rows)}'
        f'<div class="progress-track"><div class="progress-fill" style="width:{bar_pct}%"></div></div>'
        f'<div class="progress-elapsed">{elapsed}</div>'
        f'{footer}'
        f'</div>'
    )


def _narration_validation_html(
    ok: bool,
    errors: list[str],
    warnings: list[str],
    slide_count: int = 0,
    json_count: int = 0,
) -> str:
    """Return styled HTML for a validation result box."""
    if not ok:
        css_cls = "narr-validation-fail"
        header_icon = "cancel"
        header_text = "Validation Failed — fix errors before generating"
    elif warnings:
        css_cls = "narr-validation-warn"
        header_icon = "warning"
        header_text = f"Validation Passed with warnings — {slide_count} pages"
    else:
        css_cls = "narr-validation-pass"
        header_icon = "check_circle"
        header_text = f"All checks passed — {slide_count} pages ready"

    rows: list[str] = []
    count_error = any("pdf has" in e.lower() and "json has" in e.lower() for e in errors)
    sequence_error = any("not sequential" in e.lower() or "slide_number" in e.lower() for e in errors)
    range_error = any("only has" in e.lower() for e in errors)

    check_labels = [
        ("Page count match", not count_error,
         f"PDF has {slide_count} pages, JSON has {json_count} entries"),
        ("JSON slide number sequence", not sequence_error, "Checked"),
        ("JSON slide numbers in range", not range_error, "Checked"),
    ]
    for label, _pass, detail in check_labels:
        icon = "check_circle" if _pass else "cancel"
        color = "color:var(--g-green)" if icon == "check_circle" else "color:var(--g-red)"
        rows.append(
            f'<div class="narr-check-row">'
            f'<span class="material-symbols-outlined narr-check-icon" style="{color}">{icon}</span>'
            f'<span>{label}</span>'
            f'</div>'
        )

    error_html = ""
    if errors:
        items = "".join(f"<li>{e}</li>" for e in errors)
        error_html = f"<br><b>Errors:</b><ul style='margin:4px 0 0 18px;padding:0'>{items}</ul>"

    warn_html = ""
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warn_html = f"<br><b>Warnings:</b><ul style='margin:4px 0 0 18px;padding:0'>{items}</ul>"

    return (
        f'<div class="{css_cls}">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;font-weight:600">'
        f'<span class="material-symbols-outlined" style="font-size:20px">{header_icon}</span>'
        f'{header_text}</div>'
        f'{"".join(rows)}'
        f'{error_html}'
        f'{warn_html}'
        f'</div>'
    )


def _toggle_narration_tts_controls(mode: str | None):
    use_mlx = mode == NARRATION_TTS_MLX_JA
    helper = (
        "Japanese narration uses local Qwen/MLX. "
        "Choose either a saved cloned voice or a built-in Qwen preset voice such as Ono_Anna."
        if use_mlx
        else
        "English narration uses the built-in Kokoro TTS voices."
    )
    return (
        gr.update(visible=not use_mlx),
        gr.update(visible=use_mlx),
        gr.update(value=helper),
    )


def _toggle_narration_japanese_source(source_mode: str | None):
    use_kokoro = source_mode == NARRATION_JA_SOURCE_KOKORO
    use_preset = source_mode == NARRATION_JA_SOURCE_PRESET
    helper = (
        "Using Kokoro's Japanese preset voices. Kumo is the default native Japanese male voice."
        if use_kokoro else
        "Using a built-in Qwen preset voice. Ono_Anna is the native Japanese option."
        if use_preset else
        "Using a saved cloned voice from the Voice Studio tab."
    )
    return (
        gr.update(visible=not use_preset and not use_kokoro),
        gr.update(visible=use_preset),
        gr.update(visible=use_kokoro),
        gr.update(
            visible=not use_kokoro,
            value=_MLX_DEFAULT_PRESET_MODEL_LABEL if use_preset else _MLX_DEFAULT_CLONE_MODEL_LABEL,
        ),
        gr.update(value=helper),
    )


def _resolve_presenter_avatar_input(uploaded_avatar: str | None, selected_avatar: str | None) -> str | None:
    resolved = _resolve_avatar_choice(selected_avatar)
    if resolved:
        return resolved
    if uploaded_avatar and Path(uploaded_avatar).exists():
        return uploaded_avatar
    fallback = AVATARS_DIR / "avatar.png"
    return str(fallback) if fallback.exists() else None


def _presenter_enhance_enabled(choice: str | bool | None) -> bool:
    if isinstance(choice, bool):
        return choice
    return str(choice or "").strip().lower() == "yes"


def validate_narration_files(pdf_file: str | None, json_file: str | None) -> str:
    """Run sync validation and return a styled HTML report."""
    if not pdf_file:
        return _narration_validation_html(False, ["No PDF file uploaded."], [])
    if not json_file:
        return _narration_validation_html(False, ["No JSON narration file uploaded."], [])
    try:
        with open(json_file) as f:
            json_data = _json.load(f)
    except Exception as exc:
        return _narration_validation_html(False, [f"Cannot parse JSON file: {exc}"], [])

    try:
        result = _narration_validate(pdf_file, json_data)
    except Exception as exc:
        return _narration_validation_html(False, [f"Validation error: {exc}"], [])

    return _narration_validation_html(
        result.ok,
        result.errors,
        result.warnings,
        result.slide_count,
        result.json_count,
    )


def generate_narration_video(
    pdf_file: str | None,
    json_file: str | None,
    narration_mode: str,
    voice_choice: str,
    japanese_source_mode: str | None,
    mlx_voice_choice: str | None,
    mlx_preset_voice: str | None,
    kokoro_ja_voice: str | None,
    mlx_model_choice: str | None,
    pause_secs: float,
    progress=gr.Progress(track_tqdm=False),
):
    """Generator: build a narrated presentation video.

    Yields ``(video_path_or_None, progress_html, report_text)`` tuples.
    """
    _cancel_event.clear()
    wall_start = time.time()
    TOTAL = 5
    states = ["waiting"] * TOTAL
    times = [""] * TOTAL
    detail = ""

    def elapsed_str() -> str:
        s = int(time.time() - wall_start)
        return f"{s // 60}m {s % 60:02d}s"

    def step_time(t0: float) -> str:
        d = time.time() - t0
        return f"{d:.1f}s" if d < 60 else f"{int(d) // 60}m {int(d) % 60}s"

    def render(pct: float, msg: str = "") -> str:
        return _build_narration_progress_html(states, times, pct, elapsed_str(), detail, msg)

    # ── Input validation ─────────────────────────────────────────────────────
    if not pdf_file:
        states[0] = "error"
        yield None, render(0, "Upload a PDF file."), ""
        return
    if not json_file:
        states[0] = "error"
        yield None, render(0, "Upload a JSON narration file."), ""
        return

    try:
        with open(json_file) as f:
            json_data = _json.load(f)
    except Exception as exc:
        states[0] = "error"
        yield None, render(0, f"Cannot parse JSON: {exc}"), ""
        return

    use_mlx = narration_mode == NARRATION_TTS_MLX_JA
    voice_id = VOICE_CHOICES.get(voice_choice, "af_heart")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(OUTPUT_DIR / f"narration_{run_id}.mp4")

    use_kokoro_ja = use_mlx and japanese_source_mode == NARRATION_JA_SOURCE_KOKORO
    use_preset_voice = use_mlx and japanese_source_mode == NARRATION_JA_SOURCE_PRESET

    if use_kokoro_ja and not kokoro_ja_voice:
        states[0] = "error"
        yield None, render(0, "Select a Japanese Kokoro voice first."), ""
        return
    if use_mlx and use_preset_voice and not mlx_preset_voice:
        states[0] = "error"
        yield None, render(0, "Select a Japanese Qwen preset voice first."), ""
        return
    if use_mlx and not use_preset_voice and not use_kokoro_ja and not mlx_voice_choice:
        states[0] = "error"
        yield None, render(0, "Select a saved Japanese MLX voice first."), ""
        return

    n_slides = 0
    step_starts: dict[int, float] = {}
    result_path: str | None = None
    report_lines = [
        f"Narrated Presentation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"PDF: {Path(pdf_file).name}",
        f"JSON: {Path(json_file).name}",
        f"Narration engine: {narration_mode}",
        f"Japanese voice source: {japanese_source_mode}" if use_mlx else "Japanese voice source: n/a",
        f"Voice: {kokoro_ja_voice if use_kokoro_ja else (mlx_preset_voice if use_preset_voice else (mlx_voice_choice if use_mlx else voice_choice))}",
        f"Model: {'Kokoro Japanese preset' if use_kokoro_ja else (mlx_model_choice if use_mlx else 'Kokoro local default')}",
        f"Default pause between slides: {pause_secs}s",
    ]

    try:
        gen = compose_narrated_video(
            pdf_path=pdf_file,
            json_data=json_data,
            output_path=output_path,
            voice=NARRATION_JA_KOKORO_CHOICES.get(kokoro_ja_voice or "", "jm_kumo") if use_kokoro_ja else voice_id,
            pause_seconds=float(pause_secs),
            tts_engine="mlx" if (use_mlx and not use_kokoro_ja) else "kokoro",
            mlx_voice_choice=mlx_voice_choice if (use_mlx and not use_preset_voice and not use_kokoro_ja) else None,
            mlx_preset_voice=mlx_preset_voice if use_preset_voice else None,
            mlx_model_id=mlx_model_choice if use_mlx else None,
            mlx_language="ja" if use_mlx else None,
        )

        for msg, result in gen:
            if _cancel_event.is_set():
                yield None, render(0, "Cancelled."), "\n".join(report_lines)
                return

            # ── State machine ────────────────────────────────────────────────
            if msg.startswith("Validating"):
                states[0] = "active"
                step_starts[0] = time.time()

            elif msg.startswith("Validation passed"):
                states[0] = "done"
                times[0] = step_time(step_starts.get(0, time.time()))
                try:
                    n_slides = int([t for t in msg.split() if t.isdigit()][0])
                except (IndexError, ValueError):
                    n_slides = 0
                report_lines.append(f"Pages: {n_slides}")
                states[1] = "active"
                step_starts[1] = time.time()

            elif msg.startswith("Loading TTS"):
                states[1] = "active"
                if 1 not in step_starts:
                    step_starts[1] = time.time()

            elif msg.startswith("TTS"):
                states[1] = "active"
                detail = msg

            elif msg.startswith("Building master"):
                states[1] = "done"
                times[1] = step_time(step_starts.get(1, time.time()))
                states[2] = "active"
                if 2 not in step_starts:
                    step_starts[2] = time.time()
                detail = ""

            elif msg.startswith("Rendering"):
                states[2] = "done"
                times[2] = step_time(step_starts.get(2, time.time()))
                states[3] = "active"
                step_starts[3] = time.time()
                detail = ""

            elif msg.startswith("Slides rendered"):
                states[3] = "done"
                times[3] = step_time(step_starts.get(3, time.time()))
                detail = ""

            elif msg.startswith("Encoding slideshow"):
                states[4] = "active"
                step_starts[4] = time.time()
                detail = ""

            elif msg == "Done!" and result:
                states[4] = "done"
                times[4] = step_time(step_starts.get(4, time.time()))
                result_path = result

            # ── Compute progress ─────────────────────────────────────────────
            done_count = states.count("done")
            pct = min(0.97, done_count / TOTAL + 0.02)
            progress(pct, desc=msg)
            yield None, render(pct, msg), "\n".join(report_lines)

        if result_path:
            report_lines.append(f"Output: {Path(result_path).name}")
            report_text = "\n".join(report_lines)
            yield result_path, render(1.0, "Complete!"), report_text
        else:
            yield None, render(0, "No output was produced."), "\n".join(report_lines)

    except ValueError as exc:
        # Validation failure
        states[0] = "error"
        err_msg = str(exc)
        report_lines.append(f"FAILED: {err_msg}")
        yield None, render(0, err_msg[:240]), "\n".join(report_lines)

    except Exception as exc:
        logger.error(f"Narration generation failed: {exc}")
        for i, s in enumerate(states):
            if s == "active":
                states[i] = "error"
        err_msg = str(exc)
        report_lines.append(f"FAILED: {err_msg}")
        yield None, render(0, f"Error: {err_msg[:200]}"), "\n".join(report_lines)


def generate_slide_presenter(
    pdf_file: str | None,
    json_file: str | None,
    logo_upload: str | None,
    avatar_upload: str | None,
    avatar_choice: str | None,
    project_tag: str,
    slide_selection: str,
    output_mode: str,
    narration_mode: str,
    voice_choice: str,
    japanese_source_mode: str | None,
    mlx_voice_choice: str | None,
    mlx_preset_voice: str | None,
    kokoro_ja_voice: str | None,
    mlx_model_choice: str | None,
    pause_secs: float,
    lipsync_engine: str,
    enhance_face: str | bool,
    mt_batch_size: int = 8,
    mt_bbox_shift: int = 0,
    st_expression_scale: float = 1.0,
    st_pose_style: int = 0,
    st_still: bool = True,
    st_preprocess: str = "full",
    progress=gr.Progress(track_tqdm=False),
):
    _cancel_event.clear()
    wall_start = time.time()
    total = len(PRESENTER_STEP_NAMES)
    states = ["waiting"] * total
    times = [""] * total
    detail = ""

    def elapsed_str() -> str:
        s = int(time.time() - wall_start)
        return f"{s // 60}m {s % 60:02d}s"

    def step_time(t0: float) -> str:
        d = time.time() - t0
        return f"{d:.1f}s" if d < 60 else f"{int(d) // 60}m {int(d) % 60}s"

    def render(pct: float, msg: str = "") -> str:
        return _build_presenter_progress_html(states, times, pct, elapsed_str(), detail, msg)

    if not pdf_file:
        states[0] = "error"
        yield None, render(0, "Upload a PDF file."), "", []
        return
    if not json_file:
        states[0] = "error"
        yield None, render(0, "Upload a JSON narration file."), "", []
        return

    avatar_path = _resolve_presenter_avatar_input(avatar_upload, avatar_choice)
    if not avatar_path:
        states[0] = "error"
        yield None, render(0, "Upload or select a presenter avatar first."), "", []
        return

    try:
        with open(json_file) as f:
            json_data = _json.load(f)
    except Exception as exc:
        states[0] = "error"
        yield None, render(0, f"Cannot parse JSON: {exc}"), "", []
        return

    use_mlx = narration_mode == NARRATION_TTS_MLX_JA
    use_kokoro_ja = use_mlx and japanese_source_mode == NARRATION_JA_SOURCE_KOKORO
    use_preset_voice = use_mlx and japanese_source_mode == NARRATION_JA_SOURCE_PRESET
    voice_id = VOICE_CHOICES.get(voice_choice, "af_heart")

    if use_kokoro_ja and not kokoro_ja_voice:
        states[0] = "error"
        yield None, render(0, "Select a Japanese Kokoro voice first."), "", []
        return
    if use_mlx and use_preset_voice and not mlx_preset_voice:
        states[0] = "error"
        yield None, render(0, "Select a Japanese Qwen preset voice first."), "", []
        return
    if use_mlx and not use_preset_voice and not use_kokoro_ja and not mlx_voice_choice:
        states[0] = "error"
        yield None, render(0, "Select a saved Japanese MLX voice first."), "", []
        return

    selected_voice = (
        kokoro_ja_voice
        if use_kokoro_ja
        else mlx_preset_voice
        if use_preset_voice
        else mlx_voice_choice
        if use_mlx
        else voice_choice
    )
    selected_model = (
        "Kokoro Japanese preset"
        if use_kokoro_ja
        else mlx_model_choice
        if use_mlx
        else "Kokoro local default"
    )

    report_lines = [
        "Slide Presenter Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"PDF: {Path(pdf_file).name}",
        f"JSON: {Path(json_file).name}",
        f"Logo: {Path(logo_upload).name if logo_upload and Path(logo_upload).exists() else 'stored project logo / none'}",
        f"Avatar: {Path(avatar_path).name}",
        f"Project tag: {(project_tag or Path(pdf_file).stem).strip()}",
        f"Slide selection: {slide_selection or 'all'}",
        f"Output mode: {output_mode}",
        f"Narration engine: {narration_mode}",
        f"Japanese voice source: {japanese_source_mode}" if use_mlx else "Japanese voice source: n/a",
        f"Voice: {selected_voice}",
        f"Model: {selected_model}",
        f"Lip-sync engine: {lipsync_engine}",
    ]

    step_starts: dict[int, float] = {}
    preview_video = None
    generated_files: list[str] = []

    try:
        states[0] = "active"
        step_starts[0] = time.time()
        progress(0.02, desc="Validating PDF and JSON")
        yield None, render(0.02, "Validating PDF and JSON"), "\n".join(report_lines), generated_files

        gen = compose_slide_presenter_video(
            pdf_path=pdf_file,
            json_data=json_data,
            avatar_path=avatar_path,
            json_source_path=json_file,
            logo_path=logo_upload,
            project_tag=project_tag or Path(pdf_file).stem,
            slide_selection=slide_selection or "all",
            output_mode=output_mode,
            voice=NARRATION_JA_KOKORO_CHOICES.get(kokoro_ja_voice or "", "jm_kumo") if use_kokoro_ja else voice_id,
            pause_seconds=float(pause_secs),
            tts_engine="mlx" if (use_mlx and not use_kokoro_ja) else "kokoro",
            mlx_voice_choice=mlx_voice_choice if (use_mlx and not use_preset_voice and not use_kokoro_ja) else None,
            mlx_preset_voice=mlx_preset_voice if use_preset_voice else None,
            mlx_model_id=mlx_model_choice if (use_mlx and not use_kokoro_ja) else None,
            mlx_language="ja" if use_mlx else None,
            lipsync_engine={"MuseTalk 1.5": "musetalk", "SadTalker 256px": "sadtalker", "SadTalker HD": "sadtalker_hd"}.get(lipsync_engine, "musetalk"),
            enhance_face=_presenter_enhance_enabled(enhance_face),
            mt_batch_size=int(mt_batch_size),
            mt_bbox_shift=int(mt_bbox_shift),
            st_expression_scale=float(st_expression_scale),
            st_pose_style=int(st_pose_style),
            st_still=bool(st_still),
            st_preprocess=st_preprocess,
        )

        for msg, payload in gen:
            if _cancel_event.is_set():
                yield None, render(0, "Cancelled."), "\n".join(report_lines), generated_files
                return

            if msg.startswith("Validation passed"):
                states[0] = "done"
                times[0] = step_time(step_starts.get(0, time.time()))
                states[1] = "active"
                step_starts.setdefault(1, time.time())
            elif msg.startswith("Generating narration audio") or msg.startswith("Reusing narration audio"):
                if states[0] == "waiting":
                    states[0] = "active"
                    step_starts[0] = time.time()
                if states[1] == "waiting":
                    states[1] = "active"
                    step_starts[1] = time.time()
                detail = msg
            elif msg.startswith("Generating lip-sync presenter") or msg.startswith("Reusing lip-sync presenter") or msg.startswith("Enhancing presenter clip"):
                if states[1] == "active":
                    states[1] = "done"
                    times[1] = step_time(step_starts.get(1, time.time()))
                states[2] = "active"
                step_starts.setdefault(2, time.time())
                detail = msg
            elif msg.startswith("Building master audio"):
                if states[2] == "active":
                    states[2] = "done"
                    times[2] = step_time(step_starts.get(2, time.time()))
                states[3] = "active"
                step_starts.setdefault(3, time.time())
                detail = ""
            elif msg.startswith("Rendering slides"):
                if states[3] == "active":
                    states[3] = "done"
                    times[3] = step_time(step_starts.get(3, time.time()))
                states[4] = "active"
                step_starts.setdefault(4, time.time())
                detail = ""
            elif msg.startswith("Composing slide"):
                if states[4] == "active":
                    states[4] = "done"
                    times[4] = step_time(step_starts.get(4, time.time()))
                states[5] = "active"
                step_starts.setdefault(5, time.time())
                detail = msg
            elif msg.startswith("Exporting combined"):
                if states[5] == "active":
                    states[5] = "done"
                    times[5] = step_time(step_starts.get(5, time.time()))
                states[6] = "active"
                step_starts.setdefault(6, time.time())
                detail = ""
            elif msg == "Done!" and payload:
                if states[5] == "active":
                    states[5] = "done"
                    times[5] = step_time(step_starts.get(5, time.time()))
                if states[6] == "active":
                    states[6] = "done"
                    times[6] = step_time(step_starts.get(6, time.time()))
                elif states[6] == "waiting":
                    states[6] = "skipped"
                    times[6] = "—"
                preview_video = payload.get("preview_video")
                generated_files = payload.get("generated_files", [])
                report_lines.append(payload.get("report", ""))

            done_count = states.count("done") + states.count("skipped")
            pct = min(0.98, max(0.03, done_count / total))
            progress(pct, desc=msg)
            yield preview_video, render(pct, msg), "\n".join(line for line in report_lines if line), generated_files

        if preview_video:
            yield preview_video, render(1.0, "Complete!"), "\n".join(line for line in report_lines if line), generated_files
        else:
            yield None, render(0, "No output was produced."), "\n".join(report_lines), generated_files

    except Exception as exc:
        for i, state in enumerate(states):
            if state == "active":
                states[i] = "error"
                break
        err_msg = str(exc)
        report_lines.append(f"FAILED: {err_msg}")
        logger.error(f"Slide presenter generation failed: {exc}")
        yield preview_video, render(0, f"Error: {err_msg[:220]}"), "\n".join(report_lines), generated_files


def save_podcast_avatar(file_path: str | None, speaker_id: str) -> str:
    """Save an uploaded avatar for podcast speaker A or B."""
    if not file_path:
        return "No file selected."
    try:
        AVATARS_DIR.mkdir(parents=True, exist_ok=True)
        dest = AVATARS_DIR / f"podcast_{speaker_id}.png"
        img = Image.open(file_path).convert("RGBA")
        img.save(str(dest), "PNG")
        return f"Avatar set ({img.width}\u00d7{img.height})"
    except Exception as e:
        return f"Error: {e}"


def _save_pod_avatar_a(f):
    return save_podcast_avatar(f, "a")


def _save_pod_avatar_b(f):
    return save_podcast_avatar(f, "b")


def _run_podcast_lipsync(
    avatar_path, audio_path, output_path, engine_key,
    mt_batch_size, mt_bbox_shift,
    st_expression_scale, st_pose_style, st_still, st_preprocess,
):
    """Run lip-sync for one podcast speaker."""
    if engine_key in ("sadtalker", "sadtalker_hd"):
        from avatarpipeline.engines.lipsync.sadtalker import SadTalkerInference
        st = SadTalkerInference(preset=engine_key)
        return st.run(
            avatar_path, audio_path, output_path=output_path,
            expression_scale=st_expression_scale, pose_style=st_pose_style,
            still=st_still, preprocess=st_preprocess,
        )
    else:
        from avatarpipeline.engines.lipsync.musetalk import MuseTalkInference
        ms = MuseTalkInference()
        ms.prepare_avatar(avatar_path)
        return ms.run(
            avatar_path, audio_path,
            batch_size=mt_batch_size, bbox_shift=mt_bbox_shift,
        )


def _build_podcast_progress_html(step_states, step_times, pct, elapsed, engine="", message=""):
    """Build progress panel HTML for the podcast pipeline."""
    rows = []
    for i, (name, state, t) in enumerate(zip(PODCAST_STEP_NAMES, step_states, step_times)):
        icon_name, css_cls = _STEP_ICONS[state]
        label = f"{name} \u2014 {engine}" if i in (1, 2) and engine else name
        rows.append(
            f'<div class="step-row {css_cls}">'
            f'<span class="material-symbols-outlined step-icon-m">{icon_name}</span>'
            f'<span class="step-name">{label}</span>'
            f'<span class="step-time-val">{t}</span>'
            f'</div>'
        )
    bar_pct = max(0, min(100, int(pct * 100)))
    footer = f'<div class="progress-message">{message}</div>' if message else ""
    return (
        f'<div class="progress-panel">'
        f'<div class="progress-header"><span class="material-symbols-outlined">podcasts</span> Podcast Progress</div>'
        f'{"".join(rows)}'
        f'<div class="progress-track"><div class="progress-fill" style="width:{bar_pct}%"></div></div>'
        f'<div class="progress-elapsed">{elapsed}</div>'
        f'{footer}'
        f'</div>'
    )


def generate_podcast(
    mode, script, audio_a, audio_b,
    voice_a, voice_b,
    layout, overlay, custom_overlay,
    orientation, lipsync_engine,
    mt_batch_size=8, mt_bbox_shift=0,
    st_expression_scale=1.0, st_pose_style=0, st_still=True, st_preprocess="full",
    progress=gr.Progress(track_tqdm=False),
):
    """Generate a two-speaker podcast video.  Yields ``(video, progress_html, metadata_html)``."""
    _cancel_event.clear()
    wall_start = time.time()
    TOTAL = 5
    states = ["waiting"] * TOTAL
    times = [""] * TOTAL
    engine_label = lipsync_engine

    def elapsed_str():
        s = int(time.time() - wall_start)
        return f"{s // 60}m {s % 60:02d}s"

    def step_time(t0):
        d = time.time() - t0
        return f"{d:.1f}s" if d < 60 else f"{int(d)//60}m {int(d)%60}s"

    def render(pct, msg=""):
        return _build_podcast_progress_html(states, times, pct, elapsed_str(), engine_label, msg)

    # Validate avatars
    avatar_a = AVATARS_DIR / "podcast_a.png"
    avatar_b = AVATARS_DIR / "podcast_b.png"
    if not avatar_a.exists():
        states[0] = "error"
        yield None, render(0, "Upload an avatar for Speaker A."), ""
        return
    if not avatar_b.exists():
        states[0] = "error"
        yield None, render(0, "Upload an avatar for Speaker B."), ""
        return

    orient_code = ORIENTATION_MAP.get(orientation, "16:9")
    engine_map = {"MuseTalk 1.5": "musetalk", "SadTalker 256px": "sadtalker", "SadTalker HD": "sadtalker_hd"}
    engine_key = engine_map.get(lipsync_engine, "musetalk")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = ROOT / "data" / "temp" / "podcast" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / f"podcast_{run_id}.mp4")

    try:
        # ── Step 1: Prepare audio ────────────────────────────────────
        states[0] = "active"
        progress(1 / TOTAL, desc="Step 1/5: Preparing audio tracks")
        yield None, render(1 / TOTAL), ""
        t0 = time.time()

        # These are populated in script mode and used in Step 4 for sequential cuts
        _pod_timeline: list[dict] | None = None
        _pod_speakers: list[str] | None = None

        if mode == "Script":
            if not script or not script.strip():
                states[0] = "error"
                yield None, render(0, "Enter a podcast script with [Speaker]: markers."), ""
                return

            segments = parse_podcast_script(script)
            speakers = get_unique_speakers(segments)
            if len(speakers) < 2:
                states[0] = "error"
                found = ", ".join(speakers) if speakers else "none"
                yield None, render(0, f"Need at least 2 speakers (found: {found}). Use [Name]: format."), ""
                return

            voice_a_id = VOICE_CHOICES.get(voice_a, "af_heart")
            voice_b_id = VOICE_CHOICES.get(voice_b, "am_adam")
            voice_map = {speakers[0]: voice_a_id, speakers[1]: voice_b_id}

            master_audio, speaker_tracks, _timeline = generate_per_speaker_audio(
                segments, speakers, voice_map, work_dir,
            )
            track_a = speaker_tracks[speakers[0]]
            track_b = speaker_tracks[speakers[1]]
            # Store for sequential composition in Step 4
            _pod_timeline = _timeline
            _pod_speakers = speakers
        else:
            # Audio upload mode
            if not audio_a or not Path(audio_a).exists():
                states[0] = "error"
                yield None, render(0, "Upload audio for Speaker A."), ""
                return
            if not audio_b or not Path(audio_b).exists():
                states[0] = "error"
                yield None, render(0, "Upload audio for Speaker B."), ""
                return

            track_a = str(work_dir / "track_a_16k.wav")
            track_b = str(work_dir / "track_b_16k.wav")
            resample_16k(audio_a, track_a)
            resample_16k(audio_b, track_b)
            master_audio = str(work_dir / "master_audio.wav")
            mix_audio_tracks([track_a, track_b], master_audio)

        states[0] = "done"; times[0] = step_time(t0)
        yield None, render(1 / TOTAL), ""

        # ── Step 2: Lip-sync Speaker A ──────────────────────────────
        if _cancel_event.is_set():
            yield None, render(1 / TOTAL, "Cancelled."), ""; return
        states[1] = "active"
        progress(2 / TOTAL, desc="Step 2/5: Lip-sync Speaker A")
        yield None, render(2 / TOTAL), ""
        t0 = time.time()

        lipsync_a = _run_podcast_lipsync(
            str(avatar_a), track_a, str(work_dir / "lipsync_a.mp4"), engine_key,
            mt_batch_size, mt_bbox_shift,
            st_expression_scale, st_pose_style, st_still, st_preprocess,
        )
        states[1] = "done"; times[1] = step_time(t0)
        yield None, render(2 / TOTAL), ""

        # ── Step 3: Lip-sync Speaker B ──────────────────────────────
        if _cancel_event.is_set():
            yield None, render(2 / TOTAL, "Cancelled."), ""; return
        states[2] = "active"
        progress(3 / TOTAL, desc="Step 3/5: Lip-sync Speaker B")
        yield None, render(3 / TOTAL), ""
        t0 = time.time()

        lipsync_b = _run_podcast_lipsync(
            str(avatar_b), track_b, str(work_dir / "lipsync_b.mp4"), engine_key,
            mt_batch_size, mt_bbox_shift,
            st_expression_scale, st_pose_style, st_still, st_preprocess,
        )
        states[2] = "done"; times[2] = step_time(t0)
        yield None, render(3 / TOTAL), ""

        # ── Step 4: Compose ─────────────────────────────────────────
        if _cancel_event.is_set():
            yield None, render(3 / TOTAL, "Cancelled."), ""; return
        states[3] = "active"
        progress(4 / TOTAL, desc="Step 4/5: Composing podcast")
        yield None, render(4 / TOTAL), ""
        t0 = time.time()

        overlay_file = None
        if custom_overlay:
            try:
                co = str(custom_overlay)
                if Path(co).exists():
                    overlay_file = co
            except Exception:
                pass

        if layout == "Sequential (Active Speaker)":
            # Build a normalised A/B timeline for sequential cuts
            if _pod_timeline is not None and _pod_speakers and len(_pod_speakers) >= 2:
                # Script mode: map speaker names → A / B
                sp_key = {_pod_speakers[0]: "A", _pod_speakers[1]: "B"}
                norm_timeline = [
                    {"speaker": sp_key.get(e["speaker"], "A"),
                     "start": e["start"], "end": e["end"]}
                    for e in _pod_timeline
                ]
            else:
                # Upload-audio mode: detect speech from each resampled track
                norm_timeline = build_timeline_from_tracks(track_a, track_b)

            compose_podcast_sequential(
                lipsync_a, lipsync_b, master_audio, norm_timeline, output_path,
                overlay=overlay, custom_overlay_path=overlay_file,
                orientation=orient_code,
            )
        else:
            compose_podcast_video(
                lipsync_a, lipsync_b, master_audio, output_path,
                layout=layout, overlay=overlay,
                custom_overlay_path=overlay_file,
                orientation=orient_code,
            )
        states[3] = "done"; times[3] = step_time(t0)
        yield None, render(4 / TOTAL), ""

        # ── Step 5: Finalize ────────────────────────────────────────
        states[4] = "done"; times[4] = "\u2014"
        wall_secs = time.time() - wall_start
        m, s = int(wall_secs) // 60, int(wall_secs) % 60
        progress(1.0, desc="Done!")
        yield (
            output_path,
            render(1.0, f"Podcast complete in {m}m {s:02d}s"),
            get_video_metadata(output_path, wall_secs),
        )

    except Exception as exc:
        for i, st in enumerate(states):
            if st == "active":
                states[i] = "error"; break
        logger.error(f"Podcast pipeline error: {exc}")
        yield None, render(0, f"Error: {exc}"), ""


# ═══════════════════════════════════════════════════════════════════════════════
# Logo
# ═══════════════════════════════════════════════════════════════════════════════

_logo_b64 = ""
_logo_path = ASSETS_DIR / "logo.png"
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()


# ═══════════════════════════════════════════════════════════════════════════════
# Google Material-inspired CSS
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Google+Sans+Text:wght@400;500&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

/* ── Foundation ──────────────────────────────────────────────────────────── */
:root {
    --g-blue: #0f6fff;
    --g-blue-light: #e9f1ff;
    --g-blue-hover: #0a5ee1;
    --g-green: #0e9f6e;
    --g-green-light: #e8fbf4;
    --g-red: #dc3b3b;
    --g-red-light: #fdecec;
    --g-yellow: #f0a500;
    --g-yellow-light: #fff5d9;
    --g-surface: #ffffff;
    --g-surface-dim: #f4f7fb;
    --g-surface-container: #edf2f9;
    --g-surface-strong: #0f172a;
    --g-on-surface: #0f172a;
    --g-on-surface-variant: #5b667a;
    --g-outline: #d9e1ec;
    --g-outline-variant: #e6ebf2;
    --g-elevation-1: 0 12px 30px rgba(15, 23, 42, 0.06);
    --g-elevation-2: 0 18px 40px rgba(15, 23, 42, 0.09);
    --g-elevation-3: 0 24px 60px rgba(15, 23, 42, 0.14);
    --g-radius: 18px;
    --g-radius-lg: 24px;
    --g-radius-xl: 999px;
}

body {
    background:
        radial-gradient(circle at top left, rgba(15,111,255,0.10), transparent 24%),
        radial-gradient(circle at top right, rgba(14,159,110,0.08), transparent 20%),
        linear-gradient(180deg, #f7faff 0%, #eef3f9 100%) !important;
}

.gradio-container {
    max-width: none !important;
    width: 100% !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 20px 28px 40px !important;
    font-family: 'Google Sans Text', 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: transparent !important;
}

.gradio-container .main,
.gradio-container .contain {
    max-width: none !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-box,
.gradio-container .gr-group {
    border-radius: var(--g-radius) !important;
}

.gradio-container .gr-box,
.gradio-container .gr-group,
.gradio-container .block.gradio-row > div,
.gradio-container .block.gradio-column > div {
    box-shadow: none;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
.app-header {
    background:
        linear-gradient(135deg, rgba(255,255,255,0.96), rgba(245,249,255,0.96)),
        var(--g-surface);
    border-radius: 28px;
    padding: 28px 32px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: var(--g-elevation-2);
    border: 1px solid rgba(217,225,236,0.85);
}
.app-header .logo-circle {
    width: 56px; height: 56px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f6fff 0%, #2cc59f 100%);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 16px 36px rgba(15,111,255,0.26);
}
.app-header .logo-circle .material-symbols-outlined {
    font-size: 28px; color: white; font-variation-settings: 'FILL' 1;
}
.app-header h1 {
    font-family: 'Google Sans', sans-serif;
    font-size: 1.85rem; font-weight: 700; margin: 0;
    color: var(--g-on-surface); letter-spacing: -0.01em;
}
.app-header .tagline {
    color: var(--g-on-surface-variant); margin: 4px 0 0 0;
    font-size: 0.95rem; font-weight: 500;
}
.app-header .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
}
.app-header .chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 7px 13px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 600;
    letter-spacing: 0.01em;
}
.chip-online { background: var(--g-green-light); color: var(--g-green); }
.chip-local  { background: var(--g-blue-light); color: var(--g-blue); }
.chip-hw     { background: var(--g-yellow-light); color: #e37400; }

/* ── Tab Overrides ────────────────────────────────────────────────────────── */
.tabs > .tab-nav {
    position: sticky;
    top: 14px;
    z-index: 20;
    background: rgba(255,255,255,0.88) !important;
    backdrop-filter: blur(18px);
    border-radius: 24px !important;
    border: 1px solid rgba(217,225,236,0.85) !important;
    box-shadow: var(--g-elevation-1) !important;
    padding: 10px !important;
    gap: 10px !important;
    margin-bottom: 16px !important;
}
.tabs > .tab-nav button {
    font-family: 'Google Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: var(--g-on-surface-variant) !important;
    border: 1px solid transparent !important;
    padding: 14px 20px !important;
    border-radius: 18px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
    min-height: 52px !important;
}
.tabs > .tab-nav button:hover {
    background: #f6f9fe !important;
    color: var(--g-on-surface) !important;
    border-color: var(--g-outline) !important;
}
.tabs > .tab-nav button.selected {
    color: white !important;
    border-color: transparent !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #0f6fff 0%, #2b86ff 100%) !important;
    box-shadow: 0 12px 26px rgba(15,111,255,0.25) !important;
}
.tabitem {
    background: rgba(255,255,255,0.94) !important;
    border-radius: 28px !important;
    border: 1px solid rgba(217,225,236,0.82) !important;
    padding: 28px !important;
    box-shadow: var(--g-elevation-1) !important;
}

/* ── Section headings ────────────────────────────────────────────────────── */
.section-title {
    font-family: 'Google Sans', sans-serif;
    font-size: 0.74rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--g-on-surface-variant);
    margin: 0 0 14px 0 ; padding: 0 0 10px 0;
    border-bottom: 1px solid var(--g-outline-variant);
    display: flex; align-items: center; gap: 6px;
}
.section-title .material-symbols-outlined {
    font-size: 17px; color: var(--g-blue);
}

.gradio-container .gr-form,
.gradio-container .form,
.gradio-container .gr-box,
.gradio-container .gr-group {
    background: rgba(255,255,255,0.74) !important;
    border: 1px solid var(--g-outline-variant) !important;
}

.gradio-container .gr-row,
.gradio-container .gr-column {
    gap: 18px !important;
}

.gradio-container .gr-group,
.gradio-container .gr-box {
    padding: 18px !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    background: #fbfdff !important;
    border: 1px solid var(--g-outline) !important;
    border-radius: 14px !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus,
.gradio-container select:focus {
    border-color: rgba(15,111,255,0.45) !important;
    box-shadow: 0 0 0 4px rgba(15,111,255,0.10) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.g-btn-primary {
    font-family: 'Google Sans', sans-serif !important;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 13px 24px !important;
    background: linear-gradient(135deg, #0f6fff 0%, #2b86ff 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--g-radius-xl) !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 14px 28px rgba(15,111,255,0.23) !important;
}
.g-btn-primary:hover {
    background: linear-gradient(135deg, #0a5ee1 0%, #1d77f0 100%) !important;
    box-shadow: 0 18px 32px rgba(15,111,255,0.26) !important;
    transform: translateY(-1px);
}
.g-btn-danger {
    border-radius: var(--g-radius-xl) !important;
    font-family: 'Google Sans', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Avatar Display ──────────────────────────────────────────────────────── */
.avatar-display img {
    object-fit: contain !important; max-height: 360px !important;
    width: 100% !important; border-radius: var(--g-radius) !important;
    background: var(--g-surface-dim) !important;
    border: 1px solid var(--g-outline-variant) !important;
}

/* ── Progress Panel ──────────────────────────────────────────────────────── */
.progress-panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(247,250,255,0.94));
    border-radius: var(--g-radius);
    padding: 20px 24px;
    border: 1px solid var(--g-outline-variant);
    box-shadow: var(--g-elevation-1);
}
.progress-header {
    font-family: 'Google Sans', sans-serif;
    font-size: 0.8rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--g-on-surface-variant);
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}
.progress-header .material-symbols-outlined { font-size: 18px; color: var(--g-blue); }
.step-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--g-outline-variant);
    transition: background 0.15s;
}
.step-row:last-of-type { border-bottom: none; }
.step-icon-m { font-size: 20px; flex-shrink: 0; }
.step-done .step-icon-m { color: var(--g-green); font-variation-settings: 'FILL' 1; }
.step-active .step-icon-m { color: var(--g-blue); animation: spin 1.2s linear infinite; }
.step-waiting .step-icon-m { color: var(--g-outline); }
.step-skipped .step-icon-m { color: var(--g-on-surface-variant); }
.step-error .step-icon-m { color: var(--g-red); font-variation-settings: 'FILL' 1; }
.step-name {
    flex: 1; font-size: 0.85rem; color: var(--g-on-surface); font-weight: 500;
}
.step-waiting .step-name { color: var(--g-on-surface-variant); }
.step-active .step-name { color: var(--g-blue); font-weight: 600; }
.step-time-val {
    font-size: 0.75rem; color: var(--g-on-surface-variant);
    font-family: 'Google Sans Mono', 'Roboto Mono', monospace;
    min-width: 50px; text-align: right;
}
.progress-track {
    width: 100%; height: 4px; background: var(--g-surface-container);
    border-radius: 2px; margin-top: 16px; overflow: hidden;
}
.progress-fill {
    height: 100%; border-radius: 2px;
    transition: width 0.4s cubic-bezier(0.4,0,0.2,1);
    background: linear-gradient(90deg, var(--g-blue), #34a853);
}
.progress-elapsed {
    font-size: 0.72rem; color: var(--g-on-surface-variant);
    margin-top: 8px; text-align: right;
    font-family: 'Google Sans Mono', monospace;
}
.progress-message {
    color: var(--g-green); font-size: 0.82rem;
    margin-top: 10px; font-weight: 500;
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* ── Output Video ────────────────────────────────────────────────────────── */
.output-video {
    border-radius: var(--g-radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--g-outline-variant) !important;
    box-shadow: var(--g-elevation-1) !important;
}

/* ── Metadata grid ───────────────────────────────────────────────────────── */
.meta-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
}
.meta-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px;
    background: var(--g-surface-dim);
    border-radius: var(--g-radius);
    border: 1px solid var(--g-outline-variant);
}
.meta-icon { font-size: 20px; color: var(--g-blue); }
.meta-label {
    display: block; font-size: 0.68rem; color: var(--g-on-surface-variant);
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
}
.meta-value {
    display: block; font-size: 0.92rem; color: var(--g-on-surface);
    font-weight: 600; margin-top: 1px;
}

/* ── Estimate card ───────────────────────────────────────────────────────── */
.estimate-card {
    display: flex; align-items: flex-start; gap: 10px;
    background: var(--g-blue-light);
    border: 1px solid #c2d7f2;
    border-radius: var(--g-radius);
    padding: 12px 16px; margin-top: 6px;
    font-size: 0.82rem; color: #174ea6;
}
.estimate-icon {
    font-family: 'Material Symbols Outlined';
    font-size: 20px; color: var(--g-blue); flex-shrink: 0; margin-top: 1px;
}
.estimate-detail { color: #5f6368; font-size: 0.78rem; }

/* ── Model info card ─────────────────────────────────────────────────────── */
.model-card {
    display: flex; align-items: center; gap: 12px;
    background: var(--g-surface);
    border: 1px solid var(--g-outline);
    border-radius: var(--g-radius);
    padding: 14px 18px; margin-bottom: 16px;
    box-shadow: var(--g-elevation-1);
}
.model-card .material-symbols-outlined {
    font-size: 28px; color: var(--g-blue); flex-shrink: 0;
}
.model-card b { color: var(--g-on-surface); }
.model-card .model-detail { font-size: 0.8rem; color: var(--g-on-surface-variant); line-height: 1.5; }

/* ── Separator ───────────────────────────────────────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--g-outline), transparent);
    margin: 24px 0;
}

/* ── Footer hide ─────────────────────────────────────────────────────────── */
footer { display: none !important; }

/* ── Inputs / Dropdowns (Google feel) ────────────────────────────────────── */
.gradio-container input, .gradio-container textarea,
.gradio-container select {
    font-family: 'Google Sans Text', sans-serif !important;
    border-radius: 14px !important;
}
.gradio-container .label-wrap {
    font-family: 'Google Sans', sans-serif !important;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 12px 12px 28px !important;
    }

    .app-header {
        padding: 22px 20px;
        border-radius: 22px;
    }

    .app-header h1 {
        font-size: 1.45rem;
    }

    .tabs > .tab-nav {
        top: 8px;
        padding: 8px !important;
    }

    .tabs > .tab-nav button {
        font-size: 0.86rem !important;
        padding: 12px 14px !important;
    }

    .tabitem {
        padding: 18px !important;
    }
}

/* ── Podcast tab ─────────────────────────────────────────────────────────── */
.podcast-speaker-label {
    font-family: 'Google Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.podcast-speaker-label .material-symbols-outlined {
    font-size: 18px;
}
.speaker-a-label {
    background: var(--g-blue-light);
    color: var(--g-blue);
    border-left: 3px solid var(--g-blue);
}
.speaker-b-label {
    background: var(--g-green-light);
    color: var(--g-green);
    border-left: 3px solid var(--g-green);
}

/* ── Slide Narrator tab ──────────────────────────────────────────────────── */
.narr-validation-pass {
    background: var(--g-green-light);
    color: #0a7a55;
    border: 1px solid var(--g-green);
    border-radius: var(--g-radius);
    padding: 16px 20px;
    font-family: 'Google Sans Text', sans-serif;
    font-size: 0.88rem;
    line-height: 1.6;
}
.narr-validation-fail {
    background: var(--g-red-light);
    color: #b91c1c;
    border: 1px solid var(--g-red);
    border-radius: var(--g-radius);
    padding: 16px 20px;
    font-family: 'Google Sans Text', sans-serif;
    font-size: 0.88rem;
    line-height: 1.6;
}
.narr-validation-warn {
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #f59e0b;
    border-radius: var(--g-radius);
    padding: 16px 20px;
    font-family: 'Google Sans Text', sans-serif;
    font-size: 0.88rem;
    line-height: 1.6;
}
.narr-check-row {
    display: flex; align-items: flex-start; gap: 8px;
    margin-bottom: 6px;
}
.narr-check-icon { font-size: 17px; flex-shrink: 0; margin-top: 1px; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Gradio UI — Google Material-styled Tabbed Layout
# ═══════════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Google Sans Text"),
    font_mono=gr.themes.GoogleFont("Roboto Mono"),
    radius_size=gr.themes.sizes.radius_md,
)

with gr.Blocks(title="Avatar Studio") as demo:

    # ── Header ───────────────────────────────────────────────────────────────
    gr.HTML("""
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet" />
        <div class="app-header">
            <div class="logo-circle">
                <span class="material-symbols-outlined">smart_toy</span>
            </div>
            <div>
                <h1>Avatar Studio</h1>
                <p class="tagline">AI-powered video, image &amp; audio generation — fully local</p>
                <div class="chip-row">
                    <span class="chip chip-online"><span class="material-symbols-outlined" style="font-size:14px">check_circle</span> Ready</span>
                    <span class="chip chip-local"><span class="material-symbols-outlined" style="font-size:14px">lock</span> Offline</span>
                    <span class="chip chip-hw"><span class="material-symbols-outlined" style="font-size:14px">memory</span> Apple Silicon · MLX</span>
                </div>
            </div>
        </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1: Text to Audio
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("Text to Audio", id="tab-tts"):
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>record_voice_over</span> Text-to-Speech with Kokoro</div>")
            gr.Markdown("Convert text to natural speech using **Kokoro TTS** — runs locally, no API key needed. Download the generated audio file when done.")

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    tts_script = gr.Textbox(label="Script", placeholder="Type or paste the text to convert to speech…", lines=6)
                    tts_char_counter = gr.Markdown("0 characters · Kokoro TTS (local, free)")
                    with gr.Row():
                        tts_voice = gr.Dropdown(
                            label="Voice", choices=list(VOICE_CHOICES.keys()),
                            value="Heart — Warm Female (default)", scale=2,
                        )
                        tts_voice_preview = gr.Audio(label="Voice Preview", type="filepath", interactive=False, scale=2)
                    tts_generate_btn = gr.Button("Generate Audio", variant="primary", elem_classes=["g-btn-primary"])

                with gr.Column(scale=2):
                    tts_output = gr.Audio(label="Generated Audio (click ⬇ to download)", type="filepath", interactive=False)
                    tts_status = gr.Markdown("")

            tts_script.change(fn=update_char_count, inputs=[tts_script], outputs=[tts_char_counter])
            tts_voice.change(fn=generate_voice_preview, inputs=[tts_voice], outputs=[tts_voice_preview])
            tts_generate_btn.click(fn=generate_audio_only, inputs=[tts_script, tts_voice], outputs=[tts_output, tts_status])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2: Voice Studio
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("Voice Studio", id="tab-voice-studio"):
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>graphic_eq</span> MLX Voice Studio</div>")
            gr.Markdown(
                "Save reference voices locally, then use them for **text-to-voice** and **voice-to-voice** conversion on Apple Silicon. "
                "The first run downloads the selected MLX model into your local Hugging Face cache. "
                "Voice-to-voice here is transcript-guided: the uploaded audio is transcribed locally and then re-synthesized in the selected saved voice."
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>library_add</span> Save New Voice</div>")
                    vs_voice_name = gr.Textbox(label="Voice Name", placeholder="e.g. My Narrator")
                    vs_reference_audio = gr.Audio(
                        label="Reference Audio",
                        type="filepath",
                        sources=["upload"],
                    )
                    vs_reference_text = gr.Textbox(
                        label="Reference Transcript (optional)",
                        placeholder="Leave blank to auto-transcribe locally with MLX Whisper…",
                        lines=3,
                    )
                    vs_model = gr.Dropdown(
                        label="MLX Model",
                        choices=_MLX_MODEL_LABELS,
                        value=_MLX_MODEL_LABELS[0],
                    )
                    vs_save_btn = gr.Button("Save Voice", variant="primary", elem_classes=["g-btn-primary"])
                    vs_save_status = gr.Markdown("")

                with gr.Column(scale=2):
                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>folder_managed</span> Saved Voices</div>")
                    with gr.Row():
                        vs_saved_voice = gr.Dropdown(
                            label="Target Saved Voice",
                            choices=_MLX_INITIAL_CHOICES,
                            value=_MLX_INITIAL_VOICE,
                            scale=4,
                        )
                        vs_refresh_btn = gr.Button("Refresh", scale=1)
                    vs_preview = gr.Audio(label="Reference Preview", type="filepath", interactive=False, value=_MLX_INITIAL_PREVIEW)
                    vs_meta = gr.Markdown(_MLX_INITIAL_SUMMARY)

            gr.HTML('<div class="divider"></div>')
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>record_voice_over</span> Text → Saved Voice</div>")
                    vs_tts_script = gr.Textbox(
                        label="Script",
                        placeholder="Type or paste the text to speak in the selected saved voice…",
                        lines=6,
                    )
                    with gr.Row():
                        vs_tts_language = gr.Dropdown(
                            label="Language",
                            choices=_MLX_LANGUAGE_LABELS,
                            value="English",
                        )
                        vs_tts_speed = gr.Slider(
                            label="Speech Speed",
                            minimum=0.8,
                            maximum=1.3,
                            step=0.05,
                            value=1.0,
                        )
                    vs_tts_pitch = gr.Slider(
                        label="Pitch Shift (semitones)",
                        minimum=-12,
                        maximum=12,
                        step=1,
                        value=0,
                    )
                    vs_tts_generate_btn = gr.Button("Generate Cloned Audio", variant="primary", elem_classes=["g-btn-primary"])
                    vs_tts_output = gr.Audio(label="Generated Audio", type="filepath", interactive=False)
                    vs_tts_status = gr.Markdown("")

                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>swap_horiz</span> Voice → Saved Voice</div>")
                    vs_v2v_audio = gr.Audio(
                        label="Source Audio",
                        type="filepath",
                        sources=["upload"],
                    )
                    vs_v2v_override = gr.Textbox(
                        label="Transcript Override (optional)",
                        placeholder="Leave blank to transcribe the uploaded audio locally first…",
                        lines=3,
                    )
                    with gr.Row():
                        vs_v2v_language = gr.Dropdown(
                            label="Language",
                            choices=_MLX_LANGUAGE_LABELS,
                            value="English",
                        )
                        vs_v2v_speed = gr.Slider(
                            label="Speech Speed",
                            minimum=0.8,
                            maximum=1.3,
                            step=0.05,
                            value=1.0,
                        )
                    vs_v2v_pitch = gr.Slider(
                        label="Pitch Shift (semitones)",
                        minimum=-12,
                        maximum=12,
                        step=1,
                        value=0,
                    )
                    vs_v2v_generate_btn = gr.Button("Convert Voice", variant="primary", elem_classes=["g-btn-primary"])
                    vs_v2v_output = gr.Audio(label="Converted Audio", type="filepath", interactive=False)
                    vs_v2v_transcript = gr.Textbox(label="Transcript Used", interactive=False, lines=4)
                    vs_v2v_status = gr.Markdown("")

            vs_saved_voice.change(
                fn=get_mlx_voice_profile_details,
                inputs=[vs_saved_voice],
                outputs=[vs_preview, vs_meta],
            )
            vs_refresh_btn.click(
                fn=refresh_mlx_voice_library,
                inputs=[vs_saved_voice],
                outputs=[vs_saved_voice, vs_preview, vs_meta],
            )
            vs_save_btn.click(
                fn=save_mlx_voice_profile,
                inputs=[vs_voice_name, vs_reference_audio, vs_reference_text, vs_model],
                outputs=[vs_save_status, vs_saved_voice, vs_preview, vs_meta],
            )
            vs_tts_generate_btn.click(
                fn=generate_mlx_voice_audio,
                inputs=[vs_tts_script, vs_saved_voice, vs_model, vs_tts_language, vs_tts_speed, vs_tts_pitch],
                outputs=[vs_tts_output, vs_tts_status],
            )
            vs_v2v_generate_btn.click(
                fn=convert_mlx_voice_audio,
                inputs=[vs_v2v_audio, vs_saved_voice, vs_v2v_override, vs_model, vs_v2v_language, vs_v2v_speed, vs_v2v_pitch],
                outputs=[vs_v2v_output, vs_v2v_transcript, vs_v2v_status],
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3: Audio to Lipsync
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("Audio to Lipsync", id="tab-a2l"):
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>lips</span> Audio → Lip-synced Video</div>")
            gr.Markdown("Upload audio and an avatar image to generate a **lip-synced talking-head video**. Skips TTS and now keeps the uploaded avatar's original framing instead of forcing it into a square preview.")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=260):
                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>face</span> Avatar</div>")
                    a2l_avatar_upload = gr.Image(label="Upload avatar", type="filepath", sources=["upload", "clipboard"], height=150, elem_classes=["avatar-display"])
                    a2l_avatar_preview = gr.Image(
                        label="Active Avatar", type="filepath", interactive=False, height=180,
                        value=str(AVATARS_DIR / "avatar.png") if (AVATARS_DIR / "avatar.png").exists() else None,
                        elem_classes=["avatar-display"],
                    )
                    a2l_avatar_status = gr.Textbox(interactive=False, lines=1, show_label=False, value="Active: avatar.png" if (AVATARS_DIR / "avatar.png").exists() else "No avatar")
                    a2l_avatar_gallery = gr.Gallery(label="Saved Avatars", value=get_avatar_gallery(), columns=4, height=120, allow_preview=False)

                with gr.Column(scale=3):
                    a2l_audio = gr.Audio(label="Upload Audio (MP3 / WAV / M4A)", type="filepath", sources=["upload"])
                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>tune</span> Video Settings</div>")
                    with gr.Row():
                        a2l_orientation = gr.Radio(label="Orientation", choices=list(ORIENTATION_MAP.keys()), value="Portrait 9:16", scale=2)
                        a2l_music_slider = gr.Slider(label="Music Volume", minimum=0.0, maximum=1.0, step=0.05, value=0.15, scale=1)
                        a2l_background = gr.File(label="Background / Music", file_types=["image", ".mp4", ".mp3", ".wav", ".m4a"], scale=1)
                    with gr.Accordion("Advanced Options", open=False):
                        a2l_engine = gr.Radio(label="Lip-sync Engine", choices=["MuseTalk 1.5", "SadTalker 256px", "SadTalker HD"], value="MuseTalk 1.5")
                        with gr.Column(visible=True) as a2l_mt_params:
                            with gr.Row():
                                a2l_mt_batch = gr.Slider(label="Batch Size", minimum=1, maximum=16, step=1, value=8)
                                a2l_mt_bbox = gr.Slider(label="Lip Region Shift", minimum=-10, maximum=10, step=1, value=0)
                        with gr.Column(visible=False) as a2l_st_params:
                            with gr.Row():
                                a2l_st_expr = gr.Slider(label="Expression Scale", minimum=0.5, maximum=3.0, step=0.1, value=1.0)
                                a2l_st_pose = gr.Slider(label="Pose Style", minimum=0, maximum=45, step=1, value=0)
                            with gr.Row():
                                a2l_st_still = gr.Checkbox(label="Still mode", value=True)
                                a2l_st_preprocess = gr.Dropdown(label="Preprocess", choices=["crop", "extcrop", "resize", "full", "extfull"], value="full")
                        with gr.Row():
                            a2l_enhance = gr.Checkbox(label="Face enhancement", value=True)
                        with gr.Row():
                            a2l_captions = gr.Checkbox(label="Auto captions", value=True)
                            a2l_preview_mode = gr.Checkbox(label="Preview mode (faster)", value=False)
                        with gr.Row():
                            a2l_caption_fontsize = gr.Slider(label="Caption Size", minimum=12, maximum=32, step=1, value=20)
                            a2l_caption_position = gr.Dropdown(label="Caption Position", choices=["Bottom", "Center", "Top"], value="Bottom")

            gr.HTML('<div class="divider"></div>')
            with gr.Row():
                a2l_generate_btn = gr.Button("Generate Lipsync Video", variant="primary", scale=3, elem_classes=["g-btn-primary"])
                a2l_cancel_btn = gr.Button("Cancel", variant="stop", scale=1, elem_classes=["g-btn-danger"])

            a2l_log = gr.HTML(value="<div class='progress-panel'><div class='progress-header'><span class='material-symbols-outlined'>manufacturing</span> Pipeline Progress</div><div style='color:var(--g-on-surface-variant);font-size:0.85rem;padding:12px 0'>Ready — upload audio and click Generate</div></div>")
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>movie</span> Output</div>")
            with gr.Row():
                a2l_output = gr.Video(label="Generated Video", elem_classes=["output-video"], scale=3)
                with gr.Column(scale=1):
                    a2l_metadata = gr.HTML(value="")
                    a2l_open_folder = gr.Button("Open Output Folder", size="sm")
                    a2l_folder_status = gr.Textbox(visible=False)
            a2l_history = gr.Gallery(label="Recent Videos", value=get_video_history(), columns=5, height=120, allow_preview=False)

            a2l_avatar_upload.change(fn=save_uploaded_avatar, inputs=[a2l_avatar_upload], outputs=[a2l_avatar_preview, a2l_avatar_status, a2l_avatar_gallery])
            a2l_avatar_gallery.select(fn=select_avatar_from_gallery, outputs=[a2l_avatar_preview, a2l_avatar_status])
            a2l_engine.change(fn=_toggle_lipsync_params, inputs=[a2l_engine], outputs=[a2l_mt_params, a2l_st_params])
            a2l_generate_btn.click(
                fn=generate_video,
                inputs=[
                    gr.Textbox(value="", visible=False), a2l_audio,
                    gr.Dropdown(value="Heart — Warm Female (default)", choices=list(VOICE_CHOICES.keys()), visible=False),
                    a2l_orientation, a2l_music_slider, a2l_background,
                    a2l_engine, a2l_enhance, a2l_captions, a2l_preview_mode,
                    a2l_caption_fontsize, a2l_caption_position,
                    a2l_mt_batch, a2l_mt_bbox,
                    a2l_st_expr, a2l_st_pose, a2l_st_still, a2l_st_preprocess,
                ],
                outputs=[a2l_output, a2l_log, a2l_metadata, a2l_history],
            )
            a2l_cancel_btn.click(fn=cancel_generation, outputs=[a2l_log])
            a2l_open_folder.click(fn=open_output_folder, outputs=[a2l_folder_status])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4: Text to Lipsync (Full Pipeline)
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("Text to Lipsync", id="tab-t2l"):
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>auto_awesome</span> Full Pipeline: Text → Video</div>")
            gr.Markdown("**End-to-end:** Text → Speech → Lip-sync → Enhancement → Composite → Captions → Final Video. Uploaded avatars keep their original aspect ratio so portrait art stays portrait.")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=260):
                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>face</span> Avatar</div>")
                    t2l_avatar_upload = gr.Image(label="Upload avatar", type="filepath", sources=["upload", "clipboard"], height=150, elem_classes=["avatar-display"])
                    t2l_avatar_preview = gr.Image(
                        label="Active Avatar", type="filepath", interactive=False, height=180,
                        value=str(AVATARS_DIR / "avatar.png") if (AVATARS_DIR / "avatar.png").exists() else None,
                        elem_classes=["avatar-display"],
                    )
                    t2l_avatar_status = gr.Textbox(interactive=False, lines=1, show_label=False, value="Active: avatar.png" if (AVATARS_DIR / "avatar.png").exists() else "No avatar")
                    t2l_avatar_gallery = gr.Gallery(label="Saved Avatars", value=get_avatar_gallery(), columns=4, height=120, allow_preview=False)

                with gr.Column(scale=3):
                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>edit_note</span> Script</div>")
                    t2l_script = gr.Textbox(label="Script", placeholder="Type or paste the text your avatar will speak…", lines=5, show_label=False)
                    t2l_char_counter = gr.Markdown("0 characters · Kokoro TTS (local, free)")
                    gr.HTML("<div style='display:flex;align-items:center;gap:8px;color:var(--g-on-surface-variant);font-size:0.78rem;padding:8px 0'><span style='flex:1;height:1px;background:var(--g-outline-variant)'></span>or upload audio directly<span style='flex:1;height:1px;background:var(--g-outline-variant)'></span></div>")
                    t2l_audio_upload = gr.Audio(label="Upload audio (bypasses TTS)", type="filepath", sources=["upload"])
                    t2l_estimate = gr.HTML(value="")

                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>record_voice_over</span> Voice</div>")
                    with gr.Row():
                        t2l_voice = gr.Dropdown(label="Voice", choices=list(VOICE_CHOICES.keys()), value="Heart — Warm Female (default)", scale=2, show_label=False)
                        t2l_voice_preview = gr.Audio(label="Preview", type="filepath", interactive=False, scale=2)

                    gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>tune</span> Video Settings</div>")
                    with gr.Row():
                        t2l_orientation = gr.Radio(label="Orientation", choices=list(ORIENTATION_MAP.keys()), value="Portrait 9:16", scale=2)
                        t2l_music_slider = gr.Slider(label="Music Volume", minimum=0.0, maximum=1.0, step=0.05, value=0.15, scale=1)
                        t2l_background = gr.File(label="Background / Music", file_types=["image", ".mp4", ".mp3", ".wav", ".m4a"], scale=1)

                    with gr.Accordion("Advanced Options", open=False):
                        t2l_engine = gr.Radio(label="Lip-sync Engine", choices=["MuseTalk 1.5", "SadTalker 256px", "SadTalker HD"], value="MuseTalk 1.5")
                        with gr.Column(visible=True) as t2l_mt_params:
                            with gr.Row():
                                t2l_mt_batch = gr.Slider(label="Batch Size", minimum=1, maximum=16, step=1, value=8)
                                t2l_mt_bbox = gr.Slider(label="Lip Region Shift", minimum=-10, maximum=10, step=1, value=0)
                        with gr.Column(visible=False) as t2l_st_params:
                            with gr.Row():
                                t2l_st_expr = gr.Slider(label="Expression Scale", minimum=0.5, maximum=3.0, step=0.1, value=1.0)
                                t2l_st_pose = gr.Slider(label="Pose Style", minimum=0, maximum=45, step=1, value=0)
                            with gr.Row():
                                t2l_st_still = gr.Checkbox(label="Still mode", value=True)
                                t2l_st_preprocess = gr.Dropdown(label="Preprocess", choices=["crop", "extcrop", "resize", "full", "extfull"], value="full")
                        with gr.Row():
                            t2l_enhance = gr.Checkbox(label="Face enhancement", value=True)
                        with gr.Row():
                            t2l_captions = gr.Checkbox(label="Auto captions", value=True)
                            t2l_preview_mode = gr.Checkbox(label="Preview mode", value=False)
                        with gr.Row():
                            t2l_caption_fontsize = gr.Slider(label="Caption Size", minimum=12, maximum=32, step=1, value=20)
                            t2l_caption_position = gr.Dropdown(label="Caption Position", choices=["Bottom", "Center", "Top"], value="Bottom")

            gr.HTML('<div class="divider"></div>')
            with gr.Row():
                t2l_generate_btn = gr.Button("Generate Video", variant="primary", scale=3, elem_classes=["g-btn-primary"])
                t2l_cancel_btn = gr.Button("Cancel", variant="stop", scale=1, elem_classes=["g-btn-danger"])

            t2l_log = gr.HTML(value="<div class='progress-panel'><div class='progress-header'><span class='material-symbols-outlined'>manufacturing</span> Pipeline Progress</div><div style='color:var(--g-on-surface-variant);font-size:0.85rem;padding:12px 0'>Ready — click Generate Video to start</div></div>")
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>movie</span> Output</div>")
            with gr.Row():
                t2l_output = gr.Video(label="Generated Video", elem_classes=["output-video"], scale=3)
                with gr.Column(scale=1):
                    t2l_metadata = gr.HTML(value="")
                    t2l_open_folder = gr.Button("Open Output Folder", size="sm")
                    t2l_folder_status = gr.Textbox(visible=False)
            t2l_history = gr.Gallery(label="Recent Videos", value=get_video_history(), columns=5, height=120, allow_preview=False)

            t2l_avatar_upload.change(fn=save_uploaded_avatar, inputs=[t2l_avatar_upload], outputs=[t2l_avatar_preview, t2l_avatar_status, t2l_avatar_gallery])
            t2l_avatar_gallery.select(fn=select_avatar_from_gallery, outputs=[t2l_avatar_preview, t2l_avatar_status])
            t2l_script.change(fn=update_char_count, inputs=[t2l_script], outputs=[t2l_char_counter])
            t2l_script.change(fn=estimate_generation_time, inputs=[t2l_script, t2l_audio_upload, t2l_engine], outputs=[t2l_estimate])
            t2l_audio_upload.change(fn=estimate_generation_time, inputs=[t2l_script, t2l_audio_upload, t2l_engine], outputs=[t2l_estimate])
            t2l_engine.change(fn=estimate_generation_time, inputs=[t2l_script, t2l_audio_upload, t2l_engine], outputs=[t2l_estimate])
            t2l_voice.change(fn=generate_voice_preview, inputs=[t2l_voice], outputs=[t2l_voice_preview])
            t2l_engine.change(fn=_toggle_lipsync_params, inputs=[t2l_engine], outputs=[t2l_mt_params, t2l_st_params])
            t2l_generate_btn.click(
                fn=generate_video,
                inputs=[
                    t2l_script, t2l_audio_upload, t2l_voice, t2l_orientation,
                    t2l_music_slider, t2l_background, t2l_engine, t2l_enhance,
                    t2l_captions, t2l_preview_mode, t2l_caption_fontsize, t2l_caption_position,
                    t2l_mt_batch, t2l_mt_bbox,
                    t2l_st_expr, t2l_st_pose, t2l_st_still, t2l_st_preprocess,
                ],
                outputs=[t2l_output, t2l_log, t2l_metadata, t2l_history],
            )
            t2l_cancel_btn.click(fn=cancel_generation, outputs=[t2l_log])
            t2l_open_folder.click(fn=open_output_folder, outputs=[t2l_folder_status])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5: Podcast Studio
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("Podcast Studio", id="tab-podcast"):
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>podcasts</span> Two-Speaker Podcast Generator</div>")
            gr.Markdown(
                "Create a **two-speaker podcast video** with animated avatars. "
                "Write a script with `[Speaker A]:` / `[Speaker B]:` markers, or upload separate audio for each speaker. "
                "Choose a frame layout, add visual overlays, and generate!"
            )

            # ── Speakers ────────────────────────────────────────────────
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>group</span> Speakers</div>")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML("<div class='podcast-speaker-label speaker-a-label'><span class='material-symbols-outlined'>person</span> Speaker A</div>")
                    pod_avatar_a_upload = gr.Image(
                        label="Avatar A", type="filepath",
                        sources=["upload", "clipboard"], height=160,
                        elem_classes=["avatar-display"],
                    )
                    pod_avatar_a_status = gr.Textbox(
                        interactive=False, lines=1, show_label=False,
                        value="Avatar set" if (AVATARS_DIR / "podcast_a.png").exists() else "No avatar",
                    )
                    pod_voice_a = gr.Dropdown(
                        label="Voice (script mode)",
                        choices=list(VOICE_CHOICES.keys()),
                        value="Heart \u2014 Warm Female (default)",
                    )
                    pod_audio_a = gr.Audio(
                        label="Audio A (upload mode)", type="filepath",
                        sources=["upload"], visible=False,
                    )

                with gr.Column(scale=1):
                    gr.HTML("<div class='podcast-speaker-label speaker-b-label'><span class='material-symbols-outlined'>person</span> Speaker B</div>")
                    pod_avatar_b_upload = gr.Image(
                        label="Avatar B", type="filepath",
                        sources=["upload", "clipboard"], height=160,
                        elem_classes=["avatar-display"],
                    )
                    pod_avatar_b_status = gr.Textbox(
                        interactive=False, lines=1, show_label=False,
                        value="Avatar set" if (AVATARS_DIR / "podcast_b.png").exists() else "No avatar",
                    )
                    pod_voice_b = gr.Dropdown(
                        label="Voice (script mode)",
                        choices=list(VOICE_CHOICES.keys()),
                        value="Adam \u2014 Deep Male",
                    )
                    pod_audio_b = gr.Audio(
                        label="Audio B (upload mode)", type="filepath",
                        sources=["upload"], visible=False,
                    )

            gr.HTML('<div class="divider"></div>')

            # ── Input mode ──────────────────────────────────────────────
            pod_mode = gr.Radio(
                label="Input Mode",
                choices=["Script", "Upload Audio"],
                value="Script",
            )

            with gr.Column(visible=True) as pod_script_section:
                pod_script = gr.Textbox(
                    label="Podcast Script",
                    placeholder=(
                        "[Host]: Welcome to our AI podcast! Today we're discussing the future of technology.\n"
                        "[Guest]: Thanks for having me. It's an exciting time!\n"
                        "[Host]: Let's dive right in."
                    ),
                    lines=8,
                )

            with gr.Column(visible=False) as pod_audio_section:
                gr.Markdown(
                    "Upload separate audio files for each speaker above. "
                    "Each file should contain only that speaker's parts "
                    "(with silence during the other speaker's turns)."
                )

            gr.HTML('<div class="divider"></div>')

            # ── Layout & Effects ────────────────────────────────────────
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>grid_view</span> Layout & Effects</div>")
            with gr.Row():
                pod_layout = gr.Radio(
                    label="Frame Layout",
                    choices=PODCAST_LAYOUTS,
                    value="Sequential (Active Speaker)",
                    scale=2,
                )
                pod_orientation = gr.Radio(
                    label="Orientation",
                    choices=list(ORIENTATION_MAP.keys()),
                    value="Landscape 16:9",
                    scale=2,
                )
            with gr.Row():
                pod_overlay = gr.Dropdown(
                    label="Visual Overlay",
                    choices=PODCAST_OVERLAYS,
                    value="None",
                    scale=2,
                )
                pod_custom_overlay = gr.File(
                    label="Custom Overlay (transparent PNG)",
                    file_types=["image"],
                    scale=2,
                )

            with gr.Accordion("Advanced Options", open=False):
                pod_engine = gr.Radio(
                    label="Lip-sync Engine",
                    choices=["MuseTalk 1.5", "SadTalker 256px", "SadTalker HD"],
                    value="MuseTalk 1.5",
                )
                with gr.Column(visible=True) as pod_mt_params:
                    with gr.Row():
                        pod_mt_batch = gr.Slider(label="Batch Size", minimum=1, maximum=16, step=1, value=8)
                        pod_mt_bbox = gr.Slider(label="Lip Region Shift", minimum=-10, maximum=10, step=1, value=0)
                with gr.Column(visible=False) as pod_st_params:
                    with gr.Row():
                        pod_st_expr = gr.Slider(label="Expression Scale", minimum=0.5, maximum=3.0, step=0.1, value=1.0)
                        pod_st_pose = gr.Slider(label="Pose Style", minimum=0, maximum=45, step=1, value=0)
                    with gr.Row():
                        pod_st_still = gr.Checkbox(label="Still mode", value=True)
                        pod_st_preprocess = gr.Dropdown(
                            label="Preprocess",
                            choices=["crop", "extcrop", "resize", "full", "extfull"],
                            value="full",
                        )

            gr.HTML('<div class="divider"></div>')

            with gr.Row():
                pod_generate_btn = gr.Button(
                    "Generate Podcast", variant="primary", scale=3,
                    elem_classes=["g-btn-primary"],
                )
                pod_cancel_btn = gr.Button(
                    "Cancel", variant="stop", scale=1,
                    elem_classes=["g-btn-danger"],
                )

            pod_log = gr.HTML(
                value=(
                    "<div class='progress-panel'>"
                    "<div class='progress-header'>"
                    "<span class='material-symbols-outlined'>podcasts</span> Podcast Progress"
                    "</div>"
                    "<div style='color:var(--g-on-surface-variant);font-size:0.85rem;padding:12px 0'>"
                    "Ready \u2014 set up speakers and click Generate"
                    "</div></div>"
                )
            )

            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>movie</span> Output</div>")
            with gr.Row():
                pod_output = gr.Video(label="Generated Podcast", elem_classes=["output-video"], scale=3)
                with gr.Column(scale=1):
                    pod_metadata = gr.HTML(value="")
                    pod_open_folder = gr.Button("Open Output Folder", size="sm")
                    pod_folder_status = gr.Textbox(visible=False)

            # ── Events ──────────────────────────────────────────────────
            def _podcast_mode_toggle(mode):
                is_script = mode == "Script"
                return (
                    gr.update(visible=is_script),       # script section
                    gr.update(visible=not is_script),   # audio section
                    gr.update(visible=is_script),       # voice A
                    gr.update(visible=is_script),       # voice B
                    gr.update(visible=not is_script),   # audio A
                    gr.update(visible=not is_script),   # audio B
                )

            pod_mode.change(
                fn=_podcast_mode_toggle,
                inputs=[pod_mode],
                outputs=[pod_script_section, pod_audio_section,
                         pod_voice_a, pod_voice_b, pod_audio_a, pod_audio_b],
            )
            pod_avatar_a_upload.change(
                fn=_save_pod_avatar_a,
                inputs=[pod_avatar_a_upload],
                outputs=[pod_avatar_a_status],
            )
            pod_avatar_b_upload.change(
                fn=_save_pod_avatar_b,
                inputs=[pod_avatar_b_upload],
                outputs=[pod_avatar_b_status],
            )
            pod_engine.change(
                fn=_toggle_lipsync_params,
                inputs=[pod_engine],
                outputs=[pod_mt_params, pod_st_params],
            )
            pod_generate_btn.click(
                fn=generate_podcast,
                inputs=[
                    pod_mode, pod_script, pod_audio_a, pod_audio_b,
                    pod_voice_a, pod_voice_b,
                    pod_layout, pod_overlay, pod_custom_overlay,
                    pod_orientation, pod_engine,
                    pod_mt_batch, pod_mt_bbox,
                    pod_st_expr, pod_st_pose, pod_st_still, pod_st_preprocess,
                ],
                outputs=[pod_output, pod_log, pod_metadata],
            )
            pod_cancel_btn.click(fn=cancel_generation, outputs=[pod_log])
            pod_open_folder.click(fn=open_output_folder, outputs=[pod_folder_status])

        # ══════════════════════════════════════════════════════════════════════
        # Tab 6 — Slide Narrator
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("Slide Narrator", id="tab-narration"):
            gr.HTML(
                "<div class='section-title'>"
                "<span class='material-symbols-outlined'>slideshow</span>"
                " PDF + JSON Narration Sync Tool</div>"
            )
            gr.Markdown(
                "Upload a **PDF** file and a **JSON narration** file. "
                "The tool validates that they are in sync, generates the narration audio first, "
                "then renders a narrated video where each page stays on screen for the generated "
                "audio length or any longer timing you specify in JSON, followed by an optional pause."
            )

            # ── Inputs ──────────────────────────────────────────────────────
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>upload_file</span> Inputs</div>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    narr_pdf = gr.File(
                        label="PDF File (.pdf)",
                        file_types=[".pdf"],
                        file_count="single",
                    )
                with gr.Column(scale=1):
                    narr_json = gr.File(
                        label="Narration JSON File",
                        file_types=[".json"],
                        file_count="single",
                    )

            gr.HTML(
                "<div style='font-size:0.8rem;color:var(--g-on-surface-variant);margin:-8px 0 12px'>"
                "JSON format: <code>{\"presentation_title\": \"…\", \"default_pause_seconds\": 1.0, "
                "\"slides\": [{\"slide_number\": 1, \"narration\": \"…\", \"duration_seconds\": 6, "
                "\"pause_seconds\": 0.5}, …]}</code><br>"
                "Also accepted: plain slide arrays, <code>text</code>/<code>script</code> instead of "
                "<code>narration</code>, and omitted <code>slide_number</code> fields when the JSON order matches the PDF page order."
                "</div>"
            )

            # ── Validate button + result ─────────────────────────────────────
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>rule</span> Validation</div>")
            narr_validate_btn = gr.Button(
                "▶  Validate Files",
                size="sm",
                variant="secondary",
            )
            narr_validation_result = gr.HTML(
                value="<div style='color:var(--g-on-surface-variant);font-size:0.85rem;padding:10px 0'>"
                      "Upload both files, then click Validate to run sync checks.</div>",
                label="Validation Result",
            )

            gr.HTML('<div class="divider"></div>')

            # ── Configuration ────────────────────────────────────────────────
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>tune</span> Voice & Timing</div>")
            with gr.Row():
                narr_mode = gr.Dropdown(
                    label="Narration Engine",
                    choices=NARRATION_TTS_CHOICES,
                    value=NARRATION_TTS_KOKORO,
                    scale=2,
                )
                narr_pause = gr.Slider(
                    label="Pause between slides (seconds)",
                    minimum=0.0,
                    maximum=5.0,
                    step=0.5,
                    value=float(NARRATION_DEFAULT_PAUSE),
                    scale=1,
                )
            with gr.Row(visible=True) as narr_kokoro_row:
                narr_voice = gr.Dropdown(
                    label="Narration Voice (Kokoro TTS)",
                    choices=list(VOICE_CHOICES.keys()),
                    value="Heart \u2014 Warm Female (default)",
                    scale=2,
                )
            with gr.Column(visible=False) as narr_mlx_col:
                narr_ja_source = gr.Radio(
                    label="Japanese Voice Source",
                    choices=[NARRATION_JA_SOURCE_KOKORO, NARRATION_JA_SOURCE_PRESET, NARRATION_JA_SOURCE_SAVED],
                    value=NARRATION_JA_SOURCE_KOKORO,
                )
                with gr.Row():
                    with gr.Column(visible=False) as narr_mlx_saved_col:
                        narr_mlx_voice = gr.Dropdown(
                            label="Japanese Saved Voice (MLX)",
                            choices=_MLX_INITIAL_CHOICES,
                            value=_MLX_INITIAL_VOICE,
                            scale=4,
                        )
                    with gr.Column(visible=True) as narr_mlx_preset_col:
                        narr_mlx_preset = gr.Dropdown(
                            label="Japanese Preset Voice (Qwen)",
                            choices=_MLX_PRESET_VOICE_LABELS,
                            value="Ono_Anna — Japanese Female (native)",
                            scale=4,
                        )
                    with gr.Column(visible=True) as narr_kokoro_ja_col:
                        narr_kokoro_ja_voice = gr.Dropdown(
                            label="Japanese Preset Voice (Kokoro)",
                            choices=list(NARRATION_JA_KOKORO_CHOICES.keys()),
                            value="Kumo — Japanese Male (default)",
                            scale=4,
                        )
                    narr_mlx_refresh = gr.Button("Refresh Saved Voices", scale=1)
                narr_ja_source_help = gr.Markdown(
                    "Using Kokoro's Japanese preset voices. Kumo is the default native Japanese male voice."
                )
                narr_mlx_model = gr.Dropdown(
                    label="Japanese TTS Model",
                    choices=_MLX_MODEL_LABELS,
                    value=_MLX_DEFAULT_CLONE_MODEL_LABEL,
                )
            narr_engine_help = gr.Markdown("English narration uses the built-in Kokoro TTS voices.")

            gr.HTML('<div class="divider"></div>')

            # ── Generate ─────────────────────────────────────────────────────
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>movie</span> Generate</div>")
            with gr.Row():
                narr_generate_btn = gr.Button(
                    "Generate Narrated Video",
                    variant="primary",
                    elem_classes=["g-btn-primary"],
                    scale=3,
                )
                narr_cancel_btn = gr.Button(
                    "Cancel",
                    variant="stop",
                    elem_classes=["g-btn-danger"],
                    scale=1,
                )

            narr_log = gr.HTML(label="Progress", value="")

            with gr.Row():
                narr_output = gr.Video(
                    label="Narrated Video",
                    height=440,
                    scale=2,
                )
                with gr.Column(scale=1):
                    narr_report = gr.Textbox(
                        label="Report",
                        lines=14,
                        interactive=False,
                        placeholder="Validation + generation report will appear here…",
                    )
                    narr_open_folder = gr.Button("Open Output Folder", size="sm", variant="secondary")
                    narr_folder_status = gr.Textbox(
                        label="",
                        interactive=False,
                        lines=1,
                        show_label=False,
                        visible=True,
                    )

            # ── Wiring ───────────────────────────────────────────────────────
            narr_validate_btn.click(
                fn=validate_narration_files,
                inputs=[narr_pdf, narr_json],
                outputs=[narr_validation_result],
            )
            narr_mode.change(
                fn=_toggle_narration_tts_controls,
                inputs=[narr_mode],
                outputs=[narr_kokoro_row, narr_mlx_col, narr_engine_help],
            )
            narr_ja_source.change(
                fn=_toggle_narration_japanese_source,
                inputs=[narr_ja_source],
                outputs=[narr_mlx_saved_col, narr_mlx_preset_col, narr_kokoro_ja_col, narr_mlx_model, narr_ja_source_help],
            )
            narr_mlx_refresh.click(
                fn=_mlx_voice_dropdown_update,
                inputs=[narr_mlx_voice],
                outputs=[narr_mlx_voice],
            )
            narr_generate_btn.click(
                fn=generate_narration_video,
                inputs=[narr_pdf, narr_json, narr_mode, narr_voice, narr_ja_source, narr_mlx_voice, narr_mlx_preset, narr_kokoro_ja_voice, narr_mlx_model, narr_pause],
                outputs=[narr_output, narr_log, narr_report],
            )
            narr_cancel_btn.click(fn=cancel_generation, outputs=[narr_log])
            narr_open_folder.click(fn=open_output_folder, outputs=[narr_folder_status])

        # ══════════════════════════════════════════════════════════════════════
        # Tab 7 — Slide Presenter
        # ══════════════════════════════════════════════════════════════════════
        presenter_avatar_choices = get_avatar_choices()
        presenter_avatar_default = presenter_avatar_choices[0] if presenter_avatar_choices else None
        presenter_avatar_initial, presenter_avatar_status_initial = preview_saved_avatar(presenter_avatar_default)

        with gr.TabItem("Slide Presenter", id="tab-slide-presenter"):
            gr.HTML(
                "<div class='section-title'>"
                "<span class='material-symbols-outlined'>present_to_all</span>"
                " PDF + JSON Slide Presenter with Lip-sync</div>"
            )
            gr.Markdown(
                "Create a narrated slide video in a consulting-style 16:9 frame with a **top-left logo header**, a **bounded presentation area**, and a lip-synced presenter anchored on the **left side**. "
                "This tab keeps per-slide narration audio, presenter lip-sync clips, master audio, and slide composites "
                "under `data/presentations/<project-tag>`. The uploaded PDF, JSON, avatar, and numbered slide renders are saved there too, "
                "so reruns can reuse existing `slide_001.png`, `slide_002.png`, and so on when the PDF has not changed."
            )

            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>upload_file</span> Inputs</div>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    presenter_pdf = gr.File(
                        label="PDF File (.pdf)",
                        file_types=[".pdf"],
                        file_count="single",
                    )
                with gr.Column(scale=1):
                    presenter_json = gr.File(
                        label="Narration JSON File",
                        file_types=[".json"],
                        file_count="single",
                    )
                with gr.Column(scale=1):
                    presenter_logo = gr.Image(
                        label="Logo (optional — top-left header area)",
                        type="filepath",
                        sources=["upload", "clipboard"],
                        height=140,
                        elem_classes=["avatar-display"],
                    )

            gr.HTML(
                "<div style='font-size:0.8rem;color:var(--g-on-surface-variant);margin:-8px 0 12px'>"
                "Slide selection examples: <code>all</code>, <code>1</code>, <code>1-3</code>, <code>1,2,5</code>. "
                "If you leave Project Tag blank, the folder name defaults to the PDF filename. On rerun, Slide Presenter checks that folder first and reuses saved numbered slide PNGs when possible. "
                "Uploaded logos are placed in the consulting-style top-left header zone and saved in the same project folder."
                "</div>"
            )

            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>face</span> Presenter Avatar</div>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    presenter_avatar_upload = gr.Image(
                        label="Upload Presenter Avatar",
                        type="filepath",
                        sources=["upload", "clipboard"],
                        height=190,
                        elem_classes=["avatar-display"],
                    )
                    with gr.Row():
                        presenter_avatar_save_btn = gr.Button("Save Uploaded Avatar", size="sm", variant="secondary")
                        presenter_avatar_refresh_btn = gr.Button("Refresh Saved Avatars", size="sm", variant="secondary")
                    presenter_avatar_choice = gr.Dropdown(
                        label="Saved Presenter Avatar",
                        choices=presenter_avatar_choices,
                        value=presenter_avatar_default,
                    )
                    presenter_avatar_status = gr.Textbox(
                        interactive=False,
                        lines=2,
                        value=presenter_avatar_status_initial,
                    )
                with gr.Column(scale=1, min_width=280):
                    presenter_avatar_preview = gr.Image(
                        label="Selected Presenter",
                        type="filepath",
                        value=presenter_avatar_initial,
                        interactive=False,
                        height=320,
                        elem_classes=["avatar-display"],
                    )

            gr.HTML('<div class="divider"></div>')

            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>rule</span> Validation</div>")
            presenter_validate_btn = gr.Button(
                "▶  Validate Files",
                size="sm",
                variant="secondary",
            )
            presenter_validation_result = gr.HTML(
                value="<div style='color:var(--g-on-surface-variant);font-size:0.85rem;padding:10px 0'>"
                      "Upload the PDF and JSON, then validate before generating the presenter pipeline.</div>",
            )

            gr.HTML('<div class="divider"></div>')

            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>tune</span> Project & Output Scope</div>")
            with gr.Row():
                presenter_project_tag = gr.Textbox(
                    label="Project Tag",
                    placeholder="Optional. Reuse the same tag when you want to keep presenter audio/lip-sync assets across PDF revisions.",
                    scale=2,
                )
                presenter_slide_selection = gr.Textbox(
                    label="Slides to Render",
                    value="all",
                    placeholder="Examples: all, 1, 1-3, 1,2,5",
                    scale=2,
                )
                presenter_output_mode = gr.Radio(
                    label="Output Mode",
                    choices=PRESENTER_OUTPUT_MODE_CHOICES,
                    value=PRESENTER_OUTPUT_MODE_ONE_BY_ONE,
                    scale=2,
                )

            gr.HTML('<div class="divider"></div>')

            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>record_voice_over</span> Narration</div>")
            with gr.Row():
                presenter_mode = gr.Dropdown(
                    label="Narration Engine",
                    choices=NARRATION_TTS_CHOICES,
                    value=NARRATION_TTS_KOKORO,
                    scale=2,
                )
                presenter_pause = gr.Slider(
                    label="Pause between slides (seconds)",
                    minimum=0.0,
                    maximum=5.0,
                    step=0.5,
                    value=float(NARRATION_DEFAULT_PAUSE),
                    scale=1,
                )
            with gr.Row(visible=True) as presenter_kokoro_row:
                presenter_voice = gr.Dropdown(
                    label="Narration Voice (Kokoro TTS)",
                    choices=list(VOICE_CHOICES.keys()),
                    value="Heart — Warm Female (default)",
                    scale=2,
                )
            with gr.Column(visible=False) as presenter_mlx_col:
                presenter_ja_source = gr.Radio(
                    label="Japanese Voice Source",
                    choices=[NARRATION_JA_SOURCE_KOKORO, NARRATION_JA_SOURCE_PRESET, NARRATION_JA_SOURCE_SAVED],
                    value=NARRATION_JA_SOURCE_KOKORO,
                )
                with gr.Row():
                    with gr.Column(visible=False) as presenter_mlx_saved_col:
                        presenter_mlx_voice = gr.Dropdown(
                            label="Japanese Saved Voice (MLX)",
                            choices=_MLX_INITIAL_CHOICES,
                            value=_MLX_INITIAL_VOICE,
                            scale=4,
                        )
                    with gr.Column(visible=True) as presenter_mlx_preset_col:
                        presenter_mlx_preset = gr.Dropdown(
                            label="Japanese Preset Voice (Qwen)",
                            choices=_MLX_PRESET_VOICE_LABELS,
                            value="Ono_Anna — Japanese Female (native)",
                            scale=4,
                        )
                    with gr.Column(visible=True) as presenter_kokoro_ja_col:
                        presenter_kokoro_ja_voice = gr.Dropdown(
                            label="Japanese Preset Voice (Kokoro)",
                            choices=list(NARRATION_JA_KOKORO_CHOICES.keys()),
                            value="Kumo — Japanese Male (default)",
                            scale=4,
                        )
                    presenter_mlx_refresh = gr.Button("Refresh Saved Voices", scale=1)
                presenter_ja_source_help = gr.Markdown(
                    "Using Kokoro's Japanese preset voices. Kumo is the default native Japanese male voice."
                )
                presenter_mlx_model = gr.Dropdown(
                    label="Japanese TTS Model",
                    choices=_MLX_MODEL_LABELS,
                    value=_MLX_DEFAULT_CLONE_MODEL_LABEL,
                )
            presenter_engine_help = gr.Markdown("English narration uses the built-in Kokoro TTS voices.")

            gr.HTML('<div class="divider"></div>')

            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>lips</span> Presenter Lip-sync</div>")
            presenter_enhance = gr.Radio(
                label="Enhance presenter face after lip-sync",
                choices=["Yes", "No"],
                value="No",
            )
            gr.Markdown(
                "Select **Yes** to run face enhancement after lip-sync. Select **No** to keep the raw lip-sync output. "
                "Applies to **MuseTalk 1.5**, **SadTalker 256px**, and **SadTalker HD**."
            )
            with gr.Accordion("Advanced Presenter Controls", open=False):
                presenter_lipsync_engine = gr.Radio(
                    label="Lip-sync Engine",
                    choices=["MuseTalk 1.5", "SadTalker 256px", "SadTalker HD"],
                    value="MuseTalk 1.5",
                )
                with gr.Column(visible=True) as presenter_mt_params:
                    with gr.Row():
                        presenter_mt_batch = gr.Slider(label="Batch Size", minimum=1, maximum=16, step=1, value=8)
                        presenter_mt_bbox = gr.Slider(label="Lip Region Shift", minimum=-10, maximum=10, step=1, value=0)
                with gr.Column(visible=False) as presenter_st_params:
                    with gr.Row():
                        presenter_st_expr = gr.Slider(label="Expression Scale", minimum=0.5, maximum=3.0, step=0.1, value=1.0)
                        presenter_st_pose = gr.Slider(label="Pose Style", minimum=0, maximum=45, step=1, value=0)
                    with gr.Row():
                        presenter_st_still = gr.Checkbox(label="Still mode", value=True)
                        presenter_st_preprocess = gr.Dropdown(
                            label="Preprocess",
                            choices=["crop", "extcrop", "resize", "full", "extfull"],
                            value="full",
                        )

            gr.HTML('<div class="divider"></div>')

            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>movie</span> Generate</div>")
            with gr.Row():
                presenter_generate_btn = gr.Button(
                    "Generate Slide Presenter",
                    variant="primary",
                    elem_classes=["g-btn-primary"],
                    scale=3,
                )
                presenter_cancel_btn = gr.Button(
                    "Cancel",
                    variant="stop",
                    elem_classes=["g-btn-danger"],
                    scale=1,
                )

            presenter_log = gr.HTML(
                value=(
                    "<div class='progress-panel'>"
                    "<div class='progress-header'><span class='material-symbols-outlined'>present_to_all</span> Slide Presenter Pipeline</div>"
                    "<div style='color:var(--g-on-surface-variant);font-size:0.85rem;padding:12px 0'>"
                    "Ready — validate the PDF/JSON pair, select an avatar, and click Generate."
                    "</div></div>"
                )
            )

            with gr.Row():
                presenter_output = gr.Video(
                    label="Preview Video",
                    height=440,
                    scale=2,
                )
                with gr.Column(scale=1):
                    presenter_report = gr.Textbox(
                        label="Report",
                        lines=15,
                        interactive=False,
                        placeholder="Generation report and reusable project details will appear here…",
                    )
                    presenter_files = gr.File(
                        label="Generated Assets",
                        file_count="multiple",
                    )
                    presenter_open_folder = gr.Button("Open Presentations Folder", size="sm", variant="secondary")
                    presenter_folder_status = gr.Textbox(
                        label="",
                        interactive=False,
                        lines=1,
                        show_label=False,
                        visible=True,
                    )

            presenter_validate_btn.click(
                fn=validate_narration_files,
                inputs=[presenter_pdf, presenter_json],
                outputs=[presenter_validation_result],
            )
            presenter_avatar_upload.change(
                fn=save_presenter_avatar,
                inputs=[presenter_avatar_upload, presenter_avatar_choice],
                outputs=[presenter_avatar_choice, presenter_avatar_preview, presenter_avatar_status],
            )
            presenter_avatar_save_btn.click(
                fn=save_presenter_avatar,
                inputs=[presenter_avatar_upload, presenter_avatar_choice],
                outputs=[presenter_avatar_choice, presenter_avatar_preview, presenter_avatar_status],
            )
            presenter_avatar_refresh_btn.click(
                fn=refresh_avatar_dropdown,
                inputs=[presenter_avatar_choice],
                outputs=[presenter_avatar_choice, presenter_avatar_preview, presenter_avatar_status],
            )
            presenter_avatar_choice.change(
                fn=preview_saved_avatar,
                inputs=[presenter_avatar_choice],
                outputs=[presenter_avatar_preview, presenter_avatar_status],
            )
            presenter_mode.change(
                fn=_toggle_narration_tts_controls,
                inputs=[presenter_mode],
                outputs=[presenter_kokoro_row, presenter_mlx_col, presenter_engine_help],
            )
            presenter_ja_source.change(
                fn=_toggle_narration_japanese_source,
                inputs=[presenter_ja_source],
                outputs=[
                    presenter_mlx_saved_col,
                    presenter_mlx_preset_col,
                    presenter_kokoro_ja_col,
                    presenter_mlx_model,
                    presenter_ja_source_help,
                ],
            )
            presenter_mlx_refresh.click(
                fn=_mlx_voice_dropdown_update,
                inputs=[presenter_mlx_voice],
                outputs=[presenter_mlx_voice],
            )
            presenter_lipsync_engine.change(
                fn=_toggle_lipsync_params,
                inputs=[presenter_lipsync_engine],
                outputs=[presenter_mt_params, presenter_st_params],
            )
            presenter_generate_btn.click(
                fn=generate_slide_presenter,
                inputs=[
                    presenter_pdf,
                    presenter_json,
                    presenter_logo,
                    presenter_avatar_upload,
                    presenter_avatar_choice,
                    presenter_project_tag,
                    presenter_slide_selection,
                    presenter_output_mode,
                    presenter_mode,
                    presenter_voice,
                    presenter_ja_source,
                    presenter_mlx_voice,
                    presenter_mlx_preset,
                    presenter_kokoro_ja_voice,
                    presenter_mlx_model,
                    presenter_pause,
                    presenter_lipsync_engine,
                    presenter_enhance,
                    presenter_mt_batch,
                    presenter_mt_bbox,
                    presenter_st_expr,
                    presenter_st_pose,
                    presenter_st_still,
                    presenter_st_preprocess,
                ],
                outputs=[presenter_output, presenter_log, presenter_report, presenter_files],
            )
            presenter_cancel_btn.click(fn=cancel_generation, outputs=[presenter_log])
            presenter_open_folder.click(fn=open_presentations_folder, outputs=[presenter_folder_status])

    # ── Merge Videos ─────────────────────────────────────────────────────────
    with gr.Accordion("Merge Videos", open=False):
        gr.HTML(
            "<div class='section-title'>"
            "<span class='material-symbols-outlined'>merge</span>"
            " Combine output videos into one</div>"
        )
        gr.Markdown(
            "Select any videos from your **output folder**, in the order you want them joined. "
            "Click **Refresh** to pick up newly generated files. The merged file is saved back to the output folder."
        )
        with gr.Row():
            merge_refresh_btn = gr.Button(
                "Refresh List",
                size="sm",
                variant="secondary",
                scale=1,
            )
            merge_custom_name = gr.Textbox(
                label="Output filename (optional)",
                placeholder="e.g. my_presentation_final  (no .mp4 needed)",
                scale=3,
            )
        _initial_merge_choices = get_merge_choices()
        merge_checklist = gr.CheckboxGroup(
            label="Videos in output folder — check to include, top-to-bottom = play order",
            choices=_initial_merge_choices,
            value=_initial_merge_choices,
        )
        with gr.Row():
            merge_btn = gr.Button(
                "Merge Selected Videos",
                variant="primary",
                elem_classes=["g-btn-primary"],
                scale=3,
            )
            merge_open_folder_btn = gr.Button(
                "Open Output Folder",
                size="sm",
                variant="secondary",
                scale=1,
            )
        merge_status = gr.Textbox(
            label="Merge Status",
            lines=6,
            interactive=False,
        )
        merge_output_video = gr.Video(label="Merged Video", height=360)

        merge_refresh_btn.click(fn=refresh_merge_list, outputs=[merge_checklist])
        merge_btn.click(
            fn=merge_output_videos,
            inputs=[merge_checklist, merge_custom_name],
            outputs=[merge_output_video, merge_status],
        )
        merge_open_folder_btn.click(fn=open_output_folder, outputs=[merge_status])

    # ── Settings ─────────────────────────────────────────────────────────────
    with gr.Accordion("Settings", open=False):
        _key, _status = load_settings()
        settings_key = gr.Textbox(label="ElevenLabs API Key (optional — Kokoro is free)", type="password", value=_key)
        settings_status = gr.Markdown(_status)
        save_settings_btn = gr.Button("Save Settings", size="sm")
        save_settings_btn.click(fn=save_settings, inputs=[settings_key], outputs=[settings_status])


# ═══════════════════════════════════════════════════════════════════════════════
# Stand-alone launch
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=True,
        theme=THEME,
        css=CSS,
        favicon_path=str(ASSETS_DIR / "favicon.png"),
        allowed_paths=[str(ROOT)],
    )
