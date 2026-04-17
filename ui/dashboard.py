#!/usr/bin/env python3
"""
ui.dashboard — Gradio web dashboard for the AI Avatar Video Pipeline.

Start with:
    python scripts/run_dashboard.py
    — or —
    python -m ui.dashboard

Opens automatically at http://localhost:7860
"""

import base64
import gc
import glob
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
sys.path.insert(0, str(ROOT))

from avatarpipeline import (
    ASSETS_DIR,
    AUDIO_DIR,
    AVATARS_DIR,
    CAPTIONS_DIR,
    OUTPUT_DIR,
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
        img = Image.open(file_path).convert("RGB")
        target = 512
        ratio = min(target / img.width, target / img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target, target), (255, 255, 255))
        canvas.paste(img, ((target - new_w) // 2, (target - new_h) // 2))
        gallery_copy = AVATARS_DIR / f"{original_name}.png"
        if gallery_copy.name != "avatar.png":
            canvas.save(str(gallery_copy), "PNG")
        dest = AVATARS_DIR / "avatar.png"
        canvas.save(str(dest), "PNG")
        return str(dest), f"Avatar saved ({new_w}×{new_h} → 512×512)", get_avatar_gallery()
    except Exception as e:
        return None, f"Upload failed: {e}", get_avatar_gallery()


def get_avatar_gallery() -> list[str]:
    files = []
    for p in ["*.png", "*.jpg", "*.jpeg"]:
        files.extend(glob.glob(str(AVATARS_DIR / p)))
    return sorted(files, key=os.path.getmtime, reverse=True)


def select_avatar_from_gallery(evt: gr.SelectData) -> tuple[str | None, str]:
    gallery = get_avatar_gallery()
    if evt.index < len(gallery):
        selected = gallery[evt.index]
        dest = AVATARS_DIR / "avatar.png"
        if Path(selected).resolve() != dest.resolve():
            shutil.copy(selected, str(dest))
        return str(dest), f"Selected: {Path(selected).name}"
    return None, "Selection failed."


def get_video_history() -> list[str]:
    files = glob.glob(str(OUTPUT_DIR / "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:10]


def open_output_folder() -> str:
    subprocess.Popen(["open", str(OUTPUT_DIR)])
    return f"Opened {OUTPUT_DIR}"


def generate_voice_preview(voice_display: str) -> str | None:
    voice_id = VOICE_CHOICES.get(voice_display)
    if not voice_id:
        return None
    try:
        from avatarpipeline.voice.kokoro import VoiceGenerator
        vg = VoiceGenerator()
        name = voice_display.split("—")[0].strip()
        out = str(AUDIO_DIR / f"preview_{voice_id}.wav")
        vg.generate(f"Hello, I'm {name}. Nice to meet you!", voice=voice_id, out_path=out)
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
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=5,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(OUTPUT_DIR / f"studio_{run_id}.mp4")
    engine_map = {"MuseTalk 1.5": "musetalk", "SadTalker 256px": "sadtalker", "SadTalker HD": "sadtalker_hd"}
    engine_key = engine_map.get(lipsync_engine, "musetalk")

    progress(0, desc="Starting pipeline...")
    yield None, render(0), "", get_video_history()

    try:
        from avatarpipeline.voice.kokoro import VoiceGenerator
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
            from avatarpipeline.lipsync.sadtalker import SadTalkerInference
            st = SadTalkerInference(preset=engine_key)
            lipsync_mp4 = st.run(
                str(avatar_png), speech_16k, output_path=lipsync_mp4,
                expression_scale=st_expression_scale, pose_style=st_pose_style,
                still=st_still, preprocess=st_preprocess,
            )
        else:
            from avatarpipeline.lipsync.musetalk import MuseTalkInference
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
        from avatarpipeline.voice.kokoro import VoiceGenerator
        vg = VoiceGenerator()
        vg.generate(script, voice=voice_id, out_path=output_path)
        progress(1.0, desc="Done!")
        return output_path, f"Audio saved — {Path(output_path).name}"
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return None, f"Generation failed: {e}"


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
    --g-blue: #1a73e8;
    --g-blue-light: #e8f0fe;
    --g-blue-hover: #1765cc;
    --g-green: #1e8e3e;
    --g-green-light: #e6f4ea;
    --g-red: #d93025;
    --g-red-light: #fce8e6;
    --g-yellow: #f9ab00;
    --g-yellow-light: #fef7e0;
    --g-surface: #ffffff;
    --g-surface-dim: #f8f9fa;
    --g-surface-container: #f1f3f4;
    --g-on-surface: #202124;
    --g-on-surface-variant: #5f6368;
    --g-outline: #dadce0;
    --g-outline-variant: #e8eaed;
    --g-elevation-1: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
    --g-elevation-2: 0 1px 2px 0 rgba(60,64,67,0.3), 0 2px 6px 2px rgba(60,64,67,0.15);
    --g-elevation-3: 0 4px 8px 3px rgba(60,64,67,0.15), 0 1px 3px rgba(60,64,67,0.3);
    --g-radius: 12px;
    --g-radius-lg: 16px;
    --g-radius-xl: 28px;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Google Sans Text', 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--g-surface-dim) !important;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
.app-header {
    background: var(--g-surface);
    border-radius: var(--g-radius-lg);
    padding: 24px 32px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: var(--g-elevation-1);
    border: 1px solid var(--g-outline-variant);
}
.app-header .logo-circle {
    width: 48px; height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4285f4, #34a853, #fbbc04, #ea4335);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(66,133,244,0.3);
}
.app-header .logo-circle .material-symbols-outlined {
    font-size: 26px; color: white; font-variation-settings: 'FILL' 1;
}
.app-header h1 {
    font-family: 'Google Sans', sans-serif;
    font-size: 1.4rem; font-weight: 600; margin: 0;
    color: var(--g-on-surface); letter-spacing: -0.01em;
}
.app-header .tagline {
    color: var(--g-on-surface-variant); margin: 2px 0 0 0;
    font-size: 0.85rem; font-weight: 400;
}
.app-header .chip-row { display: flex; gap: 8px; margin-top: 10px; }
.app-header .chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 12px; border-radius: 16px;
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.01em;
}
.chip-online { background: var(--g-green-light); color: var(--g-green); }
.chip-local  { background: var(--g-blue-light); color: var(--g-blue); }
.chip-hw     { background: var(--g-yellow-light); color: #e37400; }

/* ── Tab Overrides ────────────────────────────────────────────────────────── */
.tabs > .tab-nav {
    background: var(--g-surface) !important;
    border-radius: var(--g-radius) var(--g-radius) 0 0 !important;
    border-bottom: 1px solid var(--g-outline) !important;
    box-shadow: none !important;
    padding: 0 8px !important;
}
.tabs > .tab-nav button {
    font-family: 'Google Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--g-on-surface-variant) !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 14px 20px !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.tabs > .tab-nav button:hover {
    background: var(--g-surface-container) !important;
    color: var(--g-blue) !important;
}
.tabs > .tab-nav button.selected {
    color: var(--g-blue) !important;
    border-bottom-color: var(--g-blue) !important;
    font-weight: 600 !important;
    background: transparent !important;
}
.tabitem {
    background: var(--g-surface) !important;
    border-radius: 0 0 var(--g-radius) var(--g-radius) !important;
    border: 1px solid var(--g-outline-variant) !important;
    border-top: none !important;
    padding: 24px !important;
}

/* ── Section headings ────────────────────────────────────────────────────── */
.section-title {
    font-family: 'Google Sans', sans-serif;
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--g-on-surface-variant);
    margin: 0 0 12px 0 ; padding: 0 0 8px 0;
    border-bottom: 1px solid var(--g-outline-variant);
    display: flex; align-items: center; gap: 6px;
}
.section-title .material-symbols-outlined {
    font-size: 16px; color: var(--g-blue);
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.g-btn-primary {
    font-family: 'Google Sans', sans-serif !important;
    font-size: 14px !important; font-weight: 500 !important;
    padding: 12px 24px !important;
    background: var(--g-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--g-radius-xl) !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: var(--g-elevation-1) !important;
}
.g-btn-primary:hover {
    background: var(--g-blue-hover) !important;
    box-shadow: var(--g-elevation-2) !important;
}
.g-btn-danger {
    border-radius: var(--g-radius-xl) !important;
    font-family: 'Google Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* ── Avatar Display ──────────────────────────────────────────────────────── */
.avatar-display img {
    object-fit: contain !important; max-height: 240px !important;
    width: 100% !important; border-radius: var(--g-radius) !important;
    background: var(--g-surface-dim) !important;
    border: 1px solid var(--g-outline-variant) !important;
}

/* ── Progress Panel ──────────────────────────────────────────────────────── */
.progress-panel {
    background: var(--g-surface);
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
    background: var(--g-outline-variant);
    margin: 20px 0;
}

/* ── Footer hide ─────────────────────────────────────────────────────────── */
footer { display: none !important; }

/* ── Inputs / Dropdowns (Google feel) ────────────────────────────────────── */
.gradio-container input, .gradio-container textarea,
.gradio-container select {
    font-family: 'Google Sans Text', sans-serif !important;
    border-radius: 8px !important;
}
.gradio-container .label-wrap {
    font-family: 'Google Sans', sans-serif !important;
}
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
        # TAB 2: Audio to Lipsync
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("Audio to Lipsync", id="tab-a2l"):
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>lips</span> Audio → Lip-synced Video</div>")
            gr.Markdown("Upload audio and an avatar image to generate a **lip-synced talking-head video**. Skips TTS — uses your audio directly.")

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
        # TAB 3: Text to Lipsync (Full Pipeline)
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("Text to Lipsync", id="tab-t2l"):
            gr.HTML("<div class='section-title'><span class='material-symbols-outlined'>auto_awesome</span> Full Pipeline: Text → Video</div>")
            gr.Markdown("**End-to-end:** Text → Speech → Lip-sync → Enhancement → Composite → Captions → Final Video")

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
