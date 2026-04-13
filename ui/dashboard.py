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
# ui/dashboard.py lives one level below the project root
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
    "📱 Portrait 9:16":  "9:16",
    "🖥️ Landscape 16:9": "16:9",
    "⬛ Square 1:1":     "1:1",
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
    """Save uploaded avatar, resize to 512×512, return (preview, message, gallery)."""
    if file_path is None:
        return None, "No file selected.", get_avatar_gallery()

    try:
        AVATARS_DIR.mkdir(parents=True, exist_ok=True)

        # Keep a named copy so gallery shows it
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

        # Also save the original-name version in the gallery
        gallery_copy = AVATARS_DIR / f"{original_name}.png"
        if gallery_copy.name != "avatar.png":
            canvas.save(str(gallery_copy), "PNG")

        dest = AVATARS_DIR / "avatar.png"
        canvas.save(str(dest), "PNG")
        return str(dest), f"✅ Avatar saved ({new_w}×{new_h} → 512×512)", get_avatar_gallery()
    except Exception as e:
        return None, f"❌ Upload failed: {e}", get_avatar_gallery()


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
        return str(dest), f"✅ Selected: {Path(selected).name}"
    return None, "Selection failed."


# ── Video history ────────────────────────────────────────────────────────────

def get_video_history() -> list[str]:
    files = glob.glob(str(OUTPUT_DIR / "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:10]


def open_output_folder() -> str:
    subprocess.Popen(["open", str(OUTPUT_DIR)])
    return f"📂 Opened {OUTPUT_DIR}"


# ── Voice preview ────────────────────────────────────────────────────────────

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


# ── Settings ─────────────────────────────────────────────────────────────────

def load_settings() -> tuple[str, str]:
    cfg = _load_config()
    key = cfg.get("elevenlabs_key", "") or ""
    status = "✅ Key configured" if key else "ℹ️  No API key needed — using local Kokoro TTS (free)"
    return key, status


def save_settings(api_key: str) -> str:
    cfg = _load_config()
    cfg["elevenlabs_key"] = api_key
    _save_config(cfg)
    return "✅ Settings saved"


# ── Character counter ────────────────────────────────────────────────────────

def update_char_count(text: str) -> str:
    n = len(text) if text else 0
    return f"<span style='color:#94a3b8;font-size:0.82rem'>{n:,} characters · Kokoro TTS (local, free)</span>"


# ── Video metadata card ──────────────────────────────────────────────────────

def get_video_metadata(video_path: str, gen_time_secs: float) -> str:
    if not video_path or not Path(video_path).exists():
        return ""
    try:
        size_mb = Path(video_path).stat().st_size / 1_048_576
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "default=noprint_wrappers=1",
            video_path,
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
            "<div class='metadata-card'>"
            f"<div class='stat-row'><span class='stat-label'>Resolution</span>"
            f"<span class='stat-value'>{width}×{height}</span></div>"
            f"<div class='stat-row'><span class='stat-label'>Duration</span>"
            f"<span class='stat-value'>{duration}</span></div>"
            f"<div class='stat-row'><span class='stat-label'>File Size</span>"
            f"<span class='stat-value'>{size_mb:.1f} MB</span></div>"
            f"<div class='stat-row'><span class='stat-label'>Generated in</span>"
            f"<span class='stat-value'>{gm}m {gs}s</span></div>"
            "</div>"
        )
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Main generation function (streaming progress via yield)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_video(
    script: str,
    voice_choice: str,
    orientation: str,
    music_volume: float,
    background_file: str | None,
    lipsync_engine: str,
    enhance_face: bool,
    add_captions: bool,
    preview_mode: bool,
    caption_font_size: int,
    caption_position: str,
    mt_batch_size: int = 8,
    mt_bbox_shift: int = 0,
    st_expression_scale: float = 1.0,
    st_pose_style: int = 0,
    st_still: bool = True,
    st_preprocess: str = "full",
    progress=gr.Progress(track_tqdm=False),
):
    """Run the full 7-step pipeline with rich HTML progress updates."""
    _cancel_event.clear()
    wall_start = time.time()
    TOTAL = 7

    # State for rich progress panel
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

    # ── Validation ────────────────────────────────────────────────────────
    if not script or not script.strip():
        states[0] = "error"
        yield None, render(0, "Script is empty — please enter text above."), "", get_video_history()
        return

    avatar_png = AVATARS_DIR / "avatar.png"
    if not avatar_png.exists():
        states[0] = "error"
        yield None, render(0, "No avatar found — upload a PNG/JPG in the Avatar section."), "", get_video_history()
        return

    voice_id    = VOICE_CHOICES.get(voice_choice, "af_heart")
    orient_code = ORIENTATION_MAP.get(orientation, "9:16")
    background  = str(background_file) if background_file and Path(background_file).exists() else "black"
    run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(OUTPUT_DIR / f"studio_{run_id}.mp4")
    engine_map = {
        "MuseTalk 1.5 (default)": "musetalk",
        "SadTalker (256 px)": "sadtalker",
        "SadTalker HD (512 px + GFPGAN)": "sadtalker_hd",
    }
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
        # ── STEP 1: TTS ──────────────────────────────────────────────────
        if _cancel_event.is_set():
            yield None, render(0, "Cancelled."), "", get_video_history()
            return
        states[0] = "active"
        progress(1 / TOTAL, desc="Step 1/7: Voice synthesis")
        yield None, render(1 / TOTAL), "", get_video_history()

        t0 = time.time()
        vg = VoiceGenerator()
        speech_wav = str(AUDIO_DIR / f"speech_{run_id}.wav")
        vg.generate(script, voice=voice_id, out_path=speech_wav)
        states[0] = "done"; times[0] = step_time(t0)
        yield None, render(1 / TOTAL), "", get_video_history()

        # ── STEP 2: Resample ─────────────────────────────────────────────
        if _cancel_event.is_set():
            yield None, render(1 / TOTAL, "Cancelled."), "", get_video_history()
            return
        states[1] = "active"
        progress(2 / TOTAL, desc="Step 2/7: Audio prep")
        yield None, render(2 / TOTAL), "", get_video_history()

        t0 = time.time()
        speech_16k = str(AUDIO_DIR / f"speech_{run_id}_16k.wav")
        vg.convert_to_16k(speech_wav, speech_16k)
        states[1] = "done"; times[1] = step_time(t0)
        yield None, render(2 / TOTAL), "", get_video_history()

        # ── STEP 3: Lip-sync ─────────────────────────────────────────────
        if _cancel_event.is_set():
            yield None, render(2 / TOTAL, "Cancelled."), "", get_video_history()
            return
        states[2] = "active"
        progress(3 / TOTAL, desc=f"Step 3/7: Lip-sync ({lipsync_engine})")
        yield None, render(3 / TOTAL), "", get_video_history()

        t0 = time.time()
        lipsync_mp4 = str(OUTPUT_DIR / f"lipsync_{run_id}.mp4")

        if engine_key in ("sadtalker", "sadtalker_hd"):
            from avatarpipeline.lipsync.sadtalker import SadTalkerInference
            st = SadTalkerInference(preset=engine_key)
            lipsync_mp4 = st.run(
                str(avatar_png), speech_16k,
                output_path=lipsync_mp4,
                expression_scale=st_expression_scale,
                pose_style=st_pose_style,
                still=st_still,
                preprocess=st_preprocess,
            )
        else:
            from avatarpipeline.lipsync.musetalk import MuseTalkInference
            ms = MuseTalkInference()
            ms.prepare_avatar(str(avatar_png))
            lipsync_mp4 = ms.run(
                str(avatar_png), speech_16k,
                batch_size=mt_batch_size,
                bbox_shift=mt_bbox_shift,
            )

        states[2] = "done"; times[2] = step_time(t0)
        yield None, render(3 / TOTAL), "", get_video_history()

        # ── STEP 4: Face enhancement ──────────────────────────────────────
        if enhance_face and not preview_mode:
            if _cancel_event.is_set():
                yield None, render(3 / TOTAL, "Cancelled."), "", get_video_history()
                return
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

        # ── STEP 5: Background composite ─────────────────────────────────
        if _cancel_event.is_set():
            yield None, render(4 / TOTAL, "Cancelled."), "", get_video_history()
            return
        states[4] = "active"
        progress(5 / TOTAL, desc="Step 5/7: Composite")
        yield None, render(5 / TOTAL), "", get_video_history()

        t0 = time.time()
        va = VideoAssembler()
        composed_mp4 = str(OUTPUT_DIR / f"composed_{run_id}.mp4")
        composed_mp4 = va.add_background(
            enhanced_mp4, orientation=orient_code,
            background=background, output_path=composed_mp4,
        )
        if music_volume > 0 and background_file and Path(background_file).suffix.lower() in (".mp3", ".wav", ".m4a"):
            music_out = composed_mp4.replace(".mp4", "_music.mp4")
            composed_mp4 = va.add_music(composed_mp4, str(background_file), music_volume=music_volume, output_path=music_out)
        states[4] = "done"; times[4] = step_time(t0)
        yield None, render(5 / TOTAL), "", get_video_history()

        # ── STEP 6: Captions ─────────────────────────────────────────────
        srt_path = None
        if add_captions and not preview_mode:
            if _cancel_event.is_set():
                yield None, render(5 / TOTAL, "Cancelled."), "", get_video_history()
                return
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

        # ── STEP 7: Final encode ──────────────────────────────────────────
        if _cancel_event.is_set():
            yield None, render(6 / TOTAL, "Cancelled."), "", get_video_history()
            return
        states[6] = "active"
        progress(7 / TOTAL, desc="Step 7/7: Final encode")
        yield None, render(7 / TOTAL), "", get_video_history()

        t0 = time.time()
        va.finalize(
            composed_mp4, output_path,
            srt_path=srt_path,
            include_captions=(add_captions and not preview_mode),
        )
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
        # Mark current step as error
        for i, s in enumerate(states):
            if s == "active":
                states[i] = "error"
                break
        logger.error(f"Dashboard pipeline error: {exc}")
        for p in OUTPUT_DIR.glob(f"*{run_id}*"):
            if "studio_" not in p.name:
                p.unlink(missing_ok=True)
        yield None, render(0, f"Error: {exc}"), "", get_video_history()


def cancel_generation():
    _cancel_event.set()
    return f"[{_ts()}] ⏹ Cancel requested — stopping after current step..."


def _toggle_lipsync_params(engine: str):
    """Show MuseTalk params for MuseTalk, SadTalker params for SadTalker variants."""
    is_musetalk = "MuseTalk" in engine
    return gr.update(visible=is_musetalk), gr.update(visible=not is_musetalk)


# ── Rich progress panel renderer ─────────────────────────────────────────────

_STEP_ICONS = {
    "done": ("✓", "done"),
    "active": ("⟳", "active"),
    "waiting": ("·", "waiting"),
    "skipped": ("—", "skipped"),
    "error": ("✗", "error"),
}

STEP_NAMES = [
    "Voice Synthesis (TTS)",
    "Audio Resampling",
    "Lip-sync Generation",
    "Face Enhancement",
    "Background Composite",
    "Caption Generation",
    "Final Encode",
]


def _build_progress_html(
    step_states: list[str],
    step_times: list[str],
    pct: float,
    elapsed: str,
    engine: str = "",
    message: str = "",
) -> str:
    """Render the pipeline progress as a styled HTML panel.

    Args:
        step_states: list of 7 states from ('done','active','waiting','skipped','error')
        step_times:  list of 7 time strings (e.g. '2.1s', '', '—')
        pct:         overall progress 0.0–1.0
        elapsed:     total elapsed string
        engine:      engine name for step 3 label
        message:     optional footer message
    """
    rows = []
    for i, (name, state, t) in enumerate(zip(STEP_NAMES, step_states, step_times)):
        icon_char, css_cls = _STEP_ICONS[state]
        label_cls = "active" if state == "active" else ("waiting" if state == "waiting" else "")
        label = name
        if i == 2 and engine:
            label = f"{name} — {engine}"
        rows.append(
            f'<div class="step-row">'
            f'<div class="step-icon {css_cls}">{icon_char}</div>'
            f'<div class="step-label {label_cls}">{label}</div>'
            f'<div class="step-time">{t}</div>'
            f'</div>'
        )
    bar_pct = max(0, min(100, int(pct * 100)))
    footer = f'<div style="color:#34d399;font-size:0.82rem;margin-top:12px;font-weight:600">{message}</div>' if message else ""
    return (
        f'<div class="pipeline-progress">'
        f'<div class="pipeline-title">Pipeline Progress</div>'
        f'{"".join(rows)}'
        f'<div class="progress-bar-outer"><div class="progress-bar-inner" style="width:{bar_pct}%"></div></div>'
        f'<div class="elapsed">Elapsed: {elapsed}</div>'
        f'{footer}'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Logo embedding
# ═══════════════════════════════════════════════════════════════════════════════

_logo_b64 = ""
_logo_path = ASSETS_DIR / "logo.png"
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()


# ═══════════════════════════════════════════════════════════════════════════════
# Professional CSS — BCG / McKinsey consulting aesthetic
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Foundation ──────────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
.studio-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 20px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}
.studio-header .logo-img {
    width: 56px; height: 56px;
    border-radius: 12px; flex-shrink: 0;
}
.studio-header h1 {
    font-size: 1.6rem; font-weight: 700; margin: 0;
    color: #f1f5f9; letter-spacing: -0.02em;
}
.studio-header .tagline {
    color: #94a3b8; margin: 4px 0 0 0;
    font-size: 0.88rem; font-weight: 400;
}
.studio-header .badge-row { display: flex; gap: 8px; margin-top: 10px; }
.studio-header .badge {
    display: inline-block; padding: 3px 12px;
    border-radius: 20px; font-size: 0.72rem;
    font-weight: 500; letter-spacing: 0.03em;
}
.studio-header .badge-ready {
    background: rgba(16,185,129,0.12); color: #34d399;
    border: 1px solid rgba(16,185,129,0.25);
}
.studio-header .badge-local {
    background: rgba(99,102,241,0.12); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.25);
}
.studio-header .badge-mps {
    background: rgba(251,191,36,0.12); color: #fcd34d;
    border: 1px solid rgba(251,191,36,0.25);
}

/* ── Section Labels ──────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: #64748b !important; margin: 0 0 8px 0 !important;
    padding: 0 0 6px 0 !important; border-bottom: 2px solid #e2e8f0 !important;
}

/* ── Avatar Display — full image visible ─────────────────────────────────── */
.avatar-display img {
    object-fit: contain !important; max-height: 260px !important;
    width: 100% !important; border-radius: 8px !important;
    background: #f8fafc !important;
}

/* ── Generate Button ─────────────────────────────────────────────────────── */
.generate-btn {
    font-size: 15px !important; font-weight: 600 !important;
    padding: 14px 28px !important;
    background: linear-gradient(135deg, #0d9488 0%, #10b981 100%) !important;
    border: none !important; border-radius: 10px !important;
    letter-spacing: 0.02em !important; transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(13,148,136,0.25) !important;
}
.generate-btn:hover {
    opacity: 0.92 !important;
    box-shadow: 0 4px 16px rgba(13,148,136,0.35) !important;
    transform: translateY(-1px) !important;
}
.cancel-btn { border-radius: 10px !important; font-weight: 500 !important; }

/* ── Pipeline Progress Panel ─────────────────────────────────────────────── */
.pipeline-progress {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px; padding: 20px 24px;
    border: 1px solid rgba(255,255,255,0.06);
    font-family: 'Inter', -apple-system, sans-serif;
}
.pipeline-progress .pipeline-title {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b; margin-bottom: 16px;
}
.pipeline-progress .step-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.pipeline-progress .step-row:last-child { border-bottom: none; }
.pipeline-progress .step-icon {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; flex-shrink: 0; font-weight: 600;
}
.pipeline-progress .step-icon.done { background: rgba(16,185,129,0.15); color: #34d399; }
.pipeline-progress .step-icon.active { background: rgba(99,102,241,0.2); color: #a5b4fc; animation: pulse 1.5s infinite; }
.pipeline-progress .step-icon.waiting { background: rgba(100,116,139,0.1); color: #475569; }
.pipeline-progress .step-icon.skipped { background: rgba(100,116,139,0.08); color: #64748b; }
.pipeline-progress .step-icon.error { background: rgba(239,68,68,0.15); color: #f87171; }
.pipeline-progress .step-label {
    flex: 1; font-size: 0.82rem; color: #cbd5e1; font-weight: 500;
}
.pipeline-progress .step-label.active { color: #e2e8f0; font-weight: 600; }
.pipeline-progress .step-label.waiting { color: #475569; }
.pipeline-progress .step-time {
    font-size: 0.72rem; color: #64748b; font-family: 'JetBrains Mono', monospace;
    min-width: 50px; text-align: right;
}
.pipeline-progress .progress-bar-outer {
    width: 100%; height: 4px; background: rgba(255,255,255,0.06);
    border-radius: 2px; margin-top: 16px; overflow: hidden;
}
.pipeline-progress .progress-bar-inner {
    height: 100%; border-radius: 2px; transition: width 0.5s ease;
    background: linear-gradient(90deg, #0d9488, #10b981, #34d399);
}
.pipeline-progress .elapsed {
    font-size: 0.72rem; color: #64748b; margin-top: 8px; text-align: right;
    font-family: 'JetBrains Mono', monospace;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ── Log Box ─────────────────────────────────────────────────────────────── */
.log-box textarea {
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Menlo', monospace !important;
    font-size: 11px !important; background: #0f172a !important;
    color: #cbd5e1 !important; border-radius: 8px !important;
    line-height: 1.65 !important; padding: 12px !important;
    border: 1px solid #1e293b !important;
}

/* ── Output Video ────────────────────────────────────────────────────────── */
.output-video { border-radius: 10px !important; overflow: hidden !important; }

/* ── Metadata Card ───────────────────────────────────────────────────────── */
.metadata-card { font-size: 13px; line-height: 1.7; }
.metadata-card .stat-row {
    display: flex; justify-content: space-between;
    padding: 6px 0; border-bottom: 1px solid #f1f5f9;
}
.metadata-card .stat-label { color: #64748b; font-weight: 500; }
.metadata-card .stat-value { color: #1e293b; font-weight: 600; }

/* ── Misc ────────────────────────────────────────────────────────────────── */
footer { display: none !important; }
.separator-line {
    height: 1px;
    background: linear-gradient(to right, transparent, #e2e8f0 20%, #e2e8f0 80%, transparent);
    margin: 16px 0; border: none;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Avatar Studio") as demo:

    # ── Header ───────────────────────────────────────────────────────────────
    logo_tag = (
        f'<img class="logo-img" src="data:image/png;base64,{_logo_b64}" alt="Logo"/>'
        if _logo_b64 else ""
    )
    gr.HTML(f"""
        <div class="studio-header">
            {logo_tag}
            <div>
                <h1>Avatar Studio</h1>
                <p class="tagline">AI-Powered Avatar Video Generation</p>
                <div class="badge-row">
                    <span class="badge badge-ready">● Pipeline Ready</span>
                    <span class="badge badge-local">Fully Offline</span>
                    <span class="badge badge-mps">Apple M4 Pro · MPS</span>
                </div>
            </div>
        </div>
    """)

    # ── Main content: Avatar (left) + Script/Voice/Settings (right) ──────────
    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=280):
            gr.Markdown("<p class='section-label'>Avatar</p>")
            avatar_upload = gr.Image(
                label="Upload avatar (PNG or JPG)",
                type="filepath",
                sources=["upload", "clipboard"],
                height=160,
                elem_classes=["avatar-display"],
            )
            avatar_preview = gr.Image(
                label="Active Avatar",
                type="filepath",
                interactive=False,
                height=200,
                value=str(AVATARS_DIR / "avatar.png") if (AVATARS_DIR / "avatar.png").exists() else None,
                elem_classes=["avatar-display"],
            )
            avatar_status = gr.Textbox(
                interactive=False, lines=1, show_label=False,
                value=(
                    "Active: data/avatars/avatar.png"
                    if (AVATARS_DIR / "avatar.png").exists()
                    else "No avatar uploaded"
                ),
            )
            avatar_gallery = gr.Gallery(
                label="Saved Avatars (click to select)",
                value=get_avatar_gallery(),
                columns=4, height=130, allow_preview=False,
            )

        with gr.Column(scale=3):
            gr.Markdown("<p class='section-label'>Script</p>")
            script_box = gr.Textbox(
                label="Enter script",
                placeholder="Type or paste the text your avatar will speak…",
                lines=5, show_label=False,
            )
            char_counter = gr.Markdown(
                "<span style='color:#94a3b8;font-size:0.82rem'>0 characters · Kokoro TTS (local, free)</span>"
            )

            gr.Markdown("<p class='section-label'>Voice</p>")
            with gr.Row():
                voice_dropdown = gr.Dropdown(
                    label="Select voice",
                    choices=list(VOICE_CHOICES.keys()),
                    value="Heart — Warm Female (default)",
                    scale=2, show_label=False,
                )
                voice_preview = gr.Audio(
                    label="Preview", type="filepath",
                    interactive=False, scale=2,
                )

            gr.Markdown("<p class='section-label'>Video Settings</p>")
            with gr.Row():
                orientation_radio = gr.Radio(
                    label="Orientation",
                    choices=list(ORIENTATION_MAP.keys()),
                    value="📱 Portrait 9:16", scale=2,
                )
                music_slider = gr.Slider(
                    label="Music Volume",
                    minimum=0.0, maximum=1.0, step=0.05, value=0.15, scale=1,
                )
                background_upload = gr.File(
                    label="Background / Music",
                    file_types=["image", ".mp4", ".mp3", ".wav", ".m4a"],
                    scale=1,
                )

            with gr.Accordion("Advanced Options", open=False):
                lipsync_radio = gr.Radio(
                    label="Lip-sync Engine",
                    choices=[
                        "MuseTalk 1.5 (default)",
                        "SadTalker (256 px)",
                        "SadTalker HD (512 px + GFPGAN)",
                    ],
                    value="MuseTalk 1.5 (default)",
                )

                # ── MuseTalk hyperparameters ──────────────────────────────
                with gr.Column(visible=True) as musetalk_params:
                    gr.Markdown(
                        "<span style='font-size:0.72rem;color:#64748b;font-weight:600;"
                        "text-transform:uppercase;letter-spacing:0.08em'>"
                        "MuseTalk Settings</span>"
                    )
                    with gr.Row():
                        mt_batch_size = gr.Slider(
                            label="Batch Size",
                            minimum=1, maximum=16, step=1, value=8,
                            info="Larger batch = faster inference, uses more memory",
                        )
                        mt_bbox_shift = gr.Slider(
                            label="Lip Region Shift",
                            minimum=-10, maximum=10, step=1, value=0,
                            info="Shift lip crop region up (−) or down (+)",
                        )

                # ── SadTalker hyperparameters ─────────────────────────────
                with gr.Column(visible=False) as sadtalker_params:
                    gr.Markdown(
                        "<span style='font-size:0.72rem;color:#64748b;font-weight:600;"
                        "text-transform:uppercase;letter-spacing:0.08em'>"
                        "SadTalker Settings</span>"
                    )
                    with gr.Row():
                        st_expression_scale = gr.Slider(
                            label="Expression Scale",
                            minimum=0.5, maximum=3.0, step=0.1, value=1.0,
                            info="How expressive the face movement is (1.0 = natural)",
                        )
                        st_pose_style = gr.Slider(
                            label="Pose Style (0–45)",
                            minimum=0, maximum=45, step=1, value=0,
                            info="Head pose variation preset (0 = neutral)",
                        )
                    with gr.Row():
                        st_still = gr.Checkbox(
                            label="Still mode (minimize head movement)", value=True,
                        )
                        st_preprocess = gr.Dropdown(
                            label="Preprocess Mode",
                            choices=["crop", "extcrop", "resize", "full", "extfull"],
                            value="full",
                            info="full = paste back to original; crop = face-only output",
                        )

                with gr.Row():
                    enhance_face_cb = gr.Checkbox(
                        label="CodeFormer face enhancement", value=True
                    )
                with gr.Row():
                    add_captions_cb = gr.Checkbox(
                        label="Auto-generated captions", value=True
                    )
                    preview_mode_cb = gr.Checkbox(
                        label="Preview mode (faster, lower quality)", value=False
                    )
                with gr.Row():
                    caption_fontsize = gr.Slider(
                        label="Caption Font Size",
                        minimum=12, maximum=32, step=1, value=20,
                    )
                    caption_position = gr.Dropdown(
                        label="Caption Position",
                        choices=["Bottom", "Center", "Top"], value="Bottom",
                    )

    # ── Generate controls ────────────────────────────────────────────────────
    gr.HTML('<div class="separator-line"></div>')
    with gr.Row():
        generate_btn = gr.Button(
            "Generate Video", variant="primary", scale=3,
            elem_classes=["generate-btn"],
        )
        cancel_btn = gr.Button(
            "Cancel", variant="stop", scale=1,
            elem_classes=["cancel-btn"],
        )

    log_box = gr.HTML(
        value="<div class='pipeline-progress'><div class='pipeline-title'>Pipeline Progress</div>"
              "<div style='color:#475569;font-size:0.82rem;padding:12px 0'>Ready — click Generate Video to start</div></div>",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    gr.Markdown("<p class='section-label'>Output</p>")
    with gr.Row():
        output_video = gr.Video(
            label="Generated Video",
            elem_classes=["output-video"], scale=3,
        )
        with gr.Column(scale=1):
            metadata_html = gr.HTML(value="")
            open_folder_btn = gr.Button("Open Output Folder", size="sm")
            folder_status = gr.Textbox(visible=False)

    video_history = gr.Gallery(
        label="Recent Videos",
        value=get_video_history(),
        columns=5, height=120, allow_preview=False,
    )

    with gr.Accordion("Settings", open=False):
        _key, _status = load_settings()
        settings_key = gr.Textbox(
            label="ElevenLabs API Key (optional — Kokoro is free)",
            type="password", value=_key,
        )
        settings_status = gr.Markdown(_status)
        save_settings_btn = gr.Button("Save Settings", size="sm")

    # ── Event wiring ─────────────────────────────────────────────────────────
    avatar_upload.change(
        fn=save_uploaded_avatar,
        inputs=[avatar_upload],
        outputs=[avatar_preview, avatar_status, avatar_gallery],
    )
    avatar_gallery.select(
        fn=select_avatar_from_gallery,
        inputs=None,
        outputs=[avatar_preview, avatar_status],
    )
    script_box.change(
        fn=update_char_count,
        inputs=[script_box], outputs=[char_counter],
    )
    voice_dropdown.change(
        fn=generate_voice_preview,
        inputs=[voice_dropdown], outputs=[voice_preview],
    )
    generate_btn.click(
        fn=generate_video,
        inputs=[
            script_box, voice_dropdown, orientation_radio, music_slider,
            background_upload, lipsync_radio, enhance_face_cb,
            add_captions_cb, preview_mode_cb, caption_fontsize, caption_position,
            mt_batch_size, mt_bbox_shift,
            st_expression_scale, st_pose_style, st_still, st_preprocess,
        ],
        outputs=[output_video, log_box, metadata_html, video_history],
    )
    cancel_btn.click(fn=cancel_generation, outputs=[log_box])
    lipsync_radio.change(
        fn=_toggle_lipsync_params,
        inputs=[lipsync_radio],
        outputs=[musetalk_params, sadtalker_params],
    )
    open_folder_btn.click(fn=open_output_folder, outputs=[folder_status])
    save_settings_btn.click(
        fn=save_settings,
        inputs=[settings_key], outputs=[settings_status],
    )


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
        theme=gr.themes.Soft(),
        css=CSS,
        favicon_path=str(ASSETS_DIR / "favicon.png"),
        allowed_paths=[str(ROOT)],
    )
