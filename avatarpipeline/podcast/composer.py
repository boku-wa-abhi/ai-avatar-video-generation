#!/usr/bin/env python3
"""
avatarpipeline.podcast.composer — Podcast video composition engine.

Handles multi-speaker script parsing, per-speaker audio track generation,
split-screen / picture-in-picture video composition, and overlay effects.
All video processing uses ffmpeg (no extra dependencies).
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

# ── Layout & overlay presets ────────────────────────────────────────────────

LAYOUT_CHOICES = [
    "Split Screen",
    "Focus Speaker A",
    "Focus Speaker B",
]

OVERLAY_CHOICES = [
    "None",
    "Warm Candle Glow",
    "Cinematic Vignette",
    "Film Grain",
    "Warm Glow + Vignette",
    "Cool Blue Tone",
    "Soft Dreamy",
]

_OVERLAY_FILTERS: dict[str, str] = {
    "None": "",
    "Warm Candle Glow": (
        "colortemperature=temperature=4500,"
        "eq=brightness=0.03:saturation=1.2"
    ),
    "Cinematic Vignette": "vignette=PI/4",
    "Film Grain": "noise=c0s=8:c0f=u+t",
    "Warm Glow + Vignette": (
        "colortemperature=temperature=4500,"
        "eq=brightness=0.03:saturation=1.2,"
        "vignette=PI/4"
    ),
    "Cool Blue Tone": "colorbalance=bs=0.15:bm=0.08,eq=saturation=0.9",
    "Soft Dreamy": (
        "gblur=sigma=0.8,"
        "eq=brightness=0.04:saturation=1.1,"
        "vignette=PI/5"
    ),
}


# ── Script parsing ──────────────────────────────────────────────────────────

def parse_podcast_script(script: str) -> list[dict]:
    """Parse a multi-speaker podcast script into segments.

    Accepted formats::

        [Host]: Welcome to our show!
        [Guest]: Thanks for having me.

    Also supports ``Speaker:`` without brackets as a fallback.

    Returns list of ``{"speaker": str, "text": str}``.
    """
    segments: list[dict] = []

    # Primary format: [Speaker Name]: text
    pattern = re.compile(r'\[([^\]]+)\]\s*:\s*(.+?)(?=\n\s*\[|\Z)', re.DOTALL)
    for m in pattern.finditer(script.strip()):
        speaker = m.group(1).strip()
        text = m.group(2).strip()
        if text:
            segments.append({"speaker": speaker, "text": text})

    if not segments:
        # Fallback: Speaker: text  (no brackets)
        pattern2 = re.compile(
            r'^(\w[\w\s]*?):\s*(.+?)(?=\n\w[\w\s]*?:|\Z)',
            re.DOTALL | re.MULTILINE,
        )
        for m in pattern2.finditer(script.strip()):
            speaker = m.group(1).strip()
            text = m.group(2).strip()
            if text:
                segments.append({"speaker": speaker, "text": text})

    return segments


def get_unique_speakers(segments: list[dict]) -> list[str]:
    """Return unique speaker names in order of first appearance."""
    seen: set[str] = set()
    speakers: list[str] = []
    for seg in segments:
        if seg["speaker"] not in seen:
            seen.add(seg["speaker"])
            speakers.append(seg["speaker"])
    return speakers


# ── Audio utilities ─────────────────────────────────────────────────────────

def _get_audio_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _generate_silence(duration: float, output_dir: Path) -> str:
    """Create a silent WAV of the given duration."""
    key = f"{duration:.2f}".replace('.', '_')
    path = str(output_dir / f"_silence_{key}s.wav")
    if not Path(path).exists():
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"anullsrc=r=16000:cl=mono", "-t", str(duration),
             "-c:a", "pcm_s16le", path],
            capture_output=True, text=True, timeout=10,
        )
    return path


def resample_16k(input_path: str, output_path: str) -> str:
    """Resample audio to 16 kHz mono PCM."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Audio resample failed:\n{r.stderr[-400:]}")
    return output_path


def mix_audio_tracks(track_paths: list[str], output_path: str) -> str:
    """Mix multiple audio tracks into one."""
    if len(track_paths) == 1:
        shutil.copy(track_paths[0], output_path)
        return output_path

    inputs: list[str] = []
    for p in track_paths:
        inputs.extend(["-i", p])

    n = len(track_paths)
    r = subprocess.run(
        ["ffmpeg", "-y", *inputs,
         "-filter_complex", f"amix=inputs={n}:duration=longest:normalize=0",
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Audio mix failed:\n{r.stderr[-400:]}")
    return output_path


# ── Per-speaker audio from script ───────────────────────────────────────────

def generate_per_speaker_audio(
    segments: list[dict],
    speakers: list[str],
    voice_map: dict[str, str],
    output_dir: str | Path,
    gap: float = 0.4,
) -> tuple[str, dict[str, str], list[dict]]:
    """TTS each segment, then build per-speaker audio tracks.

    Returns ``(master_audio, {speaker: track_path}, timeline)``.
    Timeline entries: ``{"speaker", "start", "end", "audio_file"}``.
    """
    from avatarpipeline.voice.kokoro import VoiceGenerator

    vg = VoiceGenerator()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. TTS every segment
    seg_files: list[dict] = []
    for i, seg in enumerate(segments):
        voice_id = voice_map.get(seg["speaker"], "af_heart")
        seg_path = str(output_dir / f"seg_{i:03d}.wav")
        vg.generate(seg["text"], voice=voice_id, out_path=seg_path)
        dur = _get_audio_duration(seg_path)
        seg_files.append({
            "speaker": seg["speaker"],
            "path": seg_path,
            "duration": dur,
        })
        logger.info(f"TTS seg {i}: {seg['speaker']} → {dur:.1f}s")

    # 2. Build timeline
    timeline: list[dict] = []
    cursor = 0.0
    for sf in seg_files:
        timeline.append({
            "speaker": sf["speaker"],
            "start": cursor,
            "end": cursor + sf["duration"],
            "audio_file": sf["path"],
        })
        cursor += sf["duration"] + gap

    # 3. Per-speaker tracks (speech + silence for the other speaker)
    speaker_tracks: dict[str, str] = {}
    for speaker in speakers:
        track_path = str(output_dir / f"track_{_slugify(speaker)}.wav")
        _build_speaker_track(timeline, speaker, track_path, output_dir)
        speaker_tracks[speaker] = track_path

    # 4. Master mix (all speakers combined)
    master_path = str(output_dir / "master_audio.wav")
    mix_audio_tracks(list(speaker_tracks.values()), master_path)

    return master_path, speaker_tracks, timeline


def _build_speaker_track(
    timeline: list[dict],
    speaker: str,
    output_path: str,
    work_dir: Path,
) -> None:
    """Concat a speaker's audio segments with silence padding for others."""
    concat_path = str(work_dir / f"_concat_{_slugify(speaker)}.txt")
    try:
        with open(concat_path, "w") as f:
            cursor = 0.0
            for entry in timeline:
                gap = entry["start"] - cursor
                if gap > 0.01:
                    f.write(f"file '{_generate_silence(gap, work_dir)}'\n")

                if entry["speaker"] == speaker:
                    f.write(f"file '{entry['audio_file']}'\n")
                else:
                    dur = entry["end"] - entry["start"]
                    f.write(f"file '{_generate_silence(dur, work_dir)}'\n")

                cursor = entry["end"]

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path],
            capture_output=True, text=True, timeout=120,
        )
    finally:
        if os.path.exists(concat_path):
            os.unlink(concat_path)


# ── Video composition ───────────────────────────────────────────────────────

def compose_podcast_video(
    video_a: str,
    video_b: str,
    master_audio: str,
    output_path: str,
    layout: str = "Split Screen",
    overlay: str = "None",
    custom_overlay_path: str | None = None,
    orientation: str = "16:9",
) -> str:
    """Compose two lip-synced videos into a single podcast video.

    Parameters
    ----------
    video_a, video_b : str
        Paths to each speaker's lip-synced MP4.
    master_audio : str
        Mixed audio track.
    output_path : str
        Destination MP4.
    layout : str
        One of :data:`LAYOUT_CHOICES`.
    overlay : str
        One of :data:`OVERLAY_CHOICES`.
    custom_overlay_path : str | None
        Optional PNG/WebP with alpha to overlay on top.
    orientation : str
        ``"16:9"``, ``"9:16"``, or ``"1:1"``.
    """
    res_map = {
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "1:1":  (1080, 1080),
    }
    W, H = res_map.get(orientation, (1920, 1080))
    overlay_ff = _OVERLAY_FILTERS.get(overlay, "")

    if layout == "Focus Speaker A":
        fc = _pip_filter(W, H, overlay_ff, main=0, pip=1)
    elif layout == "Focus Speaker B":
        fc = _pip_filter(W, H, overlay_ff, main=1, pip=0)
    else:
        fc = _split_screen_filter(W, H, overlay_ff, orientation)

    # Custom image overlay (transparent PNG on top)
    extra_inputs: list[str] = []
    if custom_overlay_path and Path(custom_overlay_path).exists():
        extra_inputs = ["-i", custom_overlay_path]
        idx = 3  # 0=video_a, 1=video_b, 2=audio, 3=overlay image
        fc = fc.replace(
            "[vout]",
            f"[pre_overlay];[{idx}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black@0.0,format=rgba[ovl];"
            f"[pre_overlay][ovl]overlay=0:0:format=auto[vout]",
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_a,
        "-i", video_b,
        "-i", master_audio,
        *extra_inputs,
        "-filter_complex", fc,
        "-map", "[vout]",
        "-map", "2:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]

    logger.info(f"Composing podcast: layout={layout}, overlay={overlay}, {W}x{H}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg compose failed:\n{result.stderr[-800:]}")

    return output_path


def _split_screen_filter(
    W: int, H: int, overlay_ff: str, orientation: str,
) -> str:
    """Build ffmpeg filter_complex for side-by-side layout."""
    if orientation == "9:16":
        # Vertical split: top / bottom
        half = H // 2
        parts = [
            f"[0:v]scale={W}:{half}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{half}:(ow-iw)/2:(oh-ih)/2:black[a]",
            f"[1:v]scale={W}:{half}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{half}:(ow-iw)/2:(oh-ih)/2:black[b]",
            "[a][b]vstack[composed]",
        ]
    else:
        # Horizontal split: left / right
        half = W // 2
        parts = [
            f"[0:v]scale={half}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={half}:{H}:(ow-iw)/2:(oh-ih)/2:black[a]",
            f"[1:v]scale={half}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={half}:{H}:(ow-iw)/2:(oh-ih)/2:black[b]",
            "[a][b]hstack[composed]",
        ]

    if overlay_ff:
        parts.append(f"[composed]{overlay_ff}[vout]")
    else:
        parts.append("[composed]copy[vout]")

    return ";".join(parts)


def _pip_filter(
    W: int, H: int, overlay_ff: str, main: int = 0, pip: int = 1,
) -> str:
    """Build ffmpeg filter_complex for picture-in-picture layout."""
    pip_w = W // 4
    pip_h = H // 4
    margin = 20
    border = 3

    parts = [
        f"[{main}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black[main]",
        # PiP with thin white border
        f"[{pip}:v]scale={pip_w - 2*border}:{pip_h - 2*border}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={pip_w}:{pip_h}:(ow-iw)/2:(oh-ih)/2:color=white@0.6[pip]",
        f"[main][pip]overlay={W - pip_w - margin}:{H - pip_h - margin}[composed]",
    ]

    if overlay_ff:
        parts.append(f"[composed]{overlay_ff}[vout]")
    else:
        parts.append("[composed]copy[vout]")

    return ";".join(parts)
