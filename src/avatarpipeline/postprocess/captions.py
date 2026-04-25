"""
avatarpipeline.postprocess.captions — Subtitle generation using faster-whisper.

Transcribes audio locally (large-v3 model by default) and produces SRT files
that can be burned into video via the FFmpeg subtitles filter.
"""

import os
import subprocess
from pathlib import Path

from loguru import logger

from avatarpipeline import CAPTIONS_DIR


class CaptionGenerator:
    """Generate SRT captions from audio and burn them into video."""

    def __init__(self, model_size: str = "large-v3", device: str = "auto") -> None:
        """Load the Whisper transcription model.

        Args:
            model_size: Whisper variant (tiny/base/small/medium/large-v3).
            device:     "auto" → CPU with int8 (fastest on Apple Silicon).
        """
        from faster_whisper import WhisperModel

        # faster-whisper doesn't support MPS — CPU int8 is fastest on M4 Pro
        resolved_device = "cpu"
        compute_type = "int8"

        logger.info(f"Loading Whisper {model_size} on {resolved_device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=resolved_device, compute_type=compute_type)
        logger.info("Whisper model ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_wav: str,
        output_srt: str | None = None,
    ) -> str:
        """Transcribe audio and write a properly formatted SRT file.

        Args:
            audio_wav:   Path to the 16 kHz mono WAV.
            output_srt:  Destination SRT path. Defaults to data/captions/output.srt.

        Returns:
            Absolute path to the generated SRT file.
        """
        audio_wav = Path(audio_wav).resolve()
        if not audio_wav.exists():
            raise FileNotFoundError(f"Audio not found: {audio_wav}")

        srt_path = Path(output_srt) if output_srt else CAPTIONS_DIR / "output.srt"
        if not srt_path.is_absolute():
            from avatarpipeline import ROOT
            srt_path = ROOT / srt_path
        srt_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Transcribing: {audio_wav}")
        segments, info = self.model.transcribe(
            str(audio_wav),
            word_timestamps=True,
            vad_filter=True,
        )
        logger.info(f"Detected language: {info.language} (p={info.language_probability:.2f})")

        blocks = []
        idx = 1
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            blocks.append(
                f"{idx}\n"
                f"{self._fmt_ts(seg.start)} --> {self._fmt_ts(seg.end)}\n"
                f"{text}\n"
            )
            idx += 1

        srt_path.write_text("\n".join(blocks), encoding="utf-8")
        logger.info(f"SRT written ({idx - 1} segments): {srt_path}")
        return str(srt_path.resolve())

    def burn_captions(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
        font_size: int = 20,
        font_color: str = "white",
        outline_color: str = "black",
        outline_width: int = 2,
    ) -> str:
        """Burn SRT subtitles into a video using FFmpeg.

        Args:
            video_path:    Source MP4.
            srt_path:      SRT subtitle file.
            output_path:   Destination MP4.
            font_size:     Subtitle font size in points.
            font_color:    Text colour (name or #RRGGBB).
            outline_color: Outline colour.
            outline_width: Outline thickness in pixels.

        Returns:
            Absolute path of the output video.
        """
        srt_abs = str(Path(srt_path).resolve()).replace("\\", "\\\\").replace(":", "\\:")
        force_style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&H00{self._hex_color(font_color)}&,"
            f"OutlineColour=&H00{self._hex_color(outline_color)}&,"
            f"Outline={outline_width},"
            f"Alignment=2"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{srt_abs}':force_style='{force_style}'",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Caption burn failed:\n{result.stderr[-2000:]}")
        logger.info(f"Captioned video: {output_path}")
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
        ms = int(round(seconds * 1000))
        hours, ms = divmod(ms, 3_600_000)
        minutes, ms = divmod(ms, 60_000)
        secs, ms = divmod(ms, 1_000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    @staticmethod
    def _hex_color(name: str) -> str:
        """Convert colour name / #RRGGBB to FFmpeg ASS BBGGRR hex."""
        named = {
            "white": "FFFFFF", "black": "000000",
            "yellow": "00FFFF", "red": "0000FF",
            "blue": "FF0000", "green": "00FF00",
        }
        rgb = named.get(name.lower(), name.lstrip("#").upper() if name.startswith("#") else "FFFFFF")
        return rgb[4:6] + rgb[2:4] + rgb[0:2]
