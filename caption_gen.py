"""
caption_gen.py — Subtitle generation and caption burning using faster-whisper.

Transcribes audio locally (large-v3 model) and produces SRT files that can
be burned into video via FFmpeg subtitles filter.
"""

import os
import subprocess
from pathlib import Path

from loguru import logger


class CaptionGenerator:
    """Generate SRT captions from audio and burn them into video."""

    def __init__(self, model_size: str = "large-v3", device: str = "auto"):
        """Initialise the Whisper transcription model.

        Args:
            model_size: Whisper model variant (tiny/base/small/medium/large-v3).
            device: "auto" picks MPS on Apple Silicon, falls back to CPU.
        """
        from faster_whisper import WhisperModel

        # Apple Silicon: faster-whisper doesn't support MPS directly —
        # use CPU with int8 quantisation which is still fast on M4 Pro.
        if device == "auto":
            resolved_device = "cpu"
            compute_type = "int8"
        else:
            resolved_device = device
            compute_type = "int8"

        logger.info(f"Loading Whisper {model_size} on {resolved_device} ({compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=resolved_device,
            compute_type=compute_type,
        )
        logger.info("Whisper model ready")

    def transcribe(
        self,
        audio_wav: str,
        output_srt: str = "captions/output.srt",
    ) -> str:
        """Transcribe audio and write a properly formatted SRT file.

        Args:
            audio_wav: Path to the 16 kHz mono WAV file.
            output_srt: Destination SRT path (relative to this file's dir or absolute).

        Returns:
            Absolute path to the generated SRT file.
        """
        pipeline_dir = Path(__file__).resolve().parent
        audio_wav = Path(audio_wav).resolve()

        srt_path = Path(output_srt)
        if not srt_path.is_absolute():
            srt_path = pipeline_dir / srt_path
        srt_path.parent.mkdir(parents=True, exist_ok=True)

        if not audio_wav.exists():
            raise FileNotFoundError(f"Audio not found: {audio_wav}")

        logger.info(f"Transcribing: {audio_wav}")
        segments, info = self.model.transcribe(
            str(audio_wav),
            word_timestamps=True,
            vad_filter=True,
        )
        logger.info(f"Detected language: {info.language} (p={info.language_probability:.2f})")

        srt_blocks = []
        idx = 1
        for segment in segments:
            start_ts = self.get_srt_timestamp(segment.start)
            end_ts = self.get_srt_timestamp(segment.end)
            text = segment.text.strip()
            if not text:
                continue
            srt_blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
            idx += 1

        srt_content = "\n".join(srt_blocks)
        srt_path.write_text(srt_content, encoding="utf-8")
        logger.info(f"SRT written ({idx - 1} segments): {srt_path}")
        return str(srt_path.resolve())

    def get_srt_timestamp(self, seconds: float) -> str:
        """Convert a float number of seconds to SRT timestamp format.

        Args:
            seconds: Time in seconds (may be fractional).

        Returns:
            Formatted string like "HH:MM:SS,mmm".
        """
        ms = int(round(seconds * 1000))
        hours, ms = divmod(ms, 3_600_000)
        minutes, ms = divmod(ms, 60_000)
        secs, ms = divmod(ms, 1_000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

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
            video_path: Source MP4 path.
            srt_path: Path to the SRT subtitle file.
            output_path: Destination MP4 path.
            font_size: Subtitle font size in points.
            font_color: Subtitle text colour (FFmpeg colour name or hex).
            outline_color: Outline colour.
            outline_width: Outline thickness in pixels.

        Returns:
            Absolute path to the output video with captions burned in.
        """
        # FFmpeg subtitles filter needs an escaped absolute path on macOS
        srt_abs = str(Path(srt_path).resolve())
        # Escape colons and backslashes in path (required by the subtitles filter)
        srt_escaped = srt_abs.replace("\\", "\\\\").replace(":", "\\:")

        force_style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&H00{self._hex_color(font_color)}&,"
            f"OutlineColour=&H00{self._hex_color(outline_color)}&,"
            f"Outline={outline_width},"
            f"Alignment=2"   # bottom-centre
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{srt_escaped}':force_style='{force_style}'",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        logger.info(f"Burning captions into: {output_path}")
        logger.debug(f"FFmpeg: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Caption burn failed:\n{result.stderr[-2000:]}")

        logger.info(f"Captioned video: {output_path}")
        return os.path.abspath(output_path)

    @staticmethod
    def _hex_color(name: str) -> str:
        """Convert a colour name or #hex to BGR hex string for FFmpeg ASS style.

        Args:
            name: Colour name (white/black/yellow/red) or #RRGGBB hex.

        Returns:
            BBGGRR hex string (FFmpeg ASS format).
        """
        named = {
            "white": "FFFFFF",
            "black": "000000",
            "yellow": "00FFFF",
            "red": "0000FF",
            "blue": "FF0000",
            "green": "00FF00",
        }
        if name.lower() in named:
            rgb = named[name.lower()]
        elif name.startswith("#"):
            rgb = name.lstrip("#").upper()
        else:
            rgb = "FFFFFF"

        # Convert RRGGBB → BBGGRR
        return rgb[4:6] + rgb[2:4] + rgb[0:2]
