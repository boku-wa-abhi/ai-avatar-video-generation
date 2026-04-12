"""
video_assembler.py — Composites avatar video with background, music and captions.

Steps:
  1. add_background() — place avatar MP4 onto a solid/blur/image background
  2. add_music()       — mix in optional background music
  3. finalize()        — burn captions, encode to delivery spec
"""

import json
import subprocess
from pathlib import Path

from loguru import logger


class VideoAssembler:
    """Assemble the final deliverable MP4."""

    ORIENTATIONS: dict[str, dict[str, int]] = {
        "9:16":  {"w": 1080, "h": 1920},
        "16:9":  {"w": 1920, "h": 1080},
        "1:1":   {"w": 1080, "h": 1080},
    }

    def add_background(
        self,
        avatar_mp4: str,
        orientation: str = "9:16",
        background: str = "black",
        output_path: str | None = None,
    ) -> str:
        """Composite the avatar onto a full-frame background.

        The avatar is centered horizontally and vertically on the frame.

        Args:
            avatar_mp4:    Path to the lip-synced (enhanced) avatar MP4.
            orientation:   One of "9:16", "16:9", "1:1".
            background:    "black", "white", "blur", or an absolute path to an image.
            output_path:   Destination path. Auto-derived if not given.

        Returns:
            Path to the composited MP4.
        """
        dims = self.ORIENTATIONS.get(orientation, self.ORIENTATIONS["9:16"])
        canvas_w, canvas_h = dims["w"], dims["h"]

        avatar_mp4 = str(Path(avatar_mp4).resolve())
        if output_path is None:
            output_path = avatar_mp4.replace(".mp4", "_bg.mp4")

        avatar_info = self.get_video_info(avatar_mp4)
        av_w = avatar_info.get("width", canvas_w)
        av_h = avatar_info.get("height", canvas_h)

        # Letterbox / pillarbox the avatar to fit canvas
        scale_w = canvas_w
        scale_h = int(av_h * canvas_w / av_w)
        if scale_h > canvas_h:
            scale_h = canvas_h
            scale_w = int(av_w * canvas_h / av_h)

        pad_x = (canvas_w - scale_w) // 2
        pad_y = (canvas_h - scale_h) // 2

        if background in ("black", "white"):
            bg_color = background
            vf = (
                f"scale={scale_w}:{scale_h},"
                f"pad={canvas_w}:{canvas_h}:{pad_x}:{pad_y}:color={bg_color}"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", avatar_mp4,
                "-vf", vf,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "copy",
                output_path,
            ]
        elif background == "blur":
            # Blurred-and-zoomed version of the avatar behind the avatar itself
            vf_bg = (
                f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                f"crop={canvas_w}:{canvas_h},"
                f"boxblur=30:5"
            )
            vf_fg = f"scale={scale_w}:{scale_h}"
            cmd = [
                "ffmpeg", "-y",
                "-i", avatar_mp4,
                "-i", avatar_mp4,
                "-filter_complex",
                f"[0:v]{vf_bg}[bg];"
                f"[1:v]{vf_fg}[fg];"
                f"[bg][fg]overlay={pad_x}:{pad_y}",
                "-map", "0:a?",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "copy",
                output_path,
            ]
        else:
            # Background is an image file
            bg_path = str(Path(background).resolve())
            vf_bg = (
                f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                f"crop={canvas_w}:{canvas_h}"
            )
            vf_fg = f"scale={scale_w}:{scale_h}"
            avatar_dur = avatar_info.get("duration", 30)
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", bg_path,
                "-i", avatar_mp4,
                "-filter_complex",
                f"[0:v]{vf_bg}[bg];"
                f"[1:v]{vf_fg}[fg];"
                f"[bg][fg]overlay={pad_x}:{pad_y}",
                "-map", "1:a?",
                "-t", str(avatar_dur),
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "copy",
                output_path,
            ]

        logger.info(f"Adding background ({background}) → {output_path}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"add_background failed:\n{r.stderr[-1500:]}")

        return output_path

    def add_music(
        self,
        video_path: str,
        music_path: str,
        music_volume: float = 0.15,
        output_path: str | None = None,
    ) -> str:
        """Mix background music into the video at a reduced volume.

        Args:
            video_path:    Path to the video with dialogue audio.
            music_path:    Path to the background music file (MP3/WAV/M4A).
            music_volume:  Relative volume of music track (0.0–1.0).
            output_path:   Destination path. Auto-derived if not given.

        Returns:
            Path to the mixed video.
        """
        video_path = str(Path(video_path).resolve())
        music_path = str(Path(music_path).resolve())
        if output_path is None:
            output_path = video_path.replace(".mp4", "_music.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", music_path,
            "-filter_complex",
            f"[1:a]volume={music_volume}[bg];"
            "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[outa]",
            "-map", "0:v:0",
            "-map", "[outa]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

        logger.info(f"Mixing music at volume {music_volume} → {output_path}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"add_music failed:\n{r.stderr[-1500:]}")

        return output_path

    def finalize(
        self,
        video_path: str,
        output_path: str,
        srt_path: str | None = None,
        include_captions: bool = True,
    ) -> str:
        """Final encode: subtitles baked in, H.264 high-quality delivery spec.

        Args:
            video_path:       Input video (with background / audio already mixed).
            output_path:      Final deliverable MP4 path.
            srt_path:         SRT subtitle file, or None to skip captions.
            include_captions: If False, skip subtitle burn even if srt_path given.

        Returns:
            Absolute path of the final output MP4.
        """
        video_path = str(Path(video_path).resolve())
        output_path = str(Path(output_path).resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        has_subs = (
            include_captions
            and srt_path is not None
            and Path(srt_path).exists()
        )

        if has_subs:
            # Escape colons and backslashes for the subtitles filter
            safe_srt = str(Path(srt_path).resolve()).replace("\\", "/").replace(":", "\\:")
            vf = (
                f"subtitles={safe_srt}:force_style='"
                "FontName=Arial Bold,FontSize=16,"
                "PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,"
                "Outline=2,Alignment=2,"
                "MarginV=40'"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", vf,
                "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_path,
            ]

        logger.info(f"Finalizing video → {output_path}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"finalize failed:\n{r.stderr[-1500:]}")

        size_mb = Path(output_path).stat().st_size / 1_048_576
        logger.info(f"Final video: {output_path} ({size_mb:.1f} MB)")
        return output_path

    def get_video_info(self, video_path: str) -> dict:
        """Return width, height, duration, fps of a video via ffprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            dict with keys: width, height, duration, fps.
        """
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-of", "json",
            video_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(r.stdout)
            stream = data["streams"][0]
            num, den = stream.get("r_frame_rate", "25/1").split("/")
            fps = float(num) / float(den)
            return {
                "width": int(stream.get("width", 1080)),
                "height": int(stream.get("height", 1920)),
                "duration": float(stream.get("duration", 0)),
                "fps": fps,
            }
        except Exception:
            return {"width": 1080, "height": 1920, "duration": 0.0, "fps": 25.0}
