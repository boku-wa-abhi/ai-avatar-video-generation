"""
musetalk_infer.py — MuseTalk 1.5 lip-sync inference wrapper.

Handles avatar preparation, subprocess invocation, and output discovery
for the MuseTalk lip-sync model on Apple M4 Pro (MPS backend).
"""

import glob
import os
import subprocess
from pathlib import Path

import yaml
from loguru import logger
from PIL import Image


class MuseTalkInference:
    """Wrapper around MuseTalk 1.5 inference for avatar lip-sync generation."""

    def __init__(self, config_path: str = "configs/settings.yaml"):
        """Initialise MuseTalkInference from the pipeline settings file.

        Args:
            config_path: Path to the global pipeline settings YAML,
                         relative to the directory containing this file.
        """
        self.pipeline_dir = Path(__file__).resolve().parent
        config_file = self.pipeline_dir / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.musetalk_dir = Path(os.path.expanduser(self.config["musetalk_dir"]))
        self.venv_python = self.musetalk_dir / "musetalk-env" / "bin" / "python"
        self.default_fps = self.config.get("default_fps", 25)

        if not self.musetalk_dir.exists():
            raise FileNotFoundError(
                f"MuseTalk directory not found: {self.musetalk_dir}. "
                "Run setup.sh first."
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(
                f"MuseTalk venv Python not found: {self.venv_python}. "
                "Run setup.sh first."
            )

        logger.info(f"MuseTalk dir: {self.musetalk_dir}")
        logger.info(f"Venv Python:  {self.venv_python}")

    def prepare_avatar(self, png_path: str, size: int = 256) -> str:
        """Resize and pad the avatar PNG to a square of the given size.

        The image is resized to fit within ``size x size`` while maintaining
        aspect ratio, then centred on a white canvas.

        Args:
            png_path: Path to the source PNG image.
            size: Target square dimension in pixels (default 256).

        Returns:
            Path to the prepared avatar PNG (saved alongside the original).
        """
        png_path = Path(png_path).resolve()
        if not png_path.exists():
            raise FileNotFoundError(f"Avatar image not found: {png_path}")

        img = Image.open(png_path).convert("RGBA")
        img.thumbnail((size, size), Image.LANCZOS)

        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        offset_x = (size - img.width) // 2
        offset_y = (size - img.height) // 2
        canvas.paste(img, (offset_x, offset_y), mask=img)

        prepared_path = png_path.parent / f"{png_path.stem}_prepared.png"
        canvas.save(prepared_path, "PNG")
        logger.info(f"Prepared avatar ({size}x{size}): {prepared_path}")
        return str(prepared_path)

    def run(
        self,
        avatar_png: str,
        audio_wav: str,
        output_dir: str = "temp/musetalk_out",
    ) -> str:
        """Run MuseTalk 1.5 inference to generate a lip-synced video.

        Args:
            avatar_png: Path to the avatar PNG (already prepared).
            audio_wav: Path to the 16 kHz mono WAV audio file.
            output_dir: Directory for output video, relative to pipeline root.

        Returns:
            Absolute path to the generated MP4 video.

        Raises:
            RuntimeError: If MuseTalk inference fails.
            FileNotFoundError: If no output video is produced.
        """
        avatar_png = Path(avatar_png).resolve()
        audio_wav = Path(audio_wav).resolve()
        out_dir = (self.pipeline_dir / output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not avatar_png.exists():
            raise FileNotFoundError(f"Avatar PNG not found: {avatar_png}")
        if not audio_wav.exists():
            raise FileNotFoundError(f"Audio WAV not found: {audio_wav}")

        logger.info("Running MuseTalk inference...")
        logger.info(f"  Avatar: {avatar_png}")
        logger.info(f"  Audio:  {audio_wav}")
        logger.info(f"  Output: {out_dir}")

        cmd = [
            str(self.venv_python),
            "-m", "scripts.inference",
            "--version", "v15",
            "--video_path", str(avatar_png),
            "--audio_path", str(audio_wav),
            "--output_dir", str(out_dir),
            "--fps", str(self.default_fps),
            "--use_float16",
        ]

        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.musetalk_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("MuseTalk inference timed out after 600 seconds.")

        if result.returncode != 0:
            logger.error(f"MuseTalk STDOUT:\n{result.stdout}")
            logger.error(f"MuseTalk STDERR:\n{result.stderr}")
            raise RuntimeError(
                f"MuseTalk inference failed (exit code {result.returncode}). "
                "Check logs above for details."
            )

        logger.info("MuseTalk inference completed successfully.")
        if result.stdout:
            logger.debug(f"STDOUT tail:\n{result.stdout[-2000:]}")

        output_video = self._find_output_video(str(out_dir))
        logger.info(f"Output video: {output_video}")
        return output_video

    def _find_output_video(self, output_dir: str) -> str:
        """Return the most recently created MP4 in the output directory.

        Args:
            output_dir: Directory to search for MP4 files.

        Returns:
            Absolute path to the latest MP4 file.

        Raises:
            FileNotFoundError: If no MP4 files are found.
        """
        mp4_files = glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True)
        if not mp4_files:
            raise FileNotFoundError(
                f"No MP4 files found in {output_dir}. "
                "MuseTalk may have failed silently."
            )
        return os.path.abspath(max(mp4_files, key=os.path.getmtime))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MuseTalk 1.5 lip-sync inference")
    parser.add_argument("--avatar", required=True, help="Path to avatar PNG")
    parser.add_argument("--audio", required=True, help="Path to 16kHz mono WAV")
    parser.add_argument("--output-dir", default="temp/musetalk_out", help="Output dir (relative to pipeline root)")
    parser.add_argument("--config", default="configs/settings.yaml", help="Pipeline config path")
    parser.add_argument("--prepare", action="store_true", help="Resize/pad avatar before inference")
    args = parser.parse_args()

    infer = MuseTalkInference(config_path=args.config)

    avatar_path = args.avatar
    if args.prepare:
        avatar_path = infer.prepare_avatar(avatar_path)

    result_path = infer.run(avatar_path, args.audio, output_dir=args.output_dir)
    print(f"\nLip-synced video: {result_path}")
