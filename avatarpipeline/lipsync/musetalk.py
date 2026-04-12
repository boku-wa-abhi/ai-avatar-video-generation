"""
avatarpipeline.lipsync.musetalk — MuseTalk 1.5 lip-sync inference wrapper.

Handles avatar preparation, subprocess invocation, and output discovery
for the MuseTalk lip-sync model on Apple M4 Pro (MPS backend).
"""

import glob
import os
import subprocess
import tempfile
from pathlib import Path

import yaml
from loguru import logger
from PIL import Image

from avatarpipeline import CONFIGS_DIR, ROOT


class MuseTalkInference:
    """Wrapper around MuseTalk 1.5 inference for avatar lip-sync generation."""

    def __init__(self) -> None:
        """Load MuseTalk settings from configs/settings.yaml."""
        cfg_file = CONFIGS_DIR / "settings.yaml"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config not found: {cfg_file}")

        with open(cfg_file) as f:
            self.config = yaml.safe_load(f)

        self.musetalk_dir = Path(os.path.expanduser(self.config["musetalk_dir"]))
        self.venv_python = self.musetalk_dir / "musetalk-env" / "bin" / "python"
        self.default_fps = self.config.get("default_fps", 25)

        if not self.musetalk_dir.exists():
            raise FileNotFoundError(
                f"MuseTalk directory not found: {self.musetalk_dir}. "
                "Run install/setup.sh first."
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(
                f"MuseTalk venv Python not found: {self.venv_python}. "
                "Run install/setup.sh first."
            )

        logger.info(f"MuseTalk dir: {self.musetalk_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_output_video(self, output_dir: str) -> str:
        """Return the most recently created MP4 in the output directory.

        Args:
            output_dir: Directory to search.

        Returns:
            Absolute path to the latest MP4 file.

        Raises:
            FileNotFoundError: If no MP4 files are found.
        """
        mp4_files = glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True)
        # Filter out temp_ files — MuseTalk creates temp_<name>.mp4 intermediates
        mp4_files = [f for f in mp4_files if not os.path.basename(f).startswith("temp_")]
        if not mp4_files:
            raise FileNotFoundError(
                f"No MP4 files found in {output_dir}. "
                "MuseTalk may have failed silently."
            )
        return os.path.abspath(max(mp4_files, key=os.path.getmtime))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_avatar(self, png_path: str, size: int = 256) -> str:
        """Resize and pad the avatar PNG to a square of the given size.

        Args:
            png_path: Path to the source PNG.
            size:     Target square dimension in pixels.

        Returns:
            Path to the prepared avatar PNG.
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

        out = png_path.parent / f"{png_path.stem}_prepared.png"
        canvas.save(out, "PNG")
        logger.info(f"Prepared avatar ({size}x{size}): {out}")
        return str(out)

    def run(
        self,
        avatar_png: str,
        audio_wav: str,
        output_dir: str | None = None,
        batch_size: int = 8,
        bbox_shift: int = 0,
    ) -> str:
        """Run MuseTalk 1.5 inference to generate a lip-synced video.

        Args:
            avatar_png:  Path to the avatar PNG (already prepared).
            audio_wav:   Path to the 16 kHz mono WAV.
            output_dir:  Directory for output video. Defaults to data/temp/musetalk_out.

        Returns:
            Absolute path to the generated MP4.

        Raises:
            RuntimeError: If MuseTalk inference fails.
            FileNotFoundError: If no output video is produced.
        """
        avatar_png = Path(avatar_png).resolve()
        audio_wav = Path(audio_wav).resolve()
        out_dir = Path(output_dir).resolve() if output_dir else ROOT / "data" / "temp" / "musetalk_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not avatar_png.exists():
            raise FileNotFoundError(f"Avatar PNG not found: {avatar_png}")
        if not audio_wav.exists():
            raise FileNotFoundError(f"Audio WAV not found: {audio_wav}")

        logger.info("Running MuseTalk inference...")
        logger.info(f"  Avatar: {avatar_png}")
        logger.info(f"  Audio:  {audio_wav}")
        logger.info(f"  Output: {out_dir}")

        # Write a temporary inference config YAML (MuseTalk reads paths from a
        # YAML task config rather than accepting them as direct CLI arguments).
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(self.musetalk_dir)
        ) as cfg_f:
            cfg_f.write(
                f"task_0:\n"
                f"  video_path: \"{avatar_png}\"\n"
                f"  audio_path: \"{audio_wav}\"\n"
            )
            tmp_cfg_path = cfg_f.name

        try:
            cmd = [
                str(self.venv_python),
                "-m", "scripts.inference",
                "--version", "v15",
                "--inference_config", tmp_cfg_path,
                "--result_dir", str(out_dir),
                "--fps", str(self.default_fps),
                "--batch_size", str(batch_size),
                "--bbox_shift", str(bbox_shift),
                "--unet_model_path", "models/musetalkV15/unet.pth",
                "--unet_config", "models/musetalkV15/musetalk.json",
                "--whisper_dir", "models/whisper",
            ]

            env = os.environ.copy()
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

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
                    f"MuseTalk inference failed (exit code {result.returncode})."
                )
        finally:
            try:
                os.unlink(tmp_cfg_path)
            except OSError:
                pass

        logger.info("MuseTalk inference completed.")
        return self._find_output_video(str(out_dir))
