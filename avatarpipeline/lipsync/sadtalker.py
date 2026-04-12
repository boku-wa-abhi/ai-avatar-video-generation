"""
avatarpipeline.lipsync.sadtalker — SadTalker lip-sync inference wrapper.

Handles subprocess invocation of the SadTalker model for talking-head
video generation on Apple M4 Pro (MPS backend).
"""

import glob
import os
import subprocess
from pathlib import Path

import yaml
from loguru import logger

from avatarpipeline import CONFIGS_DIR, OUTPUT_DIR, ROOT

# ── Preset configurations ────────────────────────────────────────────────────
# Each preset maps to a combination of SadTalker CLI flags.
PRESETS = {
    "sadtalker": {
        "size": 256,
        "preprocess": "full",
        "still": True,
        "enhancer": None,
        "expression_scale": 1.0,
    },
    "sadtalker_hd": {
        "size": 512,
        "preprocess": "full",
        "still": True,
        "enhancer": "gfpgan",
        "expression_scale": 1.0,
    },
}


class SadTalkerInference:
    """Wrapper around SadTalker inference for talking-head video generation."""

    def __init__(self, preset: str = "sadtalker") -> None:
        """Initialise from configs/settings.yaml.

        Args:
            preset: One of ``"sadtalker"`` (256 px) or ``"sadtalker_hd"`` (512 px + GFPGAN).
        """
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}")

        cfg_file = CONFIGS_DIR / "settings.yaml"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config not found: {cfg_file}")

        with open(cfg_file) as f:
            self.config = yaml.safe_load(f)

        self.sadtalker_dir = Path(os.path.expanduser(
            self.config.get("sadtalker_dir", "~/SadTalker")
        ))
        self.venv_python = self.sadtalker_dir / "sadtalker-env" / "bin" / "python"

        if not self.sadtalker_dir.exists():
            raise FileNotFoundError(
                f"SadTalker directory not found: {self.sadtalker_dir}. "
                "Clone https://github.com/OpenTalker/SadTalker.git first."
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(
                f"SadTalker venv Python not found: {self.venv_python}. "
                "Create with: uv venv sadtalker-env --python 3.10"
            )

        self.preset_name = preset
        self.preset = PRESETS[preset].copy()

        # Allow per-run overrides from settings.yaml
        lipsync_cfg = self.config.get("lipsync", {})
        if "expression_scale" in lipsync_cfg:
            self.preset["expression_scale"] = float(lipsync_cfg["expression_scale"])

        logger.info(f"SadTalker dir: {self.sadtalker_dir}  preset: {preset}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        avatar_png: str,
        audio_wav: str,
        output_path: str | None = None,
        expression_scale: float | None = None,
    ) -> str:
        """Run SadTalker inference to generate a lip-synced video.

        Args:
            avatar_png:       Path to the source portrait image.
            audio_wav:        Path to the audio WAV file.
            output_path:      Optional explicit output MP4 path.
            expression_scale: Override expression intensity (0.0–3.0).

        Returns:
            Absolute path to the generated MP4.

        Raises:
            RuntimeError: If SadTalker inference fails.
            FileNotFoundError: If no output video is produced.
        """
        avatar_png = Path(avatar_png).resolve()
        audio_wav = Path(audio_wav).resolve()

        if not avatar_png.exists():
            raise FileNotFoundError(f"Avatar image not found: {avatar_png}")
        if not audio_wav.exists():
            raise FileNotFoundError(f"Audio WAV not found: {audio_wav}")

        result_dir = str(OUTPUT_DIR / "sadtalker_results")
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        expr_scale = expression_scale if expression_scale is not None else self.preset["expression_scale"]

        cmd = [
            str(self.venv_python),
            "inference.py",
            "--source_image", str(avatar_png),
            "--driven_audio", str(audio_wav),
            "--result_dir", result_dir,
            "--size", str(self.preset["size"]),
            "--preprocess", self.preset["preprocess"],
            "--expression_scale", str(expr_scale),
        ]

        if self.preset["still"]:
            cmd.append("--still")

        if self.preset["enhancer"]:
            cmd.extend(["--enhancer", self.preset["enhancer"]])

        logger.info(f"Running SadTalker ({self.preset_name})...")
        logger.info(f"  Avatar: {avatar_png}")
        logger.info(f"  Audio:  {audio_wav}")
        logger.info(f"  Size:   {self.preset['size']}  enhancer: {self.preset['enhancer']}")

        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.sadtalker_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("SadTalker inference timed out after 900 seconds.")

        if result.returncode != 0:
            logger.error(f"SadTalker STDOUT:\n{result.stdout}")
            logger.error(f"SadTalker STDERR:\n{result.stderr}")
            raise RuntimeError(
                f"SadTalker inference failed (exit code {result.returncode})."
            )

        logger.info("SadTalker inference completed.")
        video_path = self._find_output_video(result_dir)

        # Optionally copy to a specific output path
        if output_path:
            import shutil
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, output_path)
            return str(Path(output_path).resolve())

        return video_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_output_video(result_dir: str) -> str:
        """Return the most recently created MP4 in the result directory."""
        mp4_files = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
        if not mp4_files:
            raise FileNotFoundError(
                f"No MP4 files found in {result_dir}. "
                "SadTalker may have failed silently."
            )
        return os.path.abspath(max(mp4_files, key=os.path.getmtime))
