"""
latentsync_infer.py — LatentSync 1.6 lip-sync inference wrapper.

Creates a looped video from a single PNG, runs LatentSync via subprocess
against a ComfyUI-LatentSyncWrapper installation, and returns the output MP4.
Designed for Apple M4 Pro (MPS backend).
"""

import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import yaml
from loguru import logger


class LatentSyncInference:
    """Wrapper around LatentSync 1.6 for high-quality lip-sync generation."""

    def __init__(self, config_path: str = "configs/settings.yaml"):
        """Load pipeline and LatentSync settings.

        Args:
            config_path: Path to pipeline settings YAML (relative to this file's dir).
        """
        self.pipeline_dir = Path(__file__).resolve().parent
        config_file = self.pipeline_dir / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.comfyui_dir = Path(os.path.expanduser(self.config["comfyui_dir"]))
        self.wrapper_dir = self.comfyui_dir / "custom_nodes" / "ComfyUI-LatentSyncWrapper"
        self.checkpoints_dir = self.wrapper_dir / "checkpoints"

        ls_cfg = self.config.get("latentsync", {})
        self.inference_steps = ls_cfg.get("inference_steps", 25)
        self.lips_expression = ls_cfg.get("lips_expression", 1.5)
        self.input_fps = ls_cfg.get("input_fps", 25)
        self.face_resolution = ls_cfg.get("face_resolution", 512)

        if not self.wrapper_dir.exists():
            raise FileNotFoundError(
                f"LatentSyncWrapper not found at {self.wrapper_dir}. "
                "Run install_latentsync.sh first."
            )

        logger.info(f"LatentSync wrapper: {self.wrapper_dir}")
        logger.info(f"Settings: steps={self.inference_steps}, lips={self.lips_expression}, "
                     f"fps={self.input_fps}, face_res={self.face_resolution}")

    def get_audio_duration(self, wav_path: str) -> float:
        """Get the duration of a WAV file in seconds.

        Args:
            wav_path: Path to the WAV audio file.

        Returns:
            Duration in seconds.
        """
        data, samplerate = sf.read(wav_path)
        duration = len(data) / samplerate
        logger.info(f"Audio duration: {duration:.2f}s (sr={samplerate})")
        return duration

    def prepare_input_video(
        self, png_path: str, duration_secs: float, fps: int | None = None
    ) -> str:
        """Create a looped MP4 video from a single PNG image.

        Uses FFmpeg to loop the still image for the given duration. The output
        is used as the input face video for LatentSync.

        Args:
            png_path: Path to the avatar PNG.
            duration_secs: How long the video should be (matches audio duration).
            fps: Frames per second (defaults to config value).

        Returns:
            Absolute path to the looped video.
        """
        fps = fps or self.input_fps
        out_path = self.pipeline_dir / "temp" / "avatar_loop.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(png_path),
            "-c:v", "libx264",
            "-t", str(duration_secs),
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={self.face_resolution}:{self.face_resolution}:force_original_aspect_ratio=decrease,"
                   f"pad={self.face_resolution}:{self.face_resolution}:(ow-iw)/2:(oh-ih)/2:white",
            str(out_path),
        ]

        logger.info(f"Creating looped video: {duration_secs:.1f}s @ {fps}fps, {self.face_resolution}x{self.face_resolution}")
        logger.debug(f"FFmpeg: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg looped video failed:\n{result.stderr}")

        logger.info(f"Looped video: {out_path}")
        return str(out_path.resolve())

    def run(
        self,
        avatar_png: str,
        audio_wav: str,
        inference_steps: int | None = None,
        lips_expression: float | None = None,
        output_path: str = "temp/latentsync_out.mp4",
    ) -> str:
        """Run LatentSync 1.6 inference to generate a lip-synced video.

        Pipeline:
          1. Measure audio duration
          2. Create a looped still-image video matching that duration
          3. Run LatentSync via subprocess on the ComfyUI venv
          4. Return path to the output MP4

        Args:
            avatar_png: Path to the avatar PNG.
            audio_wav: Path to the 16 kHz mono WAV.
            inference_steps: Denoising steps (default from config).
            lips_expression: Lip movement intensity (default from config).
            output_path: Where to save the result (relative to pipeline dir or absolute).

        Returns:
            Absolute path to the generated MP4.
        """
        avatar_png = Path(avatar_png).resolve()
        audio_wav = Path(audio_wav).resolve()

        out = Path(output_path)
        if not out.is_absolute():
            out = self.pipeline_dir / out
        out.parent.mkdir(parents=True, exist_ok=True)

        steps = inference_steps or self.inference_steps
        lips = lips_expression or self.lips_expression

        if not avatar_png.exists():
            raise FileNotFoundError(f"Avatar PNG not found: {avatar_png}")
        if not audio_wav.exists():
            raise FileNotFoundError(f"Audio WAV not found: {audio_wav}")

        # 1. Audio duration
        duration = self.get_audio_duration(str(audio_wav))

        # 2. Create looped input video
        loop_video = self.prepare_input_video(str(avatar_png), duration)

        # 3. Run LatentSync
        logger.info(f"Running LatentSync (steps={steps}, lips={lips})...")

        # Determine the Python binary from a venv (ComfyUI or wrapper-local)
        venv_python = self._find_venv_python()

        # Build a small runner script that exercises the LatentSync pipeline
        runner_script = f"""
import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

sys.path.insert(0, "{self.wrapper_dir}")

import torch
from latentsync.utils.util import load_video_frames
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline

device = "mps"

# Load pipeline
pipe = LipsyncPipeline.from_pretrained(
    pretrained_model_name_or_path="{self.checkpoints_dir}",
    device=device,
)
pipe = pipe.to(device)

# Run inference
output = pipe(
    video_path="{loop_video}",
    audio_path="{audio_wav}",
    video_out_path="{out.resolve()}",
    num_inference_steps={steps},
    guidance_scale={lips},
)

print("LATENTSYNC_SUCCESS")
"""
        runner_path = self.pipeline_dir / "temp" / "_latentsync_runner.py"
        runner_path.write_text(runner_script)

        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        try:
            result = subprocess.run(
                [str(venv_python), str(runner_path)],
                cwd=str(self.wrapper_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=900,  # 15-minute timeout for high-quality renders
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("LatentSync inference timed out after 900 seconds.")

        if result.returncode != 0 or "LATENTSYNC_SUCCESS" not in result.stdout:
            logger.error(f"STDOUT:\n{result.stdout[-3000:]}")
            logger.error(f"STDERR:\n{result.stderr[-3000:]}")
            raise RuntimeError(
                f"LatentSync inference failed (exit code {result.returncode}). "
                "Check logs above."
            )

        # Clean up runner script
        runner_path.unlink(missing_ok=True)

        logger.info(f"LatentSync output: {out.resolve()}")
        return str(out.resolve())

    def _find_venv_python(self) -> Path:
        """Locate the Python binary for running LatentSync.

        Checks (in order): pipeline .venv (preferred — single venv design),
        ComfyUI .venv, ComfyUI venv, wrapper-local .venv.

        Returns:
            Path to the Python binary.

        Raises:
            FileNotFoundError: If no suitable venv is found.
        """
        candidates = [
            self.pipeline_dir / ".venv" / "bin" / "python",   # primary
            self.comfyui_dir / ".venv" / "bin" / "python",
            self.comfyui_dir / "venv" / "bin" / "python",
            self.wrapper_dir / ".venv" / "bin" / "python",
        ]
        for p in candidates:
            if p.exists():
                logger.debug(f"Using Python: {p}")
                return p

        raise FileNotFoundError(
            "No Python venv found for LatentSync. Run install_latentsync.sh first. "
            f"Checked: {[str(c) for c in candidates]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LatentSync 1.6 lip-sync inference")
    parser.add_argument("--avatar", required=True, help="Path to avatar PNG")
    parser.add_argument("--audio", required=True, help="Path to 16kHz mono WAV")
    parser.add_argument("--output", default="temp/latentsync_out.mp4", help="Output MP4 path")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps (default: from config)")
    parser.add_argument("--lips", type=float, default=None, help="Lips expression strength (default: from config)")
    parser.add_argument("--config", default="configs/settings.yaml", help="Pipeline config")
    args = parser.parse_args()

    infer = LatentSyncInference(config_path=args.config)
    result_path = infer.run(
        avatar_png=args.avatar,
        audio_wav=args.audio,
        inference_steps=args.steps,
        lips_expression=args.lips,
        output_path=args.output,
    )
    print(f"\nLip-synced video: {result_path}")
