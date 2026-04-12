"""
avatarpipeline.lipsync.latentsync — LatentSync 1.6 lip-sync inference wrapper.

Creates a looped video from a single PNG, runs LatentSync via subprocess
against a ComfyUI-LatentSyncWrapper installation, and returns the output MP4.
Designed for Apple M4 Pro (MPS backend).
"""

import os
import subprocess
from pathlib import Path

import soundfile as sf
import yaml
from loguru import logger

from avatarpipeline import CONFIGS_DIR, TEMP_DIR, ROOT


class LatentSyncInference:
    """Wrapper around LatentSync 1.6 for high-quality lip-sync generation."""

    def __init__(self) -> None:
        """Load pipeline and LatentSync settings from configs/settings.yaml."""
        cfg_file = CONFIGS_DIR / "settings.yaml"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config not found: {cfg_file}")

        with open(cfg_file) as f:
            self.config = yaml.safe_load(f)

        self.comfyui_dir = Path(os.path.expanduser(self.config["comfyui_dir"]))
        self.wrapper_dir = self.comfyui_dir / "custom_nodes" / "ComfyUI-LatentSyncWrapper"
        self.checkpoints_dir = self.wrapper_dir / "checkpoints"

        ls = self.config.get("latentsync", {})
        self.inference_steps = ls.get("inference_steps", 25)
        self.lips_expression = ls.get("lips_expression", 1.5)
        self.input_fps = ls.get("input_fps", 25)
        self.face_resolution = ls.get("face_resolution", 512)

        if not self.wrapper_dir.exists():
            raise FileNotFoundError(
                f"LatentSyncWrapper not found at {self.wrapper_dir}. "
                "Run install/install_latentsync.sh first."
            )

        logger.info(f"LatentSync wrapper: {self.wrapper_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_audio_duration(self, wav_path: str) -> float:
        """Return duration of a WAV file in seconds."""
        data, sr = sf.read(wav_path)
        duration = len(data) / sr
        logger.info(f"Audio duration: {duration:.2f}s (sr={sr})")
        return duration

    def prepare_input_video(
        self, png_path: str, duration_secs: float, fps: int | None = None
    ) -> str:
        """Create a looped MP4 from a single PNG matching the audio duration.

        Args:
            png_path:      Path to the avatar PNG.
            duration_secs: Desired video duration in seconds.
            fps:           Frames per second (defaults to config value).

        Returns:
            Absolute path to the looped video.
        """
        fps = fps or self.input_fps
        out_path = TEMP_DIR / "avatar_loop.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        res = self.face_resolution
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(png_path),
            "-c:v", "libx264",
            "-t", str(duration_secs),
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-vf",
            f"scale={res}:{res}:force_original_aspect_ratio=decrease,"
            f"pad={res}:{res}:(ow-iw)/2:(oh-ih)/2:white",
            str(out_path),
        ]

        logger.info(
            f"Creating looped video: {duration_secs:.1f}s @ {fps}fps, {res}x{res}"
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg looped video failed:\n{result.stderr}")

        logger.info(f"Looped video: {out_path}")
        return str(out_path.resolve())

    def _find_venv_python(self) -> Path:
        """Locate the Python binary for running LatentSync.

        Checks (in order): pipeline .venv, ComfyUI .venv, ComfyUI venv,
        wrapper-local .venv.

        Returns:
            Path to a Python binary.

        Raises:
            FileNotFoundError: If no suitable venv is found.
        """
        candidates = [
            ROOT / ".venv" / "bin" / "python",
            self.comfyui_dir / ".venv" / "bin" / "python",
            self.comfyui_dir / "venv" / "bin" / "python",
            self.wrapper_dir / ".venv" / "bin" / "python",
        ]
        for p in candidates:
            if p.exists():
                logger.debug(f"Using Python: {p}")
                return p

        raise FileNotFoundError(
            "No Python venv found for LatentSync. "
            "Run install/install_latentsync.sh first. "
            f"Checked: {[str(c) for c in candidates]}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        avatar_png: str,
        audio_wav: str,
        inference_steps: int | None = None,
        lips_expression: float | None = None,
        output_path: str | None = None,
    ) -> str:
        """Run LatentSync 1.6 to produce a lip-synced video.

        Pipeline:
          1. Measure audio duration
          2. Create a looped still-image video matching that duration
          3. Spawn a subprocess that loads LatentSync from the ComfyUI venv
          4. Return path to the output MP4

        Args:
            avatar_png:       Path to the avatar PNG.
            audio_wav:        Path to the 16 kHz mono WAV.
            inference_steps:  Denoising steps (default from config).
            lips_expression:  Lip movement intensity (default from config).
            output_path:      Absolute or ROOT-relative output path.

        Returns:
            Absolute path of the generated MP4.
        """
        avatar_png = Path(avatar_png).resolve()
        audio_wav = Path(audio_wav).resolve()

        out = Path(output_path) if output_path else TEMP_DIR / "latentsync_out.mp4"
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)

        steps = inference_steps or self.inference_steps
        lips = lips_expression or self.lips_expression

        if not avatar_png.exists():
            raise FileNotFoundError(f"Avatar PNG not found: {avatar_png}")
        if not audio_wav.exists():
            raise FileNotFoundError(f"Audio WAV not found: {audio_wav}")

        duration = self.get_audio_duration(str(audio_wav))
        loop_video = self.prepare_input_video(str(avatar_png), duration)

        logger.info(f"Running LatentSync (steps={steps}, lips={lips})...")

        venv_python = self._find_venv_python()

        unet_config = self.wrapper_dir / "configs" / "unet" / "stage2_512.yaml"
        if not unet_config.exists():
            unet_config = self.wrapper_dir / "configs" / "unet" / "stage2.yaml"
        if not unet_config.exists():
            raise FileNotFoundError(
                f"No unet config found in {self.wrapper_dir / 'configs' / 'unet'}. "
                "Run install/install_latentsync.sh first."
            )

        ckpt_path = self.checkpoints_dir / "latentsync_unet.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"UNet checkpoint not found: {ckpt_path}")

        mask_path = self.wrapper_dir / "latentsync" / "utils" / "mask.png"

        runner_script = f"""\
import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

wrapper_dir = {repr(str(self.wrapper_dir))}
sys.path.insert(0, wrapper_dir)

from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature

device = "mps"
dtype = torch.float32

config = OmegaConf.load({repr(str(unet_config))})
config.data.mask_image_path = {repr(str(mask_path))}

scheduler = DDIMScheduler(
    beta_end=0.012, beta_schedule="scaled_linear", beta_start=0.00085,
    clip_sample=False, num_train_timesteps=1000,
    prediction_type="epsilon", set_alpha_to_one=False, steps_offset=1,
)
print("Scheduler ready")

whisper_name = "tiny" if config.model.cross_attention_dim == 384 else "small"
audio_encoder = Audio2Feature(
    model_path=whisper_name,
    device="cpu",
    num_frames=config.data.num_frames,
    audio_feat_length=config.data.audio_feat_length,
)
print("Audio encoder ready")

vae_dir = os.path.join(wrapper_dir, "checkpoints", "vae")
vae_safetensor = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
if os.path.exists(vae_safetensor):
    try:
        vae = AutoencoderKL.from_single_file(vae_safetensor, torch_dtype=dtype)
    except Exception:
        vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=dtype, local_files_only=True)
elif os.path.isdir(vae_dir):
    vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=dtype, local_files_only=True)
else:
    raise FileNotFoundError(f"VAE not found at {{vae_dir}}")
vae.config.scaling_factor = 0.18215
vae.config.shift_factor = 0
print("VAE ready")

unet, _ = UNet3DConditionModel.from_pretrained(
    OmegaConf.to_container(config.model),
    {repr(str(ckpt_path))},
    device="cpu",
)
unet = unet.to(dtype=dtype)
print("UNet ready")

pipeline = LipsyncPipeline(
    vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler,
).to(device)

try:
    from DeepCache import DeepCacheSDHelper
    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()
    print("DeepCache enabled")
except ImportError:
    print("DeepCache not available")

torch.manual_seed(1247)
print("Running inference...")
pipeline(
    video_path={repr(str(loop_video))},
    audio_path={repr(str(audio_wav))},
    video_out_path={repr(str(out.resolve()))},
    video_mask_path={repr(str(out.resolve()).replace('.mp4', '_mask.mp4'))},
    num_frames=config.data.num_frames,
    num_inference_steps={steps},
    guidance_scale={lips},
    weight_dtype=dtype,
    width=config.data.resolution,
    height=config.data.resolution,
    mask_image_path=config.data.mask_image_path,
)
print("LATENTSYNC_SUCCESS")
"""
        runner_path = TEMP_DIR / "_latentsync_runner.py"
        runner_path.parent.mkdir(parents=True, exist_ok=True)
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
                timeout=900,
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

        runner_path.unlink(missing_ok=True)
        logger.info(f"LatentSync output: {out.resolve()}")
        return str(out.resolve())
