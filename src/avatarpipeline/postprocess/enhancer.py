"""
avatarpipeline.postprocess.enhancer — Face restoration for lip-synced video frames.

Primary backend: CodeFormer (via ComfyUI reactor node).
Fallback:        GFPGAN.
Fallback-fallback: passthrough (copies the video unchanged).
"""

import shutil
import subprocess
from pathlib import Path

from loguru import logger

from avatarpipeline import ROOT, TEMP_DIR


class FaceEnhancer:
    """Enhance face quality in a video using CodeFormer, GFPGAN, or passthrough."""

    def __init__(self) -> None:
        """Detect available backend: CodeFormer → GFPGAN → passthrough."""
        self.backend: str = "passthrough"

        # 1. Try CodeFormer via ComfyUI reactor node
        codeformer_path = Path.home() / "ComfyUI" / "custom_nodes" / "comfyui-reactor-node"
        if codeformer_path.exists():
            try:
                import sys
                sys.path.insert(0, str(codeformer_path))
                self.backend = "codeformer"
                logger.info("CodeFormer (reactor-node) found ✓")
            except Exception as e:
                logger.warning(f"CodeFormer found but failed to load: {e}")

        # 2. Try GFPGAN
        if self.backend == "passthrough":
            try:
                from gfpgan import GFPGANer  # noqa: F401
                self.backend = "gfpgan"
                logger.info("GFPGAN backend detected ✓")
            except ImportError:
                pass

        # 3. Try GFPGAN via SadTalker subprocess (no project-venv install needed)
        if self.backend == "passthrough":
            _sadtalker_dir = Path.home() / "SadTalker"
            _gfpgan_weights = _sadtalker_dir / "gfpgan" / "weights" / "GFPGANv1.4.pth"
            _sadtalker_python = _sadtalker_dir / "sadtalker-env" / "bin" / "python"
            if _gfpgan_weights.exists() and _sadtalker_python.exists():
                self.backend = "gfpgan_subprocess"
                self.gfpgan_python = str(_sadtalker_python)
                self.gfpgan_weights = str(_gfpgan_weights)
                self.sadtalker_dir = str(_sadtalker_dir)
                logger.info("GFPGAN (via SadTalker env subprocess) detected ✓")

        if self.backend == "passthrough":
            logger.warning(
                "No face enhancement backend available "
                "(CodeFormer/GFPGAN not installed). "
                "Video will be passed through unchanged."
            )

        logger.info(f"FaceEnhancer backend: {self.backend}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(
        self,
        video_path: str,
        output_path: str,
        fidelity_weight: float = 0.7,
    ) -> str:
        """Enhance faces in every frame of the video.

        Steps:
          1. Extract frames at source FPS → data/temp/enhance_frames/
          2. Run backend on each frame
          3. Reassemble frames and mux original audio back

        Args:
            video_path:       Source MP4.
            output_path:      Destination MP4.
            fidelity_weight:  0.0 = max enhancement, 1.0 = max fidelity.

        Returns:
            Absolute path to the enhanced video.
        """
        frames_dir = TEMP_DIR / "enhance_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for f in frames_dir.glob("*.png"):
            f.unlink()

        video_path = str(Path(video_path).resolve())
        output_path = str(Path(output_path).resolve())
        fps = self._get_fps(video_path)

        # 1. Extract frames
        logger.info(f"Extracting frames at {fps} fps...")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}",
             str(frames_dir / "frame_%04d.png")],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Frame extraction failed:\n{r.stderr[-1000:]}")

        frames = sorted(frames_dir.glob("frame_*.png"))
        logger.info(f"Extracted {len(frames)} frames")

        # 2. Enhance frames
        if self.backend == "codeformer":
            self._enhance_codeformer(frames_dir, fidelity_weight)
        elif self.backend == "gfpgan":
            self._enhance_gfpgan(frames_dir, fidelity_weight)
        elif self.backend == "gfpgan_subprocess":
            self._enhance_gfpgan_subprocess(frames_dir)
        else:
            logger.info("Passthrough — skipping per-frame enhancement")

        # 3. Reassemble
        no_audio = str(frames_dir / "enhanced_no_audio.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps),
             "-i", str(frames_dir / "frame_%04d.png"),
             "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p",
             no_audio],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Frame reassembly failed:\n{r.stderr[-1000:]}")

        # Mux original audio back
        r = subprocess.run(
            ["ffmpeg", "-y",
             "-i", no_audio, "-i", video_path,
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-map", "0:v:0", "-map", "1:a:0", "-shortest",
             output_path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            shutil.copy(no_audio, output_path)
            logger.warning("No audio stream to mux — video only")

        shutil.rmtree(frames_dir, ignore_errors=True)
        logger.info(f"Enhanced video: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enhance_codeformer(self, frames_dir: Path, fidelity_weight: float) -> None:
        try:
            import sys
            sys.path.insert(0, str(
                Path.home() / "ComfyUI" / "custom_nodes" / "comfyui-reactor-node"
            ))
            from scripts.codeformer_infer import inference_app  # type: ignore
            for frame in sorted(frames_dir.glob("frame_*.png")):
                inference_app(
                    input_path=str(frame), output_path=str(frame),
                    fidelity_weight=fidelity_weight, has_aligned=False,
                    only_center_face=False, bg_upsampler=None, device="mps",
                )
        except Exception as e:
            logger.warning(f"CodeFormer failed: {e} — using passthrough")

    def _enhance_gfpgan(self, frames_dir: Path, fidelity_weight: float) -> None:
        try:
            import cv2
            from gfpgan import GFPGANer
            restorer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
            )
            for frame in sorted(frames_dir.glob("frame_*.png")):
                img = cv2.imread(str(frame))
                if img is None:
                    continue
                _, _, restored = restorer.enhance(
                    img, has_aligned=False, only_center_face=False,
                    paste_back=True, weight=fidelity_weight,
                )
                cv2.imwrite(str(frame), restored)
        except Exception as e:
            logger.warning(f"GFPGAN failed: {e} — frames left unchanged")

    def _enhance_gfpgan_subprocess(self, frames_dir: Path) -> None:
        """Run GFPGAN on frames using the SadTalker venv's Python (no local install needed)."""
        runner = ROOT / "tools" / "gfpgan_runner.py"
        if not runner.exists():
            logger.warning(f"GFPGAN runner not found at {runner} — skipping enhancement")
            return
        logger.info("Running GFPGAN enhancement via SadTalker env...")
        r = subprocess.run(
            [
                self.gfpgan_python, str(runner),
                "--frames_dir", str(frames_dir),
                "--weights", self.gfpgan_weights,
                "--sadtalker_dir", self.sadtalker_dir,
            ],
            capture_output=False,
        )
        if r.returncode != 0:
            logger.warning("GFPGAN subprocess returned non-zero — frames may be unenhanced")

    def _get_fps(self, video_path: str) -> float:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True,
        )
        try:
            num, den = r.stdout.strip().split("/")
            return float(num) / float(den)
        except Exception:
            return 25.0
