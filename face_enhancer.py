"""
face_enhancer.py — Face restoration for lip-synced video frames.

Primary backend: CodeFormer (via ComfyUI reactor node).
Fallback: GFPGAN.
Fallback-fallback: passthrough (no enhancement, just copies the video).
"""

import glob
import os
import shutil
import subprocess
from pathlib import Path

from loguru import logger


class FaceEnhancer:
    """Enhance face quality in a video using CodeFormer or GFPGAN."""

    def __init__(self):
        """Detect available backend: CodeFormer → GFPGAN → passthrough."""
        self.backend = None
        self.enhance_fn = None

        # Try CodeFormer via ComfyUI reactor node
        codeformer_path = Path.home() / "ComfyUI" / "custom_nodes" / "comfyui-reactor-node"
        if codeformer_path.exists():
            try:
                import sys
                sys.path.insert(0, str(codeformer_path))
                logger.info("CodeFormer (reactor-node) found ✓")
                self.backend = "codeformer"
            except Exception as e:
                logger.warning(f"CodeFormer found but failed to load: {e}")

        # Try GFPGAN
        if not self.backend:
            try:
                from gfpgan import GFPGANer  # noqa: F401
                self.backend = "gfpgan"
                logger.info("GFPGAN backend detected ✓")
            except ImportError:
                pass

        if not self.backend:
            logger.warning(
                "No face enhancement backend available "
                "(CodeFormer/GFPGAN not installed). "
                "Enhancement will be skipped — video passed through unchanged."
            )
            self.backend = "passthrough"

        logger.info(f"FaceEnhancer backend: {self.backend}")

    def enhance(
        self,
        video_path: str,
        output_path: str,
        fidelity_weight: float = 0.7,
    ) -> str:
        """Enhance faces in every frame of the video.

        Pipeline:
          1. Extract frames at 25 fps → temp/enhance_frames/
          2. Run backend on each frame (CodeFormer / GFPGAN / passthrough)
          3. Reassemble frames → video with original audio muxed back

        Args:
            video_path: Source MP4 path.
            output_path: Destination MP4 path.
            fidelity_weight: 0.0 = max enhancement, 1.0 = max fidelity to original.

        Returns:
            Absolute path to the enhanced video.
        """
        pipeline_dir = Path(__file__).resolve().parent
        frames_dir = pipeline_dir / "temp" / "enhance_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Clean previous frames
        for f in frames_dir.glob("*.png"):
            f.unlink()

        video_path = str(Path(video_path).resolve())
        output_path = str(Path(output_path).resolve())

        # Get source FPS
        fps = self._get_fps(video_path)

        # 1. Extract frames
        logger.info(f"Extracting frames at {fps} fps...")
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps={fps}",
            str(frames_dir / "frame_%04d.png"),
        ]
        r = subprocess.run(extract_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Frame extraction failed:\n{r.stderr[-1000:]}")

        frame_files = sorted(frames_dir.glob("frame_*.png"))
        logger.info(f"Extracted {len(frame_files)} frames")

        # 2. Enhance frames
        if self.backend == "codeformer":
            self._enhance_codeformer(frames_dir, fidelity_weight)
        elif self.backend == "gfpgan":
            self._enhance_gfpgan(frames_dir, fidelity_weight)
        else:
            logger.info("Passthrough — skipping per-frame enhancement")

        # 3. Reassemble video + re-add audio
        enhanced_no_audio = str(frames_dir / "enhanced_no_audio.mp4")
        reassemble_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-crf", "17",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            enhanced_no_audio,
        ]
        r = subprocess.run(reassemble_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Frame reassembly failed:\n{r.stderr[-1000:]}")

        # Mux original audio back
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", enhanced_no_audio,
            "-i", video_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
        ]
        r = subprocess.run(mux_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            # If no audio stream in source, just copy without mux
            shutil.copy(enhanced_no_audio, output_path)
            logger.warning("No audio stream to mux — video only")

        # Clean up temp frames
        shutil.rmtree(frames_dir, ignore_errors=True)

        logger.info(f"Enhanced video: {output_path}")
        return output_path

    def _enhance_codeformer(self, frames_dir: Path, fidelity_weight: float) -> None:
        """Run CodeFormer on all frames in-place."""
        try:
            import sys
            codeformer_root = (
                Path.home() / "ComfyUI" / "custom_nodes" / "comfyui-reactor-node"
            )
            sys.path.insert(0, str(codeformer_root))
            from scripts.codeformer_infer import inference_app  # type: ignore

            for frame_path in sorted(frames_dir.glob("frame_*.png")):
                inference_app(
                    input_path=str(frame_path),
                    output_path=str(frame_path),  # overwrite in-place
                    fidelity_weight=fidelity_weight,
                    has_aligned=False,
                    only_center_face=False,
                    bg_upsampler=None,
                    device="mps",
                )
        except Exception as e:
            logger.warning(f"CodeFormer failed: {e} — falling through to passthrough")

    def _enhance_gfpgan(self, frames_dir: Path, fidelity_weight: float) -> None:
        """Run GFPGAN on all frames in-place."""
        try:
            import cv2
            import numpy as np
            from gfpgan import GFPGANer

            restorer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            for frame_path in sorted(frames_dir.glob("frame_*.png")):
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                _, _, restored = restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=fidelity_weight,
                )
                cv2.imwrite(str(frame_path), restored)
        except Exception as e:
            logger.warning(f"GFPGAN failed: {e} — frames left unchanged")

    def _get_fps(self, video_path: str) -> float:
        """Read the frame rate of a video file via ffprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            Frames per second as a float (default 25.0 on failure).
        """
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            num, den = r.stdout.strip().split("/")
            return float(num) / float(den)
        except Exception:
            return 25.0
