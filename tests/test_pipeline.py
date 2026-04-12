#!/usr/bin/env python3
"""
tests/test_pipeline.py — Unit tests for each pipeline module.

Run all:        python -m pytest tests/ -v
Run one module: python -m pytest tests/test_pipeline.py::test_voice_gen -v
Smoke test:     bash scripts/smoke_test.sh --no-enhance --no-captions
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

AVATAR_PNG = ROOT / "data" / "avatars" / "avatar.png"
TEST_OUT   = ROOT / "tests" / "_test_output"


@pytest.fixture(autouse=True)
def setup_dirs():
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# 1. Package structure
# ─────────────────────────────────────────────────────────────────────────────

def test_package_imports():
    """All top-level package imports succeed."""
    import avatarpipeline
    assert avatarpipeline.ROOT == ROOT
    assert avatarpipeline.AVATARS_DIR.parent.name == "data"
    assert avatarpipeline.AUDIO_DIR.parent.name == "data"
    assert avatarpipeline.OUTPUT_DIR.parent.name == "data"


def test_config_loads():
    """configs/settings.yaml loads with all required keys."""
    import yaml
    cfg_path = ROOT / "configs" / "settings.yaml"
    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "default_fps" in cfg
    assert "lipsync" in cfg
    assert "tts" in cfg
    assert cfg["tts"]["engine"] == "kokoro"
    assert "data/avatars" in cfg["avatar_path"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Voice generation (Kokoro)
# ─────────────────────────────────────────────────────────────────────────────

def test_voice_gen_init():
    """VoiceGenerator initializes without error."""
    from avatarpipeline.voice.kokoro import VoiceGenerator
    vg = VoiceGenerator()
    assert vg.default_voice == "af_heart"
    assert len(vg.VOICES) >= 10


def test_voice_gen_generate():
    """Generates a WAV file with correct properties."""
    import soundfile as sf
    from avatarpipeline.voice.kokoro import VoiceGenerator

    vg = VoiceGenerator()
    out = str(TEST_OUT / "test_tts.wav")
    result = vg.generate("Hello, this is a test.", voice="af_heart", out_path=out)
    assert Path(result).exists()
    data, sr = sf.read(result)
    assert sr == 16000
    assert len(data) > 0
    assert len(data) / sr > 0.5


def test_voice_gen_convert_16k():
    """Resamples audio to 16 kHz mono."""
    import soundfile as sf
    from avatarpipeline.voice.kokoro import VoiceGenerator

    vg = VoiceGenerator()
    src = str(TEST_OUT / "test_tts.wav")
    if not Path(src).exists():
        vg.generate("Quick test.", voice="af_heart", out_path=src)
    dst = str(TEST_OUT / "test_16k.wav")
    result = vg.convert_to_16k(src, dst)
    assert Path(result).exists()
    _, sr = sf.read(result)
    assert sr == 16000


# ─────────────────────────────────────────────────────────────────────────────
# 3. Caption generation
# ─────────────────────────────────────────────────────────────────────────────

def test_caption_gen_init():
    """CaptionGenerator initializes with base model."""
    from avatarpipeline.postprocess.captions import CaptionGenerator
    cg = CaptionGenerator(model_size="base")
    assert cg.model is not None


def test_caption_gen_transcribe():
    """Transcribes audio to a valid SRT file."""
    from avatarpipeline.voice.kokoro import VoiceGenerator
    from avatarpipeline.postprocess.captions import CaptionGenerator

    vg = VoiceGenerator()
    wav = str(TEST_OUT / "caption_test.wav")
    vg.generate("Testing the caption system.", voice="af_heart", out_path=wav)

    cg = CaptionGenerator(model_size="base")
    srt = str(TEST_OUT / "test_captions.srt")
    result = cg.transcribe(wav, srt)
    assert Path(result).exists()
    content = Path(result).read_text()
    assert "1\n" in content
    assert "-->" in content


# ─────────────────────────────────────────────────────────────────────────────
# 4. Video assembler
# ─────────────────────────────────────────────────────────────────────────────

def test_video_assembler_orientations():
    """VideoAssembler has correct orientation dimension mappings."""
    from avatarpipeline.postprocess.assembler import VideoAssembler
    va = VideoAssembler()
    assert va.ORIENTATIONS["9:16"]["w"] == 1080
    assert va.ORIENTATIONS["9:16"]["h"] == 1920
    assert va.ORIENTATIONS["16:9"]["w"] == 1920
    assert va.ORIENTATIONS["1:1"]["w"] == 1080


def test_video_assembler_get_info():
    """get_video_info returns expected keys for a test video."""
    from avatarpipeline.postprocess.assembler import VideoAssembler

    va = VideoAssembler()
    out = str(TEST_OUT / "tiny_test.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "color=c=black:s=320x240:d=1", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", out],
        capture_output=True, check=True,
    )
    info = va.get_video_info(out)
    assert info["width"] == 320
    assert info["height"] == 240
    assert "fps" in info


def test_video_assembler_add_background():
    """add_background composites to target orientation dimensions."""
    from avatarpipeline.postprocess.assembler import VideoAssembler

    va = VideoAssembler()
    src = str(TEST_OUT / "tiny_test.mp4")
    if not Path(src).exists():
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "color=c=blue:s=320x240:d=1", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", src],
            capture_output=True, check=True,
        )
    out = str(TEST_OUT / "bg_test.mp4")
    result = va.add_background(src, orientation="9:16", background="black", output_path=out)
    assert Path(result).exists()
    info = va.get_video_info(result)
    assert info["width"] == 1080
    assert info["height"] == 1920


# ─────────────────────────────────────────────────────────────────────────────
# 5. Face enhancer
# ─────────────────────────────────────────────────────────────────────────────

def test_face_enhancer_init():
    """FaceEnhancer detects a valid backend."""
    from avatarpipeline.postprocess.enhancer import FaceEnhancer
    fe = FaceEnhancer()
    assert fe.backend in ("codeformer", "gfpgan", "passthrough")


def test_face_enhancer_passthrough():
    """Passthrough backend produces a valid output video."""
    from avatarpipeline.postprocess.enhancer import FaceEnhancer

    fe = FaceEnhancer()
    src = str(TEST_OUT / "tiny_test.mp4")
    if not Path(src).exists():
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "color=c=red:s=320x240:d=1", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", src],
            capture_output=True, check=True,
        )
    out = str(TEST_OUT / "enhance_test.mp4")
    assert Path(fe.enhance(src, out)).exists()


# ─────────────────────────────────────────────────────────────────────────────
# 6. SadTalker inference (structure only — full inference takes minutes)
# ─────────────────────────────────────────────────────────────────────────────

def test_sadtalker_init():
    """SadTalkerInference loads config and finds SadTalker directory."""
    from avatarpipeline.lipsync.sadtalker import SadTalkerInference
    st = SadTalkerInference(preset="sadtalker")
    assert st.sadtalker_dir.exists()
    assert st.venv_python.exists()
    assert st.preset["size"] == 256


def test_sadtalker_hd_init():
    """SadTalkerInference HD preset uses 512 px and GFPGAN."""
    from avatarpipeline.lipsync.sadtalker import SadTalkerInference
    st = SadTalkerInference(preset="sadtalker_hd")
    assert st.preset["size"] == 512
    assert st.preset["enhancer"] == "gfpgan"


# ─────────────────────────────────────────────────────────────────────────────
# 7. MuseTalk inference (structure only)
# ─────────────────────────────────────────────────────────────────────────────

def test_musetalk_init():
    """MuseTalkInference loads config and locates the MuseTalk directory."""
    try:
        from avatarpipeline.lipsync.musetalk import MuseTalkInference
        ms = MuseTalkInference()
        assert ms.musetalk_dir.exists()
    except FileNotFoundError:
        pytest.skip("MuseTalk not installed")


# ─────────────────────────────────────────────────────────────────────────────
# 8. UI dashboard
# ─────────────────────────────────────────────────────────────────────────────

def test_dashboard_import():
    """ui.dashboard imports without error and exposes expected attributes."""
    import importlib
    mod = importlib.import_module("ui.dashboard")
    assert hasattr(mod, "demo")
    assert hasattr(mod, "generate_video")
    assert hasattr(mod, "VOICE_CHOICES")
    assert len(mod.VOICE_CHOICES) == 10


# ─────────────────────────────────────────────────────────────────────────────
# 9. Integration: TTS → VideoAssembler end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_tts_to_video_roundtrip():
    """TTS → 16k audio → background composite → finalize works end-to-end."""
    from avatarpipeline.voice.kokoro import VoiceGenerator
    from avatarpipeline.postprocess.assembler import VideoAssembler

    vg = VoiceGenerator()
    wav = str(TEST_OUT / "roundtrip.wav")
    vg.generate("Integration test.", voice="af_heart", out_path=wav)

    dummy = str(TEST_OUT / "dummy_lip.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=512x512:d=2",
         "-i", wav, "-c:v", "libx264", "-c:a", "aac", "-shortest",
         "-pix_fmt", "yuv420p", dummy],
        capture_output=True, check=True,
    )

    va = VideoAssembler()
    composed = str(TEST_OUT / "roundtrip_bg.mp4")
    va.add_background(dummy, orientation="9:16", output_path=composed)
    assert Path(composed).exists()

    final = str(TEST_OUT / "roundtrip_final.mp4")
    va.finalize(composed, final)
    assert Path(final).exists()
    info = va.get_video_info(final)
    assert info["width"] == 1080


# ─────────────────────────────────────────────────────────────────────────────
# 10. MuseTalk end-to-end: TTS → lip-sync → valid output video
# ─────────────────────────────────────────────────────────────────────────────

MUSETALK_DIR = Path("/Users/abhijeetanand/MuseTalk")
MUSETALK_ENV_PY = MUSETALK_DIR / "musetalk-env" / "bin" / "python"
MUSETALK_HF_CLI = MUSETALK_DIR / "musetalk-env" / "bin" / "huggingface-cli"
MUSETALK_GDOWN = MUSETALK_DIR / "musetalk-env" / "bin" / "gdown"
MODELS_DIR = MUSETALK_DIR / "models"


def _hf_download(repo_id: str, local_dir: Path, include: list[str]) -> None:
    """Download files from a HuggingFace repo into local_dir if not already present."""
    missing = [f for f in include if not (local_dir / f).exists()]
    if not missing:
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(MUSETALK_HF_CLI), "download", repo_id,
            "--local-dir", str(local_dir),
            "--include", *include,
        ],
        check=True,
        timeout=600,
    )


def _ensure_musetalk_weights() -> None:
    """Download all MuseTalk model weights if not already present."""
    # MuseTalk V1 + V1.5 UNet
    _hf_download(
        "TMElyralab/MuseTalk",
        MODELS_DIR,
        ["musetalk/musetalk.json", "musetalk/pytorch_model.bin",
         "musetalkV15/musetalk.json", "musetalkV15/unet.pth"],
    )
    # Whisper-tiny
    _hf_download(
        "openai/whisper-tiny",
        MODELS_DIR / "whisper",
        ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
    )
    # SD-VAE
    _hf_download(
        "stabilityai/sd-vae-ft-mse",
        MODELS_DIR / "sd-vae",
        ["config.json", "diffusion_pytorch_model.bin"],
    )
    # DWPose
    _hf_download(
        "yzd-v/DWPose",
        MODELS_DIR / "dwpose",
        ["dw-ll_ucoco_384.pth"],
    )
    # Face-parse-bisent: resnet18 backbone
    resnet_dst = MODELS_DIR / "face-parse-bisent" / "resnet18-5c106cde.pth"
    if not resnet_dst.exists():
        resnet_dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["curl", "-fL",
             "https://download.pytorch.org/models/resnet18-5c106cde.pth",
             "-o", str(resnet_dst)],
            check=True,
            timeout=300,
        )
    # Face-parse-bisent: segmentation model
    bisent_dst = MODELS_DIR / "face-parse-bisent" / "79999_iter.pth"
    if not bisent_dst.exists():
        bisent_dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(MUSETALK_GDOWN),
             "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812",
             "-O", str(bisent_dst)],
            check=True,
            timeout=300,
        )


def _check_video_valid(path: str) -> dict:
    """Return ffprobe info dict; raises AssertionError if video is invalid."""
    r = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=0",
         path],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"ffprobe failed on {path}: {r.stderr}"
    assert Path(path).stat().st_size > 10_000, "Output video is suspiciously small"
    return r.stdout


@pytest.mark.slow
def test_musetalk_e2e():
    """End-to-end: TTS → MuseTalk lip-sync → valid MP4."""
    if not MUSETALK_DIR.exists():
        pytest.skip("MuseTalk not installed at /Users/abhijeetanand/MuseTalk")
    if not AVATAR_PNG.exists():
        pytest.skip(f"No avatar image at {AVATAR_PNG}")

    # 1. Ensure all model weights are present (downloads if needed)
    _ensure_musetalk_weights()

    # 2. Generate short TTS audio
    from avatarpipeline.voice.kokoro import VoiceGenerator
    vg = VoiceGenerator()
    wav_path = str(TEST_OUT / "musetalk_e2e_tts.wav")
    wav_16k = str(TEST_OUT / "musetalk_e2e_tts_16k.wav")
    vg.generate("Hello, this is an end-to-end MuseTalk test.", voice="af_heart", out_path=wav_path)
    vg.convert_to_16k(wav_path, wav_16k)
    assert Path(wav_16k).exists(), "16k WAV not created"

    # 3. Run MuseTalk inference
    from avatarpipeline.lipsync.musetalk import MuseTalkInference
    ms = MuseTalkInference()
    out_dir = str(TEST_OUT / "musetalk_e2e_out")
    output_video = ms.run(str(AVATAR_PNG), wav_16k, output_dir=out_dir)

    # 4. Validate output
    assert Path(output_video).exists(), f"Output video not found: {output_video}"
    probe = _check_video_valid(output_video)
    assert "width" in probe or "height" in probe or "duration" in probe, (
        f"ffprobe output unexpected: {probe}"
    )
