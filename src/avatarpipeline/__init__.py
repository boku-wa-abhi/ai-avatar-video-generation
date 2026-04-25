"""
avatarpipeline — Local AI Avatar Video Generation Library.

Public API surface:

    from avatarpipeline.pipelines.avatar import run_pipeline
    from avatarpipeline.engines.tts.kokoro import VoiceGenerator
    from avatarpipeline.engines.lipsync.musetalk import MuseTalkInference
    from avatarpipeline.engines.lipsync.sadtalker import SadTalkerInference
    from avatarpipeline.postprocess.enhancer import FaceEnhancer
    from avatarpipeline.postprocess.captions import CaptionGenerator
    from avatarpipeline.postprocess.assembler import VideoAssembler
"""

from pathlib import Path

__version__ = "1.0.0"
__author__ = "Avatar Pipeline"

# Project root for src/ layout:
# src/avatarpipeline/__init__.py -> avatarpipeline/ -> src/ -> project_root/
ROOT: Path = Path(__file__).resolve().parents[2]

# Canonical sub-directory layout (all runtime data lives under data/)
CONFIGS_DIR: Path = ROOT / "configs"
ASSETS_DIR: Path = ROOT / "assets"
DATA_DIR: Path = ROOT / "data"
AVATARS_DIR: Path = DATA_DIR / "avatars"
AUDIO_DIR: Path = DATA_DIR / "audio"
VOICES_DIR: Path = DATA_DIR / "voices"
PRESENTATIONS_DIR: Path = DATA_DIR / "presentations"
OUTPUT_DIR: Path = DATA_DIR / "output"
CAPTIONS_DIR: Path = DATA_DIR / "captions"
IMAGES_DIR: Path = DATA_DIR / "images"
TEMP_DIR: Path = DATA_DIR / "temp"

# Ensure runtime directories exist on import
for _d in (AVATARS_DIR, AUDIO_DIR, VOICES_DIR, PRESENTATIONS_DIR, OUTPUT_DIR, CAPTIONS_DIR, IMAGES_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

__all__ = [
    "ASSETS_DIR",
    "AUDIO_DIR",
    "AVATARS_DIR",
    "CAPTIONS_DIR",
    "CONFIGS_DIR",
    "DATA_DIR",
    "IMAGES_DIR",
    "OUTPUT_DIR",
    "PRESENTATIONS_DIR",
    "ROOT",
    "TEMP_DIR",
    "VOICES_DIR",
    "__version__",
]
