"""
avatarpipeline — Local AI Avatar Video Generation Library.

Public API surface:

    from avatarpipeline.pipeline import run_pipeline
    from avatarpipeline.voice.kokoro import VoiceGenerator
    from avatarpipeline.lipsync.musetalk import MuseTalkInference
    from avatarpipeline.lipsync.sadtalker import SadTalkerInference
    from avatarpipeline.postprocess.enhancer import FaceEnhancer
    from avatarpipeline.postprocess.captions import CaptionGenerator
    from avatarpipeline.postprocess.assembler import VideoAssembler
"""

from pathlib import Path

__version__ = "1.0.0"
__author__ = "Avatar Pipeline"

# Project root — two levels up from this file
# avatarpipeline/__init__.py → avatarpipeline/ → project_root/
ROOT: Path = Path(__file__).resolve().parent.parent

# Canonical sub-directory layout (all runtime data lives under data/)
CONFIGS_DIR: Path = ROOT / "configs"
ASSETS_DIR: Path = ROOT / "assets"
DATA_DIR: Path = ROOT / "data"
AVATARS_DIR: Path = DATA_DIR / "avatars"
AUDIO_DIR: Path = DATA_DIR / "audio"
OUTPUT_DIR: Path = DATA_DIR / "output"
CAPTIONS_DIR: Path = DATA_DIR / "captions"
TEMP_DIR: Path = DATA_DIR / "temp"

# Ensure runtime directories exist on import
for _d in (AVATARS_DIR, AUDIO_DIR, OUTPUT_DIR, CAPTIONS_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)
