"""avatarpipeline.voice — Text-to-speech module."""
from avatarpipeline.voice.kokoro import VoiceGenerator
from avatarpipeline.voice.mlx_voice import MlxVoiceStudio

__all__ = ["VoiceGenerator", "MlxVoiceStudio"]
