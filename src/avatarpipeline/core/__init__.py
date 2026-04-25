"""Shared foundations for configuration, media utilities, and protocols."""

from avatarpipeline.core.config import ConfigError, PipelineConfig, load_config
from avatarpipeline.core.media import (
    audio_duration,
    concat_audio,
    generate_silence,
    normalize_to_16k_mono,
    resample_audio,
    video_info,
)

__all__ = [
    "ConfigError",
    "PipelineConfig",
    "audio_duration",
    "concat_audio",
    "generate_silence",
    "load_config",
    "normalize_to_16k_mono",
    "resample_audio",
    "video_info",
]
