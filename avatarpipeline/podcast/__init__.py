"""Podcast generation module — two-speaker video composition."""

from .composer import (
    LAYOUT_CHOICES,
    OVERLAY_CHOICES,
    build_timeline_from_tracks,
    compose_podcast_sequential,
    compose_podcast_video,
    detect_speech_segments,
    generate_per_speaker_audio,
    get_unique_speakers,
    mix_audio_tracks,
    parse_podcast_script,
    resample_16k,
)

__all__ = [
    "LAYOUT_CHOICES",
    "OVERLAY_CHOICES",
    "build_timeline_from_tracks",
    "compose_podcast_sequential",
    "compose_podcast_video",
    "detect_speech_segments",
    "generate_per_speaker_audio",
    "get_unique_speakers",
    "mix_audio_tracks",
    "parse_podcast_script",
    "resample_16k",
]
