"""Engine registry for TTS and lip-sync backends."""

from __future__ import annotations

from collections.abc import Callable

from avatarpipeline.core.interfaces import LipsyncEngine, TtsEngine
from avatarpipeline.engines.lipsync.musetalk import MuseTalkInference
from avatarpipeline.engines.lipsync.sadtalker import SadTalkerInference
from avatarpipeline.engines.tts.kokoro import VoiceGenerator
from avatarpipeline.engines.tts.mlx import MlxVoiceStudio

TTS_REGISTRY: dict[str, type[TtsEngine]] = {
    "kokoro": VoiceGenerator,
    "mlx": MlxVoiceStudio,
}

LIPSYNC_REGISTRY: dict[str, Callable[[], LipsyncEngine]] = {
    "musetalk": MuseTalkInference,
    "sadtalker": lambda: SadTalkerInference(preset="sadtalker"),
    "sadtalker_hd": lambda: SadTalkerInference(preset="sadtalker_hd"),
}


def get_tts_engine(name: str) -> TtsEngine:
    key = (name or "").strip().lower()
    cls = TTS_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown TTS engine {name!r}. Available: {sorted(TTS_REGISTRY)}")
    return cls()


def get_lipsync_engine(name: str) -> LipsyncEngine:
    key = (name or "").strip().lower()
    factory = LIPSYNC_REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"Unknown lip-sync engine {name!r}. Available: {sorted(LIPSYNC_REGISTRY)}")
    return factory()


__all__ = [
    "LIPSYNC_REGISTRY",
    "TTS_REGISTRY",
    "get_lipsync_engine",
    "get_tts_engine",
]
