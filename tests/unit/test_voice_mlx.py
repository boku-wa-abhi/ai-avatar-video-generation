#!/usr/bin/env python3
"""
tests.test_mlx_voice — Lightweight tests for the MLX voice helper layer.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _write_test_wav(path: Path, seconds: float = 1.0, sample_rate: int = 24_000) -> None:
    samples = int(seconds * sample_rate)
    t = np.linspace(0, seconds, samples, endpoint=False)
    wave = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(path, wave, sample_rate)


def test_mlx_voice_profile_save_and_load(tmp_path):
    from avatarpipeline.engines.tts.mlx import MlxVoiceStudio

    source = tmp_path / "reference_source.wav"
    _write_test_wav(source)

    studio = MlxVoiceStudio(voice_store=tmp_path / "voices")
    profile = studio.save_voice_profile(
        name="Test Voice",
        audio_path=str(source),
        transcript="Hello from the saved voice.",
    )

    assert profile["slug"] == "test-voice"
    assert Path(profile["reference_audio_path"]).exists()
    assert studio.list_voice_choices() == ["Test Voice [test-voice]"]

    loaded = studio.get_voice_profile("Test Voice [test-voice]")
    assert loaded is not None
    assert loaded["reference_text"] == "Hello from the saved voice."
    assert loaded["duration_seconds"] > 0


def test_mlx_voice_duplicate_names_get_unique_slugs(tmp_path):
    from avatarpipeline.engines.tts.mlx import MlxVoiceStudio

    source = tmp_path / "reference_source.wav"
    _write_test_wav(source)

    studio = MlxVoiceStudio(voice_store=tmp_path / "voices")
    first = studio.save_voice_profile("Narrator", str(source), transcript="One")
    second = studio.save_voice_profile("Narrator", str(source), transcript="Two")

    assert first["slug"] == "narrator"
    assert second["slug"] == "narrator-2"


def test_mlx_voice_pitch_filter_builder():
    from avatarpipeline.engines.tts.mlx import MlxVoiceStudio

    filters = MlxVoiceStudio._build_pitch_filters(24_000, 2.0)
    assert "asetrate=48000" in filters
    assert "aresample=24000" in filters
    assert "atempo=0.500000" in filters
