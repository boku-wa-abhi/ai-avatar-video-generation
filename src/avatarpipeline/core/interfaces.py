"""Protocol definitions for swappable pipeline engines."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TtsEngine(Protocol):
    def generate(self, text: str, voice: str, out_path: str) -> str: ...

    def convert_to_16k(self, wav_path: str, out_path: str) -> str: ...

    def list_voices(self) -> list[str]: ...


@runtime_checkable
class LipsyncEngine(Protocol):
    def run(self, avatar_png: str, audio_wav: str, **kwargs) -> str: ...


@runtime_checkable
class FaceEnhancerEngine(Protocol):
    def enhance(self, video_path: str, output_path: str) -> str: ...
