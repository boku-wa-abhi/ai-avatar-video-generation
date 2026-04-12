"""
avatarpipeline.voice.kokoro — Local TTS using Kokoro (hexgrad/Kokoro-82M).

Generates speech from text using the Kokoro model which runs fully locally
on Apple Silicon via MPS. Output is automatically converted to 16 kHz mono
WAV for downstream lip-sync models.

System requirement: brew install espeak-ng
Model: hexgrad/Kokoro-82M (~330 MB, MIT license, downloaded automatically)
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from loguru import logger

from avatarpipeline import AUDIO_DIR, CONFIGS_DIR


class VoiceGenerator:
    """Local text-to-speech using Kokoro (runs 100% offline on Apple Silicon)."""

    # Available voices — https://huggingface.co/hexgrad/Kokoro-82M
    VOICES = {
        # American English — Female
        "af_heart":    "af_heart",    # warm, clear (recommended default)
        "af_bella":    "af_bella",    # smooth, confident
        "af_sarah":    "af_sarah",    # natural, conversational
        "af_nicole":   "af_nicole",   # soft, friendly
        # American English — Male
        "am_adam":     "am_adam",
        "am_michael":  "am_michael",
        # British English — Female
        "bf_emma":     "bf_emma",
        "bf_isabella": "bf_isabella",
        # British English — Male
        "bm_george":   "bm_george",
        "bm_lewis":    "bm_lewis",
    }

    SAMPLE_RATE = 24_000

    def __init__(self) -> None:
        """Load TTS settings from configs/settings.yaml."""
        self.default_voice = "af_heart"
        self.speed = 1.0
        self.lang_code = "a"

        cfg_file = CONFIGS_DIR / "settings.yaml"
        if cfg_file.exists():
            with open(cfg_file) as f:
                cfg = yaml.safe_load(f) or {}
            tts = cfg.get("tts", {})
            self.default_voice = tts.get("default_voice", self.default_voice)
            self.speed = float(tts.get("speed", self.speed))
            self.lang_code = tts.get("lang_code", self.lang_code)

        self._pipeline = None  # lazy-loaded on first use
        logger.info(f"VoiceGenerator ready (engine=kokoro, voice={self.default_voice})")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        """Lazy-load the Kokoro pipeline (downloads ~330 MB model on first call)."""
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
            except ImportError:
                raise ImportError(
                    "Kokoro not installed. Run:\n"
                    "  brew install espeak-ng\n"
                    "  uv pip install kokoro"
                )
            logger.info("Loading Kokoro pipeline...")
            self._pipeline = KPipeline(lang_code=self.lang_code)
            logger.info("Kokoro pipeline ready")
        return self._pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        voice: str | None = None,
        out_path: str | None = None,
    ) -> str:
        """Generate speech and save as 16 kHz mono WAV.

        Args:
            text:     The script to speak.
            voice:    Voice name from VOICES dict (default from config).
            out_path: Output WAV path (absolute or relative to project root).
                      Defaults to data/audio/output.wav.

        Returns:
            Absolute path to the 16 kHz mono WAV file.
        """
        voice = voice or self.default_voice
        voice_id = self.VOICES.get(voice, voice)

        dest = Path(out_path) if out_path else AUDIO_DIR / "output.wav"
        if not dest.is_absolute():
            from avatarpipeline import ROOT
            dest = ROOT / dest
        dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating TTS ({len(text)} chars, voice={voice_id}, speed={self.speed})...")

        pipe = self._get_pipeline()
        chunks = []
        for _, _, audio in pipe(text, voice=voice_id, speed=self.speed):
            if audio is not None and audio.ndim > 0 and len(audio) > 0:
                chunks.append(audio)

        if not chunks:
            raise RuntimeError("Kokoro produced no audio — check the text and voice settings.")

        audio = np.concatenate(chunks)

        # Save at native 24 kHz, then downsample to 16 kHz
        raw_path = dest.with_suffix(".24k.wav")
        sf.write(str(raw_path), audio, self.SAMPLE_RATE)

        final = self.convert_to_16k(str(raw_path), str(dest))
        raw_path.unlink(missing_ok=True)
        return final

    def convert_to_16k(self, input_path: str, output_wav: str) -> str:
        """Convert any audio to 16 kHz mono PCM WAV.

        Args:
            input_path:  Source audio file (WAV, MP3, etc.).
            output_wav:  Destination WAV path.

        Returns:
            Absolute path to the converted WAV.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 16 kHz conversion failed:\n{result.stderr}")
        logger.info(f"16 kHz WAV saved: {output_wav}")
        return os.path.abspath(output_wav)

    def list_voices(self) -> list[str]:
        """Return all available voice name strings."""
        return list(self.VOICES.keys())

    def estimate_cost(self, text: str) -> float:
        """Character count. Kokoro is always free and local — cost is $0."""
        return 0.0
