"""
voice_gen.py — Local TTS using Kokoro (hexgrad/Kokoro-82M).

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


class VoiceGenerator:
    """Local text-to-speech using Kokoro (runs 100% offline on Apple Silicon)."""

    # Available voices — see https://huggingface.co/hexgrad/Kokoro-82M
    VOICES = {
        # American English — Female
        "af_heart":   "af_heart",    # warm, clear (recommended default)
        "af_bella":   "af_bella",    # smooth, confident
        "af_sarah":   "af_sarah",    # natural, conversational
        "af_nicole":  "af_nicole",   # soft, friendly
        # American English — Male
        "am_adam":    "am_adam",
        "am_michael": "am_michael",
        # British English — Female
        "bf_emma":    "bf_emma",
        "bf_isabella":"bf_isabella",
        # British English — Male
        "bm_george":  "bm_george",
        "bm_lewis":   "bm_lewis",
    }

    # Kokoro sample rate
    SAMPLE_RATE = 24_000

    def __init__(self, config_path: str = "configs/settings.yaml"):
        """Load TTS settings from the pipeline config.

        Args:
            config_path: Path to pipeline settings YAML (relative to this file's dir).
        """
        self.pipeline_dir = Path(__file__).resolve().parent
        cfg_file = self.pipeline_dir / config_path

        self.default_voice = "af_heart"
        self.speed = 1.0
        self.lang_code = "a"

        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
            tts_cfg = cfg.get("tts", {})
            self.default_voice = tts_cfg.get("default_voice", self.default_voice)
            self.speed = float(tts_cfg.get("speed", self.speed))
            self.lang_code = tts_cfg.get("lang_code", self.lang_code)

        self._pipeline = None  # lazy-loaded on first use
        logger.info(f"VoiceGenerator ready (engine=kokoro, voice={self.default_voice})")

    def _get_pipeline(self):
        """Lazy-load the Kokoro pipeline (downloads model on first call)."""
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
            except ImportError:
                raise ImportError(
                    "Kokoro not installed. Run:\n"
                    "  brew install espeak-ng\n"
                    "  uv pip install kokoro"
                )
            logger.info("Loading Kokoro pipeline (first run downloads ~330 MB model)...")
            self._pipeline = KPipeline(lang_code=self.lang_code)
            logger.info("Kokoro pipeline ready")
        return self._pipeline

    def generate(
        self,
        text: str,
        voice: str | None = None,
        out_path: str = "audio/output.wav",
    ) -> str:
        """Generate speech and save as 16 kHz mono WAV.

        Args:
            text: The script to speak.
            voice: Voice name from VOICES dict (default from config).
            out_path: Output WAV path (relative to pipeline dir or absolute).

        Returns:
            Absolute path to the 16 kHz mono WAV file.
        """
        voice = voice or self.default_voice
        voice_id = self.VOICES.get(voice, voice)

        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = self.pipeline_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating TTS ({len(text)} chars, voice={voice_id}, speed={self.speed})...")

        pipeline = self._get_pipeline()

        # Kokoro yields (graphemes, phonemes, audio_array) per text chunk
        chunks = []
        for _, _, audio in pipeline(text, voice=voice_id, speed=self.speed):
            if audio is not None and audio.ndim > 0 and len(audio) > 0:
                chunks.append(audio)

        if not chunks:
            raise RuntimeError("Kokoro produced no audio output. Check the text and voice settings.")

        audio = np.concatenate(chunks)

        # Save at native 24 kHz first, then downsample to 16 kHz
        raw_path = out_path.with_suffix(".24k.wav")
        sf.write(str(raw_path), audio, self.SAMPLE_RATE)
        logger.debug(f"Raw 24 kHz audio: {raw_path} ({raw_path.stat().st_size / 1024:.1f} KB)")

        # Convert to 16 kHz mono WAV for lip-sync models
        final_path = self.convert_to_16k(str(raw_path), str(out_path))

        raw_path.unlink(missing_ok=True)
        return final_path

    def convert_to_16k(self, input_path: str, output_wav: str) -> str:
        """Convert any audio file to 16 kHz mono PCM WAV.

        Args:
            input_path: Source audio file (WAV, MP3, etc.).
            output_wav: Destination WAV path.

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
        logger.debug(f"FFmpeg: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed:\n{result.stderr}")

        logger.info(f"16 kHz WAV saved: {output_wav}")
        return os.path.abspath(output_wav)

    def estimate_cost(self, text: str) -> float:
        """Return character count. Kokoro is free — no API cost.

        Args:
            text: The text to be synthesised.

        Returns:
            Character count (always free with Kokoro).
        """
        char_count = len(text)
        logger.info(f"Kokoro is free and local — {char_count} chars, $0 cost")
        return 0.0

    def list_voices(self) -> list[str]:
        """Return all available voice names."""
        return list(self.VOICES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate speech using Kokoro (local, free)")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--voice", default=None,
                        help="Voice name (af_heart/af_bella/am_adam/bf_emma/etc.)")
    parser.add_argument("--out", default="audio/output.wav", help="Output WAV path")
    parser.add_argument("--config", default="configs/settings.yaml", help="Pipeline config")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    args = parser.parse_args()

    gen = VoiceGenerator(config_path=args.config)

    if args.list_voices:
        print("Available voices:")
        for v in gen.list_voices():
            print(f"  {v}")
    else:
        wav_path = gen.generate(text=args.text, voice=args.voice, out_path=args.out)
        print(f"\nGenerated: {wav_path}")

    FREE_TIER_CHARS = 10_000

    def __init__(self, api_key: str | None = None, config_path: str = "configs/settings.yaml"):
        """Resolve the ElevenLabs API key from param → env → config file.

        Args:
            api_key: Explicit API key (highest priority).
            config_path: Path to pipeline settings YAML (relative to this file's dir).
        """
        self.pipeline_dir = Path(__file__).resolve().parent

        # 1. Explicit param
        key = api_key

        # 2. Environment variable
        if not key:
            key = os.environ.get("ELEVENLABS_KEY")

        # 3. Config file
        if not key:
            cfg_file = self.pipeline_dir / config_path
            if cfg_file.exists():
                with open(cfg_file, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                key = cfg.get("elevenlabs_key") or None

        if not key:
            raise ValueError(
                "ElevenLabs API key not found. Set ELEVENLABS_KEY env var, "
                "pass api_key=, or fill elevenlabs_key in configs/settings.yaml."
            )

        self.api_key = key

        # Lazy import so the module can be loaded even without the SDK installed
        from elevenlabs.client import ElevenLabs
        self.client = ElevenLabs(api_key=self.api_key)

        # Load model preference from config
        cfg_file = self.pipeline_dir / config_path
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
            el_cfg = cfg.get("elevenlabs", {})
            self.model = el_cfg.get("model", "eleven_multilingual_v2")
            self.default_voice = el_cfg.get("default_voice", "sarah")
        else:
            self.model = "eleven_multilingual_v2"
            self.default_voice = "sarah"

        logger.info(f"VoiceGenerator ready — model={self.model}, default_voice={self.default_voice}")

    def generate(
        self,
        text: str,
        voice: str | None = None,
        out_path: str = "audio/output.wav",
    ) -> str:
        """Generate speech and save as 16 kHz mono WAV.

        Args:
            text: The script to speak.
            voice: Voice name (key in VOICES) or raw ElevenLabs voice ID.
            out_path: Output WAV path (relative to pipeline dir or absolute).

        Returns:
            Absolute path to the 16 kHz mono WAV file.
        """
        voice = voice or self.default_voice
        voice_id = self.VOICES.get(voice, voice)

        self.estimate_cost(text)

        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = self.pipeline_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # ElevenLabs returns an iterator of audio bytes (MP3 by default)
        raw_path = out_path.with_suffix(".raw.mp3")
        logger.info(f"Generating TTS ({len(text)} chars, voice={voice}, model={self.model})...")

        audio_iter = self.client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=self.model,
        )

        with open(raw_path, "wb") as f:
            for chunk in audio_iter:
                f.write(chunk)

        logger.info(f"Raw audio saved: {raw_path} ({raw_path.stat().st_size / 1024:.1f} KB)")

        # Convert to 16 kHz mono WAV
        final_path = self.convert_to_16k(str(raw_path), str(out_path))

        # Clean up raw MP3
        raw_path.unlink(missing_ok=True)

        return final_path

    def convert_to_16k(self, input_path: str, output_wav: str) -> str:
        """Convert any audio file to 16 kHz mono PCM WAV.

        Args:
            input_path: Source audio file (MP3, WAV, etc.).
            output_wav: Destination WAV path.

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
        logger.debug(f"FFmpeg: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed:\n{result.stderr}")

        logger.info(f"16 kHz WAV: {output_wav}")
        return os.path.abspath(output_wav)

    def estimate_cost(self, text: str) -> float:
        """Estimate character usage and warn if approaching free-tier limit.

        Args:
            text: The text to be synthesised.

        Returns:
            Number of characters in the text.
        """
        char_count = len(text)
        if char_count > self.FREE_TIER_CHARS:
            logger.warning(
                f"Text is {char_count} chars — exceeds ElevenLabs free tier "
                f"({self.FREE_TIER_CHARS} chars/month). You may be charged."
            )
        elif char_count > self.FREE_TIER_CHARS * 0.8:
            logger.warning(
                f"Text is {char_count} chars — approaching free tier limit "
                f"({self.FREE_TIER_CHARS} chars/month)."
            )
        else:
            logger.info(f"Text: {char_count} chars (free tier: {self.FREE_TIER_CHARS})")
        return float(char_count)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate speech from text via ElevenLabs")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--voice", default=None, help="Voice name (sarah/rachel/adam) or ElevenLabs voice ID")
    parser.add_argument("--out", default="audio/output.wav", help="Output WAV path")
    parser.add_argument("--config", default="configs/settings.yaml", help="Pipeline config")
    args = parser.parse_args()

    gen = VoiceGenerator(config_path=args.config)
    wav_path = gen.generate(text=args.text, voice=args.voice, out_path=args.out)
    print(f"\nGenerated: {wav_path}")
