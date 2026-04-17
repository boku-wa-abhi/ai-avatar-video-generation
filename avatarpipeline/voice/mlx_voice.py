"""
avatarpipeline.voice.mlx_voice — Local MLX voice cloning and voice conversion.

This module adds a simple local voice library on top of `mlx-audio` so the UI
can:
  - save reference voices locally
  - synthesize text using a saved cloned voice
  - convert uploaded speech into a saved cloned voice by transcribing locally
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from loguru import logger

from avatarpipeline import AUDIO_DIR, TEMP_DIR, VOICES_DIR


class MlxVoiceStudio:
    """Manage saved reference voices and MLX-based speech synthesis."""

    DEFAULT_TTS_MODEL = "mlx-community/chatterbox-fp16"
    DEFAULT_STT_MODEL = "mlx-community/whisper-large-v3-turbo-asr-fp16"
    OUTPUT_SAMPLE_RATE = 24_000

    MODEL_CHOICES = {
        "Chatterbox MLX — multilingual voice cloning (~2.6 GB download)": DEFAULT_TTS_MODEL,
        "Qwen3-TTS MLX 0.6B 8-bit — lighter voice cloning (~2.0 GB download)": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    }

    LANGUAGE_CHOICES = {
        "English": "en",
        "Japanese": "ja",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Hindi": "hi",
        "Italian": "it",
        "Portuguese": "pt",
        "Chinese": "zh",
        "Korean": "ko",
    }

    _TTS_MODEL_CACHE: ClassVar[dict[str, object]] = {}
    _STT_MODEL_CACHE: ClassVar[dict[str, object]] = {}

    def __init__(self, voice_store: str | Path | None = None) -> None:
        self.voice_store = Path(voice_store or VOICES_DIR).resolve()
        self.voice_store.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @classmethod
    def model_labels(cls) -> list[str]:
        return list(cls.MODEL_CHOICES.keys())

    @classmethod
    def language_labels(cls) -> list[str]:
        return list(cls.LANGUAGE_CHOICES.keys())

    @classmethod
    def resolve_model_id(cls, label_or_id: str | None) -> str:
        if not label_or_id:
            return cls.DEFAULT_TTS_MODEL
        return cls.MODEL_CHOICES.get(label_or_id, label_or_id)

    @classmethod
    def resolve_language_code(cls, label_or_code: str | None) -> str:
        if not label_or_code:
            return "en"
        return cls.LANGUAGE_CHOICES.get(label_or_code, label_or_code)

    @staticmethod
    def extract_slug(choice_or_slug: str | None) -> str:
        if not choice_or_slug:
            return ""
        choice_or_slug = choice_or_slug.strip()
        if choice_or_slug.endswith("]") and "[" in choice_or_slug:
            return choice_or_slug.rsplit("[", 1)[1][:-1].strip()
        return choice_or_slug

    def list_voice_profiles(self) -> list[dict]:
        profiles: list[dict] = []
        for meta_path in self.voice_store.glob("*/profile.json"):
            try:
                profile = json.loads(meta_path.read_text())
                ref_name = profile.get("reference_audio_file", "reference.wav")
                profile["reference_audio_path"] = str((meta_path.parent / ref_name).resolve())
                profiles.append(profile)
            except Exception as exc:
                logger.warning(f"Skipping invalid voice profile at {meta_path}: {exc}")
        profiles.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return profiles

    def list_voice_choices(self) -> list[str]:
        return [self._format_choice(profile) for profile in self.list_voice_profiles()]

    def get_voice_profile(self, choice_or_slug: str | None) -> dict | None:
        slug = self.extract_slug(choice_or_slug)
        if not slug:
            return None

        meta_path = self.voice_store / slug / "profile.json"
        if not meta_path.exists():
            return None

        profile = json.loads(meta_path.read_text())
        ref_name = profile.get("reference_audio_file", "reference.wav")
        profile["reference_audio_path"] = str((meta_path.parent / ref_name).resolve())
        return profile

    def save_voice_profile(
        self,
        name: str,
        audio_path: str,
        transcript: str | None = None,
        model_id: str | None = None,
    ) -> dict:
        """Save a reference voice locally for later cloning."""
        voice_name = (name or "").strip()
        if not voice_name:
            raise ValueError("Enter a name for the saved voice.")

        src = Path(audio_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Reference audio not found: {src}")

        slug = self._reserve_slug(voice_name)
        voice_dir = self.voice_store / slug
        voice_dir.mkdir(parents=True, exist_ok=True)

        reference_wav = voice_dir / "reference.wav"
        self._convert_audio(src, reference_wav, sample_rate=self.OUTPUT_SAMPLE_RATE)

        final_transcript = (transcript or "").strip()
        if not final_transcript:
            final_transcript = self.transcribe_audio(str(reference_wav))

        profile = {
            "name": voice_name,
            "slug": slug,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_hint": self.resolve_model_id(model_id),
            "reference_audio_file": reference_wav.name,
            "reference_text": final_transcript,
            "duration_seconds": round(self._probe_duration(reference_wav), 2),
        }
        (voice_dir / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=True))
        logger.info(f"Saved voice profile: {voice_name} ({slug})")
        profile["reference_audio_path"] = str(reference_wav.resolve())
        return profile

    def transcribe_audio(self, audio_path: str, model_id: str | None = None) -> str:
        """Transcribe audio locally using MLX Whisper."""
        _, _, generate_transcription, _ = self._mlx_audio_api()
        stt_model_id = model_id or self.DEFAULT_STT_MODEL
        model = self._get_stt_model(stt_model_id)

        stem = datetime.now().strftime("mlx_transcript_%Y%m%d_%H%M%S")
        output_base = TEMP_DIR / stem
        segments = generate_transcription(
            model=model,
            audio=audio_path,
            output_path=str(output_base),
            format="txt",
            verbose=False,
        )
        Path(f"{output_base}.txt").unlink(missing_ok=True)

        text = (getattr(segments, "text", "") or "").strip()
        if not text:
            raise RuntimeError("Transcription returned empty text.")
        return text

    def synthesize_with_voice(
        self,
        text: str,
        voice_choice: str,
        model_id: str | None = None,
        lang_code: str | None = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        output_path: str | None = None,
    ) -> str:
        """Generate speech using a saved reference voice."""
        script = (text or "").strip()
        if not script:
            raise ValueError("Enter text to synthesize.")

        profile = self.get_voice_profile(voice_choice)
        if not profile:
            raise ValueError("Select a saved voice first.")

        tts_model_id = self.resolve_model_id(model_id or profile.get("model_hint"))
        lang = self.resolve_language_code(lang_code)
        output = Path(output_path) if output_path else AUDIO_DIR / self._timestamped_name("mlx_tts")
        output = output.with_suffix(".wav").resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        raw_output = output.with_name(f"{output.stem}_raw.wav")
        self._generate_audio(
            text=script,
            model_id=tts_model_id,
            ref_audio=profile["reference_audio_path"],
            ref_text=profile.get("reference_text"),
            lang_code=lang,
            speed=float(speed),
            output_path=raw_output,
        )

        if abs(float(pitch_shift)) > 1e-6:
            self.apply_pitch_shift(raw_output, output, float(pitch_shift))
            raw_output.unlink(missing_ok=True)
        else:
            shutil.move(str(raw_output), str(output))

        logger.info(f"Generated MLX speech: {output}")
        return str(output)

    def convert_voice(
        self,
        source_audio: str,
        voice_choice: str,
        transcript_override: str | None = None,
        model_id: str | None = None,
        lang_code: str | None = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        output_path: str | None = None,
    ) -> tuple[str, str]:
        """Convert uploaded speech into a saved cloned voice."""
        src = Path(source_audio).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source audio not found: {src}")

        transcript = (transcript_override or "").strip()
        if not transcript:
            transcript = self.transcribe_audio(str(src))

        result = self.synthesize_with_voice(
            text=transcript,
            voice_choice=voice_choice,
            model_id=model_id,
            lang_code=lang_code,
            speed=speed,
            pitch_shift=pitch_shift,
            output_path=output_path,
        )
        return result, transcript

    def apply_pitch_shift(self, input_audio: str | Path, output_audio: str | Path, semitones: float) -> str:
        """Shift pitch up/down while preserving duration."""
        src = Path(input_audio).resolve()
        dest = Path(output_audio).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = self._probe_sample_rate(src) or self.OUTPUT_SAMPLE_RATE
        factor = 2 ** (float(semitones) / 12.0)
        filters = self._build_pitch_filters(sample_rate, factor)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-af", filters,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(dest),
        ]
        self._run(cmd, "Pitch shifting failed")
        return str(dest)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _mlx_audio_api():
        try:
            from mlx_audio.stt import load as load_stt_model
            from mlx_audio.stt.generate import generate_transcription
            from mlx_audio.tts.generate import generate_audio
            from mlx_audio.tts.utils import load_model as load_tts_model
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is not installed. Run: uv pip install mlx-audio>=0.3.0"
            ) from exc
        return generate_audio, load_tts_model, generate_transcription, load_stt_model

    @classmethod
    def _format_choice(cls, profile: dict) -> str:
        return f"{profile.get('name', profile.get('slug', 'voice'))} [{profile.get('slug', 'voice')}]"

    @staticmethod
    def _slugify(name: str) -> str:
        chars = []
        for ch in name.lower():
            if ch.isalnum():
                chars.append(ch)
            else:
                chars.append("-")
        slug = "".join(chars).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug or "voice"

    def _reserve_slug(self, name: str) -> str:
        base = self._slugify(name)
        candidate = base
        idx = 2
        while (self.voice_store / candidate).exists():
            candidate = f"{base}-{idx}"
            idx += 1
        return candidate

    @staticmethod
    def _timestamped_name(prefix: str) -> str:
        return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S.wav")

    def _get_tts_model(self, model_id: str):
        _, load_tts_model, _, _ = self._mlx_audio_api()
        if model_id not in self._TTS_MODEL_CACHE:
            logger.info(f"Loading MLX TTS model: {model_id}")
            self._TTS_MODEL_CACHE[model_id] = load_tts_model(model_id)
        return self._TTS_MODEL_CACHE[model_id]

    def _get_stt_model(self, model_id: str):
        _, _, _, load_stt_model = self._mlx_audio_api()
        if model_id not in self._STT_MODEL_CACHE:
            logger.info(f"Loading MLX STT model: {model_id}")
            self._STT_MODEL_CACHE[model_id] = load_stt_model(model_id)
        return self._STT_MODEL_CACHE[model_id]

    def _generate_audio(
        self,
        text: str,
        model_id: str,
        ref_audio: str,
        ref_text: str | None,
        lang_code: str,
        speed: float,
        output_path: Path,
    ) -> None:
        generate_audio, _, _, _ = self._mlx_audio_api()
        model = self._get_tts_model(model_id)
        generate_audio(
            text=text,
            model=model,
            ref_audio=ref_audio,
            ref_text=ref_text,
            lang_code=lang_code,
            speed=speed,
            output_path=str(output_path.parent),
            file_prefix=output_path.stem,
            audio_format=output_path.suffix.lstrip(".") or "wav",
            join_audio=True,
            verbose=False,
            play=False,
        )

    def _convert_audio(self, src: Path, dest: Path, sample_rate: int) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(dest),
        ]
        self._run(cmd, "Audio conversion failed")

    @staticmethod
    def _build_pitch_filters(sample_rate: int, factor: float) -> str:
        return ",".join(
            [
                f"asetrate={max(1000, int(round(sample_rate * factor)))}",
                f"aresample={sample_rate}",
                f"atempo={1 / factor:.6f}",
            ]
        )

    @staticmethod
    def _probe_duration(audio_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    @staticmethod
    def _probe_sample_rate(audio_path: Path) -> int:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return int(result.stdout.strip())
        except ValueError:
            return 0

    @staticmethod
    def _run(cmd: list[str], error_message: str) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{error_message}:\n{result.stderr}")
