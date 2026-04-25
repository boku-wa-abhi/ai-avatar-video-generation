#!/usr/bin/env python3
"""
tests.test_narration — Focused tests for the slide narrator pipeline.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _make_pdf(path: Path, page_count: int) -> None:
    import fitz

    doc = fitz.open()
    for idx in range(page_count):
        page = doc.new_page(width=1280, height=720)
        page.draw_rect(
            fitz.Rect(0, 0, 1280, 720),
            color=(0.15 + idx * 0.05, 0.35, 0.75 - idx * 0.08),
            fill=(0.15 + idx * 0.05, 0.35, 0.75 - idx * 0.08),
        )
        page.insert_textbox(
            fitz.Rect(80, 80, 1200, 240),
            f"Test Page {idx + 1}",
            fontsize=42,
            color=(1, 1, 1),
        )
    doc.save(path)
    doc.close()


def _write_silence(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    samples = max(1, int(seconds * sample_rate))
    audio = np.zeros(samples, dtype=np.float32)
    sf.write(path, audio, sample_rate)


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def test_validate_sync_accepts_flexible_json_schema(tmp_path):
    from avatarpipeline.pipelines._validate import validate_sync

    pdf_path = tmp_path / "deck.pdf"
    _make_pdf(pdf_path, 2)

    json_data = [
        {"text": "Intro narration", "duration": 4},
        {"script": "Second narration", "pause": 0.5},
    ]

    result = validate_sync(pdf_path, json_data)

    assert result.ok
    assert result.slide_count == 2
    assert result.json_count == 2
    assert result.json_data["slides"][0]["slide_number"] == 1
    assert result.json_data["slides"][0]["display_seconds"] == 4.0
    assert result.json_data["slides"][1]["pause_seconds"] == 0.5
    assert any("JSON order was used" in warning for warning in result.warnings)


def test_validate_sync_accepts_slide_number_keyed_dict(tmp_path):
    from avatarpipeline.pipelines._validate import validate_sync

    pdf_path = tmp_path / "deck.pdf"
    _make_pdf(pdf_path, 2)

    json_data = {
        "slides": {
            "1": {"narration": "First"},
            "2": {"voiceover": "Second", "duration_seconds": 2.5},
        }
    }

    result = validate_sync(pdf_path, json_data)

    assert result.ok
    assert [entry["slide_number"] for entry in result.json_data["slides"]] == [1, 2]
    assert result.json_data["slides"][1]["display_seconds"] == 2.5


def test_presenter_slide_selection_parser_supports_single_ranges_and_lists():
    from avatarpipeline.pipelines.presenter import parse_slide_selection

    assert parse_slide_selection("all", 5) == [1, 2, 3, 4, 5]
    assert parse_slide_selection("2", 5) == [2]
    assert parse_slide_selection("1-3", 5) == [1, 2, 3]
    assert parse_slide_selection("1,2,5", 5) == [1, 2, 5]
    assert parse_slide_selection("2,1-3,5", 5) == [2, 1, 3, 5]


def test_presenter_overlay_handles_odd_slide_dimensions(tmp_path):
    from avatarpipeline.pipelines.presenter import _compose_presenter_overlay

    slide_path = tmp_path / "slide.png"
    presenter_path = tmp_path / "presenter.mp4"
    output_path = tmp_path / "composite.mp4"

    Image.new("RGB", (641, 361), (240, 240, 240)).save(slide_path)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=white:s=201x201:d=0.4:r=25",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=0.4",
            "-c:v", "mpeg4",
            "-c:a", "aac",
            str(presenter_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = _compose_presenter_overlay(slide_path, presenter_path, output_path)

    assert Path(result).exists()
    dims = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert dims == "1920,1080"


def test_presenter_reuses_numbered_slide_renders_for_same_pdf_hash(tmp_path, monkeypatch):
    import avatarpipeline.pipelines.presenter as presenter

    pdf_path = tmp_path / "deck.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
    render_dir = tmp_path / "slides"
    calls: list[str] = []

    def fake_render_slides(_pdf_path, _output_dir):
        calls.append("render")
        paths = []
        for idx in range(2):
            out = Path(_output_dir) / f"page_{idx + 1:03d}.png"
            Image.new("RGB", (1280, 720), (30 + idx * 40, 60, 120)).save(out)
            paths.append(out)
        return paths

    monkeypatch.setattr(presenter, "render_slides", fake_render_slides)

    first_paths, first_reused = presenter._ensure_rendered_slides(pdf_path, render_dir, 2, "hash-1")
    second_paths, second_reused = presenter._ensure_rendered_slides(pdf_path, render_dir, 2, "hash-1")

    assert calls == ["render"]
    assert not first_reused
    assert second_reused
    assert [path.name for path in first_paths] == ["slide_001.png", "slide_002.png"]
    assert [path.name for path in second_paths] == ["slide_001.png", "slide_002.png"]


def test_presenter_overlay_supports_optional_logo(tmp_path):
    from avatarpipeline.pipelines.presenter import _compose_presenter_overlay

    slide_path = tmp_path / "slide.png"
    presenter_path = tmp_path / "presenter.mp4"
    logo_path = tmp_path / "logo.png"
    output_path = tmp_path / "composite_with_logo.mp4"

    Image.new("RGB", (1280, 720), (245, 245, 245)).save(slide_path)
    Image.new("RGBA", (420, 120), (24, 120, 80, 255)).save(logo_path)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=white:s=300x300:d=0.4:r=25",
            "-f", "lavfi", "-i", "sine=frequency=660:sample_rate=16000:duration=0.4",
            "-c:v", "mpeg4",
            "-c:a", "aac",
            str(presenter_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = _compose_presenter_overlay(
        slide_path,
        presenter_path,
        output_path,
        logo_image=logo_path,
    )

    assert Path(result).exists()


def test_compose_narrated_video_generates_audio_before_render_and_uses_slide_timing(tmp_path, monkeypatch):
    import avatarpipeline.pipelines.narration as composer
    import avatarpipeline.engines.tts.kokoro as kokoro_mod

    pdf_path = tmp_path / "deck.pdf"
    _make_pdf(pdf_path, 2)

    slide_images = []
    for idx, color in enumerate(((30, 64, 175), (15, 118, 110)), start=1):
        img_path = tmp_path / f"slide_{idx:03d}.png"
        Image.new("RGB", (1920, 1080), color).save(img_path)
        slide_images.append(img_path)

    def fake_render_slides(_pdf_path, _output_dir):
        return slide_images

    class FakeVoiceGenerator:
        def generate(self, text, voice=None, out_path=None):
            duration = 1.0 if "First" in text else 1.0
            _write_silence(Path(out_path), duration)
            return str(out_path)

    monkeypatch.setattr(composer, "render_slides", fake_render_slides)
    monkeypatch.setattr(kokoro_mod, "VoiceGenerator", FakeVoiceGenerator)

    json_data = {
        "default_pause_seconds": 0.25,
        "slides": [
            {"slide_number": 1, "narration": "First slide narration", "duration_seconds": 3.0},
            {"slide_number": 2, "narration": "Second slide narration", "pause_seconds": 0.5},
        ],
    }

    events = list(
        composer.compose_narrated_video(
            pdf_path=pdf_path,
            json_data=json_data,
            output_path=tmp_path / "narrated.mp4",
            voice="af_heart",
            pause_seconds=1.0,
        )
    )

    messages = [message for message, _ in events]
    render_index = next(i for i, message in enumerate(messages) if message.startswith("Rendering slides"))
    tts_index = next(i for i, message in enumerate(messages) if message.startswith("TTS"))
    output_path = Path(events[-1][1])

    assert tts_index < render_index
    assert output_path.exists()

    duration = _probe_duration(output_path)
    assert 4.4 <= duration <= 5.2


def test_compose_narrated_video_supports_japanese_mlx_voice(tmp_path, monkeypatch):
    import avatarpipeline.pipelines.narration as composer
    import avatarpipeline.engines.tts.mlx as mlx_voice_mod

    pdf_path = tmp_path / "deck.pdf"
    _make_pdf(pdf_path, 1)

    slide_image = tmp_path / "slide_001.png"
    Image.new("RGB", (1920, 1080), (90, 35, 130)).save(slide_image)

    calls: list[dict] = []

    def fake_render_slides(_pdf_path, _output_dir):
        return [slide_image]

    class FakeStudio:
        def synthesize_with_voice(
            self,
            text,
            voice_choice,
            model_id=None,
            lang_code=None,
            speed=1.0,
            pitch_shift=0.0,
            output_path=None,
        ):
            calls.append(
                {
                    "text": text,
                    "voice_choice": voice_choice,
                    "model_id": model_id,
                    "lang_code": lang_code,
                }
            )
            _write_silence(Path(output_path), 0.8, sample_rate=24000)
            return str(output_path)

    monkeypatch.setattr(composer, "render_slides", fake_render_slides)
    monkeypatch.setattr(mlx_voice_mod, "MlxVoiceStudio", FakeStudio)

    json_data = {
        "slides": [
            {"slide_number": 1, "narration": "これは日本語のナレーションです。", "duration_seconds": 1.0},
        ],
    }

    events = list(
        composer.compose_narrated_video(
            pdf_path=pdf_path,
            json_data=json_data,
            output_path=tmp_path / "narrated_ja.mp4",
            pause_seconds=0.0,
            tts_engine="mlx",
            mlx_voice_choice="Japanese Narrator [jp-narrator]",
            mlx_model_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            mlx_language="ja",
        )
    )

    assert calls
    assert calls[0]["voice_choice"] == "Japanese Narrator [jp-narrator]"
    assert calls[0]["lang_code"] == "ja"
    assert calls[0]["model_id"] == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    assert Path(events[-1][1]).exists()


def test_compose_narrated_video_supports_japanese_qwen_preset_voice(tmp_path, monkeypatch):
    import avatarpipeline.pipelines.narration as composer
    import avatarpipeline.engines.tts.mlx as mlx_voice_mod

    pdf_path = tmp_path / "deck.pdf"
    _make_pdf(pdf_path, 1)

    slide_image = tmp_path / "slide_001.png"
    Image.new("RGB", (1920, 1080), (25, 55, 95)).save(slide_image)

    calls: list[dict] = []

    def fake_render_slides(_pdf_path, _output_dir):
        return [slide_image]

    class FakeStudio:
        def synthesize_with_preset(
            self,
            text,
            preset_voice,
            model_id=None,
            lang_code=None,
            speed=1.0,
            pitch_shift=0.0,
            output_path=None,
        ):
            calls.append(
                {
                    "text": text,
                    "preset_voice": preset_voice,
                    "model_id": model_id,
                    "lang_code": lang_code,
                }
            )
            _write_silence(Path(output_path), 0.7, sample_rate=24000)
            return str(output_path)

    monkeypatch.setattr(composer, "render_slides", fake_render_slides)
    monkeypatch.setattr(mlx_voice_mod, "MlxVoiceStudio", FakeStudio)

    json_data = {
        "slides": [
            {"slide_number": 1, "narration": "量子計算の概要をご説明します。", "duration_seconds": 1.0},
        ],
    }

    events = list(
        composer.compose_narrated_video(
            pdf_path=pdf_path,
            json_data=json_data,
            output_path=tmp_path / "narrated_ja_preset.mp4",
            pause_seconds=0.0,
            tts_engine="mlx",
            mlx_preset_voice="Ono_Anna",
            mlx_model_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            mlx_language="ja",
        )
    )

    assert calls
    assert calls[0]["preset_voice"] == "Ono_Anna"
    assert calls[0]["lang_code"] == "ja"
    assert calls[0]["model_id"] == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    assert Path(events[-1][1]).exists()


def test_compose_narrated_video_supports_japanese_kokoro_voice(tmp_path, monkeypatch):
    import avatarpipeline.pipelines.narration as composer
    import avatarpipeline.engines.tts.kokoro as kokoro_mod

    pdf_path = tmp_path / "deck.pdf"
    _make_pdf(pdf_path, 1)

    slide_image = tmp_path / "slide_001.png"
    Image.new("RGB", (1920, 1080), (40, 80, 120)).save(slide_image)

    calls: list[dict] = []

    def fake_render_slides(_pdf_path, _output_dir):
        return [slide_image]

    class FakeVoiceGenerator:
        def generate(self, text, voice=None, out_path=None):
            calls.append({"text": text, "voice": voice})
            _write_silence(Path(out_path), 0.8)
            return str(out_path)

    monkeypatch.setattr(composer, "render_slides", fake_render_slides)
    monkeypatch.setattr(kokoro_mod, "VoiceGenerator", FakeVoiceGenerator)

    json_data = {
        "slides": [
            {"slide_number": 1, "narration": "これは日本語の男性ナレーションです。", "duration_seconds": 1.0},
        ],
    }

    events = list(
        composer.compose_narrated_video(
            pdf_path=pdf_path,
            json_data=json_data,
            output_path=tmp_path / "narrated_ja_kokoro.mp4",
            voice="jm_kumo",
            pause_seconds=0.0,
            tts_engine="kokoro",
        )
    )

    assert calls
    assert calls[0]["voice"] == "jm_kumo"
    assert Path(events[-1][1]).exists()
