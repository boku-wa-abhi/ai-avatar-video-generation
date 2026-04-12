# Avatar Studio

A fully local, offline AI avatar video generation pipeline for Apple Silicon (M4 Pro / MPS).
Drop in a portrait photo, type a script, and get back a lip-synced, face-enhanced, captioned MP4 — no API keys, no cloud, no per-minute charges.

---

## How It Works

The pipeline runs 7 sequential steps, each implemented as a standalone class:

```
Text script
    │
    ▼  Step 1 — Voice (Kokoro TTS)
Audio WAV (24 kHz)
    │
    ▼  Step 2 — Resample
Audio WAV (16 kHz mono)
    │
    ▼  Step 3 — Lip-sync (LatentSync 1.6 or MuseTalk 1.5)
Raw lip-synced MP4
    │
    ▼  Step 4 — Face Enhancement (CodeFormer / GFPGAN / passthrough)
Enhanced MP4
    │
    ▼  Step 5 — Background Composite (VideoAssembler)
Framed MP4 (9:16 / 16:9 / 1:1)
    │
    ▼  Step 6 — Captions (faster-whisper → SRT)
SRT subtitle file
    │
    ▼  Step 7 — Final Encode (FFmpeg H.264/AAC +faststart)
Deliverable MP4
```

**Models used (all local, all free):**

| Step | Model | Size | License |
|------|-------|------|---------|
| TTS | Kokoro-82M | ~330 MB | MIT |
| Lip-sync (default) | LatentSync 1.6 | ~7 GB | Apache 2.0 |
| Lip-sync (alt) | MuseTalk 1.5 | ~3 GB | Apache 2.0 |
| Captions | faster-whisper (large-v3) | ~3 GB | MIT |
| Face enhancement | CodeFormer / GFPGAN | optional | — |

---

## Project Structure

```
ai-avatar-video-generation/
│
├── avatarpipeline/           # Core library package
│   ├── __init__.py           # ROOT, data-dir constants, version
│   ├── pipeline.py           # 7-step orchestrator (run_pipeline)
│   ├── voice/
│   │   └── kokoro.py         # VoiceGenerator — Kokoro TTS
│   ├── lipsync/
│   │   ├── latentsync.py     # LatentSyncInference — lip-sync (default)
│   │   └── musetalk.py       # MuseTalkInference   — lip-sync (alt)
│   └── postprocess/
│       ├── enhancer.py       # FaceEnhancer — CodeFormer / GFPGAN
│       ├── captions.py       # CaptionGenerator — faster-whisper
│       └── assembler.py      # VideoAssembler — FFmpeg composite + encode
│
├── ui/
│   └── dashboard.py          # Gradio web dashboard
│
├── scripts/
│   ├── run_dashboard.py      # ← Start the dashboard (main entry point)
│   ├── run_pipeline.py       # CLI pipeline runner
│   └── smoke_test.sh         # End-to-end integration test
│
├── install/
│   ├── setup.sh              # Phase 1: system deps, Python venv, MuseTalk
│   └── install_latentsync.sh # Phase 2: LatentSync + ComfyUI wrapper
│
├── configs/
│   └── settings.yaml         # All pipeline settings
│
├── assets/
│   ├── logo.png
│   └── favicon.png
│
├── tests/
│   └── test_pipeline.py      # 17 pytest unit + integration tests
│
├── data/                     # Runtime data — git-ignored
│   ├── avatars/              # avatar.png lives here
│   ├── audio/                # TTS output WAVs
│   ├── output/               # Generated MP4s
│   ├── captions/             # SRT subtitle files
│   └── temp/                 # Intermediate files
│
├── requirements.txt
├── .env                      # Optional: HF_TOKEN etc. (git-ignored)
└── README.md
```

---

## Quick Start

### 1. Install system dependencies

```bash
brew install python@3.10 git ffmpeg uv espeak-ng
```

### 2. Set up the Python environment

```bash
cd ai-avatar-video-generation
bash install/setup.sh
```

Creates `.venv/` with Python 3.10 and all Python dependencies.

### 3. Install LatentSync (lip-sync model)

```bash
bash install/install_latentsync.sh
```

Clones ComfyUI-LatentSyncWrapper and downloads LatentSync 1.6 checkpoints (~7 GB).

### 4. Add your avatar

```bash
# Copy a portrait PNG into the data directory
cp /path/to/portrait.png data/avatars/avatar.png
```

Or upload it via the dashboard UI.

---

## Starting the Dashboard

```bash
python scripts/run_dashboard.py
```

Opens automatically at **http://localhost:7860**.

Optional flags:

```bash
python scripts/run_dashboard.py --port 7861        # custom port
python scripts/run_dashboard.py --no-browser       # no auto-open
python scripts/run_dashboard.py --host 0.0.0.0     # expose on LAN
python scripts/run_dashboard.py --share            # Gradio public link
```

### Dashboard walkthrough

1. **Avatar** — Upload a portrait PNG or select from the gallery. The full image is visible (not cropped).
2. **Script** — Type the text your avatar will speak. Character count updates live.
3. **Voice** — 10 Kokoro voices available. Use the preview button to audition each one.
4. **Video Settings** — Choose orientation (9:16 / 16:9 / 1:1), optional background music, optional background image.
5. **Advanced Options** — Toggle LatentSync vs MuseTalk, face enhancement, captions, preview mode, caption styling.
6. **Generate Video** — Starts the pipeline. Live log shows progress for all 7 steps with timing.
7. **Output** — Finished video plays inline. Metadata panel shows resolution, duration, file size, and generation time.

---

## CLI Usage

```bash
# Basic (portrait 9:16, default voice)
python scripts/run_pipeline.py \
  --script "Hello, I'm your AI avatar — nice to meet you!" \
  --out data/output/my_video.mp4

# British male voice, landscape orientation
python scripts/run_pipeline.py \
  --script "Welcome to the briefing." \
  --voice bm_george \
  --orientation 16:9

# Fast preview — skip face enhancement and captions
python scripts/run_pipeline.py \
  --script "Quick test." \
  --no-enhance --no-captions

# List all available voices
python scripts/run_pipeline.py --list-voices
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--script TEXT` | _(required)_ | Spoken script text |
| `--orientation` | `9:16` | `9:16` \| `16:9` \| `1:1` |
| `--voice TEXT` | `af_heart` | Kokoro voice ID |
| `--out TEXT` | `data/output/final.mp4` | Output MP4 path |
| `--background TEXT` | `black` | `black` \| `white` \| `blur` \| path to image |
| `--music TEXT` | _(none)_ | Background music file |
| `--preview` | off | Open result in default player |
| `--no-captions` | off | Skip subtitle generation |
| `--no-enhance` | off | Skip face enhancement |
| `--musetalk` | off | Use MuseTalk instead of LatentSync |

---

## Running Tests

```bash
# Full unit test suite (17 tests, ~30s)
python -m pytest tests/ -v

# End-to-end smoke test
bash scripts/smoke_test.sh --no-enhance --no-captions
```

---

## Configuration

Edit `configs/settings.yaml` to adjust pipeline behaviour:

```yaml
musetalk_dir: "~/MuseTalk"
comfyui_dir:  "~/ComfyUI"

latentsync:
  inference_steps: 25     # higher = better quality, slower
  lips_expression: 1.5    # lip movement intensity
  face_resolution: 512

tts:
  engine: "kokoro"
  default_voice: "af_heart"
  speed: 1.0
  lang_code: "a"          # 'a' = American, 'b' = British
```

---

## Available Voices

| ID | Description |
|----|-------------|
| `af_heart` | American Female — warm, clear (default) |
| `af_bella` | American Female — smooth, confident |
| `af_sarah` | American Female — natural, conversational |
| `af_nicole` | American Female — soft, friendly |
| `am_adam` | American Male — deep |
| `am_michael` | American Male — conversational |
| `bf_emma` | British Female — clear |
| `bf_isabella` | British Female — warm |
| `bm_george` | British Male — authoritative |
| `bm_lewis` | British Male — natural |

---

## Architecture Notes

`avatarpipeline/` is a proper Python package. Import any class directly:

```python
from avatarpipeline.voice.kokoro import VoiceGenerator
from avatarpipeline.lipsync.latentsync import LatentSyncInference
from avatarpipeline.postprocess.assembler import VideoAssembler
from avatarpipeline.pipeline import run_pipeline
```

`avatarpipeline/__init__.py` exports `ROOT`, `AVATARS_DIR`, `AUDIO_DIR`, `OUTPUT_DIR`, `CAPTIONS_DIR`, `TEMP_DIR` as `Path` constants so every module resolves paths consistently relative to the project root.

`ui/dashboard.py` is a consumer of the library, not part of it. `scripts/` contains thin entry-point wrappers that add the project root to `sys.path` and delegate to the library.

---

## MPS Troubleshooting

The pipeline automatically sets:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1       # fall back to CPU for unsupported ops
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # prevent OOM
```

| Symptom | Fix |
|---------|-----|
| `MPS backend out of memory` | Close other GPU apps; add `--no-enhance` |
| LatentSync faces blurry | Raise `inference_steps` in `configs/settings.yaml` |
| Black frames | Use a clear front-facing portrait, ≥ 256×256 px |
| `faster-whisper` slow | Expected — CPU int8 is used (MPS unsupported by faster-whisper) |

---

## Requirements

- macOS 14+ with Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.10
- FFmpeg (`brew install ffmpeg`)
- espeak-ng (`brew install espeak-ng`)
- ~10 GB free disk space for models

---

## License

MIT — see [LICENSE](LICENSE).