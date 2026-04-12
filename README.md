# AI Avatar Video Generation

A local, fully-offline HeyGen alternative running on Apple Silicon (M4 Pro / MPS).
Generate talking-head MP4 videos from a text script with one command.

```
python make_video.py --script "Hello! I'm your AI avatar." \
                     --orientation 9:16 \
                     --out output/final.mp4
```

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Pipeline Steps](#pipeline-steps)
6. [Model Sizes](#model-sizes)
7. [Project Structure](#project-structure)
8. [MPS Troubleshooting](#mps-troubleshooting)
9. [Example Commands](#example-commands)

---

## Architecture

```
Text Script
    │
    ▼
[1] Kokoro TTS (82M, local) ──────────────► audio/speech.wav
    │
    ▼
[2] Resample 16 kHz mono ─────────────────► audio/speech_16k.wav
    │
    ▼
[3] LatentSync 1.6 / MuseTalk 1.5 ───────► output/lipsync.mp4
      (lip-sync avatar.png + audio)
    │
    ▼
[4] CodeFormer / GFPGAN face enhancement ► output/enhanced.mp4
    │
    ▼
[5] FFmpeg background composite ──────────► output/composed.mp4
    │
    ▼
[6] faster-whisper → SRT captions ────────► captions/captions.srt
    │
    ▼
[7] Final H.264/AAC encode + captions ───► output/final.mp4
```

**Runtime:** Apple M4 Pro · MPS device · Python 3.10  
**Package manager:** uv

---

## Prerequisites

| Tool | Install |
|---|---|
| Python 3.10 | `brew install python@3.10` or `pyenv install 3.10` |
| uv | `brew install uv` |
| FFmpeg | `brew install ffmpeg` |
| espeak-ng | `brew install espeak-ng` (required by Kokoro TTS) |
| Git LFS | `brew install git-lfs && git lfs install` |

---

## Quick Start

### 1 — Install Phase 1 (MuseTalk + mmlab stack)

```bash
bash setup.sh
```

This clones MuseTalk, creates `.venv` (Python 3.10), installs all mmlab
dependencies with the known Apple Silicon fixes, and downloads MuseTalk weights.

### 2 — Install Phase 2 (LatentSync + Kokoro)

```bash
# Copy your HuggingFace token into .env first:
echo "HF_TOKEN=hf_xxxx" >> .env

bash install_latentsync.sh
```

Downloads LatentSync 1.6 weights (~6.5 GB) and installs Kokoro TTS.

### 3 — Add your avatar image

Place a front-facing portrait PNG at:

```
avatar/avatar.png
```

Recommended: 512×512 px, neutral expression, good lighting.

### 4 — Generate a video

```bash
source .venv/bin/activate

python make_video.py \
    --script "Hello! I'm your AI assistant, ready to help." \
    --orientation 9:16 \
    --out output/final.mp4
```

### 5 — Run the integration test

```bash
bash test_full_pipeline.sh --no-enhance   # fast smoke test
bash test_full_pipeline.sh                # full pipeline
```

---

## CLI Reference

```
python make_video.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--script TEXT` | _(required)_ | Spoken script text |
| `--orientation` | `9:16` | Canvas aspect ratio: `9:16` \| `16:9` \| `1:1` |
| `--voice TEXT` | `af_heart` | Kokoro voice ID (see voices below) |
| `--out TEXT` | `output/final.mp4` | Output MP4 path |
| `--background TEXT` | `black` | `black` \| `white` \| `blur` \| `/path/to/image.jpg` |
| `--music TEXT` | _(none)_ | Background music file (mixed at 15% volume) |
| `--preview` | off | Open result in macOS default player |
| `--no-captions` | off | Skip subtitle generation |
| `--no-enhance` | off | Skip face enhancement step |
| `--musetalk-only` | off | Use MuseTalk instead of LatentSync |

### Available voices

| ID | Style |
|---|---|
| `af_heart` | American female — warm, natural (default) |
| `af_bella` | American female — clear |
| `af_sarah` | American female — professional |
| `af_nicole` | American female — calm |
| `am_adam` | American male — deep |
| `am_michael` | American male — conversational |
| `bf_emma` | British female |
| `bf_isabella` | British female |
| `bm_george` | British male |
| `bm_lewis` | British male |

---

## Pipeline Steps

| # | Module | What it does |
|---|---|---|
| 1 | `voice_gen.py` | Kokoro TTS: text → WAV |
| 2 | `voice_gen.py` | FFmpeg resample: 44.1 kHz → 16 kHz mono |
| 3 | `latentsync_infer.py` | LatentSync 1.6 lip-sync (or MuseTalk 1.5 with `--musetalk-only`) |
| 4 | `face_enhancer.py` | CodeFormer frame restoration (auto-falls back to GFPGAN → passthrough) |
| 5 | `video_assembler.py` | Composite avatar onto canvas, mix music |
| 6 | `caption_gen.py` | faster-whisper word-level transcription → SRT |
| 7 | `video_assembler.py` | Final H.264 / AAC encode, burn subtitles |

---

## Model Sizes

| Model | Size | Location |
|---|---|---|
| Kokoro-82M | ~330 MB | HuggingFace cache |
| MuseTalk 1.5 weights | ~2.5 GB | `~/MuseTalk/models/` |
| LatentSync UNet | 4.7 GB | `~/ComfyUI/.../checkpoints/latentsync_unet.pt` |
| LatentSync SyncNet | 1.5 GB | `~/ComfyUI/.../checkpoints/stable_syncnet.pt` |
| Stable Diffusion VAE | 319 MB | `~/ComfyUI/.../checkpoints/vae/` |
| faster-whisper base | ~145 MB | HuggingFace cache |

Total first-run download: **~9.5 GB**

---

## Project Structure

```
ai-avatar-video-generation/
├── make_video.py          # Pipeline orchestrator (entry point)
├── voice_gen.py           # Kokoro TTS wrapper
├── musetalk_infer.py      # MuseTalk 1.5 wrapper
├── latentsync_infer.py    # LatentSync 1.6 wrapper
├── face_enhancer.py       # CodeFormer/GFPGAN face restoration
├── caption_gen.py         # faster-whisper captions + SRT
├── video_assembler.py     # FFmpeg composite, music mix, final encode
├── setup.sh               # Phase 1 installer (MuseTalk, mmlab, venv)
├── install_latentsync.sh  # Phase 2 installer (LatentSync, Kokoro)
├── test_full_pipeline.sh  # Integration smoke test
├── requirements.txt       # Python deps for pipeline .venv
├── configs/
│   └── settings.yaml      # Global settings (FPS, orientations, voices)
├── avatar/
│   └── avatar.png         # Your portrait image (you provide this)
├── audio/                 # Intermediate WAV files
├── captions/              # Generated SRT files
├── output/                # All generated MP4s
└── .env                   # Secrets: HF_TOKEN, ELEVENLABS_KEY (git-ignored)
```

---

## MPS Troubleshooting

The pipeline automatically sets these env vars before any model inference:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1    # fall back to CPU for unsupported ops
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # prevent OOM by avoiding memory reserve
```

**Common issues:**

| Symptom | Fix |
|---|---|
| `RuntimeError: MPS backend out of memory` | Close other GPU-heavy apps, add `--no-enhance` to free VRAM |
| `NotImplementedError: ... not implemented for MPS` | Already handled by `MPS_FALLBACK=1`; if still failing, open an issue |
| `faster-whisper` slow on CPU | Expected — faster-whisper uses CPU/int8 on Apple Silicon (MPS not supported natively) |
| LatentSync output faces are blurry | Increase `inference_steps` in `configs/settings.yaml` (default: 25) |
| Black frames in output | Check that `avatar/avatar.png` is a clear front-facing portrait, ≥ 256×256 px |

---

## Example Commands

```bash
# Vertical short-form video (TikTok / Reels)
python make_video.py \
    --script "Top 5 Python tips every developer should know!" \
    --orientation 9:16 \
    --voice am_adam \
    --out output/python_tips.mp4

# Landscape explainer (YouTube / LinkedIn)
python make_video.py \
    --script "Let me walk you through our Q3 results." \
    --orientation 16:9 \
    --background blur \
    --out output/quarterly_review.mp4

# Square post with background music
python make_video.py \
    --script "Welcome to our product launch!" \
    --orientation 1:1 \
    --music assets/bg_music.mp3 \
    --out output/launch.mp4

# Fast preview (skip enhancement + captions)
python make_video.py \
    --script "Quick test run." \
    --no-enhance --no-captions \
    --preview \
    --out output/preview.mp4

# Force MuseTalk (faster, lower quality)
python make_video.py \
    --script "MuseTalk test." \
    --musetalk-only \
    --out output/musetalk_test.mp4
```

---

## License

MIT — see [LICENSE](LICENSE).