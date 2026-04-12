# Avatar Studio — Pipeline Architecture & Model Documentation

> **Executive Summary:** Avatar Studio is a fully local, offline AI video generation pipeline that converts a text script + portrait photo into a lip-synced, face-enhanced, captioned MP4. It runs 7 sequential stages on Apple Silicon (MPS) using ~14 GB of open-source models. Every component has free alternatives that can be swapped in. Zero API keys, zero cloud costs, zero per-minute charges.

---

## Table of Contents

1. [Pipeline Flow Diagram](#1-pipeline-flow-diagram)
2. [Stage-by-Stage Deep Dive](#2-stage-by-stage-deep-dive)
   - [Stage 1: Voice Generation (TTS)](#stage-1-voice-generation-tts)
   - [Stage 2: Audio Resampling](#stage-2-audio-resampling)
   - [Stage 3: Lip-Sync](#stage-3-lip-sync)
   - [Stage 4: Face Enhancement](#stage-4-face-enhancement)
   - [Stage 5: Background Composite](#stage-5-background-composite)
   - [Stage 6: Caption Generation](#stage-6-caption-generation)
   - [Stage 7: Final Encode](#stage-7-final-encode)
3. [Model Registry](#3-model-registry)
4. [Model Locations & Checkpoints](#4-model-locations--checkpoints)
5. [Free Alternatives & Swap Guide](#5-free-alternatives--swap-guide)
6. [Internal Health Checks](#6-internal-health-checks)
7. [Decision Matrix](#7-decision-matrix)
8. [Data Flow Diagram](#8-data-flow-diagram)

---

## 1. Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     USER INPUT                                    │
│   Portrait PNG + Text Script + Voice + Orientation                │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 1 │ Voice Generation                                      │
│  Model:  │ Kokoro-82M (TTS)                                      │
│  Input:  │ Text script + voice ID                                 │
│  Output: │ 24 kHz WAV audio                                      │
│  Time:   │ ~5–15 sec                                              │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 2 │ Audio Resampling                                      │
│  Tool:   │ FFmpeg                                                 │
│  Input:  │ 24 kHz WAV                                             │
│  Output: │ 16 kHz mono WAV (required by lip-sync)                 │
│  Time:   │ <1 sec                                                 │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 3 │ Lip-Sync Inference                                    │
│  Model:  │ LatentSync 1.6 (default) │ MuseTalk 1.5 (alt)         │
│  Input:  │ Portrait PNG + 16 kHz WAV                              │
│  Output: │ Raw lip-synced MP4 (25 fps)                            │
│  Time:   │ ~3–8 min                                               │
│  Sub:    │ VAE (SD-VAE-FT-MSE) + UNet + Whisper audio encoder     │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 4 │ Face Enhancement       (optional — can skip)           │
│  Model:  │ CodeFormer → GFPGAN v1.3 → passthrough                 │
│  Input:  │ Raw lip-synced MP4                                     │
│  Output: │ Enhanced MP4 (cleaner face details)                    │
│  Time:   │ ~1–4 min                                               │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 5 │ Background Composite                                  │
│  Tool:   │ FFmpeg (scale + pad / overlay / blur)                  │
│  Input:  │ Enhanced MP4 + orientation (9:16 / 16:9 / 1:1)        │
│  Output: │ Framed MP4 at target resolution                       │
│  Time:   │ ~5–15 sec                                              │
│  Option: │ black / white / blur / custom image background         │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 6 │ Caption Generation     (optional — can skip)           │
│  Model:  │ faster-whisper large-v3 (Whisper ASR)                  │
│  Input:  │ 16 kHz WAV audio                                      │
│  Output: │ SRT subtitle file (word-level timestamps)              │
│  Time:   │ ~10–30 sec                                             │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 7 │ Final Encode                                          │
│  Tool:   │ FFmpeg (H.264 + AAC + subtitles burn-in)              │
│  Input:  │ Framed MP4 + SRT + optional music                     │
│  Output: │ Deliverable MP4 (CRF 18, AAC 192k, +faststart)        │
│  Time:   │ ~10–30 sec                                             │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FINAL OUTPUT                                  │
│   studio_<timestamp>.mp4 — ready to publish                      │
└──────────────────────────────────────────────────────────────────┘
```

**Total generation time:** ~5–12 minutes for a 30-second video on Apple M4 Pro.

---

## 2. Stage-by-Stage Deep Dive

### Stage 1: Voice Generation (TTS)

| Property | Value |
|----------|-------|
| **Code** | `avatarpipeline/voice/kokoro.py` → `VoiceGenerator` |
| **Model** | Kokoro-82M |
| **Source** | PyPI: `kokoro>=0.9.4` / HuggingFace: [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| **Size** | ~330 MB |
| **License** | MIT |
| **Input** | Text string + voice ID (e.g., `af_bella`) |
| **Output** | 24 kHz WAV file → `data/audio/output.wav` |
| **Config** | `configs/settings.yaml` → `tts.default_voice`, `tts.speed`, `tts.lang_code` |
| **System Dep** | `espeak-ng` (phoneme backend) |

**How it works:**
1. `VoiceGenerator.__init__()` reads `configs/settings.yaml` for voice settings
2. On first call, lazy-loads `KPipeline(lang_code='a')` (American) or `'b'` (British)
3. `generate()` splits text into chunks, passes to Kokoro, concatenates audio
4. Saves 24 kHz WAV, then calls `convert_to_16k()` via FFmpeg for Stage 3 compatibility
5. Returns path to the 16 kHz WAV

**10 built-in voices:**
| Voice ID | Description | Language |
|----------|-------------|----------|
| `af_heart` | Warm female (default) | American |
| `af_bella` | Smooth, confident female | American |
| `af_sarah` | Natural, conversational female | American |
| `af_nicole` | Soft, friendly female | American |
| `am_adam` | Deep male | American |
| `am_michael` | Conversational male | American |
| `bf_emma` | Clear female | British |
| `bf_isabella` | Warm female | British |
| `bm_george` | Authoritative male | British |
| `bm_lewis` | Natural male | British |

---

### Stage 2: Audio Resampling

| Property | Value |
|----------|-------|
| **Code** | Inside `VoiceGenerator.convert_to_16k()` |
| **Tool** | FFmpeg |
| **No model needed** | Pure signal processing |
| **Input** | 24 kHz WAV |
| **Output** | 16 kHz mono PCM WAV |

**How it works:**
```bash
ffmpeg -i input_24k.wav -ar 16000 -ac 1 -c:a pcm_s16le output_16k.wav
```
LatentSync and MuseTalk both require 16 kHz mono audio. This step ensures compatibility regardless of the TTS output format.

---

### Stage 3: Lip-Sync

This is the core stage. Two models are available:

#### Option A: LatentSync 1.6 (Default)

| Property | Value |
|----------|-------|
| **Code** | `avatarpipeline/lipsync/latentsync.py` → `LatentSyncInference` |
| **Model** | LatentSync 1.6 UNet (diffusion-based) |
| **Source** | HuggingFace: [`ByteDance/LatentSync-1.6`](https://huggingface.co/ByteDance/LatentSync-1.6) (gated — requires HF_TOKEN) |
| **Size** | ~7 GB total (UNet + VAE + Whisper + SyncNet) |
| **License** | Apache 2.0 |
| **Checkpoint Dir** | `~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/` |

**Sub-models loaded inside LatentSync:**

| Sub-model | File | Source | Purpose |
|-----------|------|--------|---------|
| UNet3D | `latentsync_unet.pt` | ByteDance/LatentSync-1.6 | Main lip-sync diffusion model |
| VAE | `vae/diffusion_pytorch_model.safetensors` | [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse) | Encode/decode video frames |
| Audio Encoder | `whisper/` (tiny model) | [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | Convert audio → feature vectors for conditioning |
| SyncNet | `stable_syncnet.pt` | ByteDance/LatentSync-1.6 | Validate lip-sync quality |
| Scheduler | DDIMScheduler | diffusers library | Denoising schedule (25 steps default) |
| Config | `configs/unet/stage2_512.yaml` | In wrapper repo | UNet architecture definition |

**How it works:**
1. `prepare_input_video()` creates a looped MP4 from the portrait PNG matching audio duration
2. Spawns a subprocess with a generated Python runner script
3. Runner loads all models via OmegaConf (not `from_pretrained` — manual assembly)
4. Patches `F.scaled_dot_product_attention` → chunked SDPA for MPS 8 GB Metal buffer limit
5. Runs `LipsyncPipeline(video, audio, num_frames=16, inference_steps=25, guidance_scale=1.5)`
6. Pipeline processes 16 frames at a time through the UNet → VAE decoder → output video
7. Returns raw lip-synced MP4

**Key config parameters:**
```yaml
latentsync:
  inference_steps: 25       # More steps = better quality, slower
  lips_expression: 1.5      # Higher = more lip movement
  input_fps: 25             # Frame rate
  face_resolution: 512      # 3D face crop resolution
```

#### Option B: MuseTalk 1.5 (Alternative)

| Property | Value |
|----------|-------|
| **Code** | `avatarpipeline/lipsync/musetalk.py` → `MuseTalkInference` |
| **Model** | MuseTalk v1.5 |
| **Source** | GitHub: [`TMElyralab/MuseTalk`](https://github.com/TMElyralab/MuseTalk) |
| **Size** | ~3 GB |
| **License** | Apache 2.0 |
| **Install Dir** | `~/MuseTalk/` with separate venv `~/MuseTalk/musetalk-env/` |

**How it works:**
1. Resizes + pads portrait to 256×256
2. Shells out to MuseTalk's inference script in its own virtual environment:
   ```bash
   ~/MuseTalk/musetalk-env/bin/python -m scripts.inference \
     --version v15 --video_path <png> --audio_path <wav> \
     --output_dir <dir> --fps 25 --use_float16
   ```
3. Returns the output MP4

**LatentSync vs MuseTalk comparison:**

| Aspect | LatentSync 1.6 | MuseTalk 1.5 |
|--------|---------------|--------------|
| Quality | Higher (diffusion-based) | Good (GAN-based) |
| Speed | Slower (~5–8 min) | Faster (~2–4 min) |
| VRAM | ~8 GB peak | ~4 GB peak |
| Model size | ~7 GB | ~3 GB |
| Lip accuracy | Very accurate | Good |
| Resolution | 512×512 face | 256×256 face |

---

### Stage 4: Face Enhancement

| Property | Value |
|----------|-------|
| **Code** | `avatarpipeline/postprocess/enhancer.py` → `FaceEnhancer` |
| **Optional** | Skippable with `--no-enhance` |
| **Input** | Raw lip-synced MP4 |
| **Output** | Enhanced MP4 with restored facial details |

**Auto-detection priority (uses first available):**

| Priority | Model | Source | Size | Location |
|----------|-------|--------|------|----------|
| 1st | CodeFormer | ComfyUI reactor-node | ~300 MB | `~/ComfyUI/custom_nodes/comfyui-reactor-node/` |
| 2nd | GFPGAN v1.3 | GitHub release | ~350 MB | Auto-downloaded from GitHub |
| 3rd | Passthrough | — | — | No enhancement applied |

**How it works:**
1. `FaceEnhancer.__init__()` checks which backend is available
2. Extract all frames from video → PNG files in `data/temp/enhance_frames/`
3. Run face restoration on each frame individually:
   - **CodeFormer**: `inference_app(input, output, fidelity_weight=0.7, device="mps")`
   - **GFPGAN**: `GFPGANer(model_path, upscale=1, arch='clean').enhance(frame)`
4. Reassemble enhanced frames → MP4 with `libx264 CRF 17`
5. Mux original audio back onto the enhanced video

---

### Stage 5: Background Composite

| Property | Value |
|----------|-------|
| **Code** | `avatarpipeline/postprocess/assembler.py` → `VideoAssembler` |
| **Tool** | FFmpeg (no ML model) |

**Supported orientations:**

| Orientation | Resolution | Use Case |
|-------------|-----------|----------|
| `9:16` | 1080×1920 | TikTok, Reels, Shorts (default) |
| `16:9` | 1920×1080 | YouTube, presentations |
| `1:1` | 1080×1080 | Instagram posts |

**Background options:**
- `black` — Black letterbox padding (default)
- `white` — White letterbox padding
- `blur` — Recursive blur effect (boxblur 30:5)
- `/path/to/image.png` — Custom image background

**How it works:**
1. FFmpeg scales avatar video to fit inside the target canvas
2. Pads or overlays with selected background style
3. Output: properly framed MP4 at target resolution

---

### Stage 6: Caption Generation

| Property | Value |
|----------|-------|
| **Code** | `avatarpipeline/postprocess/captions.py` → `CaptionGenerator` |
| **Model** | faster-whisper large-v3 (CTranslate2 port of Whisper) |
| **Source** | PyPI: `faster-whisper>=1.0.0` / HuggingFace: [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3) |
| **Size** | ~3 GB |
| **License** | MIT |
| **Optional** | Skippable with `--no-captions` |
| **Device** | CPU (int8) — MPS not supported by CTranslate2 |

**How it works:**
1. `CaptionGenerator("large-v3", device="auto")` — auto-selects CPU with int8 quantization
2. `transcribe(audio_wav)` → runs Whisper ASR, generates word-level timestamps
3. Outputs standard SRT format to `data/captions/output.srt`
4. Later burned into video in Stage 7 via FFmpeg ASS subtitles filter

**Subtitle styling (hardcoded in assembler):**
- Font: Arial Bold, 16pt
- Color: White text, black outline (2px)
- Position: Bottom center, 40px margin

---

### Stage 7: Final Encode

| Property | Value |
|----------|-------|
| **Code** | `VideoAssembler.finalize()` + `VideoAssembler.add_music()` |
| **Tool** | FFmpeg |
| **No model needed** | Pure video encoding |

**How it works:**
1. If music provided → `add_music()` mixes audio tracks (`amix` filter, default 15% volume)
2. If SRT provided → burns subtitles via FFmpeg `subtitles` filter
3. Final encode: `libx264 CRF 18, preset slow, AAC 192k, +faststart`
4. Output: `data/output/studio_<timestamp>.mp4`

**Encoding parameters:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Video codec | libx264 | Universal compatibility |
| CRF | 18 | High quality (lower = better, 0–51 scale) |
| Preset | slow | Better compression ratio |
| Audio codec | AAC | Universal compatibility |
| Audio bitrate | 192 kbps | High quality |
| `+faststart` | enabled | Streaming-ready (moov atom at start) |

---

## 3. Model Registry

Complete inventory of all models used in the pipeline:

| # | Model | Version | Parameters | Size | License | Source | Purpose |
|---|-------|---------|------------|------|---------|--------|---------|
| 1 | Kokoro | 82M | 82M | 330 MB | MIT | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Text-to-Speech |
| 2 | LatentSync UNet | 1.6 | ~1B | 1.5 GB | Apache 2.0 | [ByteDance/LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6) | Lip-sync diffusion |
| 3 | SD-VAE-FT-MSE | — | 83M | 170 MB | OpenRAIL | [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | Frame encode/decode |
| 4 | Whisper Tiny | tiny | 39M | 140 MB | MIT | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | Audio → feature vectors (lip-sync) |
| 5 | Stable SyncNet | — | — | 200 MB | Apache 2.0 | ByteDance/LatentSync-1.6 | Lip-sync quality validation |
| 6 | MuseTalk | 1.5 | — | 3 GB | Apache 2.0 | [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) | Lip-sync (alternative) |
| 7 | CodeFormer | — | — | 300 MB | — | ComfyUI reactor-node | Face restoration (primary) |
| 8 | GFPGAN | v1.3 | — | 350 MB | Apache 2.0 | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | Face restoration (fallback) |
| 9 | faster-whisper | large-v3 | 1.55B | 3 GB | MIT | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | Speech recognition (captions) |

**Total disk footprint: ~14 GB**

---

## 4. Model Locations & Checkpoints

### Where every model file lives on disk:

```
~/
├── ComfyUI/
│   └── custom_nodes/
│       ├── ComfyUI-LatentSyncWrapper/
│       │   ├── checkpoints/
│       │   │   ├── latentsync_unet.pt          ← LatentSync UNet (1.5 GB)
│       │   │   ├── stable_syncnet.pt           ← SyncNet validator (200 MB)
│       │   │   ├── config.json                 ← Model config
│       │   │   ├── vae/
│       │   │   │   ├── diffusion_pytorch_model.safetensors  ← SD VAE (170 MB)
│       │   │   │   └── config.json
│       │   │   └── whisper/
│       │   │       ├── tiny.pt                 ← Whisper Tiny (140 MB)
│       │   │       └── ...
│       │   ├── configs/
│       │   │   └── unet/
│       │   │       └── stage2_512.yaml         ← UNet architecture config
│       │   └── latentsync/                     ← Source code (Python packages)
│       │
│       └── comfyui-reactor-node/               ← CodeFormer backend
│           └── scripts/
│               └── codeformer_infer.py
│
├── MuseTalk/                                   ← MuseTalk repo clone
│   ├── musetalk-env/                           ← Separate Python venv
│   │   └── bin/python
│   ├── scripts/inference.py
│   └── models/                                 ← MuseTalk weights (~3 GB)
│
└── .cache/huggingface/                         ← Auto-cached models
    └── hub/
        ├── models--hexgrad--Kokoro-82M/        ← Kokoro TTS
        └── models--Systran--faster-whisper-large-v3/  ← Caption model
```

### Project-level paths:

```
ai-avatar-video-generation/
├── configs/settings.yaml        ← All pipeline settings
├── data/
│   ├── avatars/avatar.png       ← Default avatar
│   ├── audio/                   ← TTS output WAVs
│   ├── output/                  ← Generated videos
│   ├── captions/                ← SRT files
│   └── temp/                    ← Intermediate files
│       ├── avatar_loop.mp4      ← Looped avatar video
│       ├── _latentsync_runner.py ← Generated subprocess script
│       └── enhance_frames/      ← Extracted frames for enhancement
└── .env                         ← HF_TOKEN for gated models
```

---

## 5. Free Alternatives & Swap Guide

For every model in the pipeline, here are the free, open-source alternatives you can use. All are available on HuggingFace or GitHub.

### Stage 1: Text-to-Speech (currently: Kokoro-82M)

| Alternative | Parameters | Size | License | HuggingFace / Source | Pros | Cons |
|-------------|-----------|------|---------|---------------------|------|------|
| **Kokoro-82M** *(current)* | 82M | 330 MB | MIT | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Tiny, fast, 10 voices, runs on CPU | Limited voice cloning |
| **F5-TTS** | 330M | 1.3 GB | CC-BY-NC-4.0 | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | Zero-shot voice cloning, natural prosody | Larger, non-commercial license |
| **Parler-TTS** | 880M | 3.5 GB | Apache 2.0 | [parler-tts/parler-tts-large-v1](https://huggingface.co/parler-tts/parler-tts-large-v1) | Text-described voice control, commercial OK | Slow on CPU |
| **Coqui XTTS-v2** | 467M | 1.8 GB | MPL 2.0 | [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | Multilingual, voice cloning from 6s | Heavier |
| **Piper** | — | 15–60 MB | MIT | [rhasspy/piper](https://github.com/rhasspy/piper) | Ultra-lightweight, offline | Less natural quality |
| **StyleTTS 2** | — | 200 MB | MIT | [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) | Human-level naturalness, style transfer | Complex setup |
| **VoxCPM2** | — | — | Apache 2.0 | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | Newest, trending on HF | Recently released |
| **OmniVoice** | — | — | Apache 2.0 | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | Multi-task voice model | Recently released |
| **Fish Speech** | 1B | 4 GB | Apache 2.0 | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | Pro-quality, multilingual | Heavy |

**How to swap:** Replace `VoiceGenerator` in `avatarpipeline/voice/kokoro.py` with a new class that exposes the same `generate(text, voice, out_path) → str` interface. The output must be a WAV file (any sample rate — Stage 2 handles resampling).

---

### Stage 3: Lip-Sync (currently: LatentSync 1.6 / MuseTalk 1.5)

| Alternative | Type | Size | License | Source | Pros | Cons |
|-------------|------|------|---------|--------|------|------|
| **LatentSync 1.6** *(current default)* | Diffusion | 7 GB | Apache 2.0 | [ByteDance/LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6) | Best quality, 512px face | Slow, heavy, gated |
| **MuseTalk 1.5** *(current alt)* | GAN | 3 GB | Apache 2.0 | [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) | Faster, lighter | 256px face |
| **SadTalker** | — | 1.5 GB | MIT | [OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker) | Widely used, stable, head motion | Lower lip accuracy |
| **Wav2Lip** | GAN | 400 MB | — | [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Lightweight, fast, pioneer model | Visible artifacts, low res |
| **VideoReTalking** | — | 2 GB | Apache 2.0 | [OpenTalker/video-retalking](https://github.com/OpenTalker/video-retalking) | Good quality, post-process built-in | Complex pipeline |
| **DINet** | — | 500 MB | — | [MRzzm/DINet](https://github.com/MRzzm/DINet) | Deformation-based, lightweight | Less natural |
| **AniPortrait** | Diffusion | 5 GB | Apache 2.0 | [Zejun-Yang/AniPortrait](https://github.com/Zejun-Yang/AniPortrait) | Full head animation, not just lips | Very slow |
| **Hallo** | Diffusion | 8 GB | MIT | [fudan-generative-vision/hallo](https://github.com/fudan-generative-vision/hallo) | High-quality talking head | Heavy, new |
| **EchoMimic** | Diffusion | 6 GB | Apache 2.0 | [BadToBest/EchoMimic](https://github.com/BadToBest/EchoMimic) | Expressive, pose-driven | New, less tested |

**How to swap:** Create a new class in `avatarpipeline/lipsync/` that implements `run(avatar_png, audio_wav, output_path) → str`. The output must be an MP4 with audio. Update `avatarpipeline/pipeline.py` to import your new class.

---

### Stage 4: Face Enhancement (currently: CodeFormer / GFPGAN)

| Alternative | Size | License | Source | Pros | Cons |
|-------------|------|---------|--------|------|------|
| **CodeFormer** *(current 1st)* | 300 MB | — | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | Best fidelity, adjustable weight | Slower |
| **GFPGAN v1.3** *(current 2nd)* | 350 MB | Apache 2.0 | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | Fast, reliable | Less detail control |
| **RestoreFormer++** | 300 MB | — | [wzhouxiff/RestoreFormerPlusPlus](https://github.com/wzhouxiff/RestoreFormerPlusPlus) | Better on severe damage | Complex setup |
| **PMRF** | — | — | [ohayonguy/PMRF](https://huggingface.co/ohayonguy/PMRF_blind_face_image_restoration) | Newest approach, posterior mean | Experimental |
| **Real-ESRGAN (face)** | 65 MB | BSD-3 | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Good at upscaling, lightweight | Not face-specific |
| **DFDNet** | 350 MB | MIT | [csxmli2016/DFDNet](https://github.com/csxmli2016/DFDNet) | Dictionary-based restoration | Older |
| **None (skip)** | 0 | — | `--no-enhance` flag | Fastest, no artifacts | No enhancement |

**How to swap:** Modify `FaceEnhancer` in `avatarpipeline/postprocess/enhancer.py`. The class needs `enhance(video_path, output_path) → str` which processes frames and returns an enhanced MP4.

---

### Stage 6: Speech Recognition / Captions (currently: faster-whisper large-v3)

| Alternative | Parameters | Size | License | Source | Pros | Cons |
|-------------|-----------|------|---------|--------|------|------|
| **faster-whisper large-v3** *(current)* | 1.55B | 3 GB | MIT | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | Best accuracy, fast (CTranslate2) | CPU only (no MPS) |
| **whisper-large-v3-turbo** | 809M | 1.6 GB | MIT | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 50% smaller, nearly same accuracy | Slightly less accurate |
| **Qwen3-ASR** | 1.7B | 3.4 GB | Apache 2.0 | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Trending, multilingual | Newer, less tested |
| **VibeVoice ASR** | 9B | 18 GB | MIT | [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | State-of-the-art Microsoft | Too large for local |
| **Cohere Transcribe** | — | — | — | [CohereLabs/cohere-transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Newest, trending #1 | May require API |
| **faster-whisper small** | 244M | 500 MB | MIT | PyPI: `faster-whisper` | 6× smaller, still decent | Reduced accuracy |
| **Distil-Whisper** | 756M | 1.5 GB | MIT | [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) | 6× faster than Whisper large | Slightly less accurate |
| **Moonshine** | — | 60 MB | MIT | [usefulsensors/moonshine](https://huggingface.co/usefulsensors/moonshine) | Ultra small, edge device | Lower accuracy |

**How to swap:** Change `CaptionGenerator.__init__()` in `avatarpipeline/postprocess/captions.py`. The `model_size` parameter already supports: `tiny`, `base`, `small`, `medium`, `large-v3`. For non-Whisper models, replace the `faster_whisper.WhisperModel` call with your preferred ASR library. Output must be SRT format.

**Quick size/accuracy tradeoff:**
```python
# In captions.py — just change the model_size parameter:
CaptionGenerator(model_size="large-v3")   # 3 GB — best accuracy (current)
CaptionGenerator(model_size="medium")     # 1.5 GB — good accuracy
CaptionGenerator(model_size="small")      # 500 MB — decent accuracy
CaptionGenerator(model_size="base")       # 150 MB — acceptable
CaptionGenerator(model_size="tiny")       # 75 MB — fast but rough
```

---

## 6. Internal Health Checks

Run these checks to verify every component is working:

### Quick Diagnostic Commands

```bash
# 1. Verify all Python imports work
python -c "
from avatarpipeline.voice.kokoro import VoiceGenerator
from avatarpipeline.lipsync.latentsync import LatentSyncInference
from avatarpipeline.lipsync.musetalk import MuseTalkInference
from avatarpipeline.postprocess.enhancer import FaceEnhancer
from avatarpipeline.postprocess.captions import CaptionGenerator
from avatarpipeline.postprocess.assembler import VideoAssembler
from avatarpipeline.pipeline import run_pipeline
print('All imports OK')
"

# 2. Run unit tests (17 tests)
python -m pytest tests/ -v

# 3. Check model checkpoints exist
ls -lh ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/latentsync_unet.pt
ls -lh ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/vae/diffusion_pytorch_model.safetensors
ls -lh ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/whisper/

# 4. Check MuseTalk installation
ls ~/MuseTalk/musetalk-env/bin/python

# 5. Check FFmpeg
ffmpeg -version | head -1

# 6. Check espeak-ng (required by Kokoro)
espeak-ng --version

# 7. Verify avatar exists
ls -lh data/avatars/avatar.png

# 8. End-to-end smoke test (skip heavy stages)
bash scripts/smoke_test.sh --no-enhance --no-captions

# 9. Check HuggingFace token (needed for LatentSync gated model)
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('HF_TOKEN:', 'SET' if os.getenv('HF_TOKEN') else 'MISSING')"

# 10. Check disk space for models
du -sh ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/
du -sh ~/MuseTalk/
du -sh ~/.cache/huggingface/hub/
```

### Troubleshooting Matrix

| Symptom | Check | Fix |
|---------|-------|-----|
| `No module named 'kokoro'` | `pip list \| grep kokoro` | `pip install kokoro>=0.9.4` |
| `espeak-ng not found` | `which espeak-ng` | `brew install espeak-ng` |
| `latentsync_unet.pt not found` | `ls ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/` | `bash install/install_latentsync.sh` |
| `Exit code 1` on LatentSync | Run `_latentsync_runner.py` directly | Check `data/temp/_latentsync_runner.py` for detailed traceback |
| `MPS backend out of memory` | Activity Monitor → GPU memory | Close other GPU apps, use `--no-enhance` |
| `Invalid buffer size: 16 GiB` | Metal buffer limit | Chunked SDPA already patched — ensure runner uses it |
| `Port 7860 in use` | `lsof -i :7860` | `python scripts/run_dashboard.py --port 7861` |
| Black frames in output | Portrait quality | Use clear, front-facing portrait ≥256×256px |
| `SSL: CERTIFICATE_VERIFY_FAILED` | macOS cert issue | `pip install certifi` and set `SSL_CERT_FILE` |
| Caption model slow | Expected | CPU int8 is used (MPS unsupported by CTranslate2) |
| MuseTalk not found | `ls ~/MuseTalk` | `bash install/setup.sh` |

---

## 7. Decision Matrix

Use this to decide which models/settings to use for different scenarios:

### Speed vs Quality

| Scenario | Lip-Sync | Enhancement | Captions | Est. Time (30s video) |
|----------|----------|-------------|----------|-----------------------|
| **Maximum quality** | LatentSync (steps=50) | CodeFormer (0.8) | large-v3 | ~15 min |
| **Balanced** (default) | LatentSync (steps=25) | CodeFormer (0.7) | large-v3 | ~8 min |
| **Quick preview** | MuseTalk | Skip | Skip | ~3 min |
| **Draft / debug** | MuseTalk | Skip | Skip + `--no-captions` | ~2 min |

### Resource Usage

| Model | Peak VRAM | Peak RAM | Disk |
|-------|-----------|----------|------|
| Kokoro TTS | 500 MB | 1 GB | 330 MB |
| LatentSync | 8 GB | 4 GB | 7 GB |
| MuseTalk | 4 GB | 3 GB | 3 GB |
| CodeFormer | 2 GB | 2 GB | 300 MB |
| GFPGAN | 1.5 GB | 1.5 GB | 350 MB |
| faster-whisper large-v3 | 0 (CPU) | 4 GB | 3 GB |

### "I want to replace X" Quick Reference

| If you want to... | Replace this class | With interface | File to modify |
|---|---|---|---|
| Use a different TTS | `VoiceGenerator` | `generate(text, voice, out_path) → str` (WAV) | `avatarpipeline/voice/kokoro.py` |
| Use a different lip-sync | `LatentSyncInference` | `run(avatar_png, audio_wav, output_path) → str` (MP4) | `avatarpipeline/lipsync/latentsync.py` |
| Use a different face restorer | `FaceEnhancer` | `enhance(video_path, output_path) → str` (MP4) | `avatarpipeline/postprocess/enhancer.py` |
| Use a different ASR | `CaptionGenerator` | `transcribe(audio_wav, output_srt) → str` (SRT) | `avatarpipeline/postprocess/captions.py` |
| Change video assembly | `VideoAssembler` | `finalize(video, output, srt) → str` (MP4) | `avatarpipeline/postprocess/assembler.py` |

---

## 8. Data Flow Diagram

Complete data flow showing every intermediate file:

```
INPUT
  ├── data/avatars/avatar.png                    # User-provided portrait
  └── "Hello, I'm your AI avatar."               # User-provided script
       │
       ▼
STAGE 1: VoiceGenerator.generate()
  └── data/audio/output_24k.wav                  # 24 kHz stereo WAV
       │
       ▼
STAGE 2: VoiceGenerator.convert_to_16k()         # FFmpeg resample
  └── data/audio/output.wav                      # 16 kHz mono WAV
       │
       ▼
STAGE 3A: LatentSyncInference.prepare_input_video()
  └── data/temp/avatar_loop.mp4                  # Looped portrait video
       │
STAGE 3B: LatentSyncInference.run()              # Subprocess inference
  ├── data/temp/_latentsync_runner.py             # Generated runner script
  └── data/output/lipsync_<timestamp>.mp4         # Raw lip-synced video
       │
       ▼
STAGE 4: FaceEnhancer.enhance()
  ├── data/temp/enhance_frames/frame_0001.png     # Extracted frames
  ├── data/temp/enhance_frames/frame_0002.png
  ├── ...
  └── data/output/enhanced_<timestamp>.mp4        # Enhanced video
       │
       ▼
STAGE 5: VideoAssembler.add_background()
  └── data/output/composed_<timestamp>.mp4        # Framed 9:16 / 16:9 / 1:1
       │
       ▼
STAGE 5.5: VideoAssembler.add_music()             # Optional
  └── data/output/composed_<timestamp>_music.mp4  # With background music
       │
       ▼
STAGE 6: CaptionGenerator.transcribe()
  └── data/captions/captions_<timestamp>.srt      # Subtitle file
       │
       ▼
STAGE 7: VideoAssembler.finalize()
  └── data/output/studio_<timestamp>.mp4          # FINAL OUTPUT
```

---

## Appendix: Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `HF_TOKEN` | Yes (for LatentSync) | — | HuggingFace token for gated ByteDance/LatentSync-1.6 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Auto-set | `1` | CPU fallback for unsupported MPS ops |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | Auto-set | `0.0` | Prevent MPS OOM crashes |
| `SSL_CERT_FILE` | Auto-set | `certifi.where()` | Fix macOS SSL cert issues |
| `GRADIO_SERVER_PORT` | No | `7860` | Override dashboard port |

## Appendix: Config Reference (`configs/settings.yaml`)

```yaml
# External tool directories
musetalk_dir: "~/MuseTalk"              # MuseTalk installation
comfyui_dir:  "~/ComfyUI"               # ComfyUI with LatentSync wrapper

# Avatar
avatar_path: "data/avatars/avatar.png"   # Default portrait

# Video
default_fps: 25
default_orientation: "9:16"
output_resolution:
  "9:16": "1080x1920"
  "16:9": "1920x1080"
  "1:1":  "1080x1080"

# LatentSync tuning
latentsync:
  inference_steps: 25       # 10–50 (quality vs speed)
  lips_expression: 1.5      # 0.5–3.0 (lip movement intensity)
  input_fps: 25
  face_resolution: 512

# TTS
tts:
  engine: "kokoro"
  default_voice: "af_heart"
  speed: 1.0                # 0.5–2.0
  lang_code: "a"            # 'a' = American, 'b' = British
```
