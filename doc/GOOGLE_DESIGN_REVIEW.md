# Google-Style Design Review: AI Avatar Video Generation

> **What this document is:** A full codebase audit asked from the perspective of
> "how would a Google/Alphabet engineering team design this system?" It covers
> file structure, module responsibilities, what to merge, what to delete, and
> what patterns are missing.  It does **not** say the current code is broken —
> it says how it would evolve to production-grade.

---

## 1. Current State — What Exists

```
ai-avatar-video-generation/
├── avatarpipeline/          # core library
│   ├── __init__.py
│   ├── pipeline.py          # 7-step avatar pipeline
│   ├── lipsync/
│   │   ├── musetalk.py
│   │   └── sadtalker.py
│   ├── narration/
│   │   ├── composer.py      # PDF+JSON → narrated video
│   │   ├── presenter.py     # PDF+JSON → slide presenter w/ lipsync
│   │   ├── slide_renderer.py
│   │   └── validator.py
│   ├── podcast/
│   │   └── composer.py      # multi-speaker podcast video
│   ├── postprocess/
│   │   ├── assembler.py
│   │   ├── captions.py
│   │   └── enhancer.py
│   └── voice/
│       ├── kokoro.py
│       └── mlx_voice.py
├── ui/
│   ├── dashboard.py         # ← CURRENT (3,576 lines)
│   ├── dashboard_v1.py      # ← OBSOLETE (1,563 lines)
│   └── dashboard.py.bak     # ← OBSOLETE (1,153 lines)
├── scripts/
│   ├── build_composites.py  # ← one-off hardcoded script
│   ├── run_dashboard.py
│   ├── run_pipeline.py
│   ├── free_memory.sh
│   └── smoke_test.sh
├── configs/
│   ├── settings.yaml        # main config
│   └── musetalk_avatar.yaml # model-level runtime config
├── data/
│   └── temp/
│       └── _gfpgan_runner.py  # ← Python source living in a data dir
├── tests/
│   ├── test_mlx_voice.py
│   ├── test_narration.py
│   └── test_pipeline.py
├── install/
│   ├── setup.sh
│   └── install_latentsync.sh
├── doc/
│   └── PIPELINE.md
└── requirements.txt
```

---

## 2. Identified Problems — Root Causes

### 2.1 Dead Files That Should Be Deleted

| File | Problem |
|------|---------|
| `ui/dashboard_v1.py` | Superseded by `ui/dashboard.py`. Has image/video gen tabs that were intentionally removed. 1,563 lines of dead code. |
| `ui/dashboard.py.bak` | Backup of an even older version. 1,153 lines of dead code. |
| `data/temp/_gfpgan_runner.py` | Python source code living inside a `data/` directory. Data directories are for runtime artifacts, not source. |

### 2.2 Structural Violations

| Problem | Symptom |
|---------|---------|
| **No `pyproject.toml`** | `requirements.txt` is a flat list; the project is not installable as a package; no metadata, no entry-points, no version pinning strategy. |
| **`scripts/build_composites.py` has hardcoded paths** | References `"20260420-final-version-3"` and `"composite_f57596b2f8a1"` — absolute project-specific one-off code committed to a shared codebase. |
| **Two config files for one system** | `configs/settings.yaml` and `configs/musetalk_avatar.yaml` cover the same concern (model runtime settings). |
| **No interface/protocol layer** | `MuseTalkInference`, `SadTalkerInference`, `VoiceGenerator`, `MlxVoiceStudio` share no common interface. The UI and pipeline code branches on string keys (`"musetalk"`, `"sadtalker_hd"`) rather than typing. |
| **`ui/dashboard.py` is 3,576 lines** | The dashboard file is simultaneously: the UI definition, the pipeline orchestrator, the ffmpeg runner, the config loader, the avatar manager, and the merge utility. |

### 2.3 Code Duplication Across Modules

These utility functions are copy-pasted in multiple files:

```
_audio_duration() / _audio_duration_from_file()
    → avatarpipeline/narration/composer.py
    → avatarpipeline/narration/presenter.py
    → ui/dashboard.py

_video_info() / get_video_info()
    → avatarpipeline/narration/presenter.py
    → avatarpipeline/postprocess/assembler.py

_gen_silence() + _concat_audio() + _normalize_audio()
    → avatarpipeline/narration/composer.py
    (should be shared with assembler.py and podcast/composer.py)

VOICE_CHOICES display name → voice ID mapping
    → ui/dashboard.py  (display names)
    → avatarpipeline/voice/kokoro.py  (IDs as dict)
    → avatarpipeline/voice/mlx_voice.py  (separate dict)
    These should be a single registry.

lipsync engine name → internal key mapping
    → ui/dashboard.py  (e.g. "MuseTalk 1.5 (default)" → "musetalk")
    → avatarpipeline/pipeline.py  (e.g. "musetalk" → import path)
    → avatarpipeline/lipsync/sadtalker.py  (PRESETS dict)
```

### 2.4 Missing Patterns (Google Engineering Standard)

- **No dependency injection.** Engines are instantiated inside pipeline functions; there is no way to swap or mock them without monkey-patching.
- **No structured error hierarchy.** All failures raise generic `RuntimeError` or `FileNotFoundError`; callers cannot distinguish "model not installed" from "ffmpeg failed" from "bad input".
- **No integration tests.** `tests/` has three files covering unit behavior but no end-to-end pipeline tests (e.g. "given a real PDF and JSON, does a full narration video produce an MP4?").
- **`pipeline.py` is not used by the UI.** The Gradio dashboard re-implements the 7-step pipeline inline, meaning there are now two independent pipeline implementations that can drift.

---

## 3. How Google Would Redesign This

### 3.1 File Structure (Google-Style)

```
ai-avatar-video-generation/
│
│   ── Top-level project metadata ────────────────────────────────────
├── pyproject.toml              # replaces requirements.txt
├── README.md
├── LICENSE
│
│   ── Configuration ─────────────────────────────────────────────────
├── configs/
│   └── settings.yaml           # ONE config file — merge musetalk_avatar.yaml in
│
│   ── Static assets ────────────────────────────────────────────────
├── assets/
│
│   ── Runtime data (gitignored) ─────────────────────────────────────
├── data/
│   ├── audio/
│   ├── avatars/
│   ├── captions/
│   ├── images/
│   ├── output/
│   ├── presentations/
│   ├── temp/                   # runtime scratch — NO source files here
│   └── voices/
│
│   ── Core library (the SDK) ─────────────────────────────────────────
├── src/
│   └── avatarpipeline/
│       │
│       ├── __init__.py         # keep path constants; add __all__
│       │
│       ├── core/               # NEW — shared foundations
│       │   ├── __init__.py
│       │   ├── interfaces.py   # Protocol classes for all swappable engines
│       │   ├── config.py       # single validated config loader (replaces scattered yaml.safe_load)
│       │   └── media.py        # ALL shared ffmpeg/ffprobe utilities (no more duplication)
│       │
│       ├── engines/            # rename: lipsync + voice merged here
│       │   ├── __init__.py     # engine registry: name → class
│       │   ├── tts/
│       │   │   ├── __init__.py
│       │   │   ├── kokoro.py   # keep; implement TtsEngine protocol
│       │   │   └── mlx.py      # rename from mlx_voice.py; implement TtsEngine protocol
│       │   └── lipsync/
│       │       ├── __init__.py
│       │       ├── musetalk.py # keep; implement LipsyncEngine protocol
│       │       └── sadtalker.py # keep; implement LipsyncEngine protocol
│       │
│       ├── postprocess/        # keep; no changes needed
│       │   ├── __init__.py
│       │   ├── assembler.py
│       │   ├── captions.py
│       │   └── enhancer.py
│       │
│       ├── pipelines/          # rename: one file per end-to-end pipeline mode
│       │   ├── __init__.py
│       │   ├── avatar.py       # rename from pipeline.py (7-step avatar video)
│       │   ├── podcast.py      # move from podcast/composer.py
│       │   ├── narration.py    # merge narration/composer.py + slide_renderer.py
│       │   ├── presenter.py    # keep narration/presenter.py; pull in slide_renderer
│       │   └── _slide_pdf.py   # internal: PDF → PNG rendering (was slide_renderer.py)
│       │
│       └── narration/          # REMOVED as a top-level package
│           └── validator.py    # → move to pipelines/narration.py or pipelines/_validate.py
│
│   ── Application (the Gradio UI) ───────────────────────────────────
├── app/
│   ├── __init__.py
│   ├── dashboard.py            # single, current file moved here from ui/
│   └── _handlers/              # optional: split the 3,576-line file
│       ├── __init__.py
│       ├── avatar.py           # generate_video(), cancel_generation()
│       ├── narration.py        # generate_narration_video(), validate_narration_files()
│       ├── podcast.py          # generate_podcast()
│       ├── presenter.py        # generate_slide_presenter()
│       └── shared.py           # get_avatar_gallery(), merge_output_videos(), etc.
│
│   ── Launch scripts ─────────────────────────────────────────────────
├── scripts/
│   ├── run_dashboard.py        # keep
│   └── run_pipeline.py         # keep
│
│   ── One-off / project tooling ─────────────────────────────────────
├── tools/
│   └── build_composites.py     # MOVE from scripts/ — it is not a general script
│
│   ── Tests ──────────────────────────────────────────────────────────
├── tests/
│   ├── unit/
│   │   ├── test_config.py       # config loader
│   │   ├── test_media_utils.py  # shared ffmpeg helpers
│   │   ├── test_voice_kokoro.py # rename from test_mlx_voice.py + test_pipeline.py voice section
│   │   ├── test_voice_mlx.py
│   │   ├── test_narration_validator.py  # extracted from test_narration.py
│   │   └── test_assembler.py
│   └── integration/
│       ├── test_avatar_pipeline.py    # full 7-step smoke test
│       ├── test_narration_pipeline.py # PDF + JSON → MP4
│       └── test_podcast_pipeline.py
│
│   ── Installation ─────────────────────────────────────────────────
├── install/
│   ├── setup.sh
│   └── install_latentsync.sh
│
└── doc/
    ├── PIPELINE.md
    └── GOOGLE_DESIGN_REVIEW.md  ← this file
```

---

## 4. File-by-File Decision Map

### DELETE (no migration needed)

| File | Why |
|------|-----|
| `ui/dashboard_v1.py` | Obsolete. All features either merged into `ui/dashboard.py` or intentionally removed. Contains two tabs (Text→Image via FLUX, Text→Video via damo-vilab) that were dropped from the product. No test coverage. |
| `ui/dashboard.py.bak` | Older version of the above. Avatar handling model is weaker (force-crops to 512×512 RGB, discards alpha and aspect ratio). Completely superseded. |
| `data/temp/_gfpgan_runner.py` | Source code does not belong in a runtime data directory. If still needed, move to `src/avatarpipeline/engines/lipsync/` or a dedicated `tools/` path. If not needed (enhancer.py already handles this via subprocess), delete. |

### MERGE

| From | Into | What to consolidate |
|------|------|---------------------|
| `avatarpipeline/narration/composer.py` | `src/avatarpipeline/pipelines/narration.py` | All narration pipeline logic stays; the `slide_renderer.py` import becomes internal. |
| `avatarpipeline/narration/presenter.py` | `src/avatarpipeline/pipelines/presenter.py` | Move as-is; extract `_audio_duration()`, `_video_info()` into `core/media.py`. |
| `avatarpipeline/narration/slide_renderer.py` | `src/avatarpipeline/pipelines/_slide_pdf.py` | An internal helper, not a public API. Prefixed with `_` to signal that. |
| `avatarpipeline/narration/validator.py` | `src/avatarpipeline/pipelines/_validate.py` | Internal; still called by both narration and presenter pipelines. |
| `avatarpipeline/podcast/composer.py` | `src/avatarpipeline/pipelines/podcast.py` | Move unchanged; the `podcast/` sub-package is a single-file package (unnecessary layer). |
| `avatarpipeline/pipeline.py` | `src/avatarpipeline/pipelines/avatar.py` | Rename only; represents the 7-step avatar pipeline specifically. |
| `avatarpipeline/voice/kokoro.py` | `src/avatarpipeline/engines/tts/kokoro.py` | Move; add `TtsEngine` protocol implementation. |
| `avatarpipeline/voice/mlx_voice.py` | `src/avatarpipeline/engines/tts/mlx.py` | Rename (drop `_voice` suffix; the directory already says `tts`). |
| `avatarpipeline/lipsync/musetalk.py` | `src/avatarpipeline/engines/lipsync/musetalk.py` | Move under engines/. |
| `avatarpipeline/lipsync/sadtalker.py` | `src/avatarpipeline/engines/lipsync/sadtalker.py` | Move under engines/. |
| `configs/musetalk_avatar.yaml` | `configs/settings.yaml` | Add a `musetalk:` section to settings.yaml. Two config files for one app is unnecessary. |
| `requirements.txt` | `pyproject.toml` | Standard Python packaging. |
| `ui/dashboard.py` | `app/dashboard.py` | Move to `app/`. Optionally split handlers (see §5 below). |
| `scripts/build_composites.py` | `tools/build_composites.py` | This is a project-specific one-off tool, not a general-purpose script. Move to `tools/`. |

### KEEP (no changes needed)

| File | Notes |
|------|-------|
| `avatarpipeline/__init__.py` | Path constants are clean. Add `__all__`. |
| `avatarpipeline/postprocess/assembler.py` | Clean, single-responsibility. |
| `avatarpipeline/postprocess/captions.py` | Clean. |
| `avatarpipeline/postprocess/enhancer.py` | Clean; backend-detection pattern is fine. |
| `configs/settings.yaml` | Keep after absorbing `musetalk_avatar.yaml`. |
| `scripts/run_dashboard.py` | Clean CLI entry point. |
| `scripts/run_pipeline.py` | Clean CLI entry point. |
| `scripts/smoke_test.sh` | Keep. |
| `scripts/free_memory.sh` | Keep. |
| `install/setup.sh` | Keep. |
| `install/install_latentsync.sh` | Keep. |
| `doc/PIPELINE.md` | Keep. |
| `tests/test_mlx_voice.py` | Keep; move to `tests/unit/test_voice_mlx.py`. |
| `tests/test_narration.py` | Keep; move to `tests/unit/test_narration_validator.py`. |
| `tests/test_pipeline.py` | Keep; move to `tests/unit/`. |

---

## 5. The Big Split: `ui/dashboard.py` → `app/`

The current `dashboard.py` (3,576 lines) mixes five concerns. Google would split it:

```
app/
├── dashboard.py          # Gradio layout only: gr.Blocks, tabs, CSS, THEME, demo.launch()
└── _handlers/
    ├── shared.py         # avatar mgmt, voice gallery, video history, merge, settings
    ├── avatar.py         # generate_video(), cancel_generation(), _build_progress_html()
    ├── narration.py      # generate_narration_video(), validate_narration_files()
    ├── podcast.py        # generate_podcast(), _save_pod_avatar_*()
    └── presenter.py      # generate_slide_presenter()
```

**Why this matters:** Currently, a bug in `generate_podcast()` requires reading and understanding 3,576 lines to locate. With the split, each handler file is ~200–400 lines and covers exactly one mode. The Gradio wiring (`dashboard.py`) becomes purely structural.

---

## 6. The Missing `core/` Layer

Google would add a `core/` package that every other module depends on. Nothing else duplicates what is in `core/`.

### `core/interfaces.py` — Protocol classes

```python
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
```

This means:
- `MuseTalkInference` and `SadTalkerInference` are both `LipsyncEngine` — no `if engine == "musetalk"` branching.
- `VoiceGenerator` and `MlxVoiceStudio` are both `TtsEngine` — no `if tts_engine == "kokoro"` branching.
- Tests can use mock implementations without touching real models.

### `core/media.py` — Deduplicated ffmpeg utilities

Every file that currently calls `ffprobe` or `ffmpeg` directly would import from here:

```python
def audio_duration(path: str | Path) -> float: ...
def video_info(path: str | Path) -> dict: ...    # width, height, fps, duration
def concat_audio(inputs: list[str], output: str) -> None: ...
def generate_silence(duration: float, output: str) -> None: ...
def normalize_to_16k_mono(input: str, output: str) -> None: ...
def resample_audio(input: str, output: str, sample_rate: int, channels: int) -> None: ...
```

Currently these exist in 4+ files. One authoritative location eliminates the risk of divergent behavior.

### `core/config.py` — Single validated config loader

```python
@dataclass(frozen=True)
class PipelineConfig:
    musetalk_dir: Path
    sadtalker_dir: Path
    default_fps: int
    default_voice: str
    lipsync_engine: str
    ...

def load_config(path: Path | None = None) -> PipelineConfig:
    """Load and validate configs/settings.yaml, raise ConfigError on bad values."""
    ...
```

Currently, every module does `yaml.safe_load(cfg_file)` independently and accesses keys with no validation. A typed config dataclass means `KeyError` on missing settings is caught at startup, not mid-pipeline.

---

## 7. Engine Registry Pattern

Google would replace the string-matching in the pipeline with a registry:

### `engines/__init__.py`

```python
from avatarpipeline.engines.tts.kokoro import VoiceGenerator
from avatarpipeline.engines.tts.mlx import MlxVoiceStudio
from avatarpipeline.engines.lipsync.musetalk import MuseTalkInference
from avatarpipeline.engines.lipsync.sadtalker import SadTalkerInference

TTS_REGISTRY: dict[str, type] = {
    "kokoro": VoiceGenerator,
    "mlx":    MlxVoiceStudio,
}

LIPSYNC_REGISTRY: dict[str, type] = {
    "musetalk":    MuseTalkInference,
    "sadtalker":   SadTalkerInference,
    "sadtalker_hd": lambda: SadTalkerInference(preset="sadtalker_hd"),
}

def get_tts_engine(name: str) -> TtsEngine:
    cls = TTS_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown TTS engine '{name}'. Available: {list(TTS_REGISTRY)}")
    return cls()

def get_lipsync_engine(name: str) -> LipsyncEngine:
    ...
```

The pipeline then becomes:
```python
tts = get_tts_engine(config.tts_engine)
lipsync = get_lipsync_engine(config.lipsync_engine)
speech_wav = tts.generate(script, voice=voice_id, out_path=...)
lipsync_mp4 = lipsync.run(avatar_png, speech_wav)
```

No `if/elif` chains on strings in business logic. Adding a new engine = add one entry to the registry.

---

## 8. `pyproject.toml` (replaces `requirements.txt`)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "ai-avatar-video-generation"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "python-dotenv>=1.0.0",
    "kokoro>=0.9.4",
    "mlx-audio>=0.3.0",
    "faster-whisper>=1.0.0",
    "pyyaml",
    "pillow",
    "loguru",
    "numpy",
    "soundfile",
    "gradio>=4.0.0",
    "pymupdf>=1.24.0",
]

[project.optional-dependencies]
japanese = [
    "pykakasi>=2.3.0",
    "pyopenjtalk>=0.4.1",
    "fugashi[unidic-lite]>=1.3.2",
    "mojimoji>=0.0.13",
]

[project.scripts]
avatar-pipeline = "scripts.run_pipeline:main"
avatar-dashboard = "scripts.run_dashboard:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Benefits:
- `pip install -e .` works; modules importable without `sys.path` hacks in scripts.
- Optional Japanese deps are not forced on all users.
- `avatar-dashboard` becomes a proper CLI command.
- All scripts that currently do `sys.path.insert(0, str(ROOT))` can remove that line.

---

## 9. Test Reorganization

| Current | Proposed | Coverage |
|---------|----------|---------|
| `tests/test_pipeline.py` | `tests/unit/test_voice_kokoro.py` | VoiceGenerator unit tests |
| `tests/test_pipeline.py` | `tests/unit/test_assembler.py` | VideoAssembler unit tests |
| `tests/test_pipeline.py` | `tests/unit/test_config.py` | Config loading + validation |
| `tests/test_mlx_voice.py` | `tests/unit/test_voice_mlx.py` | MlxVoiceStudio unit tests |
| `tests/test_narration.py` | `tests/unit/test_narration_validator.py` | Validator unit tests |
| *(missing)* | `tests/unit/test_media_utils.py` | `core/media.py` — ffmpeg helpers |
| *(missing)* | `tests/integration/test_avatar_pipeline.py` | Full 7-step pipeline smoke test |
| *(missing)* | `tests/integration/test_narration_pipeline.py` | PDF + JSON → MP4 smoke test |
| *(missing)* | `tests/integration/test_podcast_pipeline.py` | Script → podcast MP4 smoke test |

Google requires integration tests that exercise the complete path ("does it produce an MP4?") separately from unit tests ("does this function return the right value?"). The current test suite has only unit-style tests.

---

## 10. Configs: Merge the Two YAML Files

`configs/musetalk_avatar.yaml` only contains:
```yaml
video_path: ""
audio_path: ""
bbox_shift: 0
fps: 25
output_dir: ""
use_float16: true
```

These are all MuseTalk runtime flags set dynamically at inference time. They belong in `settings.yaml` under a `musetalk:` key:

```yaml
# In settings.yaml — add this block:
musetalk:
  fps: 25
  use_float16: true
  default_bbox_shift: 0
  default_batch_size: 8
```

`musetalk_avatar.yaml` is deleted. `MuseTalkInference.__init__` reads from `settings.yaml`.

---

## 11. Priority Order (What to Do First)

The changes are ordered by impact-to-effort ratio:

### Immediate (delete/move with no code changes)
1. **Delete** `ui/dashboard_v1.py` and `ui/dashboard.py.bak`
2. **Move** `data/temp/_gfpgan_runner.py` → evaluate if still used; if yes, move to `tools/`; if no, delete
3. **Move** `scripts/build_composites.py` → `tools/build_composites.py`
4. **Merge** `configs/musetalk_avatar.yaml` into `configs/settings.yaml`

### Short-term (refactor, low risk)
5. **Create** `avatarpipeline/core/media.py` — consolidate all duplicate `_audio_duration`, `_video_info`, `_gen_silence`, `_concat_audio`, `_normalize_audio` into one module
6. **Replace** `requirements.txt` with `pyproject.toml` and `src/` layout
7. **Add** `src/avatarpipeline/engines/__init__.py` with the engine registry

### Medium-term (structural)
8. **Merge** `podcast/composer.py` → `pipelines/podcast.py` (collapse single-file package)
9. **Add** `core/interfaces.py` protocol definitions; implement on each engine class
10. **Split** `ui/dashboard.py` into `app/dashboard.py` + `app/_handlers/`

### Long-term (new infrastructure)
11. **Add** `core/config.py` typed config loader
12. **Add** `tests/integration/` suite
13. **Make** `pipelines/avatar.py` the single pipeline used by both CLI and UI (eliminate the duplicate 7-step inline in the dashboard)

---

## 12. What Would NOT Change

Google's engineering principle is: **don't fix what isn't broken**. The following are already well-designed:

- **`avatarpipeline/__init__.py`** — path constant pattern is clean; the auto-mkdir on import is pragmatic for a local tool
- **`avatarpipeline/postprocess/`** — clean single-responsibility classes
- **`avatarpipeline/narration/validator.py`** — strong flexible key normalization; `ValidationResult` dataclass is the right pattern
- **`avatarpipeline/lipsync/sadtalker.py`** — `PRESETS` dict pattern is a good approach for preset configurations
- **`scripts/run_dashboard.py`** — `_find_free_port()` is a clean utility; argument parsing is clean
- **The 7-step pipeline structure itself** — the pipeline graph is well-reasoned; it is the implementation boundaries around it that need work
- **The generator (yield) pattern** in `narration/composer.py` for streaming progress — this is the right architecture for long-running pipeline tasks
