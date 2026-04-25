#!/usr/bin/env bash
# =============================================================================
# setup.sh — Idempotent setup for AI Avatar Pipeline (Phase 1: MuseTalk)
# Safe to re-run. Skips already-completed steps.
# Target: Apple M4 Pro (ARM64 / MPS)
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

# Load .env if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$PIPELINE_DIR/.env" ]; then
    set -o allexport
    # shellcheck source=/dev/null
    source "$PIPELINE_DIR/.env"
    set +o allexport
    log "Loaded .env"
fi

MUSETALK_DIR="$HOME/MuseTalk"
VENV_DIR="$MUSETALK_DIR/musetalk-env"
PIPELINE_VENV="$PIPELINE_DIR/.venv"

log "Pipeline dir:    $PIPELINE_DIR"
log "Pipeline venv:   $PIPELINE_VENV"
log "MuseTalk dir:    $MUSETALK_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Step A — Install system dependencies via Homebrew
# ─────────────────────────────────────────────────────────────────────────────
log "Step A — Checking system dependencies..."

if ! command -v brew &>/dev/null; then
    fail "Homebrew not found. Install from https://brew.sh first."
    exit 1
fi

for pkg in python@3.10 git ffmpeg uv; do
    if brew list "$pkg" &>/dev/null; then
        log "  $pkg — already installed ✓"
    else
        log "  Installing $pkg..."
        brew install "$pkg"
    fi
done

# Ensure python3.10 is on PATH (Homebrew may not auto-link it)
if ! command -v python3.10 &>/dev/null; then
    export PATH="$(brew --prefix python@3.10)/bin:$PATH"
fi

log "Step A — System deps ready ✓"

# ─────────────────────────────────────────────────────────────────────────────
# Step B — Clone MuseTalk
# ─────────────────────────────────────────────────────────────────────────────
log "Step B — Checking MuseTalk repo..."

if [ -d "$MUSETALK_DIR/.git" ]; then
    log "  MuseTalk already cloned at $MUSETALK_DIR ✓"
elif [ -d "$MUSETALK_DIR" ]; then
    # Directory exists but is not a git repo (e.g. a stub we created earlier)
    warn "  $MUSETALK_DIR exists but has no .git — removing stub and re-cloning..."
    rm -rf "$MUSETALK_DIR"
    git clone https://github.com/TMElyralab/MuseTalk "$MUSETALK_DIR"
else
    log "  Cloning MuseTalk..."
    git clone https://github.com/TMElyralab/MuseTalk "$MUSETALK_DIR"
fi

log "Step B — MuseTalk repo ready ✓"

# ─────────────────────────────────────────────────────────────────────────────
# Step C — Create Python 3.10 venv
# ─────────────────────────────────────────────────────────────────────────────
log "Step C — Setting up Python 3.10 venv..."

if [ -d "$VENV_DIR" ]; then
    log "  Venv already exists at $VENV_DIR ✓"
else
    log "  Creating venv with uv..."
    cd "$MUSETALK_DIR"
    uv venv --python 3.10 musetalk-env
fi

# Inject MPS env vars (idempotent — only add if not already present)
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
if ! grep -q "PYTORCH_ENABLE_MPS_FALLBACK" "$ACTIVATE_SCRIPT" 2>/dev/null; then
    {
        echo ''
        echo '# Apple MPS fallback for PyTorch'
        echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1'
        echo 'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0'
    } >> "$ACTIVATE_SCRIPT"
    log "  Added MPS env vars to activate script ✓"
else
    log "  MPS env vars already set ✓"
fi

# Activate the venv for remaining steps
# shellcheck source=/dev/null
source "$ACTIVATE_SCRIPT"
log "Step C — Venv ready (Python: $(python --version)) ✓"

# ─────────────────────────────────────────────────────────────────────────────
# Step D — Install Python packages (exact order matters for mmlab stack)
# ─────────────────────────────────────────────────────────────────────────────
log "Step D — Installing Python packages..."

cd "$MUSETALK_DIR"

# setuptools/wheel must come first — mmcv's build backend requires pkg_resources
log "  Installing setuptools and wheel (required by mmcv build)..."
uv pip install --upgrade setuptools wheel pip

log "  Installing PyTorch stack..."
uv pip install torch torchvision torchaudio

log "  Installing openmim..."
uv pip install openmim

# mim lives in the venv bin — ensure it's on PATH after activation
MIM_BIN="$VENV_DIR/bin/mim"

log "  Installing mmengine..."
"$MIM_BIN" install mmengine

# ── mmcv ─────────────────────────────────────────────────────────────────────
# The mmcv==2.0.1 PyPI sdist is published under the name mmcv-lite (CPU-only,
# no CUDA ops). Install it directly as mmcv-lite — ships as a pre-built wheel.
log "  Installing mmcv-lite==2.0.1 (CPU/MPS compatible, pre-built wheel)..."
uv pip install "mmcv-lite==2.0.1"

# ── xtcocotools (mmpose dependency) ──────────────────────────────────────────
# The PyPI sdist is broken (missing _mask.c). Build from GitHub with Cython.
log "  Installing Cython (required to build xtcocotools)..."
uv pip install cython
log "  Installing xtcocotools from source (with Cython + no build isolation)..."
pip install "git+https://github.com/jin-s13/xtcocoapi.git" --no-build-isolation

# ── mmdet / mmpose ────────────────────────────────────────────────────────────
log "  Installing mmdet==3.1.0..."
"$MIM_BIN" install "mmdet==3.1.0"

# mmpose has legacy deps (chumpy, xtcocotools) that need --no-build-isolation.
log "  Installing chumpy (legacy mmpose dep, needs --no-build-isolation)..."
pip install chumpy --no-build-isolation
log "  Installing mmpose==1.1.0..."
pip install "mmpose==1.1.0" --no-build-isolation

# MuseTalk's own requirements
if [ -f "$MUSETALK_DIR/requirements.txt" ]; then
    log "  Installing MuseTalk requirements.txt..."
    uv pip install -r "$MUSETALK_DIR/requirements.txt"
else
    warn "  No requirements.txt in MuseTalk repo — skipping"
fi

# Pipeline package
log "  Installing avatar-pipeline package..."
uv pip install -e "$PIPELINE_DIR[japanese]"

log "Step D — Packages installed ✓"

# ─────────────────────────────────────────────────────────────────────────────
# Step D2 — Create/update pipeline venv in the repo
# ─────────────────────────────────────────────────────────────────────────────
log "Step D2 — Setting up pipeline venv at $PIPELINE_VENV..."

if [ -d "$PIPELINE_VENV" ]; then
    log "  Pipeline venv already exists ✓"
else
    log "  Creating pipeline venv with uv..."
    uv venv --python 3.10 "$PIPELINE_VENV"
fi

# Inject MPS env vars (idempotent)
PIPELINE_ACTIVATE="$PIPELINE_VENV/bin/activate"
if ! grep -q "PYTORCH_ENABLE_MPS_FALLBACK" "$PIPELINE_ACTIVATE" 2>/dev/null; then
    printf '\n# Apple MPS fallback for PyTorch\nexport PYTORCH_ENABLE_MPS_FALLBACK=1\nexport PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0\n' >> "$PIPELINE_ACTIVATE"
    log "  Added MPS env vars to pipeline venv ✓"
else
    log "  MPS env vars already set ✓"
fi

log "  Installing pipeline package into pipeline venv..."
uv pip install --python "$PIPELINE_VENV/bin/python" -e "$PIPELINE_DIR[japanese]"

log "Step D2 — Pipeline venv ready (Python: $($PIPELINE_VENV/bin/python --version)) ✓"

# ─────────────────────────────────────────────────────────────────────────────
# Step E — Download model weights
# ─────────────────────────────────────────────────────────────────────────────
log "Step E — Downloading model weights..."

cd "$MUSETALK_DIR"

if [ -f "download_weights.sh" ]; then
    chmod +x download_weights.sh
    log "  Running download_weights.sh..."
    ./download_weights.sh
else
    warn "  download_weights.sh not found — attempting manual download via huggingface_hub..."
    mkdir -p models/musetalkV15 models/dwpose models/face-parse-bisent models/sd-vae models/whisper

    python - <<'PYEOF'
import os, sys, urllib.request
try:
    from huggingface_hub import hf_hub_download
    REPO = "TMElyralab/MuseTalk"
    FILES = [
        "models/musetalkV15/unet.pth",
        "models/musetalkV15/musetalk.json",
        "models/dwpose/dw-ll_ucoco_384.pth",
        "models/face-parse-bisent/resnet18-5c106cde.pth",
        "models/sd-vae/diffusion_pytorch_model.bin",
    ]
    for f in FILES:
        if os.path.exists(f):
            print(f"  Already exists: {f}")
            continue
        print(f"  Downloading {f} ...")
        hf_hub_download(repo_id=REPO, filename=f, local_dir=".")
except ImportError:
    print("  huggingface_hub not installed — skipping HF downloads")

# Whisper tiny (direct URL)
whisper_path = "models/whisper/tiny.pt"
if not os.path.exists(whisper_path):
    print("  Downloading Whisper tiny.pt ...")
    urllib.request.urlretrieve(
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        whisper_path,
    )
PYEOF
fi

# Verify model weights
log "  Verifying model weights..."
MODEL_FILES=(
    "models/musetalkV15/unet.pth"
    "models/musetalkV15/musetalk.json"
    "models/dwpose/dw-ll_ucoco_384.pth"
    "models/face-parse-bisent/resnet18-5c106cde.pth"
    "models/sd-vae/diffusion_pytorch_model.bin"
    "models/whisper/tiny.pt"
)

ALL_PASS=true
for f in "${MODEL_FILES[@]}"; do
    if [ -f "$MUSETALK_DIR/$f" ]; then
        log "  PASS — $f"
    else
        fail "  FAIL — $f NOT FOUND"
        ALL_PASS=false
    fi
done

if [ "$ALL_PASS" = true ]; then
    log "Step E — All model weights verified ✓"
else
    warn "Step E — Some model weights missing. Download them manually if needed."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step F — Apply Apple MPS patches to MuseTalk source
# ─────────────────────────────────────────────────────────────────────────────
log "Step F — Applying MPS patches to MuseTalk source..."

patch_file() {
    local filepath="$1"
    if [ ! -f "$filepath" ]; then
        warn "  File not found, skipping: $filepath"
        return
    fi

    # Backup on first patch only
    if [ ! -f "${filepath}.bak" ]; then
        cp "$filepath" "${filepath}.bak"
    fi

    sed -i '' 's/device = "cuda"/device = "mps"/g'                          "$filepath"
    sed -i '' "s/device = 'cuda'/device = 'mps'/g"                          "$filepath"
    sed -i '' 's/\.cuda()/.to("mps")/g'                                      "$filepath"
    sed -i '' 's/torch\.cuda\.is_available()/torch.backends.mps.is_available()/g' "$filepath"
    sed -i '' 's/torch\.cuda\.empty_cache()/torch.mps.empty_cache()/g'      "$filepath"
    sed -i '' 's/torch\.cuda\.synchronize()/torch.mps.synchronize()/g'      "$filepath"
    sed -i '' 's/torch\.cuda\.amp/torch.amp/g'                               "$filepath"

    log "  Patched: $(basename "$filepath")"
}

FILES_TO_PATCH=(
    "$MUSETALK_DIR/musetalk/utils/utils.py"
    "$MUSETALK_DIR/scripts/inference.py"
)

for f in "${FILES_TO_PATCH[@]}"; do
    patch_file "$f"
done

# Find and patch any remaining files with cuda references
REMAINING=$(grep -rln '"cuda"\|\.cuda()\|torch\.cuda' \
    "$MUSETALK_DIR/musetalk/" "$MUSETALK_DIR/scripts/" 2>/dev/null \
    | grep -v '\.bak$' | grep -v '__pycache__' || true)

if [ -n "$REMAINING" ]; then
    warn "  Additional files with CUDA references — auto-patching..."
    while IFS= read -r extra_file; do
        patch_file "$extra_file"
    done <<< "$REMAINING"
fi

# Final scan for any remaining CUDA references
STILL_REMAINING=$(grep -rn '"cuda"\|\.cuda()\|torch\.cuda' \
    "$MUSETALK_DIR/musetalk/" "$MUSETALK_DIR/scripts/" 2>/dev/null \
    | grep -v '\.bak:' | grep -v '__pycache__' || true)

if [ -n "$STILL_REMAINING" ]; then
    warn "  Remaining CUDA references (review manually):"
    echo "$STILL_REMAINING" | head -10
else
    log "  No remaining CUDA references ✓"
fi

log "Step F — MPS patches applied ✓"

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
log "Phase 1 setup complete!"
echo "============================================="
log "Activate the pipeline venv with:"
log "  source $PIPELINE_VENV/bin/activate"
log ""
log "Activate the MuseTalk venv with:"
log "  source $VENV_DIR/bin/activate"
log ""
log "Run a smoke test with:"
log "  cd $PIPELINE_DIR && bash test_musetalk.sh"
echo ""
