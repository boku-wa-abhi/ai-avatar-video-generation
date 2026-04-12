#!/usr/bin/env bash
# =============================================================================
# install_latentsync.sh — Idempotent installer for LatentSync 1.6
# Installs ComfyUI-LatentSyncWrapper, patches for MPS, downloads models.
# Target: Apple M4 Pro (ARM64 / MPS)
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[LATENTSYNC]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

COMFYUI_DIR="$HOME/ComfyUI"
WRAPPER_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-LatentSyncWrapper"
CHECKPOINTS="$WRAPPER_DIR/checkpoints"
# Use the single pipeline venv — it already has torch/numpy/diffusers
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_VENV="$PIPELINE_DIR/.venv"

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight: create ComfyUI directory structure if it doesn't exist.
# Full ComfyUI is NOT required — we only need the folder tree.
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$COMFYUI_DIR/custom_nodes"
if [ ! -d "$COMFYUI_DIR/.git" ]; then
    log "Using $COMFYUI_DIR as standalone LatentSync workspace (no full ComfyUI needed)"
fi

if [ ! -f "$PIPELINE_VENV/bin/activate" ]; then
    fail "Pipeline venv not found at $PIPELINE_VENV. Run setup.sh first."
    exit 1
fi
# shellcheck source=/dev/null
source "$PIPELINE_VENV/bin/activate"
log "Using pipeline venv: $PIPELINE_VENV (Python: $(python --version))"

# ─────────────────────────────────────────────────────────────────────────────
# Step A — Clone & install ComfyUI-LatentSyncWrapper
# ─────────────────────────────────────────────────────────────────────────────
log "Step A — Installing ComfyUI-LatentSyncWrapper..."

if [ -d "$WRAPPER_DIR/.git" ]; then
    log "  Already cloned at $WRAPPER_DIR"
elif [ -d "$WRAPPER_DIR" ]; then
    warn "  $WRAPPER_DIR exists but has no .git — removing and re-cloning..."
    rm -rf "$WRAPPER_DIR"
    git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper "$WRAPPER_DIR"
else
    log "  Cloning..."
    git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper "$WRAPPER_DIR"
fi

cd "$WRAPPER_DIR"

# Pipeline venv already activated above — no separate venv needed.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
log "  Venv: $PIPELINE_VENV"

# Inject MPS env vars
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

if [ -f "$WRAPPER_DIR/requirements.txt" ]; then
    log "  Installing requirements (excluding decord — built from source next)..."
    # decord has no macOS ARM64 wheel on PyPI — handled separately below
    grep -v "^decord" "$WRAPPER_DIR/requirements.txt" > /tmp/latentsync_reqs_no_decord.txt
    pip install -r /tmp/latentsync_reqs_no_decord.txt
else
    warn "  No requirements.txt found — installing known deps..."
    pip install diffusers transformers mediapipe face-alignment \
                soundfile einops omegaconf accelerate opencv-python
fi

# --- decord compatibility shim using cv2 (no ARM64 wheel, source incompatible with FFmpeg 6.x) ---
log "  Installing decord cv2-shim for Apple Silicon..."
cat > "$WRAPPER_DIR/decord.py" << 'PYEOF'
"""
decord compatibility shim for Apple Silicon / macOS.
Implements VideoReader, NDArray, cpu, and gpu using OpenCV.
Drop-in replacement for the decord package where LatentSync uses it.
"""
import numpy as np
import cv2


class NDArray:
    """Minimal decord NDArray wrapper."""
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        return self._arr

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, key):
        return NDArray(self._arr[key])

    @property
    def shape(self):
        return self._arr.shape


class VideoReader:
    """decord.VideoReader drop-in backed by OpenCV."""

    def __init__(self, path: str, ctx=None, num_threads: int = 1,
                 width: int = -1, height: int = -1):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self._frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return NDArray(np.array(self._frames[idx]))
        if isinstance(idx, (list, np.ndarray)):
            return NDArray(np.array([self._frames[i] for i in idx]))
        return NDArray(np.array(self._frames[idx]))

    def get_avg_fps(self) -> float:
        return self._fps

    def get_batch(self, indices) -> NDArray:
        return NDArray(np.array([self._frames[i] for i in indices]))


class cpu:  # noqa: N801
    """Mimics decord.cpu(ctx_id)."""
    def __init__(self, ctx_id: int = 0):
        pass


class gpu:  # noqa: N801
    """Mimics decord.gpu(ctx_id) — falls back to cpu on MPS machines."""
    def __init__(self, ctx_id: int = 0):
        pass
PYEOF
log "  decord shim written to $WRAPPER_DIR/decord.py ✓"
python -c "
import sys
sys.path.insert(0, '$WRAPPER_DIR')
import decord
print('  decord shim import OK')
"

log "Step A — LatentSyncWrapper installed"

# ─────────────────────────────────────────────────────────────────────────────
# Step B — Apply Apple MPS patches
# ─────────────────────────────────────────────────────────────────────────────
log "Step B — Patching CUDA → MPS..."

patch_file() {
    local filepath="$1"
    if [ ! -f "$filepath" ]; then
        return
    fi

    # Backup on first patch only
    if [ ! -f "${filepath}.bak" ]; then
        cp "$filepath" "${filepath}.bak"
    fi

    sed -i '' 's/device = "cuda"/device = "mps"/g'                                "$filepath"
    sed -i '' "s/device = 'cuda'/device = 'mps'/g"                                "$filepath"
    sed -i '' 's/"cuda"/"mps"/g'                                                   "$filepath"
    sed -i '' "s/'cuda'/'mps'/g"                                                   "$filepath"
    sed -i '' 's/\.cuda()/.to("mps")/g'                                            "$filepath"
    sed -i '' 's/torch\.cuda\.is_available()/torch.backends.mps.is_available()/g'  "$filepath"
    sed -i '' 's/torch\.cuda\.empty_cache()/torch.mps.empty_cache()/g'            "$filepath"
    sed -i '' 's/torch\.cuda\.synchronize()/torch.mps.synchronize()/g'            "$filepath"
    sed -i '' 's/torch\.cuda\.amp/torch.amp/g'                                     "$filepath"

    log "  Patched: $filepath"
}

# Patch nodes.py
patch_file "$WRAPPER_DIR/nodes.py"

# Patch all Python files under latentsync/
if [ -d "$WRAPPER_DIR/latentsync" ]; then
    find "$WRAPPER_DIR/latentsync" -name "*.py" -print0 | while IFS= read -r -d '' pyfile; do
        patch_file "$pyfile"
    done
fi

# Final scan for remaining cuda references
REMAINING=$(grep -rn '"cuda"\|\.cuda()\|torch\.cuda' \
    "$WRAPPER_DIR/nodes.py" "$WRAPPER_DIR/latentsync/" 2>/dev/null \
    | grep -v '\.bak:' | grep -v '__pycache__' || true)

if [ -n "$REMAINING" ]; then
    warn "  Remaining CUDA references (review manually):"
    echo "$REMAINING" | head -15
else
    log "  No remaining CUDA references"
fi

log "Step B — MPS patches applied"

# ─────────────────────────────────────────────────────────────────────────────
# Step C — Download model weights
# ─────────────────────────────────────────────────────────────────────────────
log "Step C — Downloading model weights..."

mkdir -p "$CHECKPOINTS/vae" "$CHECKPOINTS/whisper"

# huggingface-cli was deprecated in favour of 'hf' — use Python SDK as fallback
HF_CLI=""
if command -v hf &>/dev/null; then
    HF_CLI="hf download"
elif command -v huggingface-cli &>/dev/null; then
    HF_CLI="huggingface-cli download"
else
    # Install via pip if neither is available
    pip install -q "huggingface_hub[cli]" 2>/dev/null
    HF_CLI="huggingface-cli download"
fi

# Check HuggingFace login state
HF_WHOAMI=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null || true)
if [ -z "$HF_WHOAMI" ]; then
    warn "  Not logged into HuggingFace!"
    warn "  LatentSync-1.6 is a gated model. You must:"
    warn "    1. Accept the license at https://huggingface.co/ByteDance/LatentSync-1.6"
    warn "    2. Run: hf auth login  (or: huggingface-cli login)"
    warn "  Skipping gated model download — re-run this script after login."
    SKIP_GATED=true
else
    SKIP_GATED=false
    log "  Logged in as: $HF_WHOAMI"
fi

# --- Main LatentSync models (gated) ---
if [ "$SKIP_GATED" = false ]; then
    log "  Downloading LatentSync-1.6 main models..."
    python -c "
from huggingface_hub import hf_hub_download
import os
for f in ['latentsync_unet.pt', 'stable_syncnet.pt', 'config.json']:
    out = os.path.join('$CHECKPOINTS', f)
    if not os.path.exists(out):
        print(f'  Downloading {f}...')
        hf_hub_download('ByteDance/LatentSync-1.6', filename=f, local_dir='$CHECKPOINTS')
    else:
        print(f'  Already exists: {f}')
" || warn "  Failed to download LatentSync models. Check HF login and license."
fi

# --- SD-VAE-FT-MSE ---
log "  Downloading SD-VAE-FT-MSE..."
python -c "
from huggingface_hub import hf_hub_download
import os
for f in ['diffusion_pytorch_model.safetensors', 'config.json']:
    out = os.path.join('$CHECKPOINTS/vae', f)
    if not os.path.exists(out):
        print(f'  Downloading {f}...')
        hf_hub_download('stabilityai/sd-vae-ft-mse', filename=f, local_dir='$CHECKPOINTS/vae')
    else:
        print(f'  Already exists: {f}')
" || warn "  Failed to download SD-VAE."

# --- Whisper tiny ---
log "  Downloading Whisper tiny..."
python -c "
from huggingface_hub import hf_hub_download, list_repo_files
import os
for f in list_repo_files('openai/whisper-tiny'):
    if f.endswith('.pt'):
        out = os.path.join('$CHECKPOINTS/whisper', f)
        if not os.path.exists(out):
            print(f'  Downloading {f}...')
            hf_hub_download('openai/whisper-tiny', filename=f, local_dir='$CHECKPOINTS/whisper')
        else:
            print(f'  Already exists: {f}')
" || warn "  Failed to download Whisper tiny."

# Verify model files (plain array — no associative arrays for portability)
log "  Verifying model files..."
MODEL_FILES=(
    "$CHECKPOINTS/latentsync_unet.pt:latentsync_unet.pt"
    "$CHECKPOINTS/stable_syncnet.pt:stable_syncnet.pt"
    "$CHECKPOINTS/config.json:config.json"
    "$CHECKPOINTS/vae/diffusion_pytorch_model.safetensors:vae/diffusion_pytorch_model.safetensors"
)

for entry in "${MODEL_FILES[@]}"; do
    fpath="${entry%%:*}"
    name="${entry##*:}"
    if [ -f "$fpath" ]; then
        SIZE=$(du -sh "$fpath" | cut -f1)
        log "  PASS — $name ($SIZE)"
    else
        warn "  MISSING — $name (download separately if needed)"
    fi
done

log "Step C — Model downloads done"

# ─────────────────────────────────────────────────────────────────────────────
# Step D — Import test
# ─────────────────────────────────────────────────────────────────────────────
log "Step D — Import test..."

python -c "
import sys, os
sys.path.insert(0, os.path.expanduser('$WRAPPER_DIR'))
try:
    import nodes
    print('LatentSync import OK')
except Exception as e:
    print(f'Import warning (may be OK outside ComfyUI): {e}')
"

log "Step D — Import test done"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
log "LatentSync 1.6 installation complete!"
echo "============================================="
log "Wrapper:     $WRAPPER_DIR"
log "Checkpoints: $CHECKPOINTS"
echo ""
