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

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -d "$COMFYUI_DIR" ]; then
    fail "ComfyUI not found at $COMFYUI_DIR. Install ComfyUI first."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step A — Clone & install ComfyUI-LatentSyncWrapper
# ─────────────────────────────────────────────────────────────────────────────
log "Step A — Installing ComfyUI-LatentSyncWrapper..."

mkdir -p "$COMFYUI_DIR/custom_nodes"

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

# Activate a venv for LatentSync deps. Use ComfyUI's venv if it exists,
# otherwise create one inside the wrapper dir.
if [ -f "$COMFYUI_DIR/.venv/bin/activate" ]; then
    log "  Using ComfyUI venv: $COMFYUI_DIR/.venv"
    # shellcheck source=/dev/null
    source "$COMFYUI_DIR/.venv/bin/activate"
elif [ -f "$COMFYUI_DIR/venv/bin/activate" ]; then
    log "  Using ComfyUI venv: $COMFYUI_DIR/venv"
    # shellcheck source=/dev/null
    source "$COMFYUI_DIR/venv/bin/activate"
else
    warn "  No ComfyUI venv found — creating one in $WRAPPER_DIR/.venv"
    uv venv --python 3.10 "$WRAPPER_DIR/.venv"
    # shellcheck source=/dev/null
    source "$WRAPPER_DIR/.venv/bin/activate"
fi

# Inject MPS env vars
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

if [ -f "$WRAPPER_DIR/requirements.txt" ]; then
    log "  Installing requirements.txt..."
    pip install -r "$WRAPPER_DIR/requirements.txt"
else
    warn "  No requirements.txt found — installing known deps..."
    pip install diffusers transformers mediapipe face-alignment decord \
                soundfile einops omegaconf accelerate
fi

# Also ensure huggingface-cli is available for model downloads
pip install huggingface_hub 2>/dev/null

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

# Check HuggingFace login
if ! huggingface-cli whoami &>/dev/null; then
    warn "  Not logged into HuggingFace!"
    warn "  LatentSync-1.6 is a gated model. You must:"
    warn "    1. Accept the license at https://huggingface.co/ByteDance/LatentSync-1.6"
    warn "    2. Run: huggingface-cli login"
    warn "  Skipping gated model download — you can re-run this script after login."
    SKIP_GATED=true
else
    SKIP_GATED=false
    HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
    log "  Logged in as: $HF_USER"
fi

# --- Main LatentSync models (gated) ---
if [ "$SKIP_GATED" = false ]; then
    log "  Downloading LatentSync-1.6 main models..."
    huggingface-cli download ByteDance/LatentSync-1.6 \
        latentsync_unet.pt stable_syncnet.pt config.json \
        --local-dir "$CHECKPOINTS/" || {
        warn "  Failed to download LatentSync models. Check HF login and license."
    }
fi

# --- SD-VAE-FT-MSE ---
log "  Downloading SD-VAE-FT-MSE..."
huggingface-cli download stabilityai/sd-vae-ft-mse \
    diffusion_pytorch_model.safetensors config.json \
    --local-dir "$CHECKPOINTS/vae/" || {
    warn "  Failed to download SD-VAE."
}

# --- Whisper tiny ---
log "  Downloading Whisper tiny..."
huggingface-cli download openai/whisper-tiny \
    --include "*.pt" \
    --local-dir "$CHECKPOINTS/whisper/" || {
    warn "  Failed to download Whisper tiny."
}

# Verify model files
log "  Verifying model files..."
declare -A MODEL_FILES=(
    ["latentsync_unet.pt"]="$CHECKPOINTS/latentsync_unet.pt"
    ["stable_syncnet.pt"]="$CHECKPOINTS/stable_syncnet.pt"
    ["config.json"]="$CHECKPOINTS/config.json"
    ["vae/diffusion_pytorch_model.safetensors"]="$CHECKPOINTS/vae/diffusion_pytorch_model.safetensors"
)

for name in "${!MODEL_FILES[@]}"; do
    fpath="${MODEL_FILES[$name]}"
    if [ -f "$fpath" ]; then
        SIZE=$(du -sh "$fpath" | cut -f1)
        log "  PASS — $name ($SIZE)"
    else
        fail "  FAIL — $name NOT FOUND"
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
