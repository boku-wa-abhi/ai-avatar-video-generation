#!/usr/bin/env bash
# free_memory.sh — Remove LatentSync 1.6 and free ~7 GB disk space.
set -euo pipefail

echo "═══════════════════════════════════════════════"
echo " LatentSync 1.6 Cleanup"
echo "═══════════════════════════════════════════════"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BEFORE=$(df -k ~ | awk 'NR==2{print $4}')

safe_rm() {
    if [ -e "$1" ]; then
        rm -rf "$1"
        echo "  ✅ Removed: $1"
    else
        echo "  ⏭️  Already gone: $1"
    fi
}

echo "Step 1 — Delete LatentSync model files..."
safe_rm ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/latentsync_unet.pt
safe_rm ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/stable_syncnet.pt
safe_rm ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/vae/
safe_rm ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/whisper/
safe_rm ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/config.json
echo ""

echo "Step 2 — Delete ComfyUI-LatentSyncWrapper node..."
safe_rm ~/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/
echo ""

echo "Step 3 — Delete LatentSync Python files from project..."
safe_rm "$PROJECT_DIR/src/avatarpipeline/engines/lipsync/latentsync.py"
safe_rm "$PROJECT_DIR/data/temp/_latentsync_runner.py"
echo ""

echo "Step 4 — Delete LatentSync temp files..."
safe_rm "$PROJECT_DIR/data/temp/avatar_loop.mp4"
for d in "$PROJECT_DIR"/data/temp/latentsync_*/; do
    [ -d "$d" ] && safe_rm "$d"
done
echo ""

echo "Step 5 — Clear caches..."
brew cleanup 2>/dev/null && echo "  ✅ brew cleanup done" || echo "  ⏭️  brew cleanup skipped"
pip cache purge 2>/dev/null && echo "  ✅ pip cache purged" || echo "  ⏭️  pip cache purge skipped"
safe_rm ~/.cache/huggingface/hub/models--ByteDance--LatentSync-1.6/
echo ""

AFTER=$(df -k ~ | awk 'NR==2{print $4}')
FREED_KB=$((AFTER - BEFORE))
FREED_GB=$(echo "scale=2; $FREED_KB / 1048576" | bc 2>/dev/null || echo "?")

echo "═══════════════════════════════════════════════"
echo "✅ LatentSync removed. ${FREED_GB} GB freed."
echo "═══════════════════════════════════════════════"
df -h ~/
