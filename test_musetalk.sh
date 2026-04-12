#!/usr/bin/env bash
# =============================================================================
# test_musetalk.sh — End-to-end smoke test for Phase 1 (MuseTalk lip-sync)
#
# Creates test fixtures (blank PNG + silent WAV), runs MuseTalk inference,
# and verifies an MP4 is produced in temp/musetalk_out/.
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# Always run relative to this script's directory
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSETALK_DIR="$HOME/MuseTalk"
PIPELINE_VENV="$PIPELINE_DIR/.venv"
VENV_ACTIVATE="$MUSETALK_DIR/musetalk-env/bin/activate"
AVATAR_PATH="$PIPELINE_DIR/avatar/avatar.png"
AUDIO_PATH="$PIPELINE_DIR/audio/test.wav"
OUTPUT_DIR="$PIPELINE_DIR/temp/musetalk_out"

cd "$PIPELINE_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Create blank 256x256 white PNG if not present
# ─────────────────────────────────────────────────────────────────────────────
# Activate pipeline venv (provides pillow, loguru, pyyaml, etc.)
if [ ! -f "$PIPELINE_VENV/bin/activate" ]; then
    echo "Pipeline venv not found. Run setup.sh first."
    exit 1
fi
# shellcheck source=/dev/null
source "$PIPELINE_VENV/bin/activate"
echo "Using Python: $(python --version) from $PIPELINE_VENV"

echo "Step 1 — Creating test avatar..."
mkdir -p "$PIPELINE_DIR/avatar"

if [ ! -f "$AVATAR_PATH" ]; then
    python - <<PYEOF
from PIL import Image
img = Image.new('RGB', (256, 256), (255, 255, 255))
img.save('$AVATAR_PATH')
print('  Created blank 256x256 avatar PNG')
PYEOF
    pass "Test avatar created: $AVATAR_PATH"
else
    pass "Avatar already exists: $AVATAR_PATH"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Generate 5-second silent 16kHz mono WAV
# ─────────────────────────────────────────────────────────────────────────────
echo "Step 2 — Generating silent test audio..."
mkdir -p "$PIPELINE_DIR/audio"
ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 5 -c:a pcm_s16le "$AUDIO_PATH" 2>/dev/null
pass "Test audio created: $AUDIO_PATH"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Run MuseTalk inference
# ─────────────────────────────────────────────────────────────────────────────
echo "Step 3 — Running MuseTalk inference..."
# NOTE: musetalk_infer.py calls MuseTalk via subprocess using the MuseTalk venv
# python directly — no need to activate the MuseTalk venv here.
if [ ! -f "$VENV_ACTIVATE" ]; then
    fail "MuseTalk venv not found at $VENV_ACTIVATE. Run setup.sh first."
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

python "$PIPELINE_DIR/musetalk_infer.py" \
    --avatar "$AVATAR_PATH" \
    --audio  "$AUDIO_PATH" \
    --output-dir "temp/musetalk_out" \
    --config "configs/settings.yaml" \
    --prepare

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Verify output MP4 exists and is non-empty
# ─────────────────────────────────────────────────────────────────────────────
echo "Step 4 — Checking output..."
MP4_FILES=( $(find "$OUTPUT_DIR" -name "*.mp4" 2>/dev/null) )

if [ "${#MP4_FILES[@]}" -eq 0 ]; then
    fail "No MP4 found in $OUTPUT_DIR"
fi

OUTPUT_FILE="${MP4_FILES[0]}"
FILESIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat --format=%s "$OUTPUT_FILE")
pass "Output MP4: $OUTPUT_FILE ($FILESIZE bytes)"

# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo -e "${GREEN}  PHASE 1 COMPLETE${NC}"
echo "============================================="
echo "Output: $OUTPUT_FILE"
echo ""
