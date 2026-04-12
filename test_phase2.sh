#!/usr/bin/env bash
# =============================================================================
# test_phase2.sh — End-to-end smoke test for Phase 2
#
# 1. Generate voice with ElevenLabs
# 2. Run LatentSync lip-sync
# 3. Verify output MP4 exists and duration > 4s
# 4. Open video on macOS
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_VENV="$PIPELINE_DIR/.venv"
AVATAR_PATH="$PIPELINE_DIR/avatar/avatar.png"
AUDIO_PATH="$PIPELINE_DIR/audio/test_p2.wav"
OUTPUT_PATH="$PIPELINE_DIR/temp/latentsync_out.mp4"

cd "$PIPELINE_DIR"

# Activate pipeline venv
if [ ! -f "$PIPELINE_VENV/bin/activate" ]; then
    fail "Pipeline venv not found. Run setup.sh first."
fi
# shellcheck source=/dev/null
source "$PIPELINE_VENV/bin/activate"
echo "Using Python: $(python --version) from $PIPELINE_VENV"

# Ensure avatar exists (create blank if not)
mkdir -p "$PIPELINE_DIR/avatar"
if [ ! -f "$AVATAR_PATH" ]; then
    python -c "
from PIL import Image
img = Image.new('RGB', (512, 512), (255, 255, 255))
img.save('$AVATAR_PATH')
print('Created blank 512x512 avatar')
"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generate voice via Kokoro (local, free)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Step 1 — Generating voice (Kokoro local TTS)..."

python "$PIPELINE_DIR/voice_gen.py" \
    --text "This is phase two. The voice and lip sync test." \
    --voice "af_heart" \
    --out "$AUDIO_PATH"

if [ -f "$AUDIO_PATH" ]; then
    pass "Voice generated: $AUDIO_PATH"
else
    fail "Voice generation failed — no WAV produced"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Run LatentSync inference
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Step 2 — Running LatentSync inference..."

python "$PIPELINE_DIR/latentsync_infer.py" \
    --avatar "$AVATAR_PATH" \
    --audio "$AUDIO_PATH" \
    --output "$OUTPUT_PATH"

if [ -f "$OUTPUT_PATH" ]; then
    pass "LatentSync output: $OUTPUT_PATH"
else
    fail "LatentSync failed — no MP4 produced"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Verify duration > 4 seconds
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Step 3 — Checking video duration..."

DURATION=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_PATH" 2>/dev/null)

if [ -z "$DURATION" ]; then
    fail "Could not read duration via ffprobe"
fi

# Compare as integers (bash doesn't do float comparison natively)
DURATION_INT=$(printf "%.0f" "$DURATION")
if [ "$DURATION_INT" -gt 4 ]; then
    pass "Video duration: ${DURATION}s (> 4s)"
else
    fail "Video too short: ${DURATION}s (expected > 4s)"
fi

FILESIZE=$(stat -f%z "$OUTPUT_PATH" 2>/dev/null || stat --format=%s "$OUTPUT_PATH")
pass "File size: $((FILESIZE / 1024)) KB"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Open video on macOS
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Step 4 — Opening video..."
open "$OUTPUT_PATH" 2>/dev/null || warn "Could not open video (non-macOS?)"

# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo -e "${GREEN}  PHASE 2 COMPLETE${NC}"
echo "============================================="
echo ""
echo "Voice:  $AUDIO_PATH"
echo "Video:  $OUTPUT_PATH"
echo "Duration: ${DURATION}s"
echo ""
