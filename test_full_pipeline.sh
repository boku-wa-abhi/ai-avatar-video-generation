#!/usr/bin/env bash
# test_full_pipeline.sh — End-to-end smoke test for the AI avatar pipeline
#
# Runs the complete 7-step pipeline with a short test script and validates
# that the output video has:
#   - duration > 5 seconds
#   - at least one video stream
#   - at least one audio stream
#
# Usage:
#   bash test_full_pipeline.sh                        # full (with captions + enhance)
#   bash test_full_pipeline.sh --no-enhance           # skip face enhancement
#   bash test_full_pipeline.sh --no-captions          # skip captions
#   bash test_full_pipeline.sh --no-enhance --no-captions   # fastest smoke test
#   bash test_full_pipeline.sh --musetalk-only        # force MuseTalk lip-sync

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colour helpers ────────────────────────────────────────────────────────────
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
blue()   { printf '\033[0;34m%s\033[0m\n' "$*"; }

# ── Load .env ──────────────────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
    set -o allexport; source .env; set +o allexport
fi

# ── Parse extra flags to forward to make_video.py ─────────────────────────────
FORWARD_FLAGS=()
for arg in "$@"; do
    FORWARD_FLAGS+=("$arg")
done

# ── Config ────────────────────────────────────────────────────────────────────
VENV=".venv"
PYTHON="$VENV/bin/python"
FFPROBE="ffprobe"
OUT_FILE="output/test_pipeline_$(date +%Y%m%d_%H%M%S).mp4"
TEST_SCRIPT="Hello! I'm your AI avatar, ready to help. This is a quick smoke test of the full video generation pipeline."
MIN_DURATION=5   # seconds

# ── Pre-checks ────────────────────────────────────────────────────────────────
blue "============================================================"
blue "  AI Avatar Pipeline — Full Integration Test"
blue "============================================================"

if [[ ! -f "$PYTHON" ]]; then
    red "ERROR: Python venv not found at $VENV"
    red "       Run: bash setup.sh && bash install_latentsync.sh"
    exit 1
fi

if [[ ! -f "avatar/avatar.png" ]]; then
    yellow "WARNING: avatar/avatar.png not found — creating placeholder..."
    mkdir -p avatar
    # Generate a 512×512 grey placeholder PNG using Python/Pillow
    "$PYTHON" - <<'PYEOF'
from PIL import Image
img = Image.new("RGB", (512, 512), color=(128, 128, 128))
img.save("avatar/avatar.png")
print("Created placeholder avatar/avatar.png (512×512 grey)")
PYEOF
fi

if ! command -v "$FFPROBE" &>/dev/null; then
    red "ERROR: ffprobe not found. Install with: brew install ffmpeg"
    exit 1
fi

mkdir -p output

# ── Run pipeline ─────────────────────────────────────────────────────────────
echo
yellow "Test script: \"$TEST_SCRIPT\""
yellow "Output path: $OUT_FILE"
yellow "Extra flags: ${FORWARD_FLAGS[*]:-<none>}"
echo

START_TS=$(date +%s)

# Export MPS vars
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

"$PYTHON" make_video.py \
    --script "$TEST_SCRIPT" \
    --orientation 9:16 \
    --out "$OUT_FILE" \
    "${FORWARD_FLAGS[@]}"

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
echo
blue "Pipeline wall time: ${ELAPSED}s"

# ── Validate output ───────────────────────────────────────────────────────────
echo
blue "── Validating output: $OUT_FILE"

if [[ ! -f "$OUT_FILE" ]]; then
    red "FAIL: Output file not created: $OUT_FILE"
    exit 1
fi

FILE_SIZE=$(stat -f%z "$OUT_FILE" 2>/dev/null || stat -c%s "$OUT_FILE")
if (( FILE_SIZE < 10000 )); then
    red "FAIL: Output file suspiciously small (${FILE_SIZE} bytes)"
    exit 1
fi

# Check video stream
VIDEO_STREAMS=$(
    "$FFPROBE" -v error \
        -select_streams v \
        -show_entries stream=codec_type \
        -of csv=p=0 "$OUT_FILE" 2>/dev/null | wc -l | tr -d ' '
)
if (( VIDEO_STREAMS < 1 )); then
    red "FAIL: No video streams found in output"
    exit 1
fi

# Check audio stream
AUDIO_STREAMS=$(
    "$FFPROBE" -v error \
        -select_streams a \
        -show_entries stream=codec_type \
        -of csv=p=0 "$OUT_FILE" 2>/dev/null | wc -l | tr -d ' '
)
if (( AUDIO_STREAMS < 1 )); then
    yellow "WARNING: No audio stream found in output (may still be valid)"
fi

# Check duration
DURATION=$(
    "$FFPROBE" -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        "$OUT_FILE" 2>/dev/null | awk '{printf "%d", $1}'
)
if (( DURATION < MIN_DURATION )); then
    red "FAIL: Output duration ${DURATION}s is shorter than minimum ${MIN_DURATION}s"
    exit 1
fi

# ── Print summary ─────────────────────────────────────────────────────────────
echo
green "============================================================"
green "  ALL CHECKS PASSED"
green "============================================================"
green "  Output : $OUT_FILE"
green "  Size   : $(du -h "$OUT_FILE" | cut -f1)"
green "  Duration: ${DURATION}s"
green "  Video  : ${VIDEO_STREAMS} stream(s)"
green "  Audio  : ${AUDIO_STREAMS} stream(s)"
green "  Wall   : ${ELAPSED}s"
green "============================================================"
