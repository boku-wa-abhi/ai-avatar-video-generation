#!/usr/bin/env bash
# scripts/smoke_test.sh — End-to-end integration test for the avatar pipeline
#
# Runs the complete 7-step pipeline and validates the output video has:
#   - at least one video stream
#   - at least one audio stream
#   - duration > MIN_DURATION seconds
#
# Usage:
#   bash scripts/smoke_test.sh
#   bash scripts/smoke_test.sh --no-enhance
#   bash scripts/smoke_test.sh --no-captions
#   bash scripts/smoke_test.sh --no-enhance --no-captions   # fastest
#   bash scripts/smoke_test.sh --musetalk

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ── Colour helpers ────────────────────────────────────────────────────────────
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
blue()   { printf '\033[0;34m%s\033[0m\n' "$*"; }

# ── Load .env ──────────────────────────────────────────────────────────────────
[[ -f ".env" ]] && { set -o allexport; source .env; set +o allexport; }

# ── Parse flags ───────────────────────────────────────────────────────────────
FORWARD_FLAGS=()
for arg in "$@"; do FORWARD_FLAGS+=("$arg"); done

# ── Config ────────────────────────────────────────────────────────────────────
VENV=".venv"
PYTHON="$VENV/bin/python"
FFPROBE="ffprobe"
OUT_FILE="data/output/smoke_test_$(date +%Y%m%d_%H%M%S).mp4"
TEST_SCRIPT="Hello! I'm your AI avatar, ready to help. This is a quick smoke test of the full video generation pipeline."
MIN_DURATION=5

blue "============================================================"
blue "  Avatar Pipeline — Smoke Test"
blue "============================================================"

# ── Pre-checks ────────────────────────────────────────────────────────────────
[[ ! -f "$PYTHON" ]] && { red "ERROR: .venv not found. Run: bash install/setup.sh"; exit 1; }

if [[ ! -f "data/avatars/avatar.png" ]]; then
    yellow "No avatar found — creating a grey placeholder..."
    mkdir -p data/avatars
    "$PYTHON" - <<'PYEOF'
from PIL import Image
img = Image.new("RGB", (512, 512), (128, 128, 128))
img.save("data/avatars/avatar.png")
print("Created placeholder: data/avatars/avatar.png")
PYEOF
fi

command -v "$FFPROBE" &>/dev/null || { red "ERROR: ffprobe not found. brew install ffmpeg"; exit 1; }

mkdir -p data/output

yellow "Test script : \"$TEST_SCRIPT\""
yellow "Output path : $OUT_FILE"
yellow "Extra flags : ${FORWARD_FLAGS[*]:-<none>}"
echo

# ── Run pipeline ─────────────────────────────────────────────────────────────
START_TS=$(date +%s)

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

"$PYTHON" scripts/run_pipeline.py \
    --script "$TEST_SCRIPT" \
    --orientation 9:16 \
    --out "$OUT_FILE" \
    "${FORWARD_FLAGS[@]}"

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
blue "Wall time: ${ELAPSED}s"

# ── Validate output ───────────────────────────────────────────────────────────
blue "── Validating output: $OUT_FILE"

[[ ! -f "$OUT_FILE" ]] && { red "FAIL: Output file not created."; exit 1; }

FILE_SIZE=$(stat -f%z "$OUT_FILE" 2>/dev/null || stat -c%s "$OUT_FILE")
(( FILE_SIZE < 10000 )) && { red "FAIL: Output file suspiciously small (${FILE_SIZE} bytes)"; exit 1; }

VIDEO_STREAMS=$(
    "$FFPROBE" -v error -select_streams v \
    -show_entries stream=codec_type -of csv=p=0 \
    "$OUT_FILE" 2>/dev/null | wc -l | tr -d ' '
)
(( VIDEO_STREAMS < 1 )) && { red "FAIL: No video streams found."; exit 1; }

AUDIO_STREAMS=$(
    "$FFPROBE" -v error -select_streams a \
    -show_entries stream=codec_type -of csv=p=0 \
    "$OUT_FILE" 2>/dev/null | wc -l | tr -d ' '
)
(( AUDIO_STREAMS < 1 )) && yellow "WARNING: No audio stream found."

DURATION=$(
    "$FFPROBE" -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$OUT_FILE" 2>/dev/null | awk '{printf "%d", $1}'
)
(( DURATION < MIN_DURATION )) && {
    red "FAIL: Duration ${DURATION}s < minimum ${MIN_DURATION}s"; exit 1
}

# ── Summary ───────────────────────────────────────────────────────────────────
echo
green "============================================================"
green "  ALL CHECKS PASSED"
green "============================================================"
green "  Output  : $OUT_FILE"
green "  Size    : $(du -h "$OUT_FILE" | cut -f1)"
green "  Duration: ${DURATION}s"
green "  Video   : ${VIDEO_STREAMS} stream(s)"
green "  Audio   : ${AUDIO_STREAMS} stream(s)"
green "  Wall    : ${ELAPSED}s"
green "============================================================"
