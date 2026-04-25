"""
Standalone script to build all composite presenter videos for the
20260420-final-version-3 presentation into composite_f57596b2f8a1/.

Uses slides/, lipsync/, logo and avatar assets that are already on disk.
Reuses _compose_presenter_overlay from avatarpipeline.pipelines.presenter.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Ensure the src-layout package is on sys.path when running from a checkout.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from avatarpipeline.pipelines.presenter import _compose_presenter_overlay  # noqa: E402

PRES_DIR = PROJECT_ROOT / "data" / "presentations" / "20260420-final-version-3"
SLIDES_DIR   = PRES_DIR / "slides"
LIPSYNC_DIR  = PRES_DIR / "lipsync"
LOGO_PATH    = PRES_DIR / "source" / "Screenshot 2026-04-19 at 17.36.03.png"
OUT_DIR      = PRES_DIR / "composite_f57596b2f8a1"

OUT_DIR.mkdir(parents=True, exist_ok=True)

slide_pngs = sorted(SLIDES_DIR.glob("slide_*.png"))
if not slide_pngs:
    print("ERROR: No slide PNGs found in", SLIDES_DIR)
    sys.exit(1)

errors: list[str] = []

for slide_png in slide_pngs:
    # e.g. slide_001.png → number "001"
    num_str = slide_png.stem.split("_")[1]   # "001"

    # Find matching lipsync: slide_001_<hash>.mp4
    matches = sorted(LIPSYNC_DIR.glob(f"slide_{num_str}_*.mp4"))
    if not matches:
        msg = f"  [SKIP] No lipsync found for slide {num_str}"
        print(msg)
        errors.append(msg)
        continue

    lipsync_mp4 = matches[0]

    # Build a deterministic output hash from slide_png name + lipsync name
    raw = (slide_png.name + lipsync_mp4.name).encode()
    out_hash = hashlib.sha1(raw).hexdigest()[:12]
    out_name = f"slide_{num_str}_{out_hash}.mp4"
    out_path = OUT_DIR / out_name

    if out_path.exists():
        print(f"  [OK]   {out_name} already exists, skipping.")
        continue

    print(f"  [RUN]  slide_{num_str}: {slide_png.name} + {lipsync_mp4.name} → {out_name}")
    try:
        _compose_presenter_overlay(
            slide_image=slide_png,
            presenter_video=lipsync_mp4,
            output_path=out_path,
            logo_image=LOGO_PATH if LOGO_PATH.exists() else None,
        )
        print(f"         done → {out_path.name}")
    except Exception as exc:
        msg = f"  [ERR]  slide_{num_str}: {exc}"
        print(msg)
        errors.append(msg)

print("\n=== Summary ===")
print(f"Output dir: {OUT_DIR}")
done = sorted(OUT_DIR.glob("slide_*.mp4"))
print(f"Composites created: {len(done)} / {len(slide_pngs)}")
if errors:
    print("Errors:")
    for e in errors:
        print(" ", e)
else:
    print("All slides composited successfully.")
