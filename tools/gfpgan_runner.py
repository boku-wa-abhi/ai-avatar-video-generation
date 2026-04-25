#!/usr/bin/env python3
"""
gfpgan_runner.py — Standalone GFPGAN frame enhancer.

Called via subprocess using the SadTalker env so the main project venv
doesn't need to install basicsr/gfpgan.

Usage:
    <sadtalker-python> tools/gfpgan_runner.py \\
        --frames_dir /path/to/frames \\
        --weights /path/to/GFPGANv1.4.pth \\
        --sadtalker_dir /path/to/SadTalker

Enhances every frame_*.png in-place using GFPGAN.
"""
import argparse
import glob
import os
import sys

import cv2

parser = argparse.ArgumentParser(description="GFPGAN frame enhancer")
parser.add_argument("--frames_dir", required=True, help="Directory containing frame_*.png files")
parser.add_argument("--weights", required=True, help="Path to GFPGANv1.4.pth weights")
parser.add_argument("--sadtalker_dir", required=True, help="SadTalker repository root")
args = parser.parse_args()

# Run from SadTalker dir so relative paths in GFPGAN code resolve
os.chdir(args.sadtalker_dir)
sys.path.insert(0, args.sadtalker_dir)

from gfpgan import GFPGANer  # noqa: E402

restorer = GFPGANer(
    model_path=args.weights,
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,
)

frames = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.png")))
print(f"GFPGAN enhancing {len(frames)} frames...", flush=True)

for i, fp in enumerate(frames):
    img = cv2.imread(fp)
    if img is None:
        continue
    _, _, restored = restorer.enhance(
        img, has_aligned=False, only_center_face=False, paste_back=True
    )
    cv2.imwrite(fp, restored)
    if (i + 1) % 25 == 0 or (i + 1) == len(frames):
        print(f"  {i + 1}/{len(frames)} frames done", flush=True)

print("GFPGAN enhancement complete.", flush=True)
