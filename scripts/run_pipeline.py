#!/usr/bin/env python3
"""
scripts/run_pipeline.py — Command-line interface for the avatar video pipeline.

Usage:
    python scripts/run_pipeline.py --script "Hello, I'm your AI avatar!" \\
        --voice af_heart --orientation 9:16 --out data/output/final.mp4

    python scripts/run_pipeline.py --script "..." --no-enhance --no-captions
    python scripts/run_pipeline.py --list-voices
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure the src-layout package is on PYTHONPATH when running from a checkout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# Load .env (HF_TOKEN, etc.)
env_file = ROOT / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(str(env_file))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI avatar video generation — one-command pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic (Portrait 9:16, Kokoro voice)
  python scripts/run_pipeline.py --script "Hello world" --out data/output/video.mp4

  # British male voice, landscape, no captions
  python scripts/run_pipeline.py --script "..." --voice bm_george --orientation 16:9 --no-captions

  # Fast smoke test (skip enhance + captions)
  python scripts/run_pipeline.py --script "..." --no-enhance --no-captions
        """,
    )
    p.add_argument("--script", help="Spoken script text")
    p.add_argument("--orientation", default="9:16", choices=["9:16", "16:9", "1:1"],
                   help="Canvas aspect ratio [default: 9:16]")
    p.add_argument("--voice", default="af_heart",
                   help="Kokoro voice ID [default: af_heart]")
    p.add_argument("--out", default=None,
                   help="Output MP4 path [default: data/output/final.mp4]")
    p.add_argument("--background", default="black",
                   help="black | white | blur | /path/to/image.jpg")
    p.add_argument("--music", default=None, help="Background music file")
    p.add_argument("--preview", action="store_true",
                   help="Open result in default media player when done")
    p.add_argument("--no-captions", action="store_true", help="Skip subtitle generation")
    p.add_argument("--no-enhance", action="store_true", help="Skip face enhancement")
    p.add_argument("--engine", default="musetalk",
                   choices=["musetalk", "sadtalker", "sadtalker_hd"],
                   help="Lip-sync engine [default: musetalk]")
    p.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_voices:
        from avatarpipeline.engines.tts.kokoro import VoiceGenerator
        print("\nAvailable voices:\n")
        for v in VoiceGenerator().list_voices():
            print(f"  {v}")
        print()
        return

    if not args.script:
        print("ERROR: --script is required. Use --help for usage.", file=sys.stderr)
        sys.exit(1)

    from loguru import logger
    from avatarpipeline.pipelines.avatar import run_pipeline

    try:
        run_pipeline(
            script=args.script,
            orientation=args.orientation,
            voice=args.voice,
            output_path=args.out,
            background=args.background,
            music_path=args.music,
            include_captions=not args.no_captions,
            include_enhance=not args.no_enhance,
            lipsync_engine=args.engine,
            preview=args.preview,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
