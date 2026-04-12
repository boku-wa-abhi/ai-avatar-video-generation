#!/usr/bin/env python3
"""
scripts/run_dashboard.py — Launch the Avatar Studio Gradio dashboard.

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 7861 --no-browser
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Avatar Studio dashboard")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on [default: 7860]")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind [default: 127.0.0.1]")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    import gradio as gr
    from ui.dashboard import CSS, demo
    from avatarpipeline import ASSETS_DIR

    print(f"\n  Avatar Studio — starting on http://{args.host}:{args.port}\n")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
        favicon_path=str(ASSETS_DIR / "favicon.png"),
    )


if __name__ == "__main__":
    main()
