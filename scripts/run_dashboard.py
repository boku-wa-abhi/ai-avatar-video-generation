#!/usr/bin/env python3
"""
scripts/run_dashboard.py — Launch the Avatar Studio Gradio dashboard.

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 7861 --no-browser
"""

import argparse
import os
import socket
import sys
from pathlib import Path

# Ensure the src-layout package is on PYTHONPATH when running from a checkout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def _find_free_port(start: int, end: int) -> int:
    """Return the first free TCP port in [start, end]."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start}-{end}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Avatar Studio dashboard")
    parser.add_argument("--port", type=int, default=None, help="Port to serve on [default: auto 7860-7870]")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind [default: 127.0.0.1]")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    port = args.port if args.port is not None else _find_free_port(7860, 7870)

    import gradio as gr
    from app.dashboard import CSS, demo, THEME
    from avatarpipeline import ASSETS_DIR

    print(f"\n  Avatar Studio — starting on http://{args.host}:{port}\n")

    demo.launch(
        server_name=args.host,
        server_port=port,
        inbrowser=not args.no_browser,
        share=args.share,
        show_error=True,
        theme=THEME,
        css=CSS,
        favicon_path=str(ASSETS_DIR / "favicon.png"),
        allowed_paths=[str(ROOT)],
    )


if __name__ == "__main__":
    main()
