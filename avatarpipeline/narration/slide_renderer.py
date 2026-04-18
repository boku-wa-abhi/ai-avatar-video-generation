"""
avatarpipeline.narration.slide_renderer — Render PDF pages to PNG images.

The Slide Narrator pipeline now accepts PDFs only. Each PDF page is rasterized
directly into a PNG using PyMuPDF, which is faster and more predictable than
interpreting editable slide documents at runtime.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger


def _load_pymupdf():
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is not installed. Run: uv pip install pymupdf"
        ) from exc
    return fitz


def count_pages(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF."""
    fitz = _load_pymupdf()
    pdf_path = Path(pdf_path)
    with fitz.open(str(pdf_path)) as doc:
        return len(doc)


def render_slides(
    pdf_path: str | Path,
    output_dir: str | Path,
    render_width: int = 1920,
) -> list[Path]:
    """Render all PDF pages to PNG images.

    Args:
        pdf_path:    Path to the source PDF.
        output_dir:  Directory where ``page_NNN.png`` files are written.
        render_width: Target pixel width for the rasterized pages.

    Returns:
        List of rendered PNG paths in page order.
    """
    fitz = _load_pymupdf()
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    paths: list[Path] = []
    with fitz.open(str(pdf_path)) as doc:
        total = len(doc)
        if total == 0:
            raise RuntimeError("PDF contains no pages.")

        for idx, page in enumerate(doc, start=1):
            rect = page.rect
            zoom = render_width / max(rect.width, 1.0)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            out = output_dir / f"page_{idx:03d}.png"
            pix.save(str(out))
            logger.debug(f"Rendered PDF page {idx}/{total} via PyMuPDF")
            paths.append(out)

    return paths
