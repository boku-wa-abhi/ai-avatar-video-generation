"""
avatarpipeline.narration.validator — Sync-validation for PPTX + JSON narration files.

Three checks are run:
  1. PPT slide count must equal JSON entry count.
  2. JSON slide_number values must form a gapless, duplicate-free sequence 1…N.
  3. Every JSON slide_number must reference a page that exists in the PPT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    slide_count: int = 0
    json_count: int = 0
    json_data: dict = field(default_factory=dict)


def validate_sync(pptx_path: str | Path, json_data: dict) -> ValidationResult:
    """Run all three sync-validation checks and return a :class:`ValidationResult`.

    Args:
        pptx_path:  Absolute path to the ``.pptx`` file.
        json_data:  Already-parsed narration dict with ``{"slides": [...]}`` structure.

    Returns:
        A :class:`ValidationResult` whose ``ok`` flag is ``True`` only when all
        checks pass.  ``errors`` lists blocking failures; ``warnings`` lists
        non-blocking notices (e.g. slides with empty narration).
    """
    from pptx import Presentation

    pptx_path = Path(pptx_path)
    errors: list[str] = []
    warnings: list[str] = []

    # ── Open the PPTX ───────────────────────────────────────────────────────
    try:
        prs = Presentation(str(pptx_path))
        slide_count = len(prs.slides)
    except Exception as exc:
        return ValidationResult(ok=False, errors=[f"Cannot open PPTX file: {exc}"])

    # ── Parse JSON entries ───────────────────────────────────────────────────
    slides_json: list[dict] = json_data.get("slides", [])
    json_count = len(slides_json)

    # Check 1 — Count match ──────────────────────────────────────────────────
    if slide_count != json_count:
        errors.append(
            f"Mismatch: PPT has {slide_count} slides but JSON has {json_count} entries."
        )

    # Check 2 — Slide number sequence ────────────────────────────────────────
    try:
        numbers = [int(s["slide_number"]) for s in slides_json]
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"JSON entries have a missing or invalid 'slide_number' field: {exc}")
        return ValidationResult(
            ok=False,
            errors=errors,
            warnings=warnings,
            slide_count=slide_count,
            json_count=json_count,
            json_data=json_data,
        )

    expected = list(range(1, json_count + 1))
    missing = sorted(set(expected) - set(numbers))

    seen: dict[int, int] = {}
    for n in numbers:
        seen[n] = seen.get(n, 0) + 1
    duplicates = sorted(k for k, v in seen.items() if v > 1)

    if sorted(numbers) != expected or duplicates:
        errors.append(
            f"JSON slide numbers are not sequential. "
            f"Found: {numbers}. "
            f"Missing: {missing}. "
            f"Duplicates: {duplicates}."
        )

    # Check 3 — Every JSON slide_number maps to a real PPT page ─────────────
    for entry in slides_json:
        n = int(entry.get("slide_number", 0))
        if n < 1 or n > slide_count:
            errors.append(
                f"JSON references slide number {n}, but PPT only has {slide_count} slides."
            )

    # Warnings — empty narration text ────────────────────────────────────────
    for entry in slides_json:
        narration = (entry.get("narration") or "").strip()
        if not narration:
            warnings.append(f"Slide {entry.get('slide_number')} has no narration entry.")

    logger.debug(
        f"Validation: pptx={pptx_path.name}, slides={slide_count}, "
        f"json_entries={json_count}, errors={len(errors)}, warnings={len(warnings)}"
    )
    return ValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        slide_count=slide_count,
        json_count=json_count,
        json_data=json_data,
    )
