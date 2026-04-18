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

_SLIDES_KEYS = ("slides", "scenes", "entries", "items")
_SLIDE_NUMBER_KEYS = (
    "slide_number", "slide", "slide_num", "slide_index",
    "page", "page_number", "number", "index",
)
_NARRATION_KEYS = (
    "narration", "voiceover", "voice_over", "text",
    "script", "audio_text", "notes", "speaker_notes",
)
_DISPLAY_SECONDS_KEYS = (
    "display_seconds", "display_duration", "duration_seconds",
    "slide_duration", "duration", "seconds", "time_seconds",
    "hold_seconds", "hold",
)
_PAUSE_SECONDS_KEYS = (
    "pause_seconds", "pause", "gap_seconds", "gap",
    "silence_after", "post_pause_seconds",
)
_DEFAULT_DISPLAY_KEYS = (
    "default_display_seconds", "default_display_duration",
    "default_duration_seconds", "default_slide_duration",
    "display_seconds", "duration_seconds",
)
_DEFAULT_PAUSE_KEYS = (
    "default_pause_seconds", "default_pause_duration",
    "default_pause", "slide_pause_seconds",
    "pause_seconds",
)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    slide_count: int = 0
    json_count: int = 0
    json_data: dict = field(default_factory=dict)


def _pick(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _coerce_non_negative_float(value, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}.") from exc
    if result < 0:
        raise ValueError(f"{field_name} cannot be negative (got {result}).")
    return result


def normalize_narration_json(json_data: dict | list) -> dict:
    """Normalize flexible slide-narration JSON into the canonical schema.

    Supported inputs:
      - ``{"slides": [...]}``
      - a plain list of slide entries
      - ``{"1": {...}, "2": {...}}`` style dicts keyed by slide number

    Supported slide keys:
      - slide number: ``slide_number`` / ``slide`` / ``page`` / ``number``
      - narration: ``narration`` / ``text`` / ``script`` / ``voiceover``
      - display time: ``duration_seconds`` / ``display_seconds`` / ``duration``
      - pause time: ``pause_seconds`` / ``pause`` / ``gap_seconds``
    """
    root = json_data if isinstance(json_data, dict) else {}

    if isinstance(json_data, list):
        raw_entries = json_data
    elif isinstance(json_data, dict):
        raw_entries = None
        for key in _SLIDES_KEYS:
            if key in json_data:
                raw_entries = json_data[key]
                break
        if raw_entries is None:
            if json_data and all(str(k).isdigit() for k in json_data.keys()):
                raw_entries = [
                    {"slide_number": key, **value} if isinstance(value, dict)
                    else {"slide_number": key, "narration": value}
                    for key, value in json_data.items()
                ]
            else:
                raise ValueError(
                    "JSON must be a list of slides or an object with a 'slides' array."
                )
    else:
        raise ValueError("JSON root must be an object or an array of slide entries.")

    if isinstance(raw_entries, dict):
        if not all(str(k).isdigit() for k in raw_entries.keys()):
            raise ValueError("When 'slides' is an object, its keys must be slide numbers.")
        raw_entries = [
            {"slide_number": key, **value} if isinstance(value, dict)
            else {"slide_number": key, "narration": value}
            for key, value in raw_entries.items()
        ]

    if not isinstance(raw_entries, list):
        raise ValueError("'slides' must be a list of slide entries.")

    default_display = _coerce_non_negative_float(
        _pick(root, _DEFAULT_DISPLAY_KEYS), "default display duration"
    )
    default_pause = _coerce_non_negative_float(
        _pick(root, _DEFAULT_PAUSE_KEYS), "default pause duration"
    )

    normalized_slides: list[dict] = []
    inferred_slide_numbers = 0

    for idx, raw_entry in enumerate(raw_entries, start=1):
        if isinstance(raw_entry, dict):
            slide_number_raw = _pick(raw_entry, _SLIDE_NUMBER_KEYS)
            narration_raw = _pick(raw_entry, _NARRATION_KEYS)
            display_raw = _pick(raw_entry, _DISPLAY_SECONDS_KEYS)
            pause_raw = _pick(raw_entry, _PAUSE_SECONDS_KEYS)
        elif isinstance(raw_entry, str):
            slide_number_raw = None
            narration_raw = raw_entry
            display_raw = None
            pause_raw = None
        else:
            raise ValueError(
                f"Slide entry {idx} must be an object or string, got {type(raw_entry).__name__}."
            )

        if slide_number_raw in (None, ""):
            slide_number = idx
            inferred_slide_numbers += 1
        else:
            try:
                slide_number = int(slide_number_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Slide entry {idx} has an invalid slide number: {slide_number_raw!r}."
                ) from exc

        display_seconds = _coerce_non_negative_float(
            display_raw if display_raw is not None else default_display,
            f"slide {slide_number} display duration",
        )
        pause_seconds = _coerce_non_negative_float(
            pause_raw if pause_raw is not None else default_pause,
            f"slide {slide_number} pause duration",
        )

        normalized = {
            "slide_number": slide_number,
            "narration": str(narration_raw or "").strip(),
        }
        if display_seconds is not None:
            normalized["display_seconds"] = display_seconds
        if pause_seconds is not None:
            normalized["pause_seconds"] = pause_seconds
        normalized_slides.append(normalized)

    normalized_json = {
        "presentation_title": root.get("presentation_title", ""),
        "slides": normalized_slides,
        "_inferred_slide_numbers": inferred_slide_numbers,
    }
    if default_display is not None:
        normalized_json["default_display_seconds"] = default_display
    if default_pause is not None:
        normalized_json["default_pause_seconds"] = default_pause
    return normalized_json


def validate_sync(pptx_path: str | Path, json_data: dict | list) -> ValidationResult:
    """Run all three sync-validation checks and return a :class:`ValidationResult`.

    Args:
        pptx_path:  Absolute path to the ``.pptx`` file.
        json_data:  Already-parsed narration JSON object or array.

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
    try:
        normalized_json = normalize_narration_json(json_data)
    except Exception as exc:
        return ValidationResult(ok=False, errors=[f"Cannot normalize narration JSON: {exc}"])

    slides_json: list[dict] = normalized_json.get("slides", [])
    json_count = len(slides_json)
    if normalized_json.get("_inferred_slide_numbers"):
        warnings.append(
            "Some slide numbers were not provided explicitly; JSON order was used."
        )

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
            json_data=normalized_json,
        )

    expected = list(range(1, slide_count + 1))
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
        json_data=normalized_json,
    )
