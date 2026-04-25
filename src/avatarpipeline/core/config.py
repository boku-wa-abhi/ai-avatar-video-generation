"""Typed configuration loading for the avatar pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from avatarpipeline import CONFIGS_DIR


class ConfigError(RuntimeError):
    """Raised when settings.yaml is missing or invalid."""


@dataclass(frozen=True)
class PipelineConfig:
    musetalk_dir: Path
    sadtalker_dir: Path
    avatar_path: Path
    default_fps: int
    default_orientation: str
    tts_engine: str
    default_voice: str
    tts_speed: float
    tts_lang_code: str
    lipsync_engine: str
    expression_scale: float
    musetalk_fps: int
    musetalk_use_float16: bool
    musetalk_default_bbox_shift: int
    musetalk_default_batch_size: int
    raw: dict[str, Any]


def _path(value: str, field_name: str) -> Path:
    if not value:
        raise ConfigError(f"Missing required config value: {field_name}")
    return Path(value).expanduser()


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load and validate configs/settings.yaml."""
    cfg_path = Path(path) if path else CONFIGS_DIR / "settings.yaml"
    if not cfg_path.exists():
        raise ConfigError(f"Config not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    tts = cfg.get("tts") or {}
    lipsync = cfg.get("lipsync") or {}
    musetalk = cfg.get("musetalk") or {}

    try:
        default_fps = int(cfg.get("default_fps", 25))
        musetalk_fps = int(musetalk.get("fps", default_fps))
        musetalk_batch = int(musetalk.get("default_batch_size", 8))
        musetalk_bbox = int(musetalk.get("default_bbox_shift", 0))
        expression_scale = float(lipsync.get("expression_scale", 1.0))
        tts_speed = float(tts.get("speed", 1.0))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid numeric value in {cfg_path}") from exc

    return PipelineConfig(
        musetalk_dir=_path(cfg.get("musetalk_dir", "~/MuseTalk"), "musetalk_dir"),
        sadtalker_dir=_path(cfg.get("sadtalker_dir", "~/SadTalker"), "sadtalker_dir"),
        avatar_path=Path(cfg.get("avatar_path", "data/avatars/avatar.png")),
        default_fps=default_fps,
        default_orientation=str(cfg.get("default_orientation", "9:16")),
        tts_engine=str(tts.get("engine", "kokoro")),
        default_voice=str(tts.get("default_voice", "af_heart")),
        tts_speed=tts_speed,
        tts_lang_code=str(tts.get("lang_code", "a")),
        lipsync_engine=str(lipsync.get("default_engine", "musetalk")),
        expression_scale=expression_scale,
        musetalk_fps=musetalk_fps,
        musetalk_use_float16=bool(musetalk.get("use_float16", True)),
        musetalk_default_bbox_shift=musetalk_bbox,
        musetalk_default_batch_size=musetalk_batch,
        raw=cfg,
    )
