"""End-to-end pipeline entry points."""

from avatarpipeline.pipelines.avatar import run_pipeline
from avatarpipeline.pipelines.narration import compose_narrated_video
from avatarpipeline.pipelines.presenter import compose_slide_presenter_video

__all__ = [
    "compose_narrated_video",
    "compose_slide_presenter_video",
    "run_pipeline",
]
