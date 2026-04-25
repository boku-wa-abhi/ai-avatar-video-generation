"""Lip-sync inference backends."""

from avatarpipeline.engines.lipsync.musetalk import MuseTalkInference
from avatarpipeline.engines.lipsync.sadtalker import SadTalkerInference

__all__ = ["MuseTalkInference", "SadTalkerInference"]
