"""avatarpipeline.lipsync — Lip-sync inference backends."""
from avatarpipeline.lipsync.musetalk import MuseTalkInference
from avatarpipeline.lipsync.sadtalker import SadTalkerInference

__all__ = ["MuseTalkInference", "SadTalkerInference"]
