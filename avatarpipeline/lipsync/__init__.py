"""avatarpipeline.lipsync — Lip-sync inference backends."""
from avatarpipeline.lipsync.latentsync import LatentSyncInference
from avatarpipeline.lipsync.musetalk import MuseTalkInference

__all__ = ["LatentSyncInference", "MuseTalkInference"]
