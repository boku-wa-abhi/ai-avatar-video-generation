
import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

sys.path.insert(0, '/Users/abhijeetanand/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper')

import torch
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline

device = "mps"

# Load pipeline
pipe = LipsyncPipeline.from_pretrained(
    pretrained_model_name_or_path='/Users/abhijeetanand/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints',
    device=device,
)
pipe = pipe.to(device)

# Run inference
output = pipe(
    video_path='/Users/abhijeetanand/Projects/Personal/02_Github_Projects/ai-avatar-video-generation/temp/avatar_loop.mp4',
    audio_path='/Users/abhijeetanand/Projects/Personal/02_Github_Projects/ai-avatar-video-generation/audio/speech_20260412_142749_16k.wav',
    video_out_path='/Users/abhijeetanand/Projects/Personal/02_Github_Projects/ai-avatar-video-generation/output/lipsync_20260412_142749.mp4',
    num_inference_steps=25,
    guidance_scale=1.5,
)

print("LATENTSYNC_SUCCESS")
