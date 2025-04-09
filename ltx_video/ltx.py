import torch
from diffusers import DiffusionPipeline
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image

# Determine device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model from Hugging Face Hub directly
print("Loading model from Hugging Face Hub...")
pipe = DiffusionPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

# Generate video
prompt = "A panda eating bamboo in a forest"
print(f"Generating video for prompt: '{prompt}'")
video_frames = pipe(prompt, height=320, width=576).frames[0]

# Save output
output_file = "output_video.mp4"
print(f"Saving video to {output_file}")

# Convert frames to numpy arrays directly (they already are numpy arrays)
# Don't convert to PIL Images first
imageio.mimwrite(output_file, video_frames, fps=8)
print(f"Video saved successfully to {output_file}")