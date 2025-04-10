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
prompt = "The turquoise waves crash against the dark, jagged rocks of the shore, sending white foam spraying into the air. The scene is dominated by the stark contrast between the bright blue water and the dark, almost black rocks. The water is a clear, turquoise color, and the waves are capped with white foam. The rocks are dark and jagged, and they are covered in patches of green moss. The shore is lined with lush green vegetation, including trees and bushes. In the background, there are rolling hills covered in dense forest. The sky is cloudy, and the light is dim."
print(f"Generating video for prompt: '{prompt}'")
video_frames = pipe(prompt, height=320, width=576).frames[0]

# Save output
output_file = "output_video.mp4"
print(f"Saving video to {output_file}")

# Convert frames to numpy arrays directly (they already are numpy arrays)
# Don't convert to PIL Images first
imageio.mimwrite(output_file, video_frames, fps=8)
print(f"Video saved successfully to {output_file}")