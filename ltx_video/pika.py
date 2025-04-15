import torch
from diffusers import LTXImageToVideoPipeline
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image

# Import the pose extraction module
from extraction import get_pose_reference

def generate_apose_video(
    input_image_path,
    output_video_path="output_apose_video.mp4",
    pose_reference_path=None,
    pose_mode="create",
    num_frames=24,
    fps=15,
    inference_steps=50,
    guidance_scale=9.0
):
    """
    Generate A-pose video from input image using LTX-Video model.
    
    Args:
        input_image_path: Path to input image
        output_video_path: Path to save output video
        pose_reference_path: Path to save pose reference (will use default if None)
        pose_mode: Method to get pose reference - 'extract' or 'create'
        num_frames: Number of frames to generate
        fps: Frames per second for output video
        inference_steps: Number of denoising steps
        guidance_scale: How strongly to follow the prompt
        
    Returns:
        Path to generated video
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set default pose reference path if not provided
    if pose_reference_path is None:
        pose_reference_path = "pose_reference.jpg"
    
    # Get pose reference image
    pose_ref_path = get_pose_reference(
        input_image_path, 
        output_path=pose_reference_path,
        mode=pose_mode
    )
    print(f"Using pose reference: {pose_ref_path}")
    
    # Load the image-to-video model
    print("Loading model from Hugging Face Hub...")
    pipe = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    # Load and prepare the input image
    input_image = Image.open(input_image_path).convert("RGB")
    
    # Get image dimensions
    original_width, original_height = input_image.size
    print(f"Original image dimensions: {original_width}x{original_height}")
    
    # Ensure dimensions are divisible by 32 for the model
    width = (original_width // 32) * 32
    height = (original_height // 32) * 32
    if width == 0:
        width = 32
    if height == 0:
        height = 32
    
    input_image = input_image.resize((width, height))
    print(f"Adjusted image dimensions: {width}x{height}")
    
    # Create a concise prompt describing the A-pose 
    detailed_prompt = """
    Professional animation of a person in perfect A-pose. Arms extended at exact 45-degree angles, 
    straight spine, feet shoulder-width apart. Minimal natural movement while maintaining the precise pose.
    """
    
    # Focused negative prompt
    negative_prompt = """
    poor quality, blurry, incorrect pose, bent arms, wrong arm angle, arms too high, arms too low, 
    excessive movement, T-pose, unstable, flickering
    """
    
    print(f"Generating video with A-pose prompt...")
    
    # Generate with optimized settings for accuracy
    video_frames = pipe(
        image=input_image,
        prompt=detailed_prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
    ).frames[0]
    
    # Save output
    print(f"Saving video to {output_video_path}")
    
    # Save with specified framerate
    imageio.mimwrite(output_video_path, video_frames, fps=fps)
    print(f"Video saved successfully to {output_video_path}")
    
    return output_video_path

if __name__ == "__main__":
    # This allows the script to be run directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate A-pose video using LTX-Video")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--output", "-o", default="output_apose_video.mp4", help="Output path for video")
    parser.add_argument("--pose_reference", "-p", default=None, help="Output path for pose reference")
    parser.add_argument("--pose_mode", "-m", choices=["extract", "create"], default="create", 
                        help="Mode: 'extract' to extract pose from input image, 'create' to create A-pose reference")
    parser.add_argument("--frames", "-f", type=int, default=24, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for output video")
    parser.add_argument("--steps", "-s", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", "-g", type=float, default=9.0, help="Guidance scale (how strictly to follow prompt)")
    
    args = parser.parse_args()
    
    try:
        video_path = generate_apose_video(
            args.input_image,
            args.output,
            args.pose_reference,
            args.pose_mode,
            args.frames,
            args.fps,
            args.steps,
            args.guidance
        )
        print(f"Successfully generated video at: {video_path}")
    except Exception as e:
        print(f"Error: {e}")