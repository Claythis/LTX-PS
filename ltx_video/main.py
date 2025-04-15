from pika import generate_apose_video

video_path = generate_apose_video(
    "joy.jpg",
    output_video_path="custom_output.mp4",
    pose_mode="create",
    num_frames=32,
    fps=24,
    inference_steps=75,
    guidance_scale=10.0
)