# mediapipe_pose.py
import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Input and output directories
input_dir = 'mediapipe_source'
output_dir = 'mediapipe_pose'
os.makedirs(output_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_files)} images")

for filename in image_files:
    image_path = os.path.join(input_dir, filename)
    
    output_filename = os.path.splitext(filename)[0] + '_pose.png'
    output_path = os.path.join(output_dir, output_filename)
    
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = pose.process(image_rgb)
    
    # Create black background
    black_bg = np.zeros_like(image)
    
    # Draw pose on black background
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            black_bg, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

        # Save result
        print(f"Saving to: {output_path}")
        success = cv2.imwrite(output_path, black_bg)
        print("Write success:", success)
    else:
        print(f"No pose detected in {filename}, skipping save.")

print("Pose extraction complete!")
