# mimipose_pose_sequence.py is a script that processes a video file and extract video frames to create pose sequences of each frame and stores them in the train file

import cv2
import mediapipe as mp
import os

video_path = "mickey.mp4"
out_dir = "train/"
pose_dir = os.path.join(out_dir, "pose")
source_dir = os.path.join(out_dir, "source")
os.makedirs(pose_dir, exist_ok=True)
os.makedirs(source_dir, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    source_path = os.path.join(source_dir, f"frame_{frame_idx:04}.png")
    cv2.imwrite(source_path, frame)

    if results.pose_landmarks:
        pose_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pose_frame[:] = 0  # Make black
        pose_frame = cv2.cvtColor(pose_frame, cv2.COLOR_GRAY2BGR)
        mp_drawing.draw_landmarks(pose_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_path = os.path.join(pose_dir, f"frame_{frame_idx:04}_pose.png")
        cv2.imwrite(pose_path, pose_frame)

    frame_idx += 1

cap.release()
