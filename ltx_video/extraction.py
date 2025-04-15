import os
import cv2
import numpy as np
from PIL import Image

def extract_pose_with_canny(image_path):
    """
    Extract pose using simple edge detection as a placeholder.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        numpy array containing the edge detection result
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_color

def create_apose_reference(width=576, height=320):
    """
    Create a reference A-pose stick figure programmatically.
    
    Args:
        width: Width of the output image
        height: Height of the output image
        
    Returns:
        numpy array containing the A-pose reference image
    """
    # Create a white background
    white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # A-pose stick figure coordinates
    # Head
    head_center = (width // 2, height // 4)
    head_radius = height // 12
    
    # Body
    neck = (width // 2, height // 3)
    hip = (width // 2, height * 2 // 3)
    
    # Arms in A-pose
    left_shoulder = (width // 2 - width // 10, neck[1] + height // 15)
    right_shoulder = (width // 2 + width // 10, neck[1] + height // 15)
    left_elbow = (width // 4, neck[1] + height // 8)
    right_elbow = (width * 3 // 4, neck[1] + height // 8)
    left_hand = (width // 8, neck[1] + height // 6)
    right_hand = (width * 7 // 8, neck[1] + height // 6)
    
    # Legs
    left_knee = (width // 2 - width // 10, height * 4 // 5)
    right_knee = (width // 2 + width // 10, height * 4 // 5)
    left_foot = (width // 2 - width // 8, height - 10)
    right_foot = (width // 2 + width // 8, height - 10)
    
    # Draw the stick figure
    # Head
    cv2.circle(white_bg, head_center, head_radius, (0, 0, 255), 2)
    
    # Body
    cv2.line(white_bg, neck, hip, (255, 0, 0), 3)
    
    # Arms
    cv2.line(white_bg, neck, left_shoulder, (255, 0, 0), 3)
    cv2.line(white_bg, neck, right_shoulder, (255, 0, 0), 3)
    cv2.line(white_bg, left_shoulder, left_elbow, (255, 0, 0), 3)
    cv2.line(white_bg, right_shoulder, right_elbow, (255, 0, 0), 3)
    cv2.line(white_bg, left_elbow, left_hand, (255, 0, 0), 3)
    cv2.line(white_bg, right_elbow, right_hand, (255, 0, 0), 3)
    
    # Legs
    cv2.line(white_bg, hip, left_knee, (255, 0, 0), 3)
    cv2.line(white_bg, hip, right_knee, (255, 0, 0), 3)
    cv2.line(white_bg, left_knee, left_foot, (255, 0, 0), 3)
    cv2.line(white_bg, right_knee, right_foot, (255, 0, 0), 3)
    
    # Draw joints
    joints = [neck, hip, left_shoulder, right_shoulder, left_elbow, right_elbow,
             left_hand, right_hand, left_knee, right_knee, left_foot, right_foot]
    for joint in joints:
        cv2.circle(white_bg, joint, 5, (0, 255, 0), -1)
    
    return white_bg

def get_pose_reference(input_image_path, output_path="pose_reference.jpg", mode="create"):
    """
    Get pose reference image using specified method.
    
    Args:
        input_image_path: Path to input image (used for extraction or dimensions)
        output_path: Path to save the pose reference image
        mode: Method to get pose reference - 'extract' or 'create'
        
    Returns:
        Path to the saved pose reference image
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found at {input_image_path}")
    
    # Get dimensions from input image
    input_image = Image.open(input_image_path)
    width, height = input_image.size
    
    # Ensure dimensions are divisible by 32
    width = (width // 32) * 32
    height = (height // 32) * 32
    if width == 0:
        width = 32
    if height == 0:
        height = 32
    
    # Get pose reference based on specified mode
    if mode == "extract":
        pose_data = extract_pose_with_canny(input_image_path)
    elif mode == "create":
        pose_data = create_apose_reference(width, height)
    else:
        raise ValueError("Mode must be 'extract' or 'create'")
    
    # Save pose reference
    cv2.imwrite(output_path, pose_data)
    print(f"Pose reference saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # This allows the script to be run directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract or create pose reference")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--output", "-o", default="pose_reference.jpg", help="Output path for pose reference")
    parser.add_argument("--mode", "-m", choices=["extract", "create"], default="create", 
                        help="Mode: 'extract' to extract pose from input image, 'create' to create A-pose reference")
    
    args = parser.parse_args()
    
    try:
        result_path = get_pose_reference(args.input_image, args.output, args.mode)
        print(f"Successfully created pose reference at: {result_path}")
    except Exception as e:
        print(f"Error: {e}")