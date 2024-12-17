import os
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont
import json

# Folder structure
DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"
DEBUG_FOLDER = "debug"

# Create output directories if they don't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# Load keypoint matches from kp_gmaps.mat
def load_keypoints(filepath):
    data = sio.loadmat(filepath)
    return data['kp_gmaps'][:, :2], data['kp_gmaps'][:, 2:]

# Compute homography matrix H using DLT
def compute_homography(points_video, points_map):
    A = []
    for (x, y), (X, Y) in zip(points_video, points_map):
        A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))
    return H / H[2, 2]

# Apply homography to bounding box points
def apply_homography(points, H):
    points = np.array(points)
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points_h = points_h @ H.T
    transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2:3]
    return transformed_points

# Validate bounding box coordinates
def validate_bbox(bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return False  # Discard if any coordinate is negative
    if x1 >= x2 or y1 >= y2:
        return False  # Discard if corners are invalid
    return True

# Swap bounding box coordinates if needed
def fix_bbox_coordinates(bbox):
    x1, y1, x2, y2 = bbox
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]

# Process each frame and YOLO file
def process_frames_and_yolo(H, frames_folder, output_folder, aerial_image_path):
    # Load class mappings
    with open('yolo_classes.json', 'r') as f:
        class_map = json.load(f)['class']
    
    frame_files = sorted(f for f in os.listdir(frames_folder) if f.endswith('.jpg'))

    # Load the aerial image as the canvas
    aerial_img = Image.open(aerial_image_path)
    
    # Create font object
    try:
        font = ImageFont.truetype("arial.ttf", 60)  # Doubled the size
    except:
        font = ImageFont.load_default()  # Fallback to default if arial not found

    for i, frame_file in enumerate(frame_files):
        yolo_file = frame_file.replace('img_', 'yolo_').replace('.jpg', '.mat')
        yolo_path = os.path.join(frames_folder, yolo_file)

        # Load YOLO data
        yolo_data = sio.loadmat(yolo_path)
        bboxes = yolo_data['xyxy']  # Bounding boxes
        classes = yolo_data['class'].flatten()  # Get class labels

        print(f"Processing {frame_file}: {len(bboxes)} bounding boxes found")
        for bbox in bboxes:
            print(f"Raw bbox: {bbox}")

        # Transform bounding boxes
        transformed_bboxes = []
        valid_classes = []  # Keep track of classes for valid boxes
        for bbox, cls in zip(bboxes, classes):
            blc = bbox[:2]  # Bottom-left corner
            trc = bbox[2:]  # Top-right corner
            transformed_blc = apply_homography([blc], H)[0]
            transformed_trc = apply_homography([trc], H)[0]

            # Ensure valid transformed bounding boxes
            x1, y1 = transformed_blc
            x2, y2 = transformed_trc
            if x1 > x2: x1, x2 = x2, x1  # Swap if necessary
            if y1 > y2: y1, y2 = y2, y1  # Swap if necessary

            bbox_valid = [x1, y1, x2, y2]
            if validate_bbox(bbox_valid):
                transformed_bboxes.append(bbox_valid)
                valid_classes.append(cls)  # Store class for valid box
        transformed_bboxes = np.array(transformed_bboxes)

        # Draw transformed bounding boxes and labels on aerial image
        img = aerial_img.copy()
        draw = ImageDraw.Draw(img)
        for bbox, cls in zip(transformed_bboxes, valid_classes):
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=2)
            
            # Get class label from JSON mapping
            label = class_map[str(int(cls))]
            
            # Get text size for background rectangle
            text_bbox = draw.textbbox((bbox[0], bbox[1] - 35), label, font=font)
            
            # Draw background rectangle for text
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5], 
                         fill="red")
            
            # Draw white text on red background
            draw.text((bbox[0], bbox[1] - 35), label, fill="white", font=font)
        
        # Save transformed image
        img_output_path = os.path.join(output_folder, f"output_{i+1:05d}.jpg")
        img.convert('RGB').save(img_output_path)

        # Save transformed YOLO detections
        yolo_output_path = os.path.join(output_folder, f"yolooutput_{i+1:05d}.mat")
        sio.savemat(yolo_output_path, {'bbox': transformed_bboxes})

        # DEBUG: Draw bounding boxes on the original input frame
        input_frame_path = os.path.join(frames_folder, frame_file)
        input_img = Image.open(input_frame_path)
        draw_debug = ImageDraw.Draw(input_img)
        for bbox in bboxes:
            print(f"Drawing raw bbox on debug: {bbox}")
            bbox_fixed = fix_bbox_coordinates(bbox)
            draw_debug.rectangle([bbox_fixed[0], bbox_fixed[1], bbox_fixed[2], bbox_fixed[3]], outline="blue", width=2)
        debug_output_path = os.path.join(DEBUG_FOLDER, f"debug_{i+1:05d}.jpg")
        input_img.convert('RGB').save(debug_output_path)

# Main workflow
def main():
    # Load keypoints and compute homography
    kp_video, kp_map = load_keypoints(os.path.join(DATA_FOLDER, 'kp_gmaps.mat'))
    H = compute_homography(kp_video, kp_map)

    # Save homography matrix
    sio.savemat(os.path.join(OUTPUT_FOLDER, 'homography.mat'), {'H': H})

    # Path to the aerial image
    aerial_image_path = 'gmaps_alamedaIST.png'

    # Process frames and YOLO detections
    process_frames_and_yolo(H, DATA_FOLDER, OUTPUT_FOLDER, aerial_image_path)

if __name__ == "__main__":
    main()
