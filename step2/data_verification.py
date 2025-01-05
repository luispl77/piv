import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def analyze_single_frame(frame_data, frame_idx):
    """Analyze a single frame's data structure and content."""
    print(f"\n=== Frame {frame_idx} Analysis ===")
    
    # Extract and analyze RGB
    rgb = frame_data['rgb'][0,0]
    print("\nRGB Data:")
    print(f"Shape: {rgb.shape}")
    print(f"Type: {rgb.dtype}")
    print(f"Value range: [{rgb.min():.2f}, {rgb.max():.2f}]")
    
    # Extract and analyze depth
    depth = frame_data['depth'][0,0]
    print("\nDepth Data:")
    print(f"Shape: {depth.shape}")
    print(f"Type: {depth.dtype}")
    print(f"Value range: [{depth.min():.2f}, {depth.max():.2f}] (in meters)")
    print(f"Number of valid depths (>0): {np.sum(depth > 0)}")
    
    # Extract and analyze confidence
    conf = frame_data['conf'][0,0]
    print("\nConfidence Data:")
    print(f"Shape: {conf.shape}")
    print(f"Type: {conf.dtype}")
    print(f"Value range: [{conf.min():.2f}, {conf.max():.2f}]")
    print(f"Mean confidence: {conf.mean():.2f}")
    
    # Extract and analyze focal length
    focal = frame_data['focal_lenght'][0,0]
    print("\nFocal Length:")
    print(f"Value: {focal}")
    
    # Visualize data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RGB image
    axes[0,0].imshow(rgb)
    axes[0,0].set_title('RGB Image')
    
    # Depth map
    depth_vis = depth.copy()
    depth_vis[depth_vis == 0] = np.nan  # Don't show invalid depths
    im = axes[0,1].imshow(depth_vis)
    plt.colorbar(im, ax=axes[0,1])
    axes[0,1].set_title('Depth Map (meters)')
    
    # Confidence map
    im = axes[1,0].imshow(conf)
    plt.colorbar(im, ax=axes[1,0])
    axes[1,0].set_title('Confidence Map')
    
    # Depth histogram
    valid_depths = depth[depth > 0]
    axes[1,1].hist(valid_depths.flatten(), bins=50)
    axes[1,1].set_title('Depth Distribution')
    axes[1,1].set_xlabel('Depth (meters)')
    axes[1,1].set_ylabel('Pixel Count')
    
    plt.tight_layout()
    plt.show()

def analyze_keypoints(keypoints, frame_idx):
    """Analyze keypoint data for a specific frame."""
    feature_name = f'Feature_img{frame_idx+1}_00000'
    frame_features = keypoints[feature_name][0,0]
    
    kp = frame_features['kp']
    desc = frame_features['desc']
    
    print(f"\n=== Keypoint Analysis for Frame {frame_idx} ===")
    print(f"Number of keypoints: {len(kp)}")
    print(f"Keypoint coordinates range:")
    print(f"X: [{kp[:,0].min():.1f}, {kp[:,0].max():.1f}]")
    print(f"Y: [{kp[:,1].min():.1f}, {kp[:,1].max():.1f}]")
    
    # Visualize keypoint distribution
    plt.figure(figsize=(10, 5))
    plt.scatter(kp[:,0], kp[:,1], alpha=0.5)
    plt.title(f'Keypoint Distribution - Frame {frame_idx}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def main():
    # Load dataset
    print("Loading data...")
    data = scipy.io.loadmat('office/cams_info_no_extr.mat')
    keypoints = scipy.io.loadmat('office/kp.mat')
    cams_info = data['cams_info']
    
    # Analyze first frame in detail
    frame_data = cams_info[0, 0]
    analyze_single_frame(frame_data, 0)
    
    # Analyze keypoints
    analyze_keypoints(keypoints, 0)
    
    # Print summary of all frames
    print("\n=== Dataset Summary ===")
    print(f"Total number of frames: {len(cams_info)}")
    
    # Check consistency across frames
    focal_lengths = []
    image_shapes = []
    depth_ranges = []
    
    for i in range(len(cams_info)):
        frame = cams_info[i, 0]
        focal_lengths.append(frame['focal_lenght'][0,0][0,0])
        image_shapes.append(frame['rgb'][0,0].shape)
        depth = frame['depth'][0,0]
        depth_ranges.append((depth[depth > 0].min(), depth[depth > 0].max()))
    
    print("\nFocal Length Variation:")
    print(f"Min: {min(focal_lengths):.2f}")
    print(f"Max: {max(focal_lengths):.2f}")
    print(f"Mean: {np.mean(focal_lengths):.2f}")
    
    print("\nImage Shape Consistency:")
    if len(set(str(s) for s in image_shapes)) == 1:
        print(f"All frames have consistent shape: {image_shapes[0]}")
    else:
        print("WARNING: Inconsistent image shapes across frames!")
        for i, shape in enumerate(image_shapes):
            print(f"Frame {i}: {shape}")
    
    print("\nDepth Ranges (meters):")
    print(f"Min depth across all frames: {min(d[0] for d in depth_ranges):.2f}")
    print(f"Max depth across all frames: {max(d[1] for d in depth_ranges):.2f}")

if __name__ == "__main__":
    main() 