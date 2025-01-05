import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_frame_data(frame_data):
    """Analyze and print detailed information about a frame."""
    print("\nAnalyzing frame data:")
    for field_name in frame_data.dtype.names:
        field = frame_data[field_name][0,0]
        print(f"\n{field_name}:")
        print(f"  Shape: {field.shape}")
        print(f"  Type: {field.dtype}")
        print(f"  Min: {np.min(field) if field.size > 0 else 'N/A'}")
        print(f"  Max: {np.max(field) if field.size > 0 else 'N/A'}")

def create_point_cloud(rgb, depth, focal_length, principal_point):
    """Generate colored point cloud from RGB-D data using camera intrinsics."""
    height, width = depth.shape
    fx, fy = focal_length
    cx, cy = principal_point
    
    # Create pixel coordinate grid
    v, u = np.mgrid[0:height, 0:width]
    
    # Apply inverse projection
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack coordinates and reshape
    points = np.stack([X, Y, Z], axis=-1)
    colors = rgb.astype(np.float32) / 255.0
    
    # Remove invalid points (where depth is 0 or too large)
    valid_points = Z > 0
    points = points[valid_points]
    colors = colors[valid_points]
    
    return points, colors

def load_and_process_frame(cams_info, frame_idx):
    """Load and process a single frame from the dataset."""
    frame_data = cams_info[frame_idx, 0]
    
    # Extract data from frame
    rgb = frame_data['rgb'][0,0]
    depth = frame_data['depth'][0,0]
    focal_length = frame_data['focal_lenght'][0,0]  # Note the typo in the field name
    
    # Print frame information
    print(f"\nProcessing frame {frame_idx}:")
    print(f"RGB shape: {rgb.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Focal length: {focal_length}")
    
    height, width = depth.shape
    principal_point = (width/2, height/2)  # As specified in the project description
    
    # Generate point cloud
    points, colors = create_point_cloud(
        rgb,
        depth,
        (focal_length, focal_length),
        principal_point
    )
    
    return points, colors

def visualize_point_cloud(points, colors, subsample=100):
    """Visualize point cloud using matplotlib (subsample points for speed)."""
    # Subsample points for visualization
    idx = np.random.choice(len(points), len(points)//subsample, replace=False)
    points = points[idx]
    colors = colors[idx]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=colors[...,:3], s=1)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    plt.show()

def main():
    # Load dataset
    data = scipy.io.loadmat('office/cams_info_no_extr.mat')
    cams_info = data['cams_info']
    
    # Analyze first frame
    frame_data = cams_info[0, 0]
    analyze_frame_data(frame_data)
    
    # Process and visualize first frame
    points, colors = load_and_process_frame(cams_info, 0)
    print(f"\nGenerated point cloud with {len(points)} points")
    
    visualize_point_cloud(points, colors)

if __name__ == "__main__":
    main() 