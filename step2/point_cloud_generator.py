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

def create_point_cloud(rgb, depth, conf, focal_length, principal_point):
    """Generate colored point cloud from RGB-D data using camera intrinsics."""
    height, width = depth.shape
    fx, fy = float(focal_length[0,0]), float(focal_length[0,0])  # Same focal length for both axes
    cx, cy = principal_point
    
    print(f"\nDebug shapes:")
    print(f"depth shape: {depth.shape}")
    print(f"focal_length: {fx}, {fy}")
    print(f"principal_point: {cx}, {cy}")
    
    # Create pixel coordinate grid
    # Note: u (x) should correspond to width, v (y) to height
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Filter points based on depth and confidence
    conf_threshold = 4.0  # Below mean confidence (4.57)
    depth_min, depth_max = 0.7, 4.5  # From data analysis
    
    # Create valid points mask
    valid_points = (depth > depth_min) & (depth < depth_max) & (conf > conf_threshold)
    print(f"Number of valid points: {np.sum(valid_points)}")
    
    # Get valid coordinates and depth
    v_valid = v[valid_points]  # y-coordinate in image
    u_valid = u[valid_points]  # x-coordinate in image
    z_valid = depth[valid_points]  # depth values
    
    # Calculate X and Y coordinates in 3D space
    # Using pinhole camera model:
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    
    print(f"Coordinate statistics:")
    print(f"X range: [{x_valid.min():.2f}, {x_valid.max():.2f}]")
    print(f"Y range: [{y_valid.min():.2f}, {y_valid.max():.2f}]")
    print(f"Z range: [{z_valid.min():.2f}, {z_valid.max():.2f}]")
    
    # Stack coordinates - using right-handed coordinate system
    # Convert from camera coordinates to world coordinates:
    # - X remains right
    # - Y becomes up (negative of camera Y)
    # - Z remains forward
    points = np.stack([x_valid, -y_valid, z_valid], axis=1)
    
    # Get colors for valid points
    colors = rgb[valid_points]
    
    # Store pixel indices for keypoint matching
    pixel_indices = v_valid * width + u_valid
    
    print(f"\nFinal point cloud:")
    print(f"Number of points: {len(points)}")
    print(f"Points range: X [{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
          f"Y [{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
          f"Z [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    
    return points, colors, pixel_indices

def load_and_process_frame(cams_info, frame_idx):
    """Load and process a single frame from the dataset."""
    frame_data = cams_info[frame_idx, 0]
    
    # Extract data from frame
    rgb = frame_data['rgb'][0,0]
    depth = frame_data['depth'][0,0]
    conf = frame_data['conf'][0,0]
    focal_length = frame_data['focal_lenght'][0,0]  # Note the typo in the field name
    
    # Print frame information
    print(f"\nProcessing frame {frame_idx}:")
    print(f"RGB shape: {rgb.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Focal length: {focal_length}")
    
    height, width = depth.shape
    # Principal point should be at image center
    # For a 512x256 image, this is (255.5, 127.5)
    principal_point = ((width-1)/2, (height-1)/2)
    
    # Generate point cloud
    points, colors, pixel_indices = create_point_cloud(
        rgb,
        depth,
        conf,
        focal_length,
        principal_point
    )
    
    return points, colors, pixel_indices

def visualize_point_cloud(points, colors, subsample=100):
    """Visualize point cloud using matplotlib (subsample points for speed)."""
    # Subsample points for visualization
    idx = np.random.choice(len(points), len(points)//subsample, replace=False)
    points = points[idx]
    colors = colors[idx]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with colors
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors[...,:3], s=2)
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'Point Cloud Visualization ({len(points)} points)')
    
    # Print coordinate ranges
    print(f"\nCoordinate ranges:")
    print(f"X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}] meters")
    print(f"Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}] meters")
    print(f"Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}] meters")
    
    # Set equal aspect ratio and adjust view
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=20, azim=45)  # Adjust view angle
    
    plt.show()

def main():
    # Load dataset
    data = scipy.io.loadmat('office/cams_info_no_extr.mat')
    cams_info = data['cams_info']
    
    # Analyze first frame
    frame_data = cams_info[0, 0]
    analyze_frame_data(frame_data)
    
    # Process and visualize first frame
    points, colors, pixel_indices = load_and_process_frame(cams_info, 0)
    print(f"\nGenerated point cloud with {len(points)} points")
    
    visualize_point_cloud(points, colors)

if __name__ == "__main__":
    main() 