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
    fx, fy = float(focal_length[0,0]), float(focal_length[0,0])
    cx, cy = principal_point
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Get all points where depth is valid
    valid_points = depth > 0
    
    # Get coordinates
    v_valid = v[valid_points]
    u_valid = u[valid_points]
    z_valid = depth[valid_points]
    
    # Calculate X and Y coordinates
    x_valid = -(u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    
    # Stack coordinates
    points = np.stack([x_valid, -y_valid, z_valid], axis=1)
    colors = rgb[valid_points]
    
    return points, colors

def visualize_point_cloud_with_camera(points, colors, subsample=100):
    """Visualize point cloud with camera position and orientation."""
    # Subsample points for visualization
    idx = np.random.choice(len(points), len(points)//subsample, replace=False)
    points = points[idx]
    colors = colors[idx]
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors[...,:3], s=1, alpha=0.5)
    
    # Draw camera
    camera_scale = 0.2
    # Camera center
    ax.scatter([0], [0], [0], color='red', s=100, label='Camera')
    
    # Camera axes
    ax.quiver(0, 0, 0, camera_scale, 0, 0, color='red', label='X (right)')
    ax.quiver(0, 0, 0, 0, camera_scale, 0, color='green', label='Y (up)')
    ax.quiver(0, 0, 0, 0, 0, camera_scale, color='blue', label='Z (forward)')
    
    # Camera frustum
    z = camera_scale
    x = np.array([-0.1, 0.1, 0.1, -0.1, -0.1]) * z
    y = np.array([-0.1, -0.1, 0.1, 0.1, -0.1]) * z
    z = np.ones_like(x) * z
    
    # Draw frustum lines from origin to edges
    for i in range(4):
        ax.plot3D([0, x[i]], [0, y[i]], [0, z[i]], 'gray', alpha=0.5)
    # Draw frustum front face
    ax.plot3D(x, y, z, 'gray', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X (meters) →')
    ax.set_ylabel('Y (meters) →')
    ax.set_zlabel('Z (meters) →')
    ax.set_title('Point Cloud with Camera Position')
    
    # Add legend
    ax.legend()
    
    # Print ranges
    print(f"\nCoordinate ranges:")
    print(f"X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}] meters")
    print(f"Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}] meters")
    print(f"Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}] meters")
    
    # Set equal aspect ratio and adjust view
    ax.set_box_aspect([1,1,1])
    
    # Set view to see both camera and points
    ax.view_init(elev=20, azim=45)
    
    plt.show()

def main():
    # Load dataset
    data = scipy.io.loadmat('office/cams_info_no_extr.mat')
    cams_info = data['cams_info']
    
    # Get first frame
    frame_data = cams_info[0, 0]
    
    # Extract data
    rgb = frame_data['rgb'][0,0]
    depth = frame_data['depth'][0,0]
    focal_length = frame_data['focal_lenght'][0,0]
    
    # Set principal point
    height, width = depth.shape
    principal_point = ((width-1)/2, (height-1)/2)
    
    # Generate and visualize point cloud
    points, colors = create_point_cloud(rgb, depth, focal_length, principal_point)
    print(f"Generated point cloud with {len(points)} points")
    
    visualize_point_cloud_with_camera(points, colors, subsample=50)

if __name__ == "__main__":
    main() 