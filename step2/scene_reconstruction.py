import numpy as np
import scipy.io
from point_cloud_generator import load_and_process_frame, visualize_point_cloud
from point_cloud_registration import register_consecutive_frames
import matplotlib.pyplot as plt

def chain_transformations(cams_info):
    """Chain transformations from consecutive frame registrations."""
    num_frames = len(cams_info)
    transforms = []
    
    # First frame is our reference frame
    transforms.append({
        'R': np.eye(3),
        'T': np.zeros(3)
    })
    
    # Accumulate transformations
    for i in range(num_frames - 1):
        print(f"\nRegistering frames {i} and {i+1}")
        R, t, num_matches = register_consecutive_frames(cams_info, i, i+1)
        
        # Chain with previous transformation
        prev_R = transforms[-1]['R']
        prev_T = transforms[-1]['T']
        
        # New transformation is composition of previous and current
        new_R = R @ prev_R
        new_T = R @ prev_T + t
        
        transforms.append({
            'R': new_R,
            'T': new_T
        })
    
    return transforms

def merge_point_clouds(cams_info, transforms):
    """Merge all point clouds using computed transformations."""
    all_points = []
    all_colors = []
    
    # Create distinct colors for each frame
    frame_colors = plt.cm.rainbow(np.linspace(0, 1, len(transforms)))
    
    for i, transform in enumerate(transforms):
        # Load point cloud
        points, _ = load_and_process_frame(cams_info, i)
        
        # Apply transformation
        transformed_points = (transform['R'] @ points.T).T + transform['T']
        
        # Assign frame-specific color
        colors = np.tile(frame_colors[i][:3], (len(points), 1))
        
        all_points.append(transformed_points)
        all_colors.append(colors)
        
        # Visualize intermediate result
        print(f"\nVisualing after adding frame {i}")
        merged_points_so_far = np.concatenate(all_points, axis=0)
        merged_colors_so_far = np.concatenate(all_colors, axis=0)
        visualize_point_cloud(merged_points_so_far, merged_colors_so_far, subsample=500)
    
    # Concatenate all points and colors
    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    
    return merged_points, merged_colors

def save_results(transforms, merged_points):
    """Save results in the required format."""
    # Save transformations
    transforms_dict = {}
    for i, transform in enumerate(transforms):
        transforms_dict[f'frame_{i}'] = {
            'R': transform['R'],
            'T': transform['T'].reshape(3, 1)  # Reshape to match required format
        }
    
    scipy.io.savemat('transforms.mat', transforms_dict)
    
    # Save merged point cloud
    output_dict = {
        'points': merged_points
    }
    scipy.io.savemat('output.mat', output_dict)

def visualize_point_cloud(points, colors, subsample=1000):
    """Visualize point cloud using matplotlib (subsample points for speed)."""
    # Subsample points for visualization
    idx = np.random.choice(len(points), len(points)//subsample, replace=False)
    points = points[idx]
    colors = colors[idx]
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors, s=1)
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'Point Cloud Visualization\n{len(points)} points shown')
    
    # Print coordinate ranges
    print(f"X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    
    # Set equal aspect ratio and adjust view
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=30, azim=45)
    
    plt.show()

def main():
    # Load dataset
    data = scipy.io.loadmat('office/cams_info_no_extr.mat')
    cams_info = data['cams_info']
    
    # Chain transformations
    print("Computing transformations chain...")
    transforms = chain_transformations(cams_info)
    
    # Merge point clouds
    print("\nMerging point clouds...")
    merged_points, merged_colors = merge_point_clouds(cams_info, transforms)
    print(f"Final point cloud has {len(merged_points)} points")
    
    # Save results
    print("\nSaving results...")
    save_results(transforms, merged_points)
    
    # Visualize result
    print("\nVisualizing merged point cloud...")
    visualize_point_cloud(merged_points, merged_colors, subsample=2000)

if __name__ == "__main__":
    main() 