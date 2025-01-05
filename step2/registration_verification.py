import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from point_cloud_generator import create_point_cloud

def load_keypoints(frame_idx):
    """Load keypoints for a specific frame."""
    keypoints = scipy.io.loadmat('office/kp.mat')
    feature_name = f'Feature_img{frame_idx+1}_00000'
    frame_features = keypoints[feature_name][0,0]
    return frame_features['kp'], frame_features['desc']

def find_matches(desc1, desc2, ratio_threshold=0.8):
    """Find matches between descriptors using ratio test."""
    matches = []
    for i, desc in enumerate(desc1):
        # Compute distances to all descriptors in desc2
        distances = np.linalg.norm(desc2 - desc, axis=1)
        
        # Find best and second best matches
        idx = np.argsort(distances)
        best_dist = distances[idx[0]]
        second_best = distances[idx[1]]
        
        # Apply ratio test
        if best_dist < ratio_threshold * second_best:
            matches.append((i, idx[0]))
    
    return np.array(matches)

def visualize_matches(points1, points2, matches, title="Point Cloud Matches"):
    """Visualize matched points between two point clouds."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points with low alpha
    ax.scatter(points1[:,0], points1[:,1], points1[:,2], c='blue', s=1, alpha=0.1, label='Cloud 1')
    ax.scatter(points2[:,0], points2[:,1], points2[:,2], c='red', s=1, alpha=0.1, label='Cloud 2')
    
    # Plot matched points with high alpha
    matched_points1 = points1[matches[:,0]]
    matched_points2 = points2[matches[:,1]]
    
    ax.scatter(matched_points1[:,0], matched_points1[:,1], matched_points1[:,2], 
              c='blue', s=20, alpha=1, label='Matches 1')
    ax.scatter(matched_points2[:,0], matched_points2[:,1], matched_points2[:,2], 
              c='red', s=20, alpha=1, label='Matches 2')
    
    # Draw lines between matches
    for i in range(len(matches)):
        p1 = matched_points1[i]
        p2 = matched_points2[i]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g-', alpha=0.5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    plt.show()

def visualize_registration_result(points1, points2, R, t, title="Registration Result"):
    """Visualize point clouds before and after registration."""
    # Transform second point cloud
    points2_transformed = (R @ points2.T).T + t
    
    # Create two subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Before registration
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points1[:,0], points1[:,1], points1[:,2], c='blue', s=1, alpha=0.5, label='Cloud 1')
    ax1.scatter(points2[:,0], points2[:,1], points2[:,2], c='red', s=1, alpha=0.5, label='Cloud 2')
    ax1.set_title('Before Registration')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    ax1.legend()
    ax1.set_box_aspect([1,1,1])
    
    # After registration
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points1[:,0], points1[:,1], points1[:,2], c='blue', s=1, alpha=0.5, label='Cloud 1')
    ax2.scatter(points2_transformed[:,0], points2_transformed[:,1], points2_transformed[:,2], 
                c='red', s=1, alpha=0.5, label='Cloud 2 (Transformed)')
    ax2.set_title('After Registration')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_zlabel('Z (meters)')
    ax2.legend()
    ax2.set_box_aspect([1,1,1])
    
    plt.suptitle(title)
    plt.show()
    
    # Print transformation details
    print("\nTransformation Details:")
    # Convert rotation matrix to Euler angles
    euler_angles = np.array([
        np.arctan2(R[2,1], R[2,2]),
        np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2)),
        np.arctan2(R[1,0], R[0,0])
    ]) * 180 / np.pi
    
    print(f"Rotation angles (degrees):")
    print(f"  Roll:  {euler_angles[0]:.2f}°")
    print(f"  Pitch: {euler_angles[1]:.2f}°")
    print(f"  Yaw:   {euler_angles[2]:.2f}°")
    print(f"\nTranslation (meters):")
    print(f"  X: {t[0]:.3f}")
    print(f"  Y: {t[1]:.3f}")
    print(f"  Z: {t[2]:.3f}")

def get_3d_points_from_keypoints(kp, depth, focal_length, principal_point):
    """Convert 2D keypoints to 3D points using depth data."""
    fx, fy = float(focal_length[0,0]), float(focal_length[0,0])
    cx, cy = principal_point
    
    # Round keypoint coordinates to get depth values
    u = np.round(kp[:,0]).astype(int)
    v = np.round(kp[:,1]).astype(int)
    
    # Get depth values for keypoints
    z = depth[v, u]
    
    # Back-project to 3D
    x = -(u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.stack([x, y, z], axis=1)

def verify_keypoint_mapping(frame_data, kp, title="Keypoint 3D Mapping"):
    """Visualize how 2D keypoints map to 3D space."""
    rgb = frame_data['rgb'][0,0]
    depth = frame_data['depth'][0,0]
    focal_length = frame_data['focal_lenght'][0,0]
    height, width = depth.shape
    principal_point = ((width-1)/2, (height-1)/2)
    
    # Round keypoint coordinates to get depth values
    u = np.round(kp[:,0]).astype(int)
    v = np.round(kp[:,1]).astype(int)
    
    # Ensure keypoints are within image bounds
    valid_coords = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_depths = np.zeros_like(u, dtype=bool)
    valid_depths[valid_coords] = depth[v[valid_coords], u[valid_coords]] > 0
    
    # Get full point cloud
    points, colors = create_point_cloud(rgb, depth, focal_length, principal_point)
    
    # Get 3D points for keypoints
    keypoint_3d = get_3d_points_from_keypoints(kp, depth, focal_length, principal_point)
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. RGB Image with keypoints
    ax1 = fig.add_subplot(221)
    ax1.imshow(rgb)
    ax1.scatter(kp[valid_depths,0], kp[valid_depths,1], c='g', s=20, label='Valid Keypoints')
    ax1.scatter(kp[~valid_depths,0], kp[~valid_depths,1], c='r', s=20, label='Invalid Keypoints')
    ax1.set_title('RGB Image with Keypoints')
    ax1.legend()
    
    # 2. Depth map with keypoints
    ax2 = fig.add_subplot(222)
    depth_vis = depth.copy()
    depth_vis[depth_vis == 0] = np.nan  # Make invalid depths transparent
    im = ax2.imshow(depth_vis, cmap='viridis')
    ax2.scatter(kp[valid_depths,0], kp[valid_depths,1], c='g', s=20, label='Valid Keypoints')
    ax2.scatter(kp[~valid_depths,0], kp[~valid_depths,1], c='r', s=20, label='Invalid Keypoints')
    ax2.set_title('Depth Map with Keypoints')
    plt.colorbar(im, ax=ax2, label='Depth (meters)')
    ax2.legend()
    
    # 3. 3D point cloud with keypoints
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(points[:,0], points[:,1], points[:,2], c='gray', s=1, alpha=0.1, label='Full Cloud')
    scatter = ax3.scatter(keypoint_3d[valid_depths,0], 
                         keypoint_3d[valid_depths,1], 
                         keypoint_3d[valid_depths,2],
                         c=depth[v[valid_depths], u[valid_coords]],
                         cmap='viridis',
                         s=50, alpha=1, label='Valid Keypoints')
    plt.colorbar(scatter, ax=ax3, label='Depth (meters)')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    ax3.set_zlabel('Z (meters)')
    ax3.set_title('3D Point Cloud with Keypoints')
    ax3.legend()
    ax3.set_box_aspect([1,1,1])
    
    # 4. Local patches around some sample keypoints
    ax4 = fig.add_subplot(224)
    ax4.imshow(rgb)
    # Sample 5 random valid keypoints
    sample_idx = np.random.choice(np.where(valid_depths)[0], min(5, np.sum(valid_depths)), replace=False)
    colors = ['r', 'g', 'b', 'y', 'm']
    for i, idx in enumerate(sample_idx):
        x, y = kp[idx]
        # Draw patch boundary
        patch_size = 20
        rect = plt.Rectangle((x-patch_size/2, y-patch_size/2), patch_size, patch_size, 
                           fill=False, color=colors[i])
        ax4.add_patch(rect)
        # Add 3D coordinate annotation
        coord = keypoint_3d[idx]
        ax4.annotate(f'P{i+1}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})',
                    (x+patch_size/2, y+patch_size/2), color=colors[i])
    ax4.set_title('Sample Keypoint Patches with 3D Coordinates')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nKeypoint Mapping Statistics:")
    print(f"Total keypoints: {len(kp)}")
    print(f"Keypoints within image bounds: {np.sum(valid_coords)}")
    print(f"Keypoints with valid depth: {np.sum(valid_depths)}")
    if np.sum(valid_depths) > 0:
        depths = depth[v[valid_depths], u[valid_coords]]
        print(f"Depth range for valid keypoints: {depths.min():.3f} to {depths.max():.3f} meters")
        
        # Print some sample 3D coordinates
        print("\nSample keypoint 3D coordinates:")
        for i, idx in enumerate(sample_idx):
            coord = keypoint_3d[idx]
            print(f"P{i+1}: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}) meters")

def main():
    # Load dataset
    data = scipy.io.loadmat('office/cams_info_no_extr.mat')
    cams_info = data['cams_info']
    
    # Process two consecutive frames
    frame_idx1, frame_idx2 = 0, 1
    
    # Load frames
    frame1 = cams_info[frame_idx1, 0]
    frame2 = cams_info[frame_idx2, 0]
    
    # Generate point clouds
    height, width = frame1['depth'][0,0].shape
    principal_point = ((width-1)/2, (height-1)/2)
    
    points1, _ = create_point_cloud(
        frame1['rgb'][0,0],
        frame1['depth'][0,0],
        frame1['focal_lenght'][0,0],
        principal_point
    )
    
    points2, _ = create_point_cloud(
        frame2['rgb'][0,0],
        frame2['depth'][0,0],
        frame2['focal_lenght'][0,0],
        principal_point
    )
    
    # Load and match keypoints
    kp1, desc1 = load_keypoints(frame_idx1)
    kp2, desc2 = load_keypoints(frame_idx2)
    
    # Verify keypoint mapping for both frames
    print("\nVerifying Frame 1 Keypoint Mapping:")
    verify_keypoint_mapping(frame1, kp1, "Frame 1 Keypoint Mapping")
    
    print("\nVerifying Frame 2 Keypoint Mapping:")
    verify_keypoint_mapping(frame2, kp2, "Frame 2 Keypoint Mapping")
    
    # Continue with existing matching and visualization
    matches = find_matches(desc1, desc2)
    print(f"\nFound {len(matches)} matches between frames {frame_idx1} and {frame_idx2}")
    
    # Visualize matches
    visualize_matches(points1, points2, matches, 
                     f"Matches between frames {frame_idx1} and {frame_idx2}")
    
    # TODO: Add your registration code here
    # For now, just use identity transformation
    R = np.eye(3)
    t = np.zeros(3)
    
    # Visualize registration result
    visualize_registration_result(points1, points2, R, t,
                                f"Registration between frames {frame_idx1} and {frame_idx2}")

if __name__ == "__main__":
    main() 