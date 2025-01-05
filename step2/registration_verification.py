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
    matches = find_matches(desc1, desc2)
    
    print(f"Found {len(matches)} matches between frames {frame_idx1} and {frame_idx2}")
    
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