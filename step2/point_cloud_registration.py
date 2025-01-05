import numpy as np
import scipy.io
from point_cloud_generator import load_and_process_frame, visualize_point_cloud

def load_keypoints(frame_idx):
    """Load keypoints for a specific frame."""
    keypoints = scipy.io.loadmat(f'office/kp.mat')
    feature_name = f'Feature_img{frame_idx+1}_00000'
    frame_features = keypoints[feature_name][0,0]
    return frame_features['kp'], frame_features['desc']

def find_correspondences(desc1, desc2):
    """Find corresponding points between two sets of descriptors using nearest neighbor."""
    matches = []
    # For each descriptor in first frame
    for i, desc in enumerate(desc1):
        # Find nearest neighbor in second frame
        distances = np.linalg.norm(desc2 - desc, axis=1)
        best_match = np.argmin(distances)
        best_dist = distances[best_match]
        
        # Simple ratio test to filter bad matches
        distances[best_match] = float('inf')
        second_best = np.min(distances)
        if best_dist < 0.8 * second_best:
            matches.append((i, best_match))
    
    return np.array(matches)

def procrustes_registration(points1, points2, max_iterations=1000, distance_threshold=0.5):
    """Estimate rigid transformation between two point sets using RANSAC Procrustes."""
    best_num_inliers = 0
    best_R = None
    best_t = None
    
    n_points = len(points1)
    if n_points < 4:
        raise ValueError("Need at least 4 points for registration")
    
    # Normalize point clouds to improve numerical stability
    scale1 = np.sqrt(np.sum(points1 ** 2) / n_points)
    scale2 = np.sqrt(np.sum(points2 ** 2) / n_points)
    points1_normalized = points1 / scale1
    points2_normalized = points2 / scale2
    
    for _ in range(max_iterations):
        # Randomly sample 4 points
        idx = np.random.choice(n_points, 4, replace=False)
        sample1 = points1_normalized[idx]
        sample2 = points2_normalized[idx]
        
        # Center the sampled points
        centroid1 = np.mean(sample1, axis=0)
        centroid2 = np.mean(sample2, axis=0)
        
        centered1 = sample1 - centroid1
        centered2 = sample2 - centroid2
        
        # Calculate optimal rotation
        H = centered1.T @ centered2
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = centroid2 - R @ centroid1
        
        # Count inliers on normalized points
        transformed = (R @ points1_normalized.T).T + t
        distances = np.linalg.norm(transformed - points2_normalized, axis=1)
        inliers = distances < distance_threshold
        num_inliers = np.sum(inliers)
        
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_R = R
            best_t = t * scale2 + centroid2 - best_R @ (centroid1 * scale1)  # Denormalize translation
    
    print(f"RANSAC found {best_num_inliers} inliers out of {n_points} points")
    
    # Final refinement using all inliers if we have enough
    if best_num_inliers >= 10:
        transformed = (best_R @ points1_normalized.T).T + t
        distances = np.linalg.norm(transformed - points2_normalized, axis=1)
        inliers = distances < distance_threshold
        
        inlier_points1 = points1_normalized[inliers]
        inlier_points2 = points2_normalized[inliers]
        
        centroid1 = np.mean(inlier_points1, axis=0)
        centroid2 = np.mean(inlier_points2, axis=0)
        
        centered1 = inlier_points1 - centroid1
        centered2 = inlier_points2 - centroid2
        
        H = centered1.T @ centered2
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
        
        t = centroid2 - R @ centroid1
        
        # Denormalize the transformation
        R_final = R
        t_final = t * scale2 + centroid2 * scale2 - R @ (centroid1 * scale1)
        
        return R_final, t_final
    
    return best_R, best_t

def register_consecutive_frames(cams_info, frame_idx1, frame_idx2):
    """Register two consecutive point clouds using keypoint correspondences."""
    # Load point clouds
    points1, _ = load_and_process_frame(cams_info, frame_idx1)
    points2, _ = load_and_process_frame(cams_info, frame_idx2)
    
    # Load keypoints and descriptors
    kp1, desc1 = load_keypoints(frame_idx1)
    kp2, desc2 = load_keypoints(frame_idx2)
    
    print(f"\nKeypoints shape: {kp1.shape}, {kp2.shape}")
    print(f"Descriptors shape: {desc1.shape}, {desc2.shape}")
    
    # Find correspondences
    matches = find_correspondences(desc1, desc2)
    print(f"Found {len(matches)} matches")
    
    if len(matches) < 10:
        raise ValueError(f"Not enough matches between frames {frame_idx1} and {frame_idx2}")
    
    # Convert keypoint coordinates to indices
    height, width = 512, 256  # From the image dimensions
    kp1_indices = (kp1[:, 1] * width + kp1[:, 0]).astype(int)
    kp2_indices = (kp2[:, 1] * width + kp2[:, 0]).astype(int)
    
    # Get matched 3D points
    matched_points1 = points1[kp1_indices[matches[:,0]]]
    matched_points2 = points2[kp2_indices[matches[:,1]]]
    
    print(f"Matched points shapes: {matched_points1.shape}, {matched_points2.shape}")
    
    # Estimate transformation
    R, t = procrustes_registration(matched_points1, matched_points2)
    
    return R, t, len(matches)

def main():
    # Load dataset
    data = scipy.io.loadmat('office/cams_info_no_extr.mat')
    cams_info = data['cams_info']
    
    # Try registering first two frames
    R, t, num_matches = register_consecutive_frames(cams_info, 0, 1)
    print(f"\nRegistration results between frames 0 and 1:")
    print(f"Number of matches: {num_matches}")
    print("\nRotation matrix:")
    print(R)
    print("\nTranslation vector:")
    print(t)

if __name__ == "__main__":
    main() 