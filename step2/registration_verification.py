import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# 1) Simple Functions for Loading & Matching Keypoints
###############################################################################
def load_keypoints(kp_mat_path, frame_idx):
    """
    Load keypoints for frame `frame_idx` (0-based) from kp.mat file.
    """
    keypoints = scipy.io.loadmat(kp_mat_path)
    feature_name = f'Feature_img{frame_idx+1}_00000'
    frame_features = keypoints[feature_name][0,0]
    kp = frame_features['kp']   # shape (N,2)
    desc = frame_features['desc']   # shape (N,128) or so
    return kp, desc

def find_matches(desc1, desc2, ratio_threshold=0.75):
    """
    Naive descriptor matching with ratio test.
    Returns a list of (idx1, idx2) matches.
    """
    matches = []
    for i, d1 in enumerate(desc1):
        # L2 distances from d1 to all desc2
        dist = np.linalg.norm(desc2 - d1, axis=1)
        idx_sorted = np.argsort(dist)
        best_idx = idx_sorted[0]
        second_idx = idx_sorted[1]
        best_dist = dist[best_idx]
        second_dist = dist[second_idx]
        # ratio test
        if best_dist < ratio_threshold * second_dist:
            matches.append((i, best_idx))
    return np.array(matches)

###############################################################################
# 2) Simple Functions for Loading Frames & Converting Keypoints to 3D
###############################################################################
def load_frame(cams_info, frame_idx):
    """
    Returns the rgb, depth, and focal_length for frame `frame_idx`.
    """
    frame_data = cams_info[frame_idx, 0]
    rgb = frame_data['rgb'][0,0]          # shape (H,W,3)
    depth = frame_data['depth'][0,0]      # shape (H,W)
    focal_length = frame_data['focal_lenght'][0,0]  # note the 'typo' in field name
    return rgb, depth, focal_length

def keypoints_to_3d(kp, depth, focal_length):
    """
    Convert 2D keypoints (kp) into 3D coordinates using the depth map.
    Assumes same fx, fy and principal point at center.
    """
    fx = float(focal_length[0,0]) if focal_length.size > 1 else float(focal_length)
    fy = fx
    H, W = depth.shape
    cx, cy = (W-1)/2.0, (H-1)/2.0
    
    # Round keypoints to nearest pixel
    u = np.rint(kp[:,0]).astype(int)
    v = np.rint(kp[:,1]).astype(int)
    
    # Ensure they are within bounds
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid]
    v = v[valid]
    
    z = depth[v, u]
    # Filter out zero depth
    valid_depth = z > 0
    u = u[valid_depth]
    v = v[valid_depth]
    z = z[valid_depth]
    
    # Match point_cloud_generator.py coordinate system
    X_cam = -(u - cx)*z / fx
    Y_cam = -(v - cy)*z / fy  # Note the negation to match point_cloud_generator
    Z_cam = z
    
    pts_3d = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
    return pts_3d

def get_dense_points(depth, focal_length, subsample=50):
    """Generate a denser point cloud for visualization, similar to point_cloud_generator."""
    height, width = depth.shape
    fx = float(focal_length[0,0]) if focal_length.size > 1 else float(focal_length)
    cx, cy = (width-1)/2.0, (height-1)/2.0
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Get all points where depth is valid
    valid_points = depth > 0
    
    # Subsample valid points
    valid_indices = np.where(valid_points)
    num_points = len(valid_indices[0])
    sample_size = num_points // subsample
    sample_indices = np.random.choice(num_points, sample_size, replace=False)
    
    v_valid = valid_indices[0][sample_indices]
    u_valid = valid_indices[1][sample_indices]
    z_valid = depth[v_valid, u_valid]
    
    # Calculate X and Y coordinates (matching point_cloud_generator.py)
    x_valid = -(u_valid - cx) * z_valid / fx
    y_valid = -(v_valid - cy) * z_valid / fx
    
    return np.stack([x_valid, y_valid, z_valid], axis=1)

###############################################################################
# 3) Estimate a Simple Rigid Transform (R, t) from 3D-3D Correspondences
###############################################################################
def estimate_rigid_transform(p1, p2):
    """
    p1, p2: Nx3 arrays of matched 3D points.
    Return R (3x3) and t (3,) that aligns p2 to p1, i.e.
       p1 ~ R * p2 + t
    Uses a basic Procrustes approach (SVD).
    """
    # Remove any potential outliers or zero shapes
    n = min(len(p1), len(p2))
    if n < 3:
        return np.eye(3), np.zeros(3)
    
    # 1) Compute centroids
    c1 = np.mean(p1, axis=0)
    c2 = np.mean(p2, axis=0)
    p1c = p1 - c1
    p2c = p2 - c2
    
    # 2) SVD on covariance
    H = p1c.T @ p2c
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    # fix reflection if needed
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U @ Vt
    
    t = c1 - R @ c2
    return R, t

def estimate_rigid_transform_ransac(p1, p2, max_iterations=100, distance_threshold=0.1):
    """
    Estimate rigid transform using RANSAC to handle outliers.
    """
    best_num_inliers = 0
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_inliers = None
    n_points = len(p1)
    
    if n_points < 3:
        return best_R, best_t
    
    for _ in range(max_iterations):
        # 1. Sample 3 random points
        idx = np.random.choice(n_points, 3, replace=False)
        sample1 = p1[idx]
        sample2 = p2[idx]
        
        # 2. Estimate transform from samples
        R, t = estimate_rigid_transform(sample1, sample2)
        
        # 3. Transform all points and count inliers
        p2_transformed = (R @ p2.T).T + t
        distances = np.linalg.norm(p1 - p2_transformed, axis=1)
        inliers = distances < distance_threshold
        num_inliers = np.sum(inliers)
        
        # 4. Update best if we found more inliers
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_R = R
            best_t = t
            best_inliers = inliers
    
    # Final refinement using all inliers
    if best_inliers is not None and np.sum(best_inliers) >= 3:
        R, t = estimate_rigid_transform(p1[best_inliers], p2[best_inliers])
        return R, t
    
    return best_R, best_t

###############################################################################
# 4) Visualization (Before / After Registration)
###############################################################################
def visualize_registration(points1, points2, R, t):
    """
    points1, points2: Nx3 arrays (these could be entire clouds or matched points).
    R, t: the transformation that aligns points2 to points1.
    We show side-by-side subplots: before and after.
    """
    points2_transformed = (R @ points2.T).T + t
    
    fig = plt.figure(figsize=(15,7))
    
    # Before
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points1[:,0], points1[:,1], points1[:,2], c='blue', s=1, alpha=0.3, label='Points1')
    ax1.scatter(points2[:,0], points2[:,1], points2[:,2], c='red', s=1, alpha=0.3, label='Points2')
    ax1.set_title("Before Registration")
    ax1.legend()
    ax1.set_box_aspect([1,1,1])
    
    # Set same limits for both plots
    all_points = np.vstack([points1, points2, points2_transformed])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    ranges = max_vals - min_vals
    center = (max_vals + min_vals) / 2
    max_range = np.max(ranges)
    
    for ax in [ax1]:
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        ax.view_init(elev=30, azim=45)
    
    # After - show two views
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points1[:,0], points1[:,1], points1[:,2], c='blue', s=1, alpha=0.3, label='Points1')
    ax2.scatter(points2_transformed[:,0], points2_transformed[:,1], points2_transformed[:,2],
                c='red', s=1, alpha=0.3, label='Points2 transformed')
    ax2.set_title("After Registration")
    ax2.legend()
    ax2.set_box_aspect([1,1,1])
    
    for ax in [ax2]:
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        ax.view_init(elev=30, azim=135)  # Different view angle
    
    plt.tight_layout()
    plt.show()

###############################################################################
# 5) MAIN: Putting it All Together
###############################################################################
def process_frame_pair(cams_info, frame_idx1, frame_idx2):
    """Process a pair of frames and return registration quality metrics."""
    # 1) Load frames
    rgb1, depth1, f1 = load_frame(cams_info, frame_idx1)
    rgb2, depth2, f2 = load_frame(cams_info, frame_idx2)
    
    # 2) Load keypoints
    kp1, desc1 = load_keypoints('office/kp.mat', frame_idx1)
    kp2, desc2 = load_keypoints('office/kp.mat', frame_idx2)
    
    # 3) Match descriptors
    matches = find_matches(desc1, desc2, ratio_threshold=0.85)
    
    # 4) Convert matched keypoints to 3D
    p1_list = []
    p2_list = []
    for (i1, i2) in matches:
        pt1_2d = np.array([[kp1[i1,0], kp1[i1,1]]])
        pt2_2d = np.array([[kp2[i2,0], kp2[i2,1]]])
        
        p1 = keypoints_to_3d(pt1_2d, depth1, f1)
        p2 = keypoints_to_3d(pt2_2d, depth2, f2)
        
        if len(p1) > 0 and len(p2) > 0:
            p1_list.append(p1[0])
            p2_list.append(p2[0])
    
    if len(p1_list) < 10:
        return None
        
    p1_arr = np.array(p1_list, dtype=np.float32)
    p2_arr = np.array(p2_list, dtype=np.float32)
    
    # 5) Estimate transform using RANSAC
    R, t = estimate_rigid_transform_ransac(p1_arr, p2_arr)
    
    # Calculate quality metrics
    R_error = np.linalg.norm(R - np.eye(3), 'fro')
    num_matches = len(p1_arr)
    
    return {
        'frame1': frame_idx1,
        'frame2': frame_idx2,
        'R': R,
        't': t,
        'R_error': R_error,
        'num_matches': num_matches,
        'depth1': depth1,
        'depth2': depth2,
        'f1': f1,
        'f2': f2
    }

def main():
    # Debug keypoints file first
    kp_data = scipy.io.loadmat('office/kp.mat')
    print("\nKeypoint file contents:")
    print("Keys in kp_data:", kp_data.keys())
    
    # Find all keys that start with 'Feature_img'
    feature_keys = [k for k in kp_data.keys() if k.startswith('Feature_img')]
    print("\nNumber of frames with features:", len(feature_keys))
    print("Feature frame keys:", feature_keys)
    
    if feature_keys:
        # Look at first feature frame
        first_frame = kp_data[feature_keys[0]][0,0]
        if hasattr(first_frame, 'dtype'):
            print("\nFields in first feature frame:", first_frame.dtype.names)
    
    # Original camera info loading
    cams_data = scipy.io.loadmat('office/cams_info_no_extr.mat')
    print("\nKeys in cams_data:", cams_data.keys())
    
    cams_info = cams_data['cams_info']
    print("\nShape of cams_info:", cams_info.shape)
    print("Type of cams_info:", type(cams_info))
    
    # Debug first frame
    frame0 = cams_info[0,0]
    if hasattr(frame0, 'dtype'):
        print("\nFields in first frame:", frame0.dtype.names)
    
    num_frames = len(cams_info)
    print(f"\nTotal frames available: {num_frames}")
    
    # Try all consecutive pairs
    results = []
    for i in range(num_frames - 1):
        print(f"\nProcessing frames {i} and {i+1}...")
        result = process_frame_pair(cams_info, i, i+1)
        if result is not None:
            results.append(result)
            print(f"Matches: {result['num_matches']}, R error: {result['R_error']:.3f}")
    
    if not results:
        print("No good frame pairs found!")
        return
        
    # Find best pair based on number of matches and R error
    best_result = min(results, key=lambda x: x['R_error'])
    print(f"\nBest frame pair: {best_result['frame1']} and {best_result['frame2']}")
    print(f"Number of matches: {best_result['num_matches']}")
    print(f"R error: {best_result['R_error']:.3f}")
    print("\nEstimated R:\n", best_result['R'])
    print("Estimated t:\n", best_result['t'])
    
    # Visualize best pair
    dense_points1 = get_dense_points(best_result['depth1'], best_result['f1'])
    dense_points2 = get_dense_points(best_result['depth2'], best_result['f2'])
    visualize_registration(dense_points1, dense_points2, best_result['R'], best_result['t'])

if __name__ == "__main__":
    main()
