"""
Utility functions for logging 3D information.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def get_camera_intrinsics_from_fov(
    fov_x: float, fov_y: float, width: int, height: int
) -> torch.Tensor:
    """
    Convert field of view to intrinsic matrix.

    Args:
        fov_x: Horizontal field of view in radians
        fov_y: Vertical field of view in radians
        width: Image width
        height: Image height

    Returns:
        K: (3, 3) intrinsic matrix
    """
    fx = width / (2 * np.tan(fov_x / 2))
    fy = height / (2 * np.tan(fov_y / 2))
    cx = width / 2
    cy = height / 2

    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

    return K


def unproject_depth_to_points(
    depth: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unproject depth map to 3D points in world coordinates.

    Args:
        depth: (H, W) depth map
        K: (3, 3) intrinsic matrix
        c2w: (4, 4) camera-to-world transformation matrix
        valid_mask: Optional (H, W) boolean mask for valid depth values

    Returns:
        points: (N, 3) 3D points in world coordinates
        pixel_coords: (N, 2) pixel coordinates (y, x) for each point
    """
    H, W = depth.shape
    device = depth.device

    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Flatten
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    depth_flat = depth.reshape(-1)

    # Apply valid mask if provided
    if valid_mask is not None:
        valid_flat = valid_mask.reshape(-1)
        x_flat = x_flat[valid_flat]
        y_flat = y_flat[valid_flat]
        depth_flat = depth_flat[valid_flat]
    else:
        # Default: filter out zero/invalid depth
        valid = (depth_flat > 0) & torch.isfinite(depth_flat)
        x_flat = x_flat[valid]
        y_flat = y_flat[valid]
        depth_flat = depth_flat[valid]

    if len(depth_flat) == 0:
        return torch.zeros((0, 3), device=device), torch.zeros(
            (0, 2), device=device, dtype=torch.long
        )

    # Unproject to camera coordinates
    # (x - cx) / fx * depth, (y - cy) / fy * depth, depth
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (x_flat - cx) / fx * depth_flat
    y_cam = (y_flat - cy) / fy * depth_flat
    z_cam = depth_flat

    # Stack to (N, 3)
    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    # Transform to world coordinates
    # c2w is (4, 4), points_cam is (N, 3)
    R = c2w[:3, :3]  # (3, 3)
    t = c2w[:3, 3]  # (3,)

    points_world = points_cam @ R.T + t

    # Return pixel coordinates as well (for sampling RGB/features)
    pixel_coords = torch.stack([y_flat, x_flat], dim=-1).long()

    return points_world, pixel_coords


def sample_points_with_rgb(
    depth: torch.Tensor,
    rgb_image: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    sample_ratio: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3D points from depth map with corresponding RGB values.

    Args:
        depth: (H, W) depth map
        rgb_image: (3, H, W) RGB image in [0, 1]
        K: (3, 3) intrinsic matrix
        c2w: (4, 4) camera-to-world transformation matrix
        sample_ratio: Fraction of valid pixels to sample
        valid_mask: Optional (H, W) boolean mask for valid depth values

    Returns:
        points: (N, 3) sampled 3D points in world coordinates
        rgb: (N, 3) corresponding RGB values in [0, 1]
    """
    # Get all valid points first
    points, pixel_coords = unproject_depth_to_points(depth, K, c2w, valid_mask)

    if len(points) == 0:
        return points, torch.zeros((0, 3), device=points.device)

    # Sample a fraction of points
    num_points = len(points)
    num_samples = max(1, int(num_points * sample_ratio))

    if num_samples < num_points:
        indices = torch.randperm(num_points, device=points.device)[:num_samples]
        points = points[indices]
        pixel_coords = pixel_coords[indices]

    # Get RGB values at sampled pixel locations
    # rgb_image is (3, H, W), pixel_coords is (N, 2) with (y, x)
    rgb = rgb_image[:, pixel_coords[:, 0], pixel_coords[:, 1]]  # (3, N)
    rgb = rgb.T  # (N, 3)

    return points, rgb


def sample_points_with_features(
    depth: torch.Tensor,
    feature_map: torch.Tensor,
    seg_map: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    sample_ratio: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample 3D points from depth map with corresponding language features.

    Uses the segmentation map to look up which feature each pixel belongs to,
    following the same approach as get_language_feature in cameras.py.

    Args:
        depth: (H, W) depth map
        feature_map: (num_patches, feature_dim) patch features
        seg_map: (1, H, W) segmentation map mapping pixels to patch indices (-1 for invalid)
        K: (3, 3) intrinsic matrix
        c2w: (4, 4) camera-to-world transformation matrix
        sample_ratio: Fraction of valid pixels to sample
        valid_mask: Optional (H, W) boolean mask for valid depth values

    Returns:
        points: (N, 3) sampled 3D points in world coordinates
        features: (N, feature_dim) corresponding language features
        valid_feature_mask: (N,) boolean mask indicating valid features (seg != -1)
    """
    H, W = depth.shape
    device = depth.device

    # Combine depth valid mask with feature valid mask (seg != -1)
    feature_valid = seg_map[0] != -1  # (H, W)
    if valid_mask is not None:
        combined_mask = valid_mask & feature_valid
    else:
        combined_mask = feature_valid & (depth > 0) & torch.isfinite(depth)

    # Get all valid points
    points, pixel_coords = unproject_depth_to_points(depth, K, c2w, combined_mask)

    if len(points) == 0:
        feature_dim = feature_map.shape[-1]
        return (
            points,
            torch.zeros((0, feature_dim), device=device),
            torch.zeros((0,), device=device, dtype=torch.bool),
        )

    # Sample a fraction of points
    num_points = len(points)
    num_samples = max(1, int(num_points * sample_ratio))

    if num_samples < num_points:
        indices = torch.randperm(num_points, device=points.device)[:num_samples]
        points = points[indices]
        pixel_coords = pixel_coords[indices]

    # Get segment indices at sampled pixel locations
    seg_indices = seg_map[0, pixel_coords[:, 0], pixel_coords[:, 1]]  # (N,)

    # Look up features from feature map
    # seg_indices are already valid (>= 0) due to the mask
    features = feature_map[seg_indices.long()]  # (N, feature_dim)

    # All sampled points have valid features due to the combined mask
    valid_feature_mask = torch.ones(len(points), device=device, dtype=torch.bool)

    return points, features, valid_feature_mask


def get_c2w_from_camera(viewpoint_cam) -> torch.Tensor:
    """
    Extract camera-to-world transformation from a viewpoint camera.

    The camera has world_view_transform which is world-to-camera,
    so we need to invert it to get camera-to-world.

    Args:
        viewpoint_cam: Camera object with world_view_transform attribute

    Returns:
        c2w: (4, 4) camera-to-world transformation matrix
    """
    # world_view_transform is transposed (column-major) in the Camera class
    # It represents world-to-camera (w2c)
    w2c = viewpoint_cam.world_view_transform.T  # (4, 4)
    c2w = torch.inverse(w2c)
    return c2w
