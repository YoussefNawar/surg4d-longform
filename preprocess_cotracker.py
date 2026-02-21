"""
Preprocessing script for CoTracker3 control point extraction.
This script runs CoTracker3 on video frames, lifts control points to 3D using DA3 depth,
and computes Gaussian-to-control-point associations.
"""
from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import hydra
from hydra.core.global_hydra import GlobalHydra
import sys
from loguru import logger


# Import CoTracker3 utilities
from utils.cotracker_utils import (
    track_control_points,
    lift_control_points_to_3d,
    compute_gaussian_control_point_associations,
)
from utils.cotracker_interpolation import precompute_control_point_positions
from cholec_utils import get_semantic_label_from_class_id


def extract_frame_number(filepath: Path) -> int:
    """Extract frame number from filename for proper numerical sorting."""
    import re
    match = re.search(r"frame_(\d+)", filepath.stem)
    if match:
        return int(match.group(1))
    return 0


def compute_containment_ratio(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    containment_radius: float,
) -> float:
    """Compute containment ratio using KD-tree with adaptive distance threshold.
    
    Algorithm:
    1. Build KD-tree of the smaller instance
    2. Compute average distance to nearest neighbor within smaller instance (avg_dist_small)
       This captures the local point spacing/density
    3. Build KD-tree of the larger instance
    4. For each point in smaller, find nearest neighbor in larger
    5. Count points where distance < containment_radius * avg_dist_small
    
    This is adaptive because:
    - Dense point clouds get smaller distance thresholds
    - Sparse point clouds get larger distance thresholds
    - containment_radius is a multiplier (e.g., 2-3x the average spacing)
    
    Args:
        positions_a: (N_a, 3) positions of instance A
        positions_b: (N_b, 3) positions of instance B
        containment_radius: Multiplier for the average neighbor distance threshold
    
    Returns:
        Containment ratio in [0, 1], where 1 means all points in smaller set
        have a close neighbor in the larger set
    """
    from scipy.spatial import KDTree
    
    if len(positions_a) < 2 or len(positions_b) < 2:
        return 0.0
    
    # Determine which is the smaller set (we'll check containment of smaller in larger)
    if len(positions_a) <= len(positions_b):
        query_points = positions_a.astype(np.float64)
        reference_points = positions_b.astype(np.float64)
    else:
        query_points = positions_b.astype(np.float64)
        reference_points = positions_a.astype(np.float64)
    
    # Build KD-tree of smaller instance and compute average neighbor distance
    tree_small = KDTree(query_points)
    # Query k=2 because k=1 would return the point itself (distance 0)
    distances_small, _ = tree_small.query(query_points, k=2)
    avg_dist_small = distances_small[:, 1].mean()  # Use second nearest (first is self)
    
    if avg_dist_small <= 0:
        return 0.0
    
    # Adaptive threshold based on point density of smaller instance
    distance_threshold = containment_radius * avg_dist_small
    
    # Build KD-tree from the larger set
    tree_large = KDTree(reference_points)
    
    # For each point in smaller set, find distance to nearest neighbor in larger set
    distances_to_large, _ = tree_large.query(query_points, k=1)
    
    # Count how many points have a neighbor within threshold
    n_contained = np.sum(distances_to_large <= distance_threshold)
    
    # Containment ratio
    return n_contained / len(query_points)


def get_instance_semantic_ids(
    per_view_data: list,
    instance_mask_dir: Path,
    semantic_mask_dir: Path,
) -> list[dict[int, int]]:
    """For each view, compute a mapping from original instance ID -> semantic class ID.

    Reads the instance and semantic masks for each view's frame and extracts the
    dominant semantic class ID for every instance found in that frame.

    Args:
        per_view_data: List of dicts containing frame_idx, instance_ids, positions.
        instance_mask_dir: Directory containing per-frame instance masks (.npy).
        semantic_mask_dir: Directory containing per-frame semantic masks (.npy).

    Returns:
        List (one entry per view) of dicts mapping original_instance_id -> semantic_class_id.
        Instances with no pixels in the mask get semantic ID -1.
    """
    view_semantic_maps = []

    for view_data in per_view_data:
        frame_idx = view_data["frame_idx"]
        instance_ids = view_data["instance_ids"].numpy()

        instance_mask_path = instance_mask_dir / f"frame_{frame_idx:06d}.npy"
        semantic_mask_path = semantic_mask_dir / f"frame_{frame_idx:06d}.npy"

        if not instance_mask_path.exists() or not semantic_mask_path.exists():
            logger.warning(f"Missing masks for frame {frame_idx}, all instances get semantic ID -1")
            view_semantic_maps.append({int(iid): -1 for iid in np.unique(instance_ids) if iid > 0})
            continue

        instance_mask = np.load(instance_mask_path)   # (H, W)
        semantic_mask = np.load(semantic_mask_path)   # (H, W)

        semantic_map = {}
        for orig_id in np.unique(instance_ids):
            orig_id = int(orig_id)
            if orig_id <= 0:
                continue
            pixels = instance_mask == orig_id
            if pixels.sum() == 0:
                semantic_map[orig_id] = -1
                continue
            sem_ids = semantic_mask[pixels]
            unique_sem_ids = np.unique(sem_ids)
            if len(unique_sem_ids) > 1:
                raise AssertionError(
                    f"Instance {orig_id} in frame {frame_idx} has multiple semantic IDs: "
                    f"{unique_sem_ids}. Each instance must belong to exactly one semantic class."
                )
            semantic_map[orig_id] = int(unique_sem_ids[0])

        view_semantic_maps.append(semantic_map)

    return view_semantic_maps


def merge_instances_across_views(
    per_view_data: list,
    reference_timestep: int,
    containment_threshold: float,
    containment_radius: float,
    view_semantic_maps: list[dict[int, int]],
) -> np.ndarray:
    """Merge instances across different views based on spatial overlap and semantic class.

    Two instances from different views are merged if:
      1. Their containment ratio (at the reference timestep) is >= containment_threshold, AND
      2. They belong to the same semantic class.

    There is no 1:1 matching constraint — one instance from one view may merge with
    multiple instances from other views if both criteria are satisfied.
    Uses transitive closure (Union-Find) so that merge chains are handled correctly.

    Args:
        per_view_data: List of dicts containing frame_idx, instance_ids, positions.
        reference_timestep: Timestep to use for computing overlaps.
        containment_threshold: Minimum containment ratio to merge (e.g., 0.5).
        containment_radius: Multiplier for avg neighbor distance (e.g., 3.0).
        view_semantic_maps: Per-view dicts mapping original_instance_id -> semantic_class_id.

    Returns:
        merged_instance_ids: (N_total,) array with merged instance IDs (-1 = background).
    """
    n_views = len(per_view_data)

    # Extract positions and instance IDs at reference timestep
    view_positions = []
    view_instance_ids = []
    view_offsets = [0]  # Track where each view starts in the concatenated array

    for view_data in per_view_data:
        positions = view_data["positions"][reference_timestep].numpy()  # (N_view, 3)
        instance_ids = view_data["instance_ids"].numpy()  # (N_view,)

        view_positions.append(positions)
        view_instance_ids.append(instance_ids)
        view_offsets.append(view_offsets[-1] + len(instance_ids))

    # Build a flat list of all instances across all views
    instances = []
    for view_idx in range(n_views):
        positions = view_positions[view_idx]
        instance_ids = view_instance_ids[view_idx]
        unique_instances = np.unique(instance_ids)
        unique_instances = unique_instances[unique_instances > 0]  # Exclude background

        for inst_id in unique_instances:
            mask = instance_ids == inst_id
            inst_positions = positions[mask]
            inst_gaussian_indices = np.where(mask)[0]
            semantic_id = view_semantic_maps[view_idx].get(int(inst_id), -1)

            instances.append({
                "view_idx": view_idx,
                "original_id": int(inst_id),
                "semantic_id": semantic_id,
                "gaussian_indices": inst_gaussian_indices,
                "positions": inst_positions,
            })

    logger.info(f"Found {len(instances)} instances across {n_views} views")

    # Union-Find helpers
    parent = list(range(len(instances)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    n_edges = 0
    n_instances = len(instances)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            # Only consider instances from different views
            if instances[i]["view_idx"] == instances[j]["view_idx"]:
                continue

            # Semantic class must match (and be valid)
            sem_i = instances[i]["semantic_id"]
            sem_j = instances[j]["semantic_id"]
            if sem_i == -1 or sem_j == -1 or sem_i != sem_j:
                continue

            # Check spatial containment
            ratio = compute_containment_ratio(
                instances[i]["positions"],
                instances[j]["positions"],
                containment_radius,
            )

            if ratio >= containment_threshold:
                union(i, j)
                n_edges += 1
                logger.debug(
                    f"Merge: instance {instances[i]['original_id']} (view {instances[i]['view_idx']}, "
                    f"sem={sem_i}) <-> instance {instances[j]['original_id']} (view {instances[j]['view_idx']}, "
                    f"sem={sem_j}), containment={ratio:.3f}"
                )

    logger.info(f"Applied {n_edges} merge edges (same semantic class + containment >= {containment_threshold})")

    # Assign contiguous merged IDs from connected components
    component_to_merged_id = {}
    next_merged_id = 0

    for i in range(n_instances):
        component = find(i)
        if component not in component_to_merged_id:
            component_to_merged_id[component] = next_merged_id
            next_merged_id += 1

    # Count merged (non-singleton) instances for logging
    component_sizes: dict[int, int] = {}
    for i in range(n_instances):
        c = find(i)
        component_sizes[c] = component_sizes.get(c, 0) + 1
    n_merged = sum(1 for i in range(n_instances) if component_sizes[find(i)] > 1)

    logger.info(
        f"Transitive closure: {n_merged} instances merged into "
        f"{len(component_to_merged_id)} final instances"
    )

    # Build final merged instance ID array (same ordering as concatenated views)
    total_gaussians = view_offsets[-1]
    merged_instance_ids = np.full(total_gaussians, -1, dtype=np.int64)

    for idx, inst in enumerate(instances):
        view_idx = inst["view_idx"]
        view_offset = view_offsets[view_idx]
        global_indices = view_offset + inst["gaussian_indices"]
        merged_instance_ids[global_indices] = component_to_merged_id[find(idx)]

    logger.info(
        f"Total Gaussians: {total_gaussians}, "
        f"Background: {(merged_instance_ids == -1).sum()}"
    )

    return merged_instance_ids


def compute_semantic_labels_for_merged_instances(
    merged_instance_ids: np.ndarray,
    per_view_data: list,
    combined_pixel_coords: torch.Tensor,
    combined_frame_indices: torch.Tensor,
    init_frame_indices: list,
    gaussians_per_frame: list,
    clip_dir: Path,
    cfg: DictConfig,
) -> dict[int, str]:
    """Compute semantic labels for each merged instance.
    
    For each merged instance, finds the gaussians belonging to it, loads the semantic
    mask for the appropriate frame, and extracts the semantic class ID from the instance
    mask pixels. Asserts that all pixels of an instance have the same semantic ID.
    
    Args:
        merged_instance_ids: (N_total,) array with merged instance IDs (-1 for background)
        per_view_data: List of dicts with frame_idx, instance_ids, positions for each view
        combined_pixel_coords: (N_total, 2) pixel coordinates in original image space
        combined_frame_indices: (N_total,) frame index for each gaussian
        init_frame_indices: List of init frame indices used
        gaussians_per_frame: List of number of gaussians per frame (before densification)
        clip_dir: Directory containing preprocessed clip data
        cfg: Configuration dict
    
    Returns:
        Dictionary mapping merged instance ID -> semantic label string
    """
    semantic_mask_dir = clip_dir / cfg.preprocessing.semantic_mask_subdir
    instance_mask_dir = clip_dir / cfg.preprocessing.instance_mask_subdir

    # Account for densification when mapping gaussians to views
    densify_ratio = cfg.preprocessing.da3_densify_ratio
    view_offsets = [0]
    for n_gaussians_frame in gaussians_per_frame:
        view_offsets.append(view_offsets[-1] + n_gaussians_frame * densify_ratio)
    
    # Get unique merged instance IDs (excluding background -1)
    unique_merged_instances = np.unique(merged_instance_ids)
    unique_merged_instances = unique_merged_instances[unique_merged_instances >= 0]
    
    semantic_labels = {}
    
    # Process each merged instance
    for merged_inst_id in unique_merged_instances:
        # Find all gaussians belonging to this merged instance
        gaussian_mask = merged_instance_ids == merged_inst_id
        gaussian_indices = np.where(gaussian_mask)[0]
        
        if len(gaussian_indices) == 0:
            continue
        
        # Find which view this instance belongs to (use first gaussian)
        first_gaussian_idx = gaussian_indices[0]
        view_idx = None
        for v_idx, offset in enumerate(view_offsets[:-1]):
            if first_gaussian_idx < view_offsets[v_idx + 1]:
                view_idx = v_idx
                break
        
        if view_idx is None:
            logger.warning(f"Could not find view for gaussian {first_gaussian_idx}, skipping instance {merged_inst_id}")
            continue
        
        # Get frame index and original instance IDs for this view
        frame_idx = per_view_data[view_idx]["frame_idx"]
        view_instance_ids = per_view_data[view_idx]["instance_ids"].numpy()
        
        # Get gaussians for this view (accounting for densification)
        view_offset = view_offsets[view_idx]
        view_gaussian_mask = (gaussian_indices >= view_offset) & (gaussian_indices < view_offsets[view_idx + 1])
        view_gaussian_indices = gaussian_indices[view_gaussian_mask]
        
        if len(view_gaussian_indices) == 0:
            logger.warning(f"No gaussians found in view {view_idx} for merged instance {merged_inst_id}")
            continue
        
        # Get original instance ID from the first gaussian in this view
        # Account for densification: divide by densify_ratio to get original index
        first_view_gaussian_idx = view_gaussian_indices[0] - view_offset
        original_gaussian_idx = first_view_gaussian_idx // densify_ratio
        original_inst_id = view_instance_ids[original_gaussian_idx]
        
        if original_inst_id == 0:  # Background
            continue
        
        # Load semantic mask and instance mask for this frame
        semantic_mask_path = semantic_mask_dir / f"frame_{frame_idx:06d}.npy"
        instance_mask_path = instance_mask_dir / f"frame_{frame_idx:06d}.npy"
        
        if not semantic_mask_path.exists() or not instance_mask_path.exists():
            logger.warning(f"Missing masks for frame {frame_idx}, skipping instance {merged_inst_id}")
            continue
        
        semantic_mask = np.load(semantic_mask_path)  # (H, W) with semantic class IDs
        instance_mask = np.load(instance_mask_path)  # (H, W) with instance IDs
        
        # Find all pixels belonging to this original instance in the instance mask
        instance_pixels_mask = instance_mask == original_inst_id
        
        if instance_pixels_mask.sum() == 0:
            logger.warning(f"No pixels found for instance {original_inst_id} in frame {frame_idx}")
            continue
        
        # Get semantic IDs for all pixels of this instance
        instance_semantic_ids = semantic_mask[instance_pixels_mask]
        unique_semantic_ids = np.unique(instance_semantic_ids)
        
        # Sanity check: all pixels should have the same semantic ID
        if len(unique_semantic_ids) > 1:
            raise AssertionError(
                f"Instance {merged_inst_id} (original {original_inst_id}) in frame {frame_idx} "
                f"has multiple semantic IDs: {unique_semantic_ids}. "
                f"This violates the assumption that instances are created by separating semantic classes."
            )
        
        semantic_id = int(unique_semantic_ids[0])
        semantic_label = get_semantic_label_from_class_id(semantic_id)
        semantic_labels[int(merged_inst_id)] = semantic_label
        
        logger.debug(f"Merged instance {merged_inst_id} -> semantic class {semantic_id} ({semantic_label})")
    
    logger.info(f"Computed semantic labels for {len(semantic_labels)} merged instances")
    return semantic_labels


def process_clip_cotracker(clip: DictConfig, cfg: DictConfig):
    """Process a single clip to extract CoTracker3 control points."""
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    images_dir = clip_dir / "images"
    depth_processed_dir = clip_dir / cfg.preprocessing.depth_processed_subdir
    instance_mask_dir = clip_dir / cfg.preprocessing.instance_mask_subdir
    semantic_mask_dir = clip_dir / cfg.preprocessing.semantic_mask_subdir
    
    # Output directory for CoTracker data
    cotracker_dir = clip_dir / cfg.preprocessing.cotracker_subdir
    cotracker_dir.mkdir(parents=True, exist_ok=True)
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return
    
    if not depth_processed_dir.exists():
        logger.error(f"Processed depth directory not found: {depth_processed_dir}")
        return
    
    logger.info(f"Processing CoTracker3 for clip: {clip.name}")
    
    # Load images
    image_files = sorted(
        list(images_dir.glob("*.png")), key=extract_frame_number
    )
    logger.info(f"Found {len(image_files)} frames")
    
    # Load instance masks if available (convert to torch immediately)
    instance_masks = None
    if instance_mask_dir.exists():
        mask_files = sorted(
            list(instance_mask_dir.glob("*.npy")), key=extract_frame_number
        )
        if len(mask_files) == len(image_files):
            instance_masks = [torch.from_numpy(np.load(str(mf))).int() for mf in mask_files]
            logger.info(f"Loaded {len(instance_masks)} instance masks (converted to torch)")

    # TODO: check whether instance masks are actually used properly and what happens without
    
    # Track control points across all frames
    logger.info("Running CoTracker3...")
    control_points_2d, visibility = track_control_points(
        image_files,
        n_points_per_frame=cfg.preprocessing.cotracker_n_points_per_frame,
        save_dir=cotracker_dir,
    )
    # shape of control points: (T, N_total, 2) where N_total = n_points_per_frame * 3
    # shape of visibility: (T, N_grid), those are boolean values!
    
    # Lift control points to 3D using DA3 depth and camera parameters
    logger.info("Lifting control points to 3D...")
    colmap_dir = clip_dir / "sparse" / "0"
    depth_jump_threshold = cfg.preprocessing.cotracker_depth_jump_threshold
    control_points_3d, control_points_2d = lift_control_points_to_3d(
        control_points_2d,
        visibility,
        depth_processed_dir,
        colmap_dir,
        image_files,
        depth_jump_threshold=depth_jump_threshold,
        save_dir=cotracker_dir,
    )
    # control_points_3d: (T, N_valid, 3) - only permanently valid points
    # control_points_2d: (T, N_valid, 2) - filtered to match
    
    # Save control point trajectories
    control_points_3d_path = cotracker_dir / "control_points_3d.pth"
    torch.save(control_points_3d.cpu(), control_points_3d_path)
    logger.info(f"Saved control points 3D: {control_points_3d_path}")
    
    # Compute Gaussian-to-control-point associations for multiple init frames
    # Use first, middle, and last frames (matching preprocess.py multi-frame initialization)
    T = control_points_2d.shape[0]
    init_frame_indices = [0, T // 2, T - 1]
    logger.info(f"Computing associations for init frames: {init_frame_indices}")
    
    # Load confidence directory for filtering (matches da3_to_multi_view_colmap filtering)
    confidence_dir = clip_dir / cfg.preprocessing.confidence_subdir
    
    # Compute associations for each init frame and concatenate
    all_indices = []
    all_weights = []
    all_instance_ids = []
    all_pixel_coords = []
    all_frame_indices = []
    gaussians_per_frame = []
    
    for init_frame_idx in init_frame_indices:
        logger.info(f"Computing associations for frame {init_frame_idx}...")
        associations = compute_gaussian_control_point_associations(
            control_points_2d[init_frame_idx],
            control_points_3d[init_frame_idx],
            depth_processed_dir,
            colmap_dir,
            image_files,
            init_frame_idx,
            instance_masks[init_frame_idx] if instance_masks is not None else None,
            k_neighbors=cfg.preprocessing.cotracker_k_neighbors,
            idw_power=cfg.preprocessing.cotracker_idw_power,
            pixel_stride=cfg.preprocessing.da3_pc_pixel_stride,
            confidence_dir=confidence_dir,
            conf_thresh_percentile=cfg.preprocessing.da3_conf_thresh_percentile,
        )
        
        all_indices.append(associations["indices"])
        all_weights.append(associations["weights"])
        all_instance_ids.append(associations["instance_ids"])
        all_pixel_coords.append(associations["valid_pixel_coords"])
        # Store frame index for each gaussian
        n_gaussians_frame = associations["indices"].shape[0]
        all_frame_indices.append(torch.full((n_gaussians_frame,), init_frame_idx, dtype=torch.long))
        gaussians_per_frame.append(n_gaussians_frame)
        logger.info(f"Frame {init_frame_idx}: {n_gaussians_frame} Gaussians")
    
    # Concatenate associations from all frames
    combined_indices = torch.cat(all_indices, dim=0)  # (N_total, K)
    combined_weights = torch.cat(all_weights, dim=0)  # (N_total, K)
    combined_instance_ids = torch.cat(all_instance_ids, dim=0)  # (N_total,)
    combined_pixel_coords = torch.cat(all_pixel_coords, dim=0)  # (N_total, 2)
    combined_frame_indices = torch.cat(all_frame_indices, dim=0)  # (N_total,)

    # Densify associations if needed
    if cfg.preprocessing.da3_densify_ratio > 1:
        # This works because torch.repeat behaves like np.tile
        combined_indices = combined_indices.repeat(cfg.preprocessing.da3_densify_ratio, 1)
        combined_weights = combined_weights.repeat(cfg.preprocessing.da3_densify_ratio, 1)
        combined_instance_ids = combined_instance_ids.repeat(cfg.preprocessing.da3_densify_ratio)
        combined_pixel_coords = combined_pixel_coords.repeat(cfg.preprocessing.da3_densify_ratio, 1)
        combined_frame_indices = combined_frame_indices.repeat(cfg.preprocessing.da3_densify_ratio)

    total_gaussians = combined_indices.shape[0]
    logger.info(f"Total Gaussians from all frames: {total_gaussians}")
    logger.info(f"Gaussians per frame: {dict(zip(init_frame_indices, gaussians_per_frame))}")
    
    # Save associations
    indices_path = cotracker_dir / "gaussian_control_point_indices.pth"
    weights_path = cotracker_dir / "gaussian_control_point_weights.pth"
    
    torch.save(combined_indices.cpu(), indices_path)
    torch.save(combined_weights.cpu(), weights_path)
    
    # Also save the frame-wise counts for reference
    frame_counts_path = cotracker_dir / "gaussians_per_init_frame.pth"
    torch.save({
        "init_frame_indices": init_frame_indices,
        "gaussians_per_frame": gaussians_per_frame,
    }, frame_counts_path)
    
    logger.info(f"Saved associations: {indices_path}, {weights_path}")
    
    # Precompute Gaussian positions for all timesteps
    # control_points_3d stores 3D world coordinates (only valid points)
    logger.info("Precomputing Gaussian positions...")
    
    gaussian_positions = precompute_control_point_positions(
        control_points_3d,
        combined_indices,
        combined_weights,
        save_dir=cotracker_dir,
    )
    
    positions_path = cotracker_dir / "gaussian_positions_precomputed.pth"
    torch.save(gaussian_positions.cpu(), positions_path)
    logger.info(f"Saved precomputed positions: {positions_path}")
    
    # Save per-view instance assignments and positions for visualization
    per_view_data = []
    offset = 0
    for frame_idx, n_gaussians_frame in zip(init_frame_indices, gaussians_per_frame):
        # Account for densification
        n_gaussians_densified = n_gaussians_frame * cfg.preprocessing.da3_densify_ratio
        
        # Extract data for this view
        view_instance_ids = combined_instance_ids[offset:offset + n_gaussians_densified]
        view_positions = gaussian_positions[:, offset:offset + n_gaussians_densified, :]  # (T, N_view, 3)
        
        per_view_data.append({
            "frame_idx": frame_idx,
            "instance_ids": view_instance_ids.cpu(),
            "positions": view_positions.cpu(),
        })
        
        offset += n_gaussians_densified
    
    # Save per-view data
    per_view_path = cotracker_dir / "per_view_instances.pth"
    torch.save(per_view_data, per_view_path)
    logger.info(f"Saved per-view instance data: {per_view_path}")
    
    # Visualize per-view instances in Rerun
    logger.info("Logging per-view instances to Rerun...")
    from rerun_utils import init_and_save_rerun, log_per_view_instances, log_merged_instances
    
    rrd_path = cotracker_dir / "instance_assignment.rrd"
    init_and_save_rerun(str(rrd_path))
    
    log_per_view_instances(
        per_view_data=per_view_data,
        timesteps=np.arange(T),
    )
    
    # Pre-compute semantic IDs for each instance in each view (needed for merging)
    logger.info("Computing per-view instance semantic IDs for merging...")
    view_semantic_maps = get_instance_semantic_ids(
        per_view_data=per_view_data,
        instance_mask_dir=instance_mask_dir,
        semantic_mask_dir=semantic_mask_dir,
    )
    for v_idx, sem_map in enumerate(view_semantic_maps):
        logger.info(f"  View {v_idx} (frame {per_view_data[v_idx]['frame_idx']}): {sem_map}")

    # Merge instances across views
    logger.info(
        f"Merging instances across views "
        f"(threshold={cfg.preprocessing.instance_merge_containment_threshold}, "
        f"radius={cfg.preprocessing.instance_merge_containment_radius})..."
    )
    logger.info(f"Number of views: {len(per_view_data)}")
    for i, view_data in enumerate(per_view_data):
        logger.info(f"  View {i}: frame {view_data['frame_idx']}, {view_data['instance_ids'].shape[0]} Gaussians")
    
    reference_timestep = T // 2  # Use middle frame as reference
    logger.info(f"Using reference timestep {reference_timestep} of {T}")
    
    merged_instance_ids = merge_instances_across_views(
        per_view_data=per_view_data,
        reference_timestep=reference_timestep,
        containment_threshold=cfg.preprocessing.instance_merge_containment_threshold,
        containment_radius=cfg.preprocessing.instance_merge_containment_radius,
        view_semantic_maps=view_semantic_maps,
    )
    
    logger.info(
        f"Merge complete: {merged_instance_ids.shape[0]} Gaussians, "
        f"{len(np.unique(merged_instance_ids[merged_instance_ids >= 0]))} unique instances "
        f"(excluding background)"
    )
    
    # Save merged instance IDs (one ID per Gaussian, joint for all views)
    merged_path = cotracker_dir / "merged_instance_ids.npy"
    np.save(merged_path, merged_instance_ids)
    logger.info(f"Saved merged instance IDs: {merged_path} (shape: {merged_instance_ids.shape})")
    
    # Compute semantic labels for merged instances
    logger.info("Computing semantic labels for merged instances...")
    semantic_labels = compute_semantic_labels_for_merged_instances(
        merged_instance_ids=merged_instance_ids,
        per_view_data=per_view_data,
        combined_pixel_coords=combined_pixel_coords,
        combined_frame_indices=combined_frame_indices,
        init_frame_indices=init_frame_indices,
        gaussians_per_frame=gaussians_per_frame,
        clip_dir=clip_dir,
        cfg=cfg,
    )
    
    # Save semantic labels dict
    semantic_labels_path = cotracker_dir / "merged_instance_semantic_labels.json"
    import json
    with open(semantic_labels_path, "w") as f:
        json.dump(semantic_labels, f, indent=2)
    logger.info(f"Saved semantic labels: {semantic_labels_path}")
    logger.info(f"Semantic labels: {semantic_labels}")
    
    # Log merged instances to Rerun (same file as per-view)
    logger.info("Logging merged instances to Rerun...")
    gaussian_positions_numpy = gaussian_positions.cpu().numpy() if torch.is_tensor(gaussian_positions) else gaussian_positions
    log_merged_instances(
        merged_instance_ids=merged_instance_ids,
        positions_through_time=gaussian_positions_numpy,
        timesteps=np.arange(T),
    )
    logger.info(f"Saved Rerun visualization to: {rrd_path}")
    
    logger.info(f"CoTracker preprocessing complete for {clip.name}")


def main():
    # do hydra init manually here to avoid conflicts
    config_dir = Path(__file__).parent / "conf"
    with hydra.initialize_config_dir(
        config_dir=str(config_dir.resolve()), version_base="1.3"
    ):
        overrides = sys.argv[1:]
        cfg = hydra.compose("config.yaml", overrides=overrides)
    
    # Clear after composing the main config
    GlobalHydra.instance().clear()
    
    out_dir = Path(cfg.preprocessed_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for clip in tqdm(cfg.clips, desc="Processing clips", unit="clip"):
        process_clip_cotracker(clip, cfg)


if __name__ == "__main__":
    main()