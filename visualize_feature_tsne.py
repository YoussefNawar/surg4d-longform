"""Visualize Qwen3 patch features via t-SNE colored by semantic class.

Usage:
    pixi run python visualize_feature_tsne.py <clip_dir>

Example:
    pixi run python visualize_feature_tsne.py data/preprocessed/qwen3_da3_subsampled/video01_00240
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from loguru import logger

from llm.qwen_utils import get_patch_hw, qwen3_cat_to_deepstack, QWEN_CONSTANTS
import torch

# CholecSeg8k class mapping (from cholec_utils.py)
CLASS_ID_TO_NAME = {
    0: "Black Background",
    1: "Abdominal Wall",
    2: "Liver",
    3: "Gastrointestinal Tract",
    4: "Fat",
    5: "Grasper",
    6: "Connective Tissue",
    7: "Blood",
    8: "Cystic Duct",
    9: "L-hook Electrocautery",
    10: "Gallbladder",
    11: "Hepatic Vein",
    12: "Liver Ligament",
}

# Colormap for classes (distinct colors)
CLASS_COLORS = {
    0: "#1f1f1f",  # Black Background - dark gray
    1: "#e6194b",  # Abdominal Wall - red
    2: "#8B4513",  # Liver - brown
    3: "#ffe119",  # Gastrointestinal Tract - yellow
    4: "#fabebe",  # Fat - light pink
    5: "#4363d8",  # Grasper - blue
    6: "#f58231",  # Connective Tissue - orange
    7: "#911eb4",  # Blood - purple
    8: "#46f0f0",  # Cystic Duct - cyan
    9: "#3cb44b",  # L-hook Electrocautery - green
    10: "#f032e6",  # Gallbladder - magenta
    11: "#bcf60c",  # Hepatic Vein - lime
    12: "#9a6324",  # Liver Ligament - olive
}


def get_majority_class_per_patch(
    semantic_mask: np.ndarray, patch_h: int, patch_w: int, factor: int
) -> np.ndarray:
    """For each patch, determine the majority semantic class.

    Args:
        semantic_mask: Semantic mask at original resolution (H, W)
        patch_h: Number of patches in height
        patch_w: Number of patches in width
        factor: Patch size (effective_patch_size for qwen3 = 32)

    Returns:
        Array of shape (patch_h * patch_w,) with class id per patch
    """
    # Resize semantic mask to match the patch grid resolution
    # Each patch covers factor x factor pixels in the resized image
    resized_h = patch_h * factor
    resized_w = patch_w * factor

    # Resize semantic mask using nearest neighbor to preserve class labels
    sem_img = Image.fromarray(semantic_mask.astype(np.uint8))
    sem_resized = np.array(sem_img.resize((resized_w, resized_h), Image.NEAREST))

    # For each patch, find the majority class
    patch_classes = np.zeros((patch_h, patch_w), dtype=np.int32)
    for py in range(patch_h):
        for px in range(patch_w):
            patch_region = sem_resized[
                py * factor : (py + 1) * factor, px * factor : (px + 1) * factor
            ]
            # Majority vote
            unique, counts = np.unique(patch_region, return_counts=True)
            patch_classes[py, px] = unique[counts.argmax()]

    return patch_classes.flatten()


def run_tsne(features: np.ndarray, perplexity: int = 30, seed: int = 42) -> np.ndarray:
    """Run t-SNE dimensionality reduction."""
    n_samples = features.shape[0]
    # Adjust perplexity if we have few samples
    effective_perplexity = min(perplexity, max(5, n_samples // 4))
    
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=seed,
        max_iter=1000,
        init="pca",
    )
    return tsne.fit_transform(features)


def plot_tsne(
    embeddings: np.ndarray,
    class_labels: np.ndarray,
    title: str,
    output_path: Path,
):
    """Create t-SNE scatter plot colored by semantic class."""
    fig, ax = plt.subplots(figsize=(12, 10))

    present_classes = np.unique(class_labels)
    
    # Plot each class separately for legend
    for class_id in present_classes:
        mask = class_labels == class_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=CLASS_COLORS.get(class_id, "#808080"),
            label=f"{class_id}: {CLASS_ID_TO_NAME.get(class_id, 'Unknown')}",
            alpha=0.6,
            s=20,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Qwen3 patch features via t-SNE"
    )
    parser.add_argument(
        "clip_dir",
        type=str,
        help="Path to preprocessed clip directory",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index to visualize (default: 0)",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=5,
        help="Number of frames to sample and aggregate (default: 5)",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    patch_dir = clip_dir / "qwen3_patch_features"
    semantic_dir = clip_dir / "semantic_masks"
    images_dir = clip_dir / "images"
    output_dir = clip_dir / "tsne_plots"
    output_dir.mkdir(exist_ok=True)

    # Check directories exist
    assert patch_dir.exists(), f"Patch features dir not found: {patch_dir}"
    assert semantic_dir.exists(), f"Semantic masks dir not found: {semantic_dir}"
    assert images_dir.exists(), f"Images dir not found: {images_dir}"

    # Get all available frames
    feature_files = sorted(patch_dir.glob("*_f.npy"))
    n_frames = len(feature_files)
    logger.info(f"Found {n_frames} frames in {clip_dir.name}")

    # Sample frames evenly across the clip
    sample_indices = np.linspace(0, n_frames - 1, args.sample_frames, dtype=int)
    logger.info(f"Sampling frames: {sample_indices.tolist()}")

    # Collect features and labels from sampled frames
    all_main_feats = []
    all_ds0_feats = []
    all_ds1_feats = []
    all_ds2_feats = []
    all_labels = []

    qwen_version = "qwen3"
    factor = QWEN_CONSTANTS[qwen_version]["effective_patch_size"]

    for frame_idx in sample_indices:
        frame_stem = f"{frame_idx:06d}"
        
        # Load features
        feat_path = patch_dir / f"{frame_stem}_f.npy"
        concat_feats = np.load(feat_path)  # (n_patches, hidden_dim * 4)
        
        # Get image dimensions
        img_path = list(images_dir.glob(f"frame_{frame_stem}.*"))[0]
        img = Image.open(img_path)
        patch_h, patch_w = get_patch_hw(img.height, img.width, qwen_version)
        n_patches = patch_h * patch_w
        
        logger.info(
            f"Frame {frame_idx}: image {img.width}x{img.height} -> "
            f"patches {patch_w}x{patch_h} = {n_patches}, features shape {concat_feats.shape}"
        )
        
        # Load semantic mask
        sem_path = semantic_dir / f"frame_{frame_stem}.npy"
        semantic_mask = np.load(sem_path)
        
        # Get majority class per patch
        patch_labels = get_majority_class_per_patch(
            semantic_mask, patch_h, patch_w, factor
        )
        
        # Decompose features into main + deepstack
        concat_tensor = torch.from_numpy(concat_feats)
        main_feats, deepstack_feats = qwen3_cat_to_deepstack(concat_tensor)
        
        all_main_feats.append(main_feats.numpy())
        all_ds0_feats.append(deepstack_feats[0].numpy())
        all_ds1_feats.append(deepstack_feats[1].numpy())
        all_ds2_feats.append(deepstack_feats[2].numpy())
        all_labels.append(patch_labels)

    # Concatenate all frames
    all_main_feats = np.concatenate(all_main_feats, axis=0)
    all_ds0_feats = np.concatenate(all_ds0_feats, axis=0)
    all_ds1_feats = np.concatenate(all_ds1_feats, axis=0)
    all_ds2_feats = np.concatenate(all_ds2_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    logger.info(f"Total patches: {len(all_labels)}")
    logger.info(f"Main features shape: {all_main_feats.shape}")
    logger.info(f"Deepstack feature shapes: {all_ds0_feats.shape}")

    # Filter out background (class 0) for better visualization
    non_bg_mask = all_labels != 0
    logger.info(
        f"Filtering background: {non_bg_mask.sum()}/{len(all_labels)} patches remain"
    )

    if non_bg_mask.sum() < 10:
        logger.warning("Too few non-background patches, including background")
        non_bg_mask = np.ones_like(all_labels, dtype=bool)

    # Run t-SNE and plot for each feature level
    feature_sets = [
        ("main", all_main_feats),
        ("deepstack_0", all_ds0_feats),
        ("deepstack_1", all_ds1_feats),
        ("deepstack_2", all_ds2_feats),
    ]

    for name, feats in feature_sets:
        logger.info(f"Running t-SNE for {name} features...")
        
        # Filter features and labels
        feats_filtered = feats[non_bg_mask]
        labels_filtered = all_labels[non_bg_mask]
        
        # Run t-SNE
        embeddings = run_tsne(feats_filtered, perplexity=args.perplexity, seed=args.seed)
        
        # Plot
        title = (
            f"t-SNE of Qwen3 {name} features\n"
            f"{clip_dir.name} | {args.sample_frames} frames | "
            f"{len(labels_filtered)} patches"
        )
        output_path = output_dir / f"tsne_{name}.png"
        plot_tsne(embeddings, labels_filtered, title, output_path)

    # Also create a combined plot with all 4 feature levels
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for ax, (name, feats) in zip(axes, feature_sets):
        feats_filtered = feats[non_bg_mask]
        labels_filtered = all_labels[non_bg_mask]
        embeddings = run_tsne(feats_filtered, perplexity=args.perplexity, seed=args.seed)
        
        present_classes = np.unique(labels_filtered)
        for class_id in present_classes:
            mask = labels_filtered == class_id
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=CLASS_COLORS.get(class_id, "#808080"),
                label=f"{class_id}: {CLASS_ID_TO_NAME.get(class_id, 'Unknown')}",
                alpha=0.6,
                s=15,
            )
        ax.set_title(f"{name}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, markerscale=1.5)
    fig.suptitle(
        f"t-SNE of Qwen3 features by semantic class\n"
        f"{clip_dir.name} | {args.sample_frames} frames | {non_bg_mask.sum()} patches",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    combined_path = output_dir / "tsne_combined.png"
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {combined_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
