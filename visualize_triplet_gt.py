#!/usr/bin/env python3
"""
Visualize ground truth triplets aligned to frames for verification.

Creates a grid visualization showing all frames with GT triplet annotations,
grouped by clip, to verify the GT alignment logic is working correctly.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import hydra
from omegaconf import DictConfig

from benchmark.cholect50_utils import CholecT50Loader
from benchmark.triplets import load_triplet_samples


def create_frame_with_annotation(frame_path: Path, triplets: List[Dict], second_idx: int, frame_idx: int) -> np.ndarray:
    """Create annotated frame with triplet text overlay.
    
    Args:
        frame_path: Path to frame image
        triplets: List of ground truth triplets
        second_idx: CholecT50 second index
        frame_idx: Clip-relative frame index
        
    Returns:
        Annotated frame as numpy array (RGB)
    """
    # Load image
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {frame_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL for text rendering
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw frame info
    draw.rectangle([(0, 0), (pil_img.width, 80)], fill=(0, 0, 0, 200))
    draw.text((10, 10), f"Frame: {frame_idx}", fill=(255, 255, 255), font=font_large)
    draw.text((10, 35), f"Second: {second_idx}", fill=(255, 255, 255), font=font_small)
    draw.text((10, 55), f"# Triplets: {len(triplets)}", fill=(255, 255, 255), font=font_small)
    
    # Draw triplets
    y_offset = 100
    for i, triplet in enumerate(triplets):
        inst = triplet.get('instrument', 'N/A')
        verb = triplet.get('verb', 'N/A')
        target = triplet.get('target', 'N/A')
        text = f"{i+1}. {inst} → {verb} → {target}"
        
        # Background box
        bbox = draw.textbbox((10, y_offset), text, font=font_small)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((10, y_offset), text, fill=(100, 255, 100), font=font_small)
        y_offset += 25
    
    return np.array(pil_img)


def create_clip_visualization(
    clip_name: str,
    samples: List[Dict],
    output_path: Path,
    max_cols: int = 4,
):
    """Create a grid visualization for all samples in a clip.
    
    Args:
        clip_name: Clip identifier
        samples: List of samples from load_triplet_samples
        output_path: Where to save the visualization
        max_cols: Maximum number of columns in the grid
    """
    if not samples:
        print(f"No samples for {clip_name}")
        return
    
    # Sort samples by frame
    samples = sorted(samples, key=lambda s: s['target_frame'])
    
    # Load and annotate all frames
    annotated_frames = []
    for sample in samples:
        frame_path = sample['image_paths'][sample['target_frame']]
        annotated = create_frame_with_annotation(
            frame_path=frame_path,
            triplets=sample['gt_triplets'],
            second_idx=sample['second_idx'],
            frame_idx=sample['target_frame'],
        )
        annotated_frames.append(annotated)
    
    # Determine grid size
    n_frames = len(annotated_frames)
    n_cols = min(max_cols, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    # Get frame dimensions
    h, w = annotated_frames[0].shape[:2]
    
    # Create grid canvas
    grid_h = n_rows * h + (n_rows + 1) * 10  # 10px padding
    grid_w = n_cols * w + (n_cols + 1) * 10
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid.fill(30)  # Dark gray background
    
    # Place frames in grid
    for idx, frame in enumerate(annotated_frames):
        row = idx // n_cols
        col = idx % n_cols
        
        y_start = row * h + (row + 1) * 10
        x_start = col * w + (col + 1) * 10
        
        grid[y_start:y_start+h, x_start:x_start+w] = frame
    
    # Add title
    pil_grid = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_grid)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        title_font = ImageFont.load_default()
    
    title = f"Ground Truth Triplets: {clip_name}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((grid_w - title_w) // 2, grid_h - 50), title, fill=(255, 255, 255), font=title_font)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_grid.save(output_path)
    print(f"✓ Saved visualization to {output_path}")


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    """Generate GT triplet visualizations for all clips."""
    
    output_root = Path("output/triplet_gt_visualizations")
    output_root.mkdir(parents=True, exist_ok=True)
    
    cholect50_loader = CholecT50Loader(str(cfg.cholect50_root))
    
    # Generate summary JSON
    summary = {}
    
    for clip in cfg.clips:
        clip_name = str(clip.name)
        print(f"\nProcessing {clip_name}...")
        
        # Parse video_id and clip_start
        try:
            video_id = int(clip_name.split("_")[0].replace("video", ""))
            clip_start = int(clip_name.split("_")[1])
        except Exception as e:
            print(f"  ERROR: Could not parse clip name: {e}")
            continue
        
        # Skip if video > 50 (no CholecT50 annotations)
        if video_id > 50:
            print(f"  Skipping: CholecT50 only covers videos 1-50")
            continue
        
        # Load samples
        video_dir = Path(cfg.preprocessed_root) / clip_name
        
        try:
            samples = load_triplet_samples(
                video_dir=video_dir,
                clip_start=clip_start,
                video_id=video_id,
                cholect50_loader=cholect50_loader,
                images_subdir=cfg.eval.paths.images_subdir,
                framerate=cfg.eval.triplets.FRAMERATE,
                num_frames=cfg.eval.triplets.NUM_FRAMES,
                frame_stride=cfg.eval.triplets.frame_stride,
            )
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue
        
        if not samples:
            print(f"  No samples found")
            continue
        
        print(f"  Found {len(samples)} evaluation samples")
        
        # Create visualization
        output_path = output_root / f"{clip_name}.png"
        create_clip_visualization(
            clip_name=clip_name,
            samples=samples,
            output_path=output_path,
        )
        
        # Add to summary
        summary[clip_name] = {
            'video_id': video_id,
            'clip_start': clip_start,
            'num_samples': len(samples),
            'samples': [
                {
                    'second_idx': s['second_idx'],
                    'target_frame': s['target_frame'],
                    'abs_frame': s['abs_frame'],
                    'num_triplets': len(s['gt_triplets']),
                    'triplets': s['gt_triplets'],
                }
                for s in samples
            ]
        }
    
    # Save summary JSON
    summary_path = output_root / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved summary to {summary_path}")
    print(f"✓ All visualizations saved to {output_root}")


if __name__ == "__main__":
    main()
