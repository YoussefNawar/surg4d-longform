"""
Surgical action triplet recognition evaluation.

Supports evaluation methods:
- Single Frame: Baseline using one frame
- Single Frame + Mask Overlay: Single frame with segmentation overlays
- Multi-Frame (Video): Temporal reasoning with multiple frames
- Multi-Frame + Mask Overlay: Video with segmentation overlays  
- Graph Agent (Single): Agentic exploration at target timestep
- Graph Agent (Dynamic): Agentic exploration across full temporal scene
"""

import gc
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import cv2

import torch
from omegaconf import DictConfig

from benchmark.cholect50_utils import CholecT50Loader
from benchmark.serialization_utils import sanitize_tool_calls
from llm.qwen_utils import (
    prompt_with_video_frames,
    prompt_graph_agent,
    ask_qwen_about_image,
)
from llm.tools import GraphTools
from autoencoder.model_qwen import QwenAutoencoder


# =============================================================================
# Data Loading & GT Matching
# =============================================================================

def load_triplet_samples(
    video_dir: Path,
    clip_start: int,
    video_id: int,
    cholect50_loader: CholecT50Loader,
    images_subdir: str,
    framerate: int = 25,
    num_frames: int = 80,
    frame_stride: int = 4,
) -> List[Dict]:
    """Load samples with GT triplets from CholecT50 for a clip.
    
    Args:
        video_dir: Path to clip directory
        clip_start: Absolute frame index where this clip starts in original video
        video_id: CholecT50 video ID
        cholect50_loader: Loader for CholecT50 annotations
        images_subdir: Subdirectory containing frame images
        framerate: Original video framerate (default 25fps)
        num_frames: Number of frames in clip (default 80)
        frame_stride: Stride for graph alignment (default 4)
        
    Returns:
        List of sample dicts with gt_triplets, frame info, etc.
    """
    from math import floor, ceil
    
    # Load CholecT50 annotations for this video
    video_data = cholect50_loader.load_video_annotations(video_id)
    
    # Find image paths
    images_dir = video_dir / images_subdir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    
    samples = []
    
    # Absolute frame range of this clip in the original video
    clip_abs_start = clip_start
    clip_abs_end = clip_start + num_frames - 1
    
    # CholecT50 annotations exist at every 25th frame (1 per second)
    # Convert to second indices that fall into this clip
    s_start = ceil(clip_abs_start / framerate)
    s_end = floor(clip_abs_end / framerate)
    
    for second_idx in range(s_start, s_end + 1):
        annotation_abs_frame = second_idx * framerate
        if not (clip_abs_start <= annotation_abs_frame <= clip_abs_end):
            continue
        
        # Fetch GT triplets for this second from CholecT50
        triplets = cholect50_loader.get_frame_triplets(video_data, int(second_idx))
        
        # Map to clip-relative frame
        end_rel = annotation_abs_frame - clip_abs_start  # 0..79
        
        # Snap to nearest stride-aligned frame for graph alignment
        end_rel = int(round(end_rel / frame_stride) * frame_stride)
        end_rel = max(0, min(num_frames - 1, end_rel))
        
        samples.append({
            'video_id': video_id,
            'clip_start': clip_start,
            'second_idx': second_idx,
            'target_frame': end_rel,  # Clip-relative frame index
            'abs_frame': annotation_abs_frame,
            'gt_triplets': triplets,
            'gt_phase': triplets[0]['phase'] if triplets else None,
            'image_paths': image_paths,
        })
    
    return samples


def build_semantic_mask_overlay(frame_path: Path, semantic_masks_dir: Path, alpha: float = 0.5) -> Path:
    """Build a colored overlay image with semantic masks (constant color per class).
    
    Args:
        frame_path: Path to original frame image
        semantic_masks_dir: Directory containing semantic_masks/frame_XXXXXX.npy
        alpha: Overlay transparency
        
    Returns:
        Path to overlay image (in same dir with _overlay.png suffix)
        
    Raises:
        FileNotFoundError: If semantic mask doesn't exist
    """
    # Parse frame index from filename
    frame_stem = frame_path.stem
    try:
        if "_" in frame_stem:
            frame_idx = int(frame_stem.split("_")[-1])
        else:
            frame_idx = int(frame_stem)
    except Exception as e:
        raise ValueError(f"Could not parse frame index from {frame_path}: {e}")
    
    # Look for semantic mask file
    mask_path = semantic_masks_dir / f"frame_{frame_idx:06d}.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Semantic mask not found: {mask_path}")
    
    # Load image and mask
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {frame_path}")
    
    try:
        semantic_mask = np.load(str(mask_path))
    except Exception as e:
        raise ValueError(f"Could not load semantic mask {mask_path}: {e}")
    
    if semantic_mask.ndim != 2:
        raise ValueError(f"Expected 2D semantic mask, got shape {semantic_mask.shape}")
    
    H, W = semantic_mask.shape
    if img.shape[0] != H or img.shape[1] != W:
        img = cv2.resize(img, (W, H))
    
    # CholecSeg8k semantic class colors (constant per class)
    # Mapping from cholec_utils.py parse_cholecseg8k_instance_mask
    class_colors = {
        0: [0, 0, 0],           # Background (black)
        1: [255, 100, 100],     # Abdominal Wall (light red)
        2: [100, 255, 100],     # Liver (light green)
        3: [100, 100, 255],     # Gastrointestinal Tract (light blue)
        4: [255, 255, 100],     # Fat (yellow)
        5: [255, 100, 255],     # Grasper (magenta)
        6: [100, 255, 255],     # Connective Tissue (cyan)
        7: [200, 100, 100],     # Blood (dark red)
        8: [100, 200, 100],     # Cystic Duct (dark green)
        9: [100, 100, 200],     # L-hook Electrocautery (dark blue)
        10: [200, 200, 100],    # Gallbladder (dark yellow)
        11: [200, 100, 200],    # Hepatic Vein (dark magenta)
        12: [100, 200, 200],    # Liver Ligament (dark cyan)
    }
    
    overlay = img.copy()
    for class_id, color in class_colors.items():
        if class_id == 0:  # Skip background
            continue
        mask = semantic_mask == class_id
        if not np.any(mask):
            continue
        overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * np.array(color)).astype(np.uint8)
    
    # Write overlay next to original
    out_path = frame_path.parent / f"{frame_path.stem}_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    
    return out_path


# =============================================================================
# Response Parsing
# =============================================================================

def parse_triplets_response(response: str) -> List[Dict]:
    """Parse model response to extract triplets. Expects {"triplets": [...]} format.
    
    Returns:
        List of dicts with keys: instrument, verb, target, confidence
    """
    instrument_map = {
        "0": "grasper", "1": "bipolar", "2": "hook", "3": "scissors", "4": "clipper", "5": "irrigator",
        0: "grasper", 1: "bipolar", 2: "hook", 3: "scissors", 4: "clipper", 5: "irrigator",
    }
    verb_map = {
        "0": "grasp", "1": "retract", "2": "dissect", "3": "coagulate", "4": "clip", "5": "cut",
        "6": "aspirate", "7": "irrigate", "8": "pack", "9": "null_verb",
        0: "grasp", 1: "retract", 2: "dissect", 3: "coagulate", 4: "clip", 5: "cut",
        6: "aspirate", 7: "irrigate", 8: "pack", 9: "null_verb",
    }
    target_map = {
        "0": "gallbladder", "1": "cystic_plate", "2": "cystic_duct", "3": "cystic_artery", "4": "cystic_pedicle",
        "5": "blood_vessel", "6": "fluid", "7": "abdominal_wall_cavity", "8": "liver", "9": "adhesion",
        "10": "omentum", "11": "peritoneum", "12": "gut", "13": "specimen_bag", "14": "null_target",
        0: "gallbladder", 1: "cystic_plate", 2: "cystic_duct", 3: "cystic_artery", 4: "cystic_pedicle",
        5: "blood_vessel", 6: "fluid", 7: "abdominal_wall_cavity", 8: "liver", 9: "adhesion",
        10: "omentum", 11: "peritoneum", 12: "gut", 13: "specimen_bag", 14: "null_target",
    }
    
    # Extract JSON from ```json fenced block or find first {...}
    text = response.strip()
    
    # Try fenced block first
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    
    # Find first balanced JSON object
    brace_start = text.find('{')
    if brace_start == -1:
        return []
    
    bal = 0
    for i in range(brace_start, len(text)):
        if text[i] == '{':
            bal += 1
        elif text[i] == '}':
            bal -= 1
            if bal == 0:
                try:
                    data = json.loads(text[brace_start:i+1])
                    triplets_list = data.get('triplets', [])
                    
                    # Normalize
                    normalized = []
                    for item in triplets_list:
                        if not isinstance(item, dict):
                            continue
                        
                        # Map IDs to names
                        inst = item.get('instrument')
                        verb = item.get('verb')
                        targ = item.get('target')
                        
                        if isinstance(inst, (int, str)) and inst in instrument_map:
                            inst = instrument_map[inst]
                        elif isinstance(inst, str):
                            inst = inst.strip().lower()
                        else:
                            continue
                        
                        if isinstance(verb, (int, str)) and verb in verb_map:
                            verb = verb_map[verb]
                        elif isinstance(verb, str):
                            verb = verb.strip().lower()
                        else:
                            continue
                        
                        if isinstance(targ, (int, str)) and targ in target_map:
                            targ = target_map[targ]
                        elif isinstance(targ, str):
                            targ = targ.strip().lower()
                        else:
                            continue
                        
                        confidence = float(item.get('confidence', 1.0))
                        confidence = max(0.0, min(1.0, confidence))
                        
                        normalized.append({
                            'instrument': inst,
                            'verb': verb,
                            'target': targ,
                            'confidence': confidence
                        })
                    
                    return normalized
                except Exception as e:
                    print(f"Failed to parse JSON: {e}")
                    return []
                break
    
    return []


# =============================================================================
# Evaluation Methods
# =============================================================================

def single_frame_queries(
    model,
    processor,
    samples: List[Dict],
    clip: DictConfig,
    cfg: DictConfig,
) -> List[Dict]:
    """Run single-frame triplet recognition.
    
    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        samples: List of sample dicts from load_triplet_samples
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    system_prompt = cfg.eval.triplets.single_frame_system_prompt
    prompt_template = cfg.eval.triplets.single_frame_prompt_template
    
    results = []
    for sample in samples:
        target_frame = sample['target_frame']
        frame_path = sample['image_paths'][target_frame]
        
        response = ask_qwen_about_image(
            image_path=str(frame_path),
            question=prompt_template,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
        )
        
        # Parse response
        predicted = parse_triplets_response(response)
        
        # Build message history
        message_history = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": str(frame_path)},
                {"type": "text", "text": prompt_template}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        
        results.append({
            'sample_id': f"v{sample['video_id']:02d}_s{sample['second_idx']:03d}",
            'video_id': sample['video_id'],
            'clip_start': sample['clip_start'],
            'second_idx': sample['second_idx'],
            'target_frame': target_frame,
            'predicted': predicted,
            'ground_truth': sample['gt_triplets'],
            'raw_response': response,
            'message_history': message_history,
        })
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def single_frame_mask_overlay_queries(
    model,
    processor,
    samples: List[Dict],
    clip: DictConfig,
    cfg: DictConfig,
) -> List[Dict]:
    """Run single-frame triplet recognition with semantic mask overlays.
    
    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        samples: List of sample dicts
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    system_prompt = cfg.eval.triplets.single_frame_mask_overlay_system_prompt
    prompt_template = cfg.eval.triplets.single_frame_mask_overlay_prompt_template
    
    # Get semantic masks directory
    video_dir = Path(cfg.preprocessed_root) / clip.name
    semantic_masks_dir = video_dir / "semantic_masks"
    
    if not semantic_masks_dir.exists():
        raise FileNotFoundError(f"Semantic masks directory not found: {semantic_masks_dir}")
    
    results = []
    for sample in samples:
        target_frame = sample['target_frame']
        frame_path = sample['image_paths'][target_frame]
        
        # Build semantic overlay (will raise if mask doesn't exist)
        overlay_path = build_semantic_mask_overlay(frame_path, semantic_masks_dir)
        
        response = ask_qwen_about_image(
            image_path=str(overlay_path),
            question=prompt_template,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
        )
        
        # Parse response
        predicted = parse_triplets_response(response)
        
        # Build message history
        message_history = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": str(overlay_path)},
                {"type": "text", "text": prompt_template}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        
        results.append({
            'sample_id': f"v{sample['video_id']:02d}_s{sample['second_idx']:03d}",
            'video_id': sample['video_id'],
            'clip_start': sample['clip_start'],
            'second_idx': sample['second_idx'],
            'target_frame': target_frame,
            'predicted': predicted,
            'ground_truth': sample['gt_triplets'],
            'raw_response': response,
            'message_history': message_history,
        })
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def multiframe_queries(
    model,
    processor,
    samples: List[Dict],
    clip: DictConfig,
    cfg: DictConfig,
) -> List[Dict]:
    """Run multi-frame (video) triplet recognition.
    
    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        samples: List of sample dicts
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    system_prompt = cfg.eval.triplets.multiframe_system_prompt
    prompt_template = cfg.eval.triplets.multiframe_prompt_template
    
    # Load all frames at stride for video
    frame_stride = cfg.eval.triplets.frame_stride
    
    results = []
    for sample in samples:
        image_paths = sample['image_paths']
        target_frame = sample['target_frame']
        
        # Sample frames at stride
        strided_frames = image_paths[::frame_stride]
        
        # Calculate target timestep in strided frames and convert to seconds
        target_timestep = target_frame // frame_stride
        effective_fps = cfg.eval.video_fps / frame_stride
        target_second = target_timestep / effective_fps
        
        # Build prompt with target second
        prompt = prompt_template.format(target_second=f"{target_second:.2f}")
        
        response = prompt_with_video_frames(
            question=prompt,
            image_paths=strided_frames,
            model=model,
            processor=processor,
            qwen_version=cfg.eval.qwen_version,
            system_prompt=system_prompt,
            fps=effective_fps,
        )
        
        # Parse response
        predicted = parse_triplets_response(response)
        
        # Build message history
        message_history = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "video", "video": [str(p) for p in strided_frames]},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        
        results.append({
            'sample_id': f"v{sample['video_id']:02d}_s{sample['second_idx']:03d}",
            'video_id': sample['video_id'],
            'clip_start': sample['clip_start'],
            'second_idx': sample['second_idx'],
            'target_frame': target_frame,
            'predicted': predicted,
            'ground_truth': sample['gt_triplets'],
            'raw_response': response,
            'message_history': message_history,
        })
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def multiframe_mask_overlay_queries(
    model,
    processor,
    samples: List[Dict],
    clip: DictConfig,
    cfg: DictConfig,
) -> List[Dict]:
    """Run multi-frame (video) triplet recognition with semantic mask overlays.
    
    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        samples: List of sample dicts
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    system_prompt = cfg.eval.triplets.multiframe_mask_overlay_system_prompt
    prompt_template = cfg.eval.triplets.multiframe_mask_overlay_prompt_template
    
    # Get semantic masks directory
    video_dir = Path(cfg.preprocessed_root) / clip.name
    semantic_masks_dir = video_dir / "semantic_masks"
    
    if not semantic_masks_dir.exists():
        raise FileNotFoundError(f"Semantic masks directory not found: {semantic_masks_dir}")
    
    frame_stride = cfg.eval.triplets.frame_stride
    
    results = []
    for sample in samples:
        image_paths = sample['image_paths']
        target_frame = sample['target_frame']
        
        # Sample frames at stride and build semantic overlays
        strided_frames = image_paths[::frame_stride]
        overlay_frames = [build_semantic_mask_overlay(p, semantic_masks_dir) for p in strided_frames]
        
        # Calculate target second
        target_timestep = target_frame // frame_stride
        effective_fps = cfg.eval.video_fps / frame_stride
        target_second = target_timestep / effective_fps
        
        # Build prompt
        prompt = prompt_template.format(target_second=f"{target_second:.2f}")
        
        response = prompt_with_video_frames(
            question=prompt,
            image_paths=overlay_frames,
            model=model,
            processor=processor,
            qwen_version=cfg.eval.qwen_version,
            system_prompt=system_prompt,
            fps=effective_fps,
        )
        
        # Parse response
        predicted = parse_triplets_response(response)
        
        # Build message history
        message_history = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "video", "video": [str(p) for p in overlay_frames]},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        
        results.append({
            'sample_id': f"v{sample['video_id']:02d}_s{sample['second_idx']:03d}",
            'video_id': sample['video_id'],
            'clip_start': sample['clip_start'],
            'second_idx': sample['second_idx'],
            'target_frame': target_frame,
            'predicted': predicted,
            'ground_truth': sample['gt_triplets'],
            'raw_response': response,
            'message_history': message_history,
        })
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def graph_agent_single_queries(
    model,
    processor,
    samples: List[Dict],
    graph_path: Path,
    clip: DictConfig,
    cfg: DictConfig,
) -> List[Dict]:
    """Run graph agent triplet recognition at target timestep.
    
    Args:
        model: Qwen VL model (must be qwen3)
        processor: Qwen VL processor
        samples: List of sample dicts
        graph_path: Path to graph directory
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    assert cfg.eval.qwen_version == "qwen3", "graph_agent requires qwen3"
    
    system_prompt = cfg.eval.triplets.graph_agent_single_system_prompt
    prompt_template = cfg.eval.triplets.graph_agent_single_prompt_template
    
    # Load graph artifacts
    node_feats_npz = np.load(graph_path / "c_qwen_feats.npz")
    node_centers = np.load(graph_path / "c_centers.npy")
    node_centroids = np.load(graph_path / "c_centroids.npy")
    node_extents = np.load(graph_path / "c_extents.npy")
    positions = np.load(graph_path / "positions.npy")
    clusters = np.load(graph_path / "clusters.npy")
    patch_latents = np.load(graph_path / "patch_latents_through_time.npy")
    adjacency = np.load(graph_path / "graph.npy")
    bhattacharyya = np.load(graph_path / "bhattacharyya_coeffs.npy")
    
    num_ts = adjacency.shape[0]
    frame_stride = cfg.eval.triplets.frame_stride
    
    # Load autoencoder if highres tools are enabled
    autoencoder = None
    if cfg.eval.triplets.get('enable_highres_tools', False):
        autoencoder = QwenAutoencoder.from_pretrained(cfg.autoencoder.output_dir)
        autoencoder.eval()
        if torch.cuda.is_available():
            autoencoder = autoencoder.cuda()
    
    # Tool visualization setup
    tool_viz_enabled = cfg.eval.triplets.get('tool_viz_dir') is not None
    
    results = []
    for sample in samples:
        target_frame = sample['target_frame']
        target_timestep = target_frame // frame_stride
        target_timestep = min(target_timestep, num_ts - 1)
        
        # Build prompt
        prompt = prompt_template.format(timestep=target_timestep)
        
        # Create tools
        graph_tools = GraphTools(
            node_feats_npz=node_feats_npz,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            adjacency_matrices=adjacency,
            positions=positions,
            clusters=clusters,
            patch_latents_through_time=patch_latents,
            bhattacharyya_coeffs=bhattacharyya,
            autoencoder=autoencoder,
        )
        
        # Tool call limits (single timestep exploration)
        tool_call_limits = {
            'inspect_node': cfg.eval.triplets.get('max_inspect_node_calls', 20),
            'list_neighbors': cfg.eval.triplets.get('max_list_neighbors_calls', 15),
            'inspect_edge': cfg.eval.triplets.get('max_inspect_edge_calls', 10),
            'inspect_node_highres': cfg.eval.triplets.get('max_inspect_node_highres_calls', 5) if autoencoder else 0,
        }
        
        tools = graph_tools.get_tools_for_timestep_exploration(
            timestep=target_timestep,
        )
        
        if tool_viz_enabled:
            viz_dir = Path(cfg.eval.triplets.tool_viz_dir) / clip.name
            viz_dir.mkdir(parents=True, exist_ok=True)
            graph_tools.start_recording(str(viz_dir / f"sample_{sample['sample_id']}.rrd"))
        
        agent_result = prompt_graph_agent(
            question=prompt,
            node_feats=node_feats_npz,
            initial_timestep_idx=target_timestep,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            tools=tools,
            qwen_version=cfg.eval.qwen_version,
            system_prompt=system_prompt,
            max_iterations=cfg.eval.triplets.get('graph_agent_max_iterations', 10),
            tool_call_limits=tool_call_limits,
        )
        
        if tool_viz_enabled:
            graph_tools.stop_recording()
        
        # Extract response
        if isinstance(agent_result, dict):
            response = agent_result.get("final_answer", str(agent_result))
            tool_calls = sanitize_tool_calls(agent_result.get("tool_calls", []))
            message_history = agent_result.get("message_history", [])
        else:
            response = agent_result
            tool_calls = []
            message_history = []
        
        # Parse response
        predicted = parse_triplets_response(response)
        
        results.append({
            'sample_id': f"v{sample['video_id']:02d}_s{sample['second_idx']:03d}",
            'video_id': sample['video_id'],
            'clip_start': sample['clip_start'],
            'second_idx': sample['second_idx'],
            'target_frame': target_frame,
            'predicted': predicted,
            'ground_truth': sample['gt_triplets'],
            'raw_response': response,
            'tool_calls': tool_calls,
            'message_history': message_history,
        })
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def graph_agent_dynamic_queries(
    model,
    processor,
    samples: List[Dict],
    graph_path: Path,
    clip: DictConfig,
    cfg: DictConfig,
) -> List[Dict]:
    """Run graph agent triplet recognition with full temporal exploration.
    
    Args:
        model: Qwen VL model (must be qwen3)
        processor: Qwen VL processor
        samples: List of sample dicts
        graph_path: Path to graph directory
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    assert cfg.eval.qwen_version == "qwen3", "graph_agent requires qwen3"
    
    system_prompt = cfg.eval.triplets.graph_agent_dynamic_system_prompt
    prompt_template = cfg.eval.triplets.graph_agent_dynamic_prompt_template
    
    # Load graph artifacts
    node_feats_npz = np.load(graph_path / "c_qwen_feats.npz")
    node_centers = np.load(graph_path / "c_centers.npy")
    node_centroids = np.load(graph_path / "c_centroids.npy")
    node_extents = np.load(graph_path / "c_extents.npy")
    positions = np.load(graph_path / "positions.npy")
    clusters = np.load(graph_path / "clusters.npy")
    patch_latents = np.load(graph_path / "patch_latents_through_time.npy")
    adjacency = np.load(graph_path / "graph.npy")
    bhattacharyya = np.load(graph_path / "bhattacharyya_coeffs.npy")
    
    num_ts = adjacency.shape[0]
    frame_stride = cfg.eval.triplets.frame_stride
    
    # Load autoencoder
    autoencoder = None
    if cfg.eval.triplets.get('enable_highres_tools', False):
        autoencoder = QwenAutoencoder.from_pretrained(cfg.autoencoder.output_dir)
        autoencoder.eval()
        if torch.cuda.is_available():
            autoencoder = autoencoder.cuda()
    
    # Tool visualization
    tool_viz_enabled = cfg.eval.triplets.get('tool_viz_dir') is not None
    
    results = []
    for sample in samples:
        target_frame = sample['target_frame']
        target_timestep = target_frame // frame_stride
        target_timestep = min(target_timestep, num_ts - 1)
        
        # Build prompt
        prompt = prompt_template.format(timestep=target_timestep)
        
        # Create tools for full temporal exploration
        graph_tools = GraphTools(
            node_feats_npz=node_feats_npz,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            adjacency_matrices=adjacency,
            positions=positions,
            clusters=clusters,
            patch_latents_through_time=patch_latents,
            bhattacharyya_coeffs=bhattacharyya,
            autoencoder=autoencoder,
        )
        
        # Tool call limits (temporal exploration)
        tool_call_limits = {
            'inspect_node': cfg.eval.triplets.get('max_inspect_node_calls', 30),
            'list_neighbors': cfg.eval.triplets.get('max_list_neighbors_calls', 20),
            'inspect_edge': cfg.eval.triplets.get('max_inspect_edge_calls', 15),
            'inspect_node_highres': cfg.eval.triplets.get('max_inspect_node_highres_calls', 8) if autoencoder else 0,
            'inspect_node_at_timestep': cfg.eval.triplets.get('max_inspect_node_at_timestep_calls', 20),
            'list_neighbors_at_timestep': cfg.eval.triplets.get('max_list_neighbors_at_timestep_calls', 15),
            'inspect_edge_at_timestep': cfg.eval.triplets.get('max_inspect_edge_at_timestep_calls', 10),
        }
        
        tools = graph_tools.get_tools_for_temporal_exploration(
            first_frame=0,
            last_frame=num_ts - 1,
        )
        
        if tool_viz_enabled:
            viz_dir = Path(cfg.eval.triplets.tool_viz_dir) / clip.name
            viz_dir.mkdir(parents=True, exist_ok=True)
            graph_tools.start_recording(str(viz_dir / f"sample_{sample['sample_id']}_dynamic.rrd"))
        
        agent_result = prompt_graph_agent(
            question=prompt,
            node_feats=node_feats_npz,
            initial_timestep_idx=0,  # Start at beginning, agent explores forward
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            tools=tools,
            qwen_version=cfg.eval.qwen_version,
            system_prompt=system_prompt,
            max_iterations=cfg.eval.triplets.get('graph_agent_max_iterations', 15),
            tool_call_limits=tool_call_limits,
        )
        
        if tool_viz_enabled:
            graph_tools.stop_recording()
        
        # Extract response
        if isinstance(agent_result, dict):
            response = agent_result.get("final_answer", str(agent_result))
            tool_calls = sanitize_tool_calls(agent_result.get("tool_calls", []))
            message_history = agent_result.get("message_history", [])
        else:
            response = agent_result
            tool_calls = []
            message_history = []
        
        # Parse response
        predicted = parse_triplets_response(response)
        
        results.append({
            'sample_id': f"v{sample['video_id']:02d}_s{sample['second_idx']:03d}",
            'video_id': sample['video_id'],
            'clip_start': sample['clip_start'],
            'second_idx': sample['second_idx'],
            'target_frame': target_frame,
            'predicted': predicted,
            'ground_truth': sample['gt_triplets'],
            'raw_response': response,
            'tool_calls': tool_calls,
            'message_history': message_history,
        })
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results
