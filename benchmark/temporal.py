"""
Temporal action localization evaluation for surgical videos.

Supports temporal query types:
- Action Onset: When does X start?
- Action Offset: When does X end?
- Action Duration: During which timesteps does X happen?
- Multiple Event Ordering: Which happens first?
- Count/Frequency: How many times does X happen?
"""

import gc
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import torch
from omegaconf import DictConfig
from llm.qwen_utils import (
    prompt_with_video_frames,
    prompt_graph_agent,
)
from llm.tools import GraphTools
from autoencoder.model_qwen import QwenAutoencoder
from benchmark.serialization_utils import sanitize_tool_calls
from benchmark.serialization_utils import parse_json


def load_video_frames(video_dir: Path, images_subdir: str) -> "Tuple[List[Path], int]":
    """Load video frames from directory.
    
    Args:
        video_dir: Root directory containing video data
        images_subdir: Subdirectory containing frame images
        
    Returns:
        Tuple of (sorted frame paths, num_frames)
    """
    images_dir = video_dir / images_subdir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    frames = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    num_frames = len(frames)
    
    if num_frames == 0:
        raise FileNotFoundError(f"No .jpg or .png images found in {images_dir}")
    
    return frames, num_frames


def seconds_to_timestep(seconds: float, num_timesteps: int, fps: float) -> int:
    """Convert seconds to nearest timestep using nearest neighbor rounding.
    
    Args:
        seconds: Time in seconds
        num_timesteps: Total number of timesteps (frames)
        fps: Frames per second
        
    Returns:
        Nearest timestep (integer frame index)
    """
    if seconds is None:
        return None
    # Convert seconds to frame index and round to nearest integer
    frame_idx = int(round(seconds * fps))
    # Clamp to valid range
    return max(0, min(frame_idx, num_timesteps - 1))


def load_graph_data(graph_path: Path) -> Dict[str, Any]:
    """Load precomputed 4D graph data.
    
    Args:
        graph_path: Path to graph directory
        
    Returns:
        Dict with graph data
    """
    qwen_feat_file = graph_path / "c_qwen_feats.npz"
    qwen_feats_dict = np.load(qwen_feat_file)
    adjacency_matrices = np.load(graph_path / "graph.npy")
    centers = np.load(graph_path / "c_centers.npy")
    centroids = np.load(graph_path / "c_centroids.npy")
    extents = np.load(graph_path / "c_extents.npy")
    
    return {
        'node_feats': qwen_feats_dict,
        'adjacency_matrices': adjacency_matrices,
        'node_centers': centers,
        'node_centroids': centroids,
        'node_extents': extents
    }


def get_num_timesteps_from_graph(graph_path: Path) -> int:
    """Get number of timesteps from graph data.
    
    Args:
        graph_path: Path to graph directory
        
    Returns:
        Number of timesteps
    """
    graph_data = load_graph_data(graph_path)
    return int(graph_data['adjacency_matrices'].shape[0])


def parse_single_frame(response: str, query_type: str) -> Optional[Dict]:
    """Parse response for action_onset or action_offset.
    
    Args:
        response: Model response text
        query_type: Type of query (for unwrapping nested JSON)
        
    Returns:
        Dict with 'timestep' key or None
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            
            if 'timestep' not in data and isinstance(data, dict):
                inner = data.get(query_type)
                if isinstance(inner, dict):
                    data = inner

            # Support both "timestep" (old format) and "second" (new Qwen3 format)
            if 'timestep' in data:
                timestep_val = data['timestep']
                if isinstance(timestep_val, (int, float)):
                    return {'timestep': int(timestep_val)}
            elif 'second' in data:
                second_val = data['second']
                if isinstance(second_val, (int, float)):
                    return {'second': float(second_val)}
        return None
    except Exception:
        return None


def parse_frame_ranges(response: str, query_type: str) -> Optional[Dict]:
    """Parse response for action_duration.
    
    Args:
        response: Model response text
        query_type: Type of query (for unwrapping nested JSON)
        
    Returns:
        Dict with 'ranges' or 'second_ranges' key or None
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            
            if 'ranges' not in data and 'second_ranges' not in data and isinstance(data, dict):
                inner = data.get(query_type)
                if isinstance(inner, dict):
                    data = inner
                    
            # Support both "ranges" (old format) and "second_ranges" (new Qwen3 format)
            if 'ranges' in data:
                ranges = []
                for r in data['ranges']:
                    if isinstance(r, list) and len(r) == 2:
                        ranges.append([int(r[0]), int(r[1])])
                if ranges:
                    return {'ranges': ranges}
            elif 'second_ranges' in data:
                second_ranges = []
                for r in data['second_ranges']:
                    if isinstance(r, list) and len(r) == 2:
                        second_ranges.append([float(r[0]), float(r[1])])
                if second_ranges:
                    return {'second_ranges': second_ranges}
        return None
    except Exception:
        return None


def parse_ordered_events(response: str, query_type: str) -> Optional[Dict]:
    """Parse response for multiple_event_ordering.
    
    Args:
        response: Model response text
        query_type: Type of query (for unwrapping nested JSON)
        
    Returns:
        Dict with 'events' key or None
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            
            if 'events' not in data and isinstance(data, dict):
                inner = data.get(query_type)
                if isinstance(inner, dict):
                    data = inner
                    
            if 'events' in data and isinstance(data['events'], list):
                events = []
                for event in data['events']:
                    if isinstance(event, dict) and 'order' in event and 'frame_range' in event:
                        events.append({
                            'order': int(event['order']),
                            'description': event.get('description', ''),
                            'frame_range': [int(event['frame_range'][0]), int(event['frame_range'][1])]
                        })
                if events:
                    events.sort(key=lambda x: x['order'])
                    return {'events': events}
        return None
    except Exception:
        return None


def parse_count_and_ranges(response: str, query_type: str) -> Optional[Dict]:
    """Parse response for count_frequency.
    
    Args:
        response: Model response text
        query_type: Type of query (for unwrapping nested JSON)
        
    Returns:
        Dict with 'count' and 'occurrences' keys or None
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            
            if 'count' not in data and isinstance(data, dict):
                inner = data.get(query_type)
                if isinstance(inner, dict):
                    data = inner
                    
            if 'count' in data:
                occurrences = []
                if 'occurrences' in data and isinstance(data['occurrences'], list):
                    for occ in data['occurrences']:
                        if isinstance(occ, list) and len(occ) >= 2:
                            occurrences.append([int(occ[0]), int(occ[1])])
                return {
                    'count': int(data['count']),
                    'occurrences': occurrences
                }
        return None
    except Exception:
        return None


def multiframe_queries(
    model,
    processor,
    video_frames: List[Path],
    graph_path: Path,
    annotations: List[Dict],
    clip: "DictConfig",
    cfg: "DictConfig",
) -> List[Dict]:
    """Run multiframe (video-only) temporal queries.
    
    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        video_frames: List of video frame paths (full set)
        graph_path: Path to graph directory
        annotations: List of query annotations
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    # Load graph to determine stride for frame sampling
    num_ts = get_num_timesteps_from_graph(graph_path)
    stride = max(1, round(len(video_frames) / num_ts))
    selected_frames = video_frames[::stride]
    
    # Calculate effective FPS based on stride
    # Original video is at video_fps, but we sample every stride frames
    # For 20 graph timesteps from 25 fps video: effective_fps = 25 / 4 = 6.25
    effective_fps = cfg.eval.video_fps / stride

    results = []
    for query_anno in annotations:
        query_type = query_anno['type']

        # prompt
        system_prompt = cfg.eval.temporal.multiframe_system_prompt
        if query_type == 'pit':
            template = cfg.eval.temporal.multiframe_pit_prompt_template
        elif query_type == 'range':
            template = cfg.eval.temporal.multiframe_action_duration_prompt_template
        else:
            raise ValueError(f"Unsupported query type for {clip.name} {query_anno['id']}: {query_type}")
        prompt = template.format(question=query_anno['query'])

        # llm answer
        response = prompt_with_video_frames(
            question=prompt,
            image_paths=selected_frames,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
            fps=effective_fps,
        )
        
        # parse answer and convert to timestep
        json_data = parse_json(response)
        if query_type == 'pit':
            second = json_data.get("second", None)
            prediction = seconds_to_timestep(second, num_ts, effective_fps)
        elif query_type == 'range':
            second_ranges = json_data.get("second_ranges", None)
            prediction = []
            if second_ranges is not None:
                for second_range in second_ranges:
                    if not isinstance(second_range, list) or len(second_range) != 2:
                        prediction = None
                        continue
                    prediction.append([seconds_to_timestep(second_range[0], num_ts, effective_fps), seconds_to_timestep(second_range[1], num_ts, effective_fps)])
        else:
            raise ValueError(f"Unsupported query type for {clip.name} {query_anno['id']}: {query_type}")

        # Build message history for consistency with graph_agent
        # (multiframe doesn't support tools, so this is just the initial query and response)
        message_history = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        
        results.append({
            'id': query_anno['id'],
            'type': query_type,
            'query': query_anno['query'],
            'predicted': prediction,
            'raw_response': response,
            'message_history': message_history,
            'tool_calls': [],
        })
        
        # Clear memory after each query to prevent OOM with video inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def graph_agent_queries(
    model,
    processor,
    graph_path: Path,
    annotations: List[Dict],
    clip: "DictConfig",
    cfg: "DictConfig",
    video_frames: List[Path] = None, # not used in this function, but required for compatibility with other functions
) -> List[Dict]:
    """Run graph agent temporal queries with tools.
    
    Uses prompt_graph_agent with agentic tool use (requires qwen3).
    Provides initial rough nodes at timestep 0, then agent uses temporal tools to explore.
    Loads graph artifacts and creates GraphTools instance with temporal reasoning tools.
    
    Args:
        model: Qwen VL model (must be qwen3)
        processor: Qwen VL processor
        graph_path: Path to graph directory
        annotations: List of query annotations
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    # Load required graph artifacts
    node_feats_npz_path = graph_path / "c_qwen_feats.npz"
    centers_path = graph_path / "c_centers.npy"
    centroids_path = graph_path / "c_centroids.npy"
    extents_path = graph_path / "c_extents.npy"
    positions_path = graph_path / "positions.npy"
    clusters_path = graph_path / "clusters.npy"
    patch_latents_path = graph_path / "patch_latents_through_time.npy"
    adjacency_path = graph_path / "graph.npy"
    bhattacharyya_path = graph_path / "bhattacharyya_coeffs.npy"
    
    node_feats_npz = np.load(node_feats_npz_path)
    node_centers = np.load(centers_path)
    node_centroids = np.load(centroids_path)
    node_extents = np.load(extents_path)
    positions = np.load(positions_path)
    clusters = np.load(clusters_path)
    patch_latents_through_time = np.load(patch_latents_path)
    adjacency = np.load(adjacency_path)
    bhattacharyya_coeffs = np.load(bhattacharyya_path)

    # Load autoencoder for highres inspection tools (following spatial pattern)
    if cfg.eval.temporal.graph_agent_use_global_autoencoder:
        autoencoder_path = Path(cfg.preprocessed_root) / cfg.eval.temporal.graph_agent_global_autoencoder_checkpoint_dir / "best_ckpt.pth"
    else:
        clip_dir = Path(cfg.preprocessed_root) / clip.name
        autoencoder_path = clip_dir / cfg.eval.temporal.graph_agent_autoencoder_checkpoint_subdir / "best_ckpt.pth"
    
    autoencoder = QwenAutoencoder(
        input_dim=cfg.eval.temporal.graph_agent_autoencoder_full_dim,
        latent_dim=cfg.eval.temporal.graph_agent_autoencoder_latent_dim,
    ).to(model.device)
    autoencoder.load_state_dict(
        torch.load(autoencoder_path, map_location=model.device)
    )
    autoencoder.eval()
    
    # Create GraphTools instance for tool management with fps for seconds-based timestamps
    graph_tools = GraphTools(
        positions=positions,
        clusters=clusters,
        centroids=node_centroids,
        centers=node_centers,
        extents=node_extents,
        adjacency=adjacency,
        bhattacharyya_coeffs=bhattacharyya_coeffs,
        qwen_feats=node_feats_npz,
        patch_latents_through_time=patch_latents_through_time,
        autoencoder=autoencoder,
    )
    
    # Setup tool visualization directory if configured
    tool_viz_enabled = cfg.eval.temporal.tool_viz_dir is not None
    tool_viz_dir = None
    if tool_viz_enabled:
        tool_viz_dir = Path(cfg.eval.temporal.tool_viz_dir) / clip.name
        tool_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse graph_agent_tools config (list of objects with name and max_calls)
    tool_names = []
    tool_call_limits = {}
    tool_config = cfg.eval.temporal.graph_agent_tools
    for tool_entry in tool_config:
        tool_name = tool_entry['name']
        tool_names.append(tool_name)
        max_calls = tool_entry.get('max_calls')
        if max_calls is not None:
            tool_call_limits[tool_name] = max_calls
    
    # Get the specific tools needed for graph agent
    tools = graph_tools.get_tools_by_name(tool_names)
    
    # Convert to None if no limits specified
    if len(tool_call_limits) == 0:
        tool_call_limits = None

    # Process each query
    results = []
    for query_anno in annotations:
        query_type = query_anno['type']
        query_id = query_anno['id']
        query = query_anno['query']
        
        # Start recording if tool visualization is enabled
        if tool_viz_enabled:
            # Sanitize question text for filename
            sanitized_question = re.sub(r'[^\w\s-]', '', query)  # Remove special chars
            sanitized_question = re.sub(r'\s+', '_', sanitized_question)  # Replace whitespace with _
            sanitized_question = sanitized_question[:50]  # Limit length
            rrd_file = tool_viz_dir / f"{query_id}_{sanitized_question}.rrd"
            graph_tools.start_recording(str(rrd_file))
        
        # prompt
        system_prompt = cfg.eval.temporal.graph_agent_system_prompt
        if query_type == 'pit':
            template = cfg.eval.temporal.graph_agent_pit_prompt_template
        elif query_type == 'range':
            template = cfg.eval.temporal.graph_agent_range_prompt_template
        else:
            raise ValueError(f"Unsupported query type for {clip.name} {query_anno['id']}: {query_type}")
        num_ts = graph_tools.adjacency.shape[0]
        prompt = template.format(question=query, num_frames=num_ts, last_frame=num_ts - 1)

        # Query model with graph and tools
        # Use prompt_graph_agent which gives initial rough nodes at timestep 0
        # The agent then uses temporal tools to explore across time
        agent_result = prompt_graph_agent(
            question=prompt,
            node_feats=node_feats_npz,
            initial_timestep_idx=0,  # Start at first timestep, agent explores temporally
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=cfg.eval.temporal.graph_agent_max_iterations,
            tool_call_limits=tool_call_limits,
        )
        
        # Stop recording if tool visualization is enabled
        if tool_viz_enabled:
            graph_tools.stop_recording()
        
        # Extract response (agent_result is a dict when tools are used)
        json_data = parse_json(agent_result["final_answer"])
        if query_type == 'pit':
            prediction = json_data.get("timestep", None)
        elif query_type == 'range':
            prediction = json_data.get("ranges", None)
            if not isinstance(prediction, list):
                prediction = None
            for range in prediction:
                if not isinstance(range, list) or len(range) != 2:
                    prediction = None
        else:
            raise ValueError(f"Unsupported query type for {clip.name} {query_anno['id']}: {query_type}")

        results.append({
            'id': query_anno['id'],
            'type': query_type,
            'query': query_anno['query'],
            'predicted': prediction,
            'raw_response': agent_result["final_answer"],
            'message_history': agent_result["message_history"],
            'tool_calls': sanitize_tool_calls(agent_result.get("tool_calls", [])),
        })
        
        # Clear memory after each query to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Clean up autoencoder
    del autoencoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

