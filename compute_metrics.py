import hydra
from omegaconf import DictConfig
import random
import numpy as np
import torch
from pathlib import Path
import json

def compute_spatial_metrics(cfg: DictConfig):
    if cfg.compute_metrics.spatial is None:
        return
    
    cm_cfg = cfg.compute_metrics.spatial
    gt_filename: str = cm_cfg.gt_filename
    pred_root = Path(cm_cfg.pred_root)
    out_dir = Path(cm_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = Path(cm_cfg.aggregated_output_filename)
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Dataset-wide accumulators per method (query-wise for micro average)
    method_all_distances: dict[str, list[float]] = {}
    
    # Per-clip results
    for clip in cfg.clips:
        clip_name = str(clip.name)
        gt_path = Path(cfg.preprocessed_root) / clip_name / gt_filename
        pred_path = pred_root / f"{clip_name}.json"
        
        if not gt_path.exists() or not pred_path.exists():
            continue
        
        with gt_path.open("r") as f:
            gt_data = json.load(f)
        with pred_path.open("r") as f:
            preds_data = json.load(f)
        
        annotations = gt_data.get("annotations", [])
        methods_preds = preds_data
        
        clip_results: dict[str, dict] = {}
        
        # Process each method
        for method_name, method_preds in methods_preds.items():
            # Initialize method accumulators if not exists
            if method_name not in method_all_distances:
                method_all_distances[method_name] = []
            
            method_distances: list[float] = []
            query_results: list[dict] = []
            
            # Create a mapping from query_id to predictions
            pred_by_query_id: dict[str, dict] = {}
            for pred_item in method_preds:
                query_id = pred_item.get("id")
                if query_id:
                    pred_by_query_id[query_id] = pred_item
            
            # Process each annotation
            for annotation in annotations:
                query_id = str(annotation.get("id"))
                question = str(annotation.get("query"))
                timestep = int(annotation.get("timestep"))
                
                # Ground-truth pixel point (x, y)
                pil_coords = annotation.get("pil_coords", [])
                if len(pil_coords) != 2:
                    continue
                gx = float(pil_coords[0])
                gy = float(pil_coords[1])
                gt_xy = np.array([gx, gy], dtype=np.float64)
                
                pred_item = pred_by_query_id.get(query_id)
                
                if pred_item and "predicted" in pred_item and pred_item["predicted"] is not None:
                    pred_coords = pred_item["predicted"]
                    if isinstance(pred_coords, list) and len(pred_coords) == 2:
                        px = float(pred_coords[0])
                        py = float(pred_coords[1])
                        pred_xy = np.array([px, py], dtype=np.float64)
                        
                        # Compute L2 distance
                        diff = pred_xy - gt_xy
                        l2_distance = float(np.sqrt(diff[0] ** 2 + diff[1] ** 2))
                    else:
                        # Invalid prediction format
                        l2_distance = float("inf")
                        px, py = None, None
                else:
                    # No prediction or parsing failed
                    l2_distance = float("inf")
                    px, py = None, None
                
                method_distances.append(l2_distance)
                method_all_distances[method_name].append(l2_distance)
                
                query_results.append({
                    "id": query_id,
                    "timestep": timestep,
                    "query": question,
                    "ground_truth_pixel": [gx, gy],
                    "predicted_pixel": [px, py] if px is not None and py is not None else None,
                    "l2_distance": round(l2_distance, 2) if l2_distance != float("inf") else None,
                })
            
            # Compute per-method averages for this clip
            valid_distances = [d for d in method_distances if d != float("inf")]
            clip_results[method_name] = {
                "queries": query_results,
                "num_queries": len(query_results),
                "mean_l2_distance": round(float(np.mean(valid_distances)), 2) if valid_distances else None,
                "std_l2_distance": round(float(np.std(valid_distances)), 2) if valid_distances else None,
            }
        
        # Save per-clip results
        with (out_dir / f"{clip_name}.json").open("w") as f:
            json.dump({
                "clip": clip_name,
                "methods": clip_results,
            }, f, indent=2)
    
    # Compute aggregated metrics per method (micro average over all queries)
    aggregated: dict[str, dict] = {}
    for method_name in method_all_distances.keys():
        distances_list = method_all_distances[method_name]
        valid_distances = [d for d in distances_list if d != float("inf")]
        
        aggregated[method_name] = {
            "num_queries": len(distances_list),
            "mean_l2_distance": round(float(np.mean(valid_distances)), 2) if valid_distances else None,
            "std_l2_distance": round(float(np.std(valid_distances)), 2) if valid_distances else None,
        }
    
    with aggregated_file.open("w") as f:
        json.dump({"methods": aggregated}, f, indent=2)

def compute_temporal_metrics(cfg: DictConfig):
    if cfg.compute_metrics.temporal is None:
        return
    
    cm_cfg = cfg.compute_metrics.temporal
    pred_root = Path(cm_cfg.pred_root)

    out_dir = Path(cm_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = Path(cm_cfg.aggregated_output_filename)
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Dataset-wide accumulators per method (query-wise for micro average)
    method_all_errors: dict[str, list[float]] = {}
    method_all_ious: dict[str, list[float]] = {}
    
    # Per-clip results
    for clip in cfg.clips:
        clip_name = str(clip.name)
        labels_path = Path(cfg.compute_metrics.annotations_root) / "temporal" / f"{clip_name}.json"
        pred_path = pred_root / f"{clip_name}.json"
        
        if not labels_path.exists() or not pred_path.exists():
            continue
        
        with labels_path.open("r") as f:
            labels_data = json.load(f)
        with pred_path.open("r") as f:
            preds_data = json.load(f)

        annotations = labels_data.get("annotations", [])
        methods_preds = preds_data.get("methods", {})
        
        clip_results: dict[str, dict] = {}
        
        # Process each method
        for method_name, method_preds in methods_preds.items():
            # Initialize method accumulators if not exists
            if method_name not in method_all_errors:
                method_all_errors[method_name] = []
                method_all_ious[method_name] = []
            
            method_errors: list[float] = []
            method_ious: list[float] = []
            query_results: list[dict] = []
            
            # Create a mapping from query_id to predictions
            pred_by_query_id: dict[str, dict] = {}
            for pred_item in method_preds:
                query_id = pred_item.get("id")
                if query_id:
                    pred_by_query_id[query_id] = pred_item
            
            # Process each annotation
            for annotation in annotations:
                query_id = str(annotation.get("id"))
                query_type = str(annotation.get("type"))
                question = str(annotation.get("query"))
                
                pred_item = pred_by_query_id.get(query_id)
                
                if query_type == "pit":
                    gt_timestep = int(annotation["timesstep"])
                    
                    if pred_item and "predicted" in pred_item and pred_item["predicted"] is not None:
                        pred_timestep = int(pred_item["predicted"])
                        error = float(abs(pred_timestep - gt_timestep))
                    else:
                        # No prediction or parsing failed
                        pred_timestep = None
                        error = float(cm_cfg.pit_noprediction_error)
                    
                    method_errors.append(error)
                    method_all_errors[method_name].append(error)
                    
                    query_results.append({
                        "id": query_id,
                        "type": query_type,
                        "query": question,
                        "ground_truth_timestep": gt_timestep,
                        "predicted_timestep": pred_timestep,
                        "absolute_error": error,
                    })
                    
                elif query_type == "range":
                    # Ground truth ranges (inclusive)
                    gt_ranges = annotation["ranges"]
                    
                    if pred_item and pred_item.get("predicted"):
                        pred_ranges = pred_item["predicted"]
                        iou = compute_temporal_iou(gt_ranges, pred_ranges, cfg.compute_metrics.n_timesteps)
                    else:
                        # No prediction or parsing failed
                        iou = 0.0
                    
                    method_ious.append(iou)
                    method_all_ious[method_name].append(iou)
                    
                    query_results.append({
                        "id": query_id,
                        "type": query_type,
                        "query": question,
                        "ground_truth_ranges": gt_ranges,
                        "predicted_ranges": pred_ranges,
                        "iou": iou,
                    })

                else:
                    raise ValueError(f"Unsupported query type for {clip_name} {query_id}: {query_type}")
            
            # Compute per-method averages for this clip
            clip_results[method_name] = {
                "queries": query_results,
                "num_queries": len(query_results),
                "mean_absolute_error": round(float(np.mean(method_errors)), 2) if method_errors else None,
                "mean_iou": round(float(np.mean(method_ious)), 2) if method_ious else None,
            }
        
        # Save per-clip results
        with (out_dir / f"{clip_name}.json").open("w") as f:
            json.dump({
                "clip": clip_name,
                "methods": clip_results,
            }, f, indent=2)
    
    # Compute aggregated metrics per method (micro average over all queries)
    aggregated: dict[str, dict] = {}
    for method_name in method_all_errors.keys() | method_all_ious.keys():
        errors = method_all_errors.get(method_name, [])
        ious = method_all_ious.get(method_name, [])
        
        aggregated[method_name] = {
            "mean_absolute_error": round(float(np.mean(errors)), 2) if errors else None,
            "std_absolute_error": round(float(np.std(errors)), 2) if errors else None,
            "mean_iou": round(float(np.mean(ious)), 3) if ious else None,
            "std_iou": round(float(np.std(ious)), 3) if ious else None,
            "num_pit_queries": len(errors),
            "num_range_queries": len(ious),
            "num_queries": len(errors) + len(ious),
        }
    
    with aggregated_file.open("w") as f:
        json.dump({"methods": aggregated}, f, indent=2)


def compute_temporal_iou(gt_ranges: list[list[int]], pred_ranges: list[list[int]], max_timestep: int) -> float:
    """Compute IoU between ground truth and predicted temporal ranges.
    
    Args:
        gt_ranges: List of [start, end] ranges (inclusive) from ground truth
        pred_ranges: List of [start, end] ranges (inclusive) from predictions
        max_timestep: Maximum timestep value (for clipping)
        
    Returns:
        IoU score between 0 and 1
    """
    # Convert ranges to sets of timesteps
    gt_timesteps: set[int] = set()
    for start, end in gt_ranges:
        # Ranges are inclusive
        for t in range(int(start), int(end) + 1):
            if 0 <= t < max_timestep:
                gt_timesteps.add(t)
    
    pred_timesteps: set[int] = set()
    for start, end in pred_ranges:
        # Ranges are inclusive
        for t in range(int(start), int(end) + 1):
            if 0 <= t < max_timestep:
                pred_timesteps.add(t)
    
    if len(gt_timesteps) == 0 and len(pred_timesteps) == 0:
        return 1.0
    
    if len(gt_timesteps) == 0 or len(pred_timesteps) == 0:
        return 0.0
    
    intersection = len(gt_timesteps & pred_timesteps)
    union = len(gt_timesteps | pred_timesteps)
    
    return float(intersection) / float(union)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Deterministic Torch/CUDA setup (harmless for CPU-only metrics)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    compute_spatial_metrics(cfg)
    compute_temporal_metrics(cfg)

if __name__ == "__main__":
    main()