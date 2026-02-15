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

    ks: list[int] = list(cm_cfg.l2_top_ks)
    layers_filter = {str(layer_idx) for layer_idx in cm_cfg.layers}

    methods: list[str] = list(cm_cfg.methods)

    # Dataset-wide accumulators (layerwise):
    # methods -> class -> layer_key -> {sum: np.array[K], count: int}
    dataset_stats: dict[str, dict[str, dict[str, dict[str, np.ndarray | int]]]] = {
        m: {"objects": {}, "actions": {}, "all": {}} for m in methods
    }

    def _min_l2_at_k(pred_coords: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
        # pred_coords: [N, 2] (x, y); gt_xy: [2]
        if pred_coords.size == 0:
            return np.full(len(ks), np.inf, dtype=np.float64)
        diffs = pred_coords.astype(np.float64) - gt_xy[None, :]
        dists = np.sqrt((diffs[:, 0] ** 2) + (diffs[:, 1] ** 2))  # [N]
        out = np.empty(len(ks), dtype=np.float64)
        for i, k in enumerate(ks):
            kk = min(k, dists.shape[0])
            if kk <= 0:
                out[i] = np.inf
            else:
                out[i] = float(np.min(dists[:kk]))
        return out

    for clip in cfg.clips:
        clip_name = str(clip.name)
        gt_path = Path(cfg.preprocessed_root) / clip_name / gt_filename
        pred_path = pred_root / f"{clip_name}.json"

        if not gt_path.exists() or not pred_path.exists():
            continue

        with gt_path.open("r") as f:
            gt_data = json.load(f)
        with pred_path.open("r") as f:
            preds_all = json.load(f)

        per_clip_results = {}

        for method in methods:
            if method not in preds_all:
                continue

            # Per-class, per-layer accumulators for this clip
            sums: dict[str, dict[str, np.ndarray]] = {
                "objects": {},
                "actions": {},
                "all": {},
            }
            counts: dict[str, dict[str, int]] = {"objects": {}, "actions": {}, "all": {}}
            clip_method_items: list[dict] = []

            method_preds = preds_all[method]

            # Iterate timesteps present in both GT and predictions
            for t_key, gt_entry in gt_data.items():
                if t_key not in method_preds:
                    continue
                pred_entry = method_preds[t_key]

                for group in ("objects", "actions"):
                    gt_list = gt_entry.get(group, [])
                    pred_list = pred_entry.get(group, [])
                    n = min(len(gt_list), len(pred_list))
                    if n <= 0:
                        continue

                    for i in range(n):
                        gt_item = gt_list[i]
                        pred_item = pred_list[i]

                        # Ground-truth pixel point (x, y)
                        gx = float(gt_item["pixel_x"])  # assume set in config pipeline
                        gy = float(gt_item["pixel_y"])  # assume set in config pipeline
                        gt_xy = np.array([gx, gy], dtype=np.float64)

                        # Collect per-layer min_l2@k for this query
                        preds_by_layer = pred_item.get("predictions", {})
                        per_layer_out: dict[str, dict[str, float]] = {}
                        for layer_key, layer_pred in preds_by_layer.items():
                            lkey = str(layer_key)
                            if lkey not in layers_filter:
                                continue
                            coords = np.array(layer_pred.get("pixel_coords", []), dtype=np.float64)
                            vals = _min_l2_at_k(coords, gt_xy)
                            vals = np.round(vals, 2)
                            per_layer_out[lkey] = {f"min_l2@{k}": float(v) for k, v in zip(ks, vals.tolist())}
                            # accumulate per group/layer
                            if lkey not in sums[group]:
                                sums[group][lkey] = np.zeros(len(ks), dtype=np.float64)
                                counts[group][lkey] = 0
                            sums[group][lkey] += vals
                            counts[group][lkey] += 1
                            # overall
                            if lkey not in sums["all"]:
                                sums["all"][lkey] = np.zeros(len(ks), dtype=np.float64)
                                counts["all"][lkey] = 0
                            sums["all"][lkey] += vals
                            counts["all"][lkey] += 1

                        # record per-query item if we computed any layer metrics
                        if per_layer_out:
                            query_text = pred_item.get("query") or gt_item.get("query")
                            clip_method_items.append(
                                {
                                    "timestep": t_key,
                                    "frame_number": int(gt_entry.get("frame_number", -1)),
                                    "group": group,
                                    "query": query_text,
                                    "per_layer": per_layer_out,
                                }
                            )

            # Compute averages for this clip and method
            method_out = {"per_class": {}, "counts": {}, "items": clip_method_items}
            for group in ("objects", "actions", "all"):
                per_layer_avgs = {}
                per_layer_counts = {}
                for lkey, svec in sums[group].items():
                    c = counts[group].get(lkey, 0)
                    if c > 0:
                        avg_arr = svec / c
                        avg_list = np.round(avg_arr, 2).tolist()
                    else:
                        avg_list = [float("nan")] * len(ks)
                    per_layer_avgs[lkey] = {f"min_l2@{k}": v for k, v in zip(ks, avg_list)}
                    per_layer_counts[lkey] = c

                    # Update dataset-wide accumulators
                    if lkey not in dataset_stats[method][group]:
                        dataset_stats[method][group][lkey] = {
                            "sum": np.zeros(len(ks), dtype=np.float64),
                            "count": 0,
                        }
                    dataset_stats[method][group][lkey]["sum"] += svec
                    dataset_stats[method][group][lkey]["count"] += c

                method_out["per_class"][group] = per_layer_avgs
                method_out["counts"][group] = per_layer_counts

            per_clip_results[method] = method_out

        # Save per-clip file with query-wise metrics
        clip_out = {
            "clip": clip_name,
            "methods": per_clip_results,
        }
        with (out_dir / f"{clip_name}.json").open("w") as f:
            json.dump(clip_out, f, indent=2)

    # Save dataset-wide summary
    summary = {"methods": {}}
    for method in methods:
        if method not in dataset_stats:
            continue
        mstats = dataset_stats[method]
        out_m = {"per_class": {}, "counts": {}}
        for group in ("objects", "actions", "all"):
            per_layer = {}
            per_layer_counts = {}
            for lkey, stat in mstats[group].items():
                total_count = int(stat["count"])  # type: ignore[index]
                sums_arr = stat["sum"]  # type: ignore[index]
                if total_count > 0:
                    avg_arr = sums_arr / total_count  # type: ignore[operator]
                    avg_list = np.round(avg_arr, 2).tolist()
                else:
                    avg_list = [float("nan")] * len(ks)
                per_layer[lkey] = {f"min_l2@{k}": v for k, v in zip(ks, avg_list)}
                per_layer_counts[lkey] = total_count
            out_m["per_class"][group] = per_layer
            out_m["counts"][group] = per_layer_counts
        summary["methods"][method] = out_m

    # Create overview: best @1 score for each method in "all" category
    overview = {}
    for method in methods:
        if method not in dataset_stats:
            continue
        mstats = dataset_stats[method]
        all_group = mstats.get("all", {})
        
        best_layer = None
        best_score = float("inf")
        
        for lkey, stat in all_group.items():
            total_count = int(stat["count"])  # type: ignore[index]
            sums_arr = stat["sum"]  # type: ignore[index]
            if total_count > 0:
                avg_arr = sums_arr / total_count  # type: ignore[operator]
                # Get the @1 score (first element in ks)
                score_at_1 = float(avg_arr[0])
                if score_at_1 < best_score:
                    best_score = score_at_1
                    best_layer = lkey
        
        if best_layer is not None:
            overview[method] = {
                "best_min_l2@1": round(best_score, 2),
                "best_layer": best_layer
            }

    with aggregated_file.open("w") as f:
        json.dump({"overview": overview, "summary": summary}, f, indent=2)

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
                        iou = compute_temporal_iou(gt_ranges, pred_ranges, cm_cfg.n_timesteps)
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