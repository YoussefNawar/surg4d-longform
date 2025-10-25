from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra

from benchmark.benchmark_config import BenchmarkConfig
from benchmark.frame_selectors import TripletsFrameSelector
from benchmark.frame_evaluators import TripletsFrameEvaluator


def _build_benchmark_config(cfg: DictConfig, clip: DictConfig) -> BenchmarkConfig:
    # Infer qwen version and quantization from feature_extraction group if present
    use_qwen3 = bool(cfg.get("feature_extraction", {}).get("use_qwen3", False))
    qwen_version = "qwen3" if use_qwen3 else "qwen2.5"
    use_4bit = bool(cfg.get("feature_extraction", {}).get("bnb_4bit", False))

    # Paths
    cholect50_root = Path(cfg.cholect50_root)
    preprocessed_root = Path(cfg.preprocessed_root)
    output_root = Path(cfg.output_root)

    clip_name = str(clip.name)
    video_dir = preprocessed_root / clip_name
    graph_dir = output_root / clip_name / "graph"

    # Convert nested eval configs (OmegaConf) to plain dicts
    triplets_cfg = None
    temporal_cfg = None
    spatial_cfg = None
    spatiotemporal_cfg = None

    if cfg.get("eval") is not None:
        if cfg.eval.get("triplets") is not None:
            triplets_cfg = OmegaConf.to_container(cfg.eval.triplets, resolve=True)
        if cfg.eval.get("temporal") is not None:
            temporal_cfg = OmegaConf.to_container(cfg.eval.temporal, resolve=True)
        if cfg.eval.get("spatial") is not None:
            spatial_cfg = OmegaConf.to_container(cfg.eval.spatial, resolve=True)

    bench_cfg = BenchmarkConfig(
        triplets_config=triplets_cfg,
        temporal_config=temporal_cfg,
        spatial_config=spatial_cfg,
        spatiotemporal_config=spatiotemporal_cfg,
        cholect50_root=cholect50_root,
        preprocessed_root=preprocessed_root,
        output_root=output_root,
        results_dir=output_root / "benchmark",
        video_dir=video_dir,
        graph_dir=graph_dir,
        model_name="qwen",
        qwen_version=qwen_version,
        use_4bit_quantization=use_4bit,
    )
    return bench_cfg


def evaluate_clip(clip: DictConfig, cfg: DictConfig):
    """Run benchmark evaluations for a single clip using Hydra configs."""
    if not cfg.get("eval"):
        return

    bench_cfg = _build_benchmark_config(cfg, clip)

    # Only triplets for now
    if bench_cfg.triplets_config is None:
        return

    selector = TripletsFrameSelector(bench_cfg)
    samples = selector.select_sequences()
    if not samples:
        print("ERROR: No samples selected! Check if preprocessed data is available.")
        return
    selector.print_summary(samples)

    evaluator = TripletsFrameEvaluator(bench_cfg)
    results = evaluator.run_ablation_study(
        samples, ablations=bench_cfg.triplets_config["ablations"]  # type: ignore[index]
    )

    # Save to required output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.eval.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{clip.name}_ablation_{ts}.json"
    evaluator.save_results(results, out_file)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    for clip in tqdm(cfg.clips, desc="Evaluating clips", unit="clip"):
        evaluate_clip(clip, cfg)


if __name__ == "__main__":
    main()