"""
Configuration for the surgical VQA benchmark
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation"""

    # Either none or a dict witht the config for that task
    triplets_config: Optional[dict] = None
    temporal_config: Optional[dict] = None
    spatial_config: Optional[dict] = None
    spatiotemporal_config: Optional[dict] = None

    # Data paths
    # TODO: most of these should come from the prior config files
    cholect50_root: Path = Path("/home/students/lmu_proj/shared_data/data/cholect50")
    preprocessed_root: Path = Path("/home/students/lmu_proj/shared_data/data/cholecseg8k/preprocessed_ssg")
    output_root: Path = Path("/home/students/lmu_proj/shared_data/output/cholecseg8k")
    results_dir: Path = Path("/home/students/lmu_proj/surgery-scene-graphs/benchmark/results")

    # TODO: this needs to come from the config files that were used in creating the model
    video_dir: Path = Path("/home/students/lmu_proj/shared_data/data/cholecseg8k/preprocessed_ssg/video01")
    graph_dir: Path = Path("/home/students/lmu_proj/surgery-scene-graphs/output/cholecseg8k/video01_00080_qwen_cat/graph")
    
    # Model settings
    model_name: Literal["qwen", "gpt4"] = "qwen"
    qwen_version: Literal["qwen2.5", "qwen3"] = "qwen2.5"  # Differentiate between Qwen versions
    use_4bit_quantization: bool = False
    device: str = "auto"
    
    # Evaluation settings
    # TODO: remove?
    # num_test_frames: int = 5  # Start small
    seed: int = 42
    
    # Question types to evaluate
    # TODO: remove? not sure if this is currently used
    test_triplet_recognition: bool = True
    test_component_recognition: bool = True  # Individual I, V, T
    test_spatial_reasoning: bool = False  # Requires scene graph
    test_temporal_reasoning: bool = False  # Requires multiple frames
    
    # Scene graph settings
    # TODO: remove, should be derived from the ablations in the task specific configs?
    # use_scene_graph: bool = False
    graph_encoding: Literal["xml", "natural_language", "both"] = "xml"
    include_object_features: bool = True
    include_spatial_relationships: bool = True
    include_3d_positions: bool = True
    
    # Evaluation strictness
    # exact_match: bool = True  # Use exact string matching only (no fuzzy matching/synonyms)
    # case_sensitive: bool = False
    
    # Output settings
    save_responses: bool = True
    save_prompts: bool = True
    verbose: bool = True


@dataclass
class TestSample:
    """A single test sample"""
    video_id: int
    frame_num: int
    clip_start: int  # For accessing pre-computed graphs
    image_path: Optional[Path] = None
    graph_path: Optional[Path] = None
    
    # Ground truth
    gt_triplets: List[dict] = field(default_factory=list)
    gt_phase: Optional[str] = None
    
    # For identification
    sample_id: str = ""
    
    def __post_init__(self):
        if not self.sample_id:
            self.sample_id = f"vid{self.video_id:02d}_frame{self.frame_num:06d}"


# Note: Synonym matching has been removed - using exact string matching only
# The model is provided with exact options in the system prompt

def normalize_for_matching(text: str) -> str:
    """Normalize text for comparison (basic normalization only)"""
    return text.lower().strip().replace("_", " ").replace("-", " ")




