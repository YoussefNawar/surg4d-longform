from pathlib import Path
from PIL import Image
import numpy as np
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra
from loguru import logger
import re
from depth_anything_3.api import DepthAnything3

from utils.da3_utils import filter_prediction_edge_artifacts

REPO_ROOT = Path(__file__).resolve().parent
DA3_STREAMING_SCRIPT = (
    REPO_ROOT / "submodules" / "depth-anything-3" / "da3_streaming" / "da3_streaming.py"
)
DA3_STREAMING_BASE_CONFIG = (
    REPO_ROOT
    / "submodules"
    / "depth-anything-3"
    / "da3_streaming"
    / "configs"
    / "base_config.yaml"
)
DA3_STREAMING_WEIGHTS_DIR = (
    REPO_ROOT / "submodules" / "depth-anything-3" / "da3_streaming" / "scripts" / "weights"
)

def extract_frame_number(filepath: Path) -> int:
    """Extract frame number from filename for proper numerical sorting."""
    match = re.search(r"frame_(\d+)", filepath.stem)
    if match:
        return int(match.group(1))
    return 0


def convert_streaming_outputs(
    output_dir: Path,
    num_frames: int,
    edge_gradient_threshold: Optional[float] = None,
) -> dict:
    """
    Convert DA3-Streaming outputs to the format expected by the rest of the codebase.

    DA3-Streaming outputs:
    - camera_poses.txt: Each line is 12 floats (3x4 extrinsic matrix, row-major)
    - intrinsic.txt: Each line is "fx fy cx cy"
    - results_output/: Per-frame npz files with depth and conf arrays

    Returns dict with:
    - depth: (T, H, W)
    - conf: (T, H, W)
    - intrinsics: (T, 3, 3)
    - extrinsics: (T, 3, 4)
    """
    from utils.da3_utils import filter_depth_edge_artifacts

    camera_poses_file = output_dir / "camera_poses.txt"
    intrinsics_file = output_dir / "intrinsic.txt"
    results_output_dir = output_dir / "results_output"

    extrinsics_list = []
    with open(camera_poses_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(float, line.split()))
            if len(values) == 16:
                # DA3-Streaming outputs 4x4 c2w matrices, convert to 3x4 w2c
                c2w = np.array(values, dtype=np.float32).reshape(4, 4)
                w2c = np.linalg.inv(c2w)
                extrinsic = w2c[:3, :4]
            elif len(values) == 12:
                extrinsic = np.array(values, dtype=np.float32).reshape(3, 4)
            else:
                raise ValueError(
                    f"Expected 12 or 16 values per line in camera_poses.txt, got {len(values)}"
                )
            extrinsics_list.append(extrinsic)

    intrinsics_list = []
    with open(intrinsics_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(float, line.split()))
            if len(values) != 4:
                raise ValueError(
                    f"Expected 4 values (fx fy cx cy) per line in intrinsic.txt, got {len(values)}"
                )
            fx, fy, cx, cy = values
            K = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )
            intrinsics_list.append(K)

    depth_list = []
    conf_list = []
    frame_files = sorted(results_output_dir.glob("*.npz"))

    if len(frame_files) != num_frames:
        logger.warning(
            f"Expected {num_frames} frame files, found {len(frame_files)}. "
            "Using found files."
        )

    for frame_file in frame_files:
        frame_data = np.load(frame_file)
        depth = frame_data["depth"]
        conf = frame_data["conf"] if "conf" in frame_data else frame_data.get(
            "confidence", np.ones_like(depth)
        )

        if edge_gradient_threshold is not None:
            depth = filter_depth_edge_artifacts(
                depth, gradient_threshold=edge_gradient_threshold
            )

        depth_list.append(depth)
        conf_list.append(conf)

    depth_all = np.stack(depth_list, axis=0)
    conf_all = np.stack(conf_list, axis=0)
    intrinsics_all = np.stack(intrinsics_list, axis=0)
    extrinsics_all = np.stack(extrinsics_list, axis=0)

    return {
        "depth": depth_all,
        "conf": conf_all,
        "intrinsics": intrinsics_all,
        "extrinsics": extrinsics_all,
    }


def extract_geometry_streaming(clip: DictConfig, cfg: DictConfig):
    """
    Extract geometry using DA3-Streaming for long videos.

    This function calls da3_streaming.py as a subprocess and converts
    its outputs to the format expected by the rest of the codebase.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    clip_dir = Path(cfg.preprocessed_root) / clip.name
    images_dir = clip_dir / cfg.extract_geometry.image_subdir
    export_dir = clip_dir / cfg.extract_geometry.da3_subdir

    image_filenames = sorted(list(images_dir.glob("*.png")), key=extract_frame_number)
    num_frames = len(image_filenames)

    if not DA3_STREAMING_SCRIPT.exists():
        raise FileNotFoundError(
            f"DA3-Streaming script not found at {DA3_STREAMING_SCRIPT}. "
            "Please run: git submodule update --init --recursive"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_config = temp_dir / "streaming_config.yaml"
        temp_output = temp_dir / "output"

        if DA3_STREAMING_BASE_CONFIG.exists():
            with open(DA3_STREAMING_BASE_CONFIG, "r") as f:
                streaming_config = yaml.safe_load(f) or {}
        else:
            streaming_config = {}

        streaming_config["chunk_size"] = cfg.extract_geometry.da3_streaming_chunk_size
        streaming_config["overlap"] = cfg.extract_geometry.da3_streaming_overlap
        streaming_config["similarity_threshold"] = (
            cfg.extract_geometry.da3_streaming_similarity_thresh
        )
        streaming_config["save_depth_conf_result"] = (
            cfg.extract_geometry.da3_streaming_save_depth_conf
        )

        # Set weights paths dynamically based on repo location
        streaming_config["Weights"] = {
            "DA3": str(DA3_STREAMING_WEIGHTS_DIR / "model.safetensors"),
            "DA3_CONFIG": str(DA3_STREAMING_WEIGHTS_DIR / "config.json"),
            "SALAD": str(DA3_STREAMING_WEIGHTS_DIR / "dino_salad.ckpt"),
        }

        with open(temp_config, "w") as f:
            yaml.safe_dump(streaming_config, f)

        logger.info(
            f"Running DA3-Streaming on {num_frames} frames with "
            f"chunk_size={streaming_config['chunk_size']}, "
            f"overlap={streaming_config['overlap']}"
        )

        cmd = [
            sys.executable,
            str(DA3_STREAMING_SCRIPT),
            "--image_dir",
            str(images_dir),
            "--config",
            str(temp_config),
            "--output_dir",
            str(temp_output),
        ]

        # Pass environment variables to subprocess (including CUDA_VISIBLE_DEVICES)
        env = os.environ.copy()
        print(f"CUDA_VISIBLE_DEVICES being passed to subprocess: {env.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], capture_output=True, text=True)
        print(result.stdout)
        print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        # Prevent parent's CUDA context from leaking
        env.pop("CUDA_SETUP_ALREADY_RUN", None)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=DA3_STREAMING_SCRIPT.parent,
            env=env,
        )

        if result.returncode != 0:
            logger.error(f"DA3-Streaming stderr: {result.stderr}")
            raise RuntimeError(
                f"DA3-Streaming failed with return code {result.returncode}"
            )

        logger.info("DA3-Streaming completed, converting outputs...")

        edge_gradient_threshold = cfg.extract_geometry.da3_edge_gradient_threshold
        export_dict = convert_streaming_outputs(
            temp_output,
            num_frames,
            edge_gradient_threshold=edge_gradient_threshold,
        )

        export_dir.mkdir(parents=True, exist_ok=True)
        np.savez(export_dir / "results.npz", **export_dict)

    logger.info(f"Exported geometry to {export_dir / 'results.npz'}")


def extract_geometry_standard(
    clip: DictConfig, cfg: DictConfig, model: DepthAnything3
):
    """
    Extract geometry using standard DA3 inference (for short videos).
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    clip_dir = Path(cfg.preprocessed_root) / clip.name
    images_dir = clip_dir / cfg.extract_geometry.image_subdir

    image_filenames = sorted(list(images_dir.glob("*.png")), key=extract_frame_number)
    image_filenames = [str(img_file) for img_file in image_filenames]
    orig_w, orig_h = Image.open(image_filenames[0]).size
    processing_res = max(orig_w, orig_h)

    prediction = model.inference(
        image=image_filenames,
        ref_view_strategy="middle",
        process_res=processing_res,
        process_res_method="upper_bound_resize",
    )

    edge_gradient_threshold = cfg.extract_geometry.da3_edge_gradient_threshold
    if edge_gradient_threshold is not None:
        logger.info(
            f"Applying depth edge filtering with threshold: {edge_gradient_threshold}"
        )
        prediction = filter_prediction_edge_artifacts(
            prediction,
            gradient_threshold=edge_gradient_threshold,
        )

    export_dir = clip_dir / cfg.extract_geometry.da3_subdir
    export_dir.mkdir(parents=True, exist_ok=True)
    export_dict = {
        "depth": prediction.depth,
        "conf": prediction.conf,
        "intrinsics": prediction.intrinsics,
        "extrinsics": prediction.extrinsics,
    }

    np.savez(export_dir / "results.npz", **export_dict)


def extract_geometry(
    clip: DictConfig, cfg: DictConfig, model: Optional[DepthAnything3] = None
):
    """
    Extract geometry from video frames using DA3.

    Automatically chooses between streaming (for long videos) and standard
    inference (for short videos) based on configuration and frame count.
    """
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    images_dir = clip_dir / cfg.extract_geometry.image_subdir
    image_filenames = list(images_dir.glob("*.png"))
    num_frames = len(image_filenames)

    if model is None:
        logger.info(
            f"Using DA3-Streaming for {clip.name} ({num_frames} frames)"
        )
        extract_geometry_streaming(clip, cfg)
    else:
        use_streaming = getattr(cfg.extract_geometry, "da3_use_streaming", False)
        streaming_threshold = 80

        if use_streaming and num_frames > streaming_threshold:
            logger.info(
                f"Using DA3-Streaming for {clip.name} ({num_frames} frames > {streaming_threshold})"
            )
            extract_geometry_streaming(clip, cfg)
        else:
            logger.info(
                f"Using standard DA3 inference for {clip.name} ({num_frames} frames)"
            )
            extract_geometry_standard(clip, cfg, model)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = Path(cfg.preprocessed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config_dump in cfg.config_dumps or []:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    logger.info("Using DA3-Streaming for all clips (no standard model loading)")

    for clip in tqdm(cfg.clips, desc="Processing clips", unit="clip"):
        extract_geometry(clip, cfg, model=None)


if __name__ == "__main__":
    main()
