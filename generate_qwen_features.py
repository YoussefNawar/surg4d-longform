from pathlib import Path
from PIL import Image
import numpy as np

from qwen_vl import qwen_encode_image, get_patch_segmasks

scene_root = Path("data/hypernerf/chickchicken_qwen")
frame_files = (scene_root / "rgb/2x").glob("*.png")

lf_out_dir = scene_root / "qwen_features"
lf_out_dir.mkdir(parents=True, exist_ok=True)

for frame_file in frame_files:
    name = frame_file.name.split(".")[0]
    feature_file_name = lf_out_dir / f"{name}_f.npy"
    seg_map_file_name = lf_out_dir / f"{name}_s.npy"

    image = Image.open(frame_file)
    features = qwen_encode_image(image).detach().cpu().float().numpy()
    seg_map = get_patch_segmasks(image.height, image.width).unsqueeze(0)
    np.save(feature_file_name, features)
    np.save(seg_map_file_name, seg_map)