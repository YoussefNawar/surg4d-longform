from pathlib import Path
from PIL import Image
import numpy as np

# Edit this path to your directory
directory = Path("/home/data/long_form_surgery_Cholec80/5fps_samples/video01/video01_00000")

for endo_file in directory.glob("frame_*_endo.png"):
    img = Image.open(endo_file)
    black_mask = Image.fromarray(np.zeros((img.height, img.width, 3), dtype=np.uint8))
    
    stem = endo_file.stem  # e.g., "frame_000000_endo"
    black_mask.save(endo_file.parent / f"{stem}_watershed_mask.png")
    black_mask.save(endo_file.parent / f"{stem}_color_mask.png")
    print(f"Created masks for {endo_file.name}")