import numpy as np
import os

dirr = '/home/data/tumai/splatgraph/data/preprocessed/clean/test_output/video01_00000/semantic_masks_sasvi'
file_names = sorted([f for f in os.listdir(dirr)])
print(f"Reading Masks with {len(file_names)} frames from {dirr}")
# Load all masks
for f in file_names:
    x = np.load(os.path.join(dirr, f))
    print(x.shape)
    print(np.unique(x))
    break
    # print()

# all_masks = [dict(np.load(os.path.join(dirr, f), allow_pickle=True)) for f in file_names]
