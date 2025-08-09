from sklearn.cluster import HDBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import argparse
import os
import numpy as np
import torchvision
from pathlib import Path

from arguments import ModelParams, PipelineParams, ModelHiddenParams
from cluster import clusters_to_rgb
from scene import GaussianModel, Scene
from gaussian_renderer import render as gs_render
from utils.sh_utils import RGB2SH
import random


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

os.environ["use_discrete_lang_f"] = "f"

out = Path("cluster_outputs")
out.mkdir(parents=True, exist_ok=True)

# mock argparse from render.py
arglist = ["render.py"]
arglist.extend(["-s", "data/hypernerf/chickchicken"])
arglist.extend(["--language_features_name", "clip_features-language_features_dim3"])
arglist.extend(
    ["--model_path", "output/robert_nico/hypernerf/chickchicken/chickchicken_3"]
)
arglist.extend(["--feature_level", "3"])
arglist.extend(["--skip_train"])
arglist.extend(["--skip_test"])
arglist.extend(["--quiet"])
arglist.extend(["--skip_video"])
arglist.extend(["--configs", "arguments/hypernerf/chicken.py"])
arglist.extend(["--mode", "lang"])
arglist.extend(["--no_dlang", "1"])
arglist.extend(["--load_stage", "fine-lang"])
parser = argparse.ArgumentParser(arglist)  # type:ignore
model_params = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
hyperparam = ModelHiddenParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--skip_video", action="store_true")
# parser.add_argument("--skip_new_view", action="store_true")
parser.add_argument("--configs", type=str)
parser.add_argument("--mode", choices=["rgb", "lang"], default="rgb")
parser.add_argument("--novideo", type=int, default=0)
parser.add_argument("--noimage", type=int, default=0)
parser.add_argument("--nonpy", type=int, default=0)
parser.add_argument("--load_stage", type=str, default="fine-lang")
parser.add_argument("--num_views", type=int, default=5)
args = parser.parse_args(arglist[1:])
if args.configs:
    import mmcv
    from utils.params_utils import merge_hparams

    config = mmcv.Config.fromfile(args.configs)
    args = merge_hparams(args, config)

# Build dataset/scene
dataset = model_params.extract(args)
pipe = pipeline.extract(args)
hyper = hyperparam.extract(args)
gaussians = GaussianModel(dataset.sh_degree, hyper)  # type:ignore
scene = Scene(
    dataset,
    gaussians,
    load_iteration=args.iteration,
    shuffle=False,
    load_stage=args.load_stage,
)  # type:ignore

# filter gaussians
mask = (gaussians.get_opacity > 0.1).squeeze()
for prop in dir(gaussians):
    attribute = getattr(gaussians, prop)
    a_type = type(attribute)
    if a_type == torch.Tensor or a_type == torch.nn.Parameter:
        if attribute.shape[0] == len(mask):
            setattr(gaussians, prop, attribute[mask])
            print(f"Applied filter to property {prop}")

# cluster
pos = gaussians.get_xyz.detach().cpu().numpy()
lf = gaussians.get_language_feature.detach().cpu().numpy()
pos = (pos - pos.mean()) / pos.std()
lf = (lf - lf.mean(axis=0)) / lf.std(axis=0)
# graph = build_graph(pos, lf, k=10)
# labels = ng_jordan_weiss_spectral_clustering(graph, min_cluster_size=100, d_spectral=10)
clusters = HDBSCAN(min_cluster_size=100, metric="euclidean").fit_predict(
    np.concatenate([pos, lf], axis=1)
)  # type:ignore

# cluster stats and filtering
i = 0
for cluster_id in np.unique(clusters):
    cluster_mask = clusters == cluster_id
    opacity = gaussians.get_opacity[cluster_mask].mean()
    std_pos = pos[cluster_mask].std()
    std_lang = lf[cluster_mask].std()

    print(f"Cluster {cluster_id}\tn_points {cluster_mask.sum()}\topacity {opacity:.4f}\tstd_pos {std_pos:.4f}\tstd_lang {std_lang:.4f}")

    if cluster_id >= 0:
        # filter clusters
        if std_lang > 1.2:
            clusters[cluster_mask] = -1
            print(f"Filtered because std_lang > 1.0")
            continue
        if opacity < 0.4:
            clusters[cluster_mask] = -1
            print(f"Filtered because opacity < 0.5")
            continue

        # remap to 0-based
        clusters[cluster_mask] = i
        i += 1

# override spherical harmonics with cluster colors
colors = np.zeros((len(pos), 3))  # outliers black
all_clusters_mask = clusters >= 0
cluster_colors, palette = clusters_to_rgb(clusters[all_clusters_mask])
sh_dc = RGB2SH(cluster_colors)  # (N,3)
colors[all_clusters_mask] = cluster_colors
colors = torch.tensor(colors, dtype=torch.float32, device="cuda")
gaussians._features_dc.data = colors.unsqueeze(
    1
)  # (N,1,3) constant part becomes cluster color
gaussians._features_rest.data = torch.zeros_like(
    gaussians._features_rest
)  # higher order coefficients (handle view dependence) become 0

# def save_with_legend(img, legend_texts, legend_colors, path):
#     fig, ax = plt.subplots()
#     im = ax.imshow(img)
#     patches = []
#     for text, col in zip(legend_texts, legend_colors):
#         patches.append(mpatches.Patch(color=col, label=text))
#     ax.legend(handles=patches, loc='upper right')
#     ax.axis('off')
#     plt.savefig(path, bbox_inches='tight', pad_inches=0)

# store palette and print leftover cluster info
print("=======Final clusters=======")
for cluster_id in np.unique(clusters):
    cluster_mask = clusters == cluster_id
    print(f"Cluster {cluster_id}\tn_points {cluster_mask.sum()}\topacity {gaussians.get_opacity[cluster_mask].mean():.4f}\tstd_pos {pos[cluster_mask].std():.4f}\tstd_lang {lf[cluster_mask].std():.4f}")
sns_palette = sns.palettes._ColorPalette(palette)
fig, ax = plt.subplots(figsize=(8, 2))
for i, color in enumerate(sns_palette):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
ax.set_xlim(0, len(palette))
ax.set_ylim(0, 1)
ax.axis("off")  # Hide axes
plt.savefig(str(out / "cluster_palette.png"), bbox_inches="tight", dpi=300)


# render some views
random_idx = random.sample(range(len(scene.getTestCameras())), args.num_views)
test_cams = scene.getTestCameras()
views = [test_cams[i] for i in random_idx]
bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
for idx, view in enumerate(views):
    pkg = gs_render(
        view,
        gaussians,
        pipe,
        background,
        None,
        stage=args.load_stage,
        cam_type=scene.dataset_type,
        args=args,
    )
    img = torch.clamp(pkg["render"], 0.0, 1.0)
    torchvision.utils.save_image(img, out / f"cluster_{idx:03d}.png")
    # save_with_legend(img.detach().cpu().permute(1, 2, 0).numpy(), [f"Cluster {cluster_id}" for cluster_id in cluster_ids[1:]], cluster_colors, out / f"cluster_{idx:03d}_legend.png")
print(f"Saved {len(views)} cluster-colored views to: {out}")

# save gaussians
gaussians.save_ply(out / "cluster_cols_with_positions.ply")
