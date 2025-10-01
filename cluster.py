from typing import List
from sklearn.cluster import HDBSCAN
import torch
import argparse
import numpy as np
import torchvision
from pathlib import Path
import copy
import logging
import mmcv
import rerun as rr
import random

from eval.openclip_encoder import OpenCLIPNetwork
from scene.cameras import Camera
from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from cluster_utils import store_palette, clusters_to_rgb
from scene import GaussianModel, Scene
from gaussian_renderer import render as gs_render
from utils.sh_utils import RGB2SH
from autoencoder.model import Autoencoder
from autoencoder.model_qwen import QwenAutoencoder
from rerun_utils import (
    log_cluster_pointcloud_through_time,
    log_graph_structure_through_time,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Select which frame index to cluster and render
N_TIMESTEPS = 5  # change this to choose the frame index
TIMES = np.linspace(0 + 1e-6, 1 - 1e-6, N_TIMESTEPS)


def init_params():
    """Setup parameters similar to the train_eval.sh script"""
    parser = argparse.ArgumentParser()

    # these register parameters to the parser
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
    parser.add_argument("--clip_autoencoder_ckpt_path", type=str, default=None)
    parser.add_argument("--qwen_autoencoder_ckpt_path", type=str, default=None)
    # Additional model paths/stages for loading multiple GaussianModels
    parser.add_argument("--rgb_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--qwen_model_path", type=str, default=None)
    parser.add_argument("--rgb_load_stage", type=str, default=None)
    parser.add_argument("--clip_load_stage", type=str, default=None)
    parser.add_argument("--qwen_load_stage", type=str, default=None)

    # load config file if specified
    args = parser.parse_args()
    if args.configs:
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    return args, model_params, pipeline, hyperparam


def load_all_models(
    args: argparse.Namespace,
    model_params: ModelParams,
    pipeline: PipelineParams,
    hyperparam: ModelHiddenParams,
):
    """Load and return three GaussianModels and Scenes: clip, rgb, qwen.

    Returns:
        Tuple[GaussianModel, Scene, GaussianModel, Scene, GaussianModel, Scene]
        In order: (clip_gaussians, clip_scene, rgb_gaussians, rgb_scene, qwen_gaussians, qwen_scene)
    """
    # Extract base configs
    _ = model_params.extract(args)
    hyper = hyperparam.extract(args)

    # Primary (clip) model — preserve current behavior
    clip_model_path = args.clip_model_path or args.model_path
    clip_load_stage = args.clip_load_stage or args.load_stage

    args_clip = copy.copy(args)
    args_clip.model_path = clip_model_path
    dataset_clip = model_params.extract(args_clip)
    gaussians_clip = GaussianModel(dataset_clip.sh_degree, hyper)  # type:ignore
    scene_clip = Scene(
        dataset_clip,
        gaussians_clip,
        load_iteration=args.iteration,
        shuffle=False,
        load_stage=clip_load_stage,
    )
    logger.info(f"Loaded CLIP model from {clip_model_path} at stage {clip_load_stage}")

    # RGB model
    rgb_model_path = args.rgb_model_path or args.model_path
    rgb_load_stage = args.rgb_load_stage or "fine-base"

    try:
        args_rgb = copy.copy(args)
        args_rgb.model_path = rgb_model_path
        dataset_rgb = model_params.extract(args_rgb)
        gaussians_rgb = GaussianModel(dataset_rgb.sh_degree, hyper)  # type:ignore
        scene_rgb = Scene(
            dataset_rgb,
            gaussians_rgb,
            load_iteration=args.iteration,
            shuffle=False,
            load_stage=rgb_load_stage,
        )
        logger.info(f"Loaded RGB model from {rgb_model_path} at stage {rgb_load_stage}")
    except Exception as e:
        logger.warning(f"Failed to load RGB model: {e}")
        gaussians_rgb = None
        scene_rgb = None

    # Qwen model
    qwen_model_path = args.qwen_model_path or args.model_path
    qwen_load_stage = args.qwen_load_stage or "fine-lang"

    try:
        args_qwen = copy.copy(args)
        args_qwen.model_path = qwen_model_path
        dataset_qwen = model_params.extract(args_qwen)
        gaussians_qwen = GaussianModel(dataset_qwen.sh_degree, hyper)  # type:ignore
        scene_qwen = Scene(
            dataset_qwen,
            gaussians_qwen,
            load_iteration=args.iteration,
            shuffle=False,
            load_stage=qwen_load_stage,
        )
        logger.info(
            f"Loaded Qwen model from {qwen_model_path} at stage {qwen_load_stage}"
        )
    except Exception as e:
        logger.warning(f"Failed to load Qwen model: {e}")
        gaussians_qwen = None
        scene_qwen = None

    return (
        gaussians_clip,
        scene_clip,
        gaussians_rgb,
        scene_rgb,
        gaussians_qwen,
        scene_qwen,
    )


def filter_gaussians(gaussians: GaussianModel, mask: torch.Tensor):
    """Filter set of gaussians based on a mask.

    Args:
        gaussians (GaussianModel): The gaussian model to filter.
        mask (torch.Tensor): The mask to filter the gaussians. Shape (n_gaussians,)
    """
    for prop in dir(gaussians):
        attribute = getattr(gaussians, prop)
        a_type = type(attribute)
        if a_type == torch.Tensor or a_type == torch.nn.Parameter:
            if attribute.shape[0] == len(mask):
                setattr(gaussians, prop, attribute[mask])
                logger.info(f"Filtered {prop} with shape {attribute.shape}")


def normalize_indep_dim(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def normalize_dep_dim(x):
    return (x - x.mean()) / x.std()


def positions_at_timestep(gaussians: GaussianModel, timestep: float, scene: Scene):
    with torch.no_grad():
        means3D = gaussians.get_xyz
        # Short-circuit if no gaussians remain after filtering
        if means3D.shape[0] == 0:
            return means3D.detach().cpu().numpy()
        scales = gaussians._scaling
        rotations = gaussians._rotation
        opacity = gaussians._opacity
        shs = gaussians.get_features
        lang = gaussians.get_language_feature
        # Ensure time has the same dtype/device as model tensors
        time = torch.full(
            (means3D.shape[0], 1),
            float(timestep),
            device=means3D.device,
            dtype=means3D.dtype,
        )
        # Ensure language deformation is disabled for positional query
        try:
            gaussians._deformation.deformation_net.args.no_dlang = 1
        except Exception:
            pass
        means3D_final, _, _, _, _, _, _ = gaussians._deformation(
            means3D, scales, rotations, opacity, shs, lang, time
        )
    return means3D_final.detach().cpu().numpy()


def cluster_gaussians(gaussians: GaussianModel, timestep: float, scene: Scene):
    pos = normalize_indep_dim(positions_at_timestep(gaussians, timestep, scene))
    lf = gaussians.get_language_feature.detach().cpu().numpy()
    lf = normalize_dep_dim(lf)

    # graph = build_graph(pos, lf, k=10)
    # clusters = ng_jordan_weiss_spectral_clustering(graph, min_cluster_size=100, d_spectral=10)
    clusters = HDBSCAN(min_cluster_size=100, metric="euclidean").fit_predict(
        np.concatenate([pos, lf], axis=1)
    )

    return clusters


def filter_clusters(clusters, gaussians, scene):
    pos = normalize_indep_dim(positions_at_timestep(gaussians, 0.0, scene))
    lf = gaussians.get_language_feature.detach().cpu().numpy()
    lf = normalize_dep_dim(lf)

    i = 0
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        opacity = gaussians.get_opacity[cluster_mask].mean()
        std_pos = pos[cluster_mask].std()
        std_lang = lf[cluster_mask].std()

        logger.info(
            f"Cluster {cluster_id}\tn_points {cluster_mask.sum()}\topacity {opacity:.4f}\tstd_pos {std_pos:.4f}\tstd_lang {std_lang:.4f}"
        )

        if cluster_id >= 0:
            # filter clusters
            if std_lang < 0.1:
                clusters[cluster_mask] = -1
                logger.info("\tFiltered because std_lang < 0.1")
                continue
            if opacity < 0.4:
                clusters[cluster_mask] = -1
                logger.info("\tFiltered because opacity < 0.4")
                continue

            # restore contiguousness of cluster ids
            clusters[cluster_mask] = i
            i += 1


def set_cluster_colors(gaussians: GaussianModel, clusters: np.ndarray):
    colors = torch.zeros_like(gaussians._features_dc)  # outliers black
    cluster_colors, palette = clusters_to_rgb(clusters)
    sh_dc = RGB2SH(cluster_colors)  # (N,3)
    colors[:, 0, :] = torch.tensor(sh_dc, device=colors.device, dtype=colors.dtype)
    gaussians._features_dc.data = colors  # constant part becomes cluster color
    gaussians._features_rest.data = torch.zeros_like(
        gaussians._features_rest
    )  # higher order coefficients (handle view dependence) become 0

    return palette


def render(
    cam: Camera,
    timestep: float,
    gaussians: GaussianModel,
    pipe: PipelineParams,
    scene: Scene,
    args: argparse.Namespace,
    dataset: ModelParams,
):
    cam.time = timestep
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pkg = gs_render(
        cam,
        gaussians,
        pipe,
        background,
        None,
        stage=args.load_stage,
        cam_type=scene.dataset_type,
        args=args,
    )
    img = torch.clamp(pkg["render"], 0.0, 1.0)
    return img


def render_and_save_all(
    gaussians: GaussianModel,
    pipe: PipelineParams,
    scene: Scene,
    args: argparse.Namespace,
    dataset: ModelParams,
    out: Path,
):
    save_dir = out / "cluster_renders"
    save_dir.mkdir(parents=True, exist_ok=True)

    # pick random views
    test_cams = scene.getVideoCameras()  # test + train
    random_idx = random.sample(range(len(test_cams)), args.num_views)
    cams = [test_cams[i] for i in random_idx]

    # evenly spaced timesteps
    timesteps = np.linspace(0, 1, args.num_views, dtype=np.float32)

    # render and save
    for i, cam in enumerate(cams):
        cam_dir = save_dir / f"cam_{i:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for j, timestep in enumerate(timesteps):
            img = render(cam, timestep, gaussians, pipe, scene, args, dataset)
            torchvision.utils.save_image(img, cam_dir / f"timestep_{j:02d}.png")


def bhattacharyya_coefficient(mu1, Sigma1, mu2, Sigma2):
    mu1, mu2 = np.asarray(mu1), np.asarray(mu2)
    Sigma1, Sigma2 = np.asarray(Sigma1), np.asarray(Sigma2)

    # Average covariance
    Sigma = 0.5 * (Sigma1 + Sigma2)

    # Cholesky factorization for stability
    L = np.linalg.cholesky(Sigma)
    # Solve for (mu2 - mu1) without explicit inverse
    diff = mu2 - mu1
    sol = np.linalg.solve(L, diff)
    sol = np.linalg.solve(L.T, sol)
    term1 = 0.125 * np.dot(diff, sol)  # (1/8) Δμᵀ Σ⁻¹ Δμ

    # log-determinants via Cholesky
    logdet_Sigma = 2.0 * np.sum(np.log(np.diag(L)))
    logdet_Sigma1 = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma1))))
    logdet_Sigma2 = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma2))))
    term2 = 0.5 * (logdet_Sigma - 0.5 * (logdet_Sigma1 + logdet_Sigma2))

    DB = term1 + term2
    return np.exp(-DB)  # Bhattacharyya coefficient


def decode_clip(lfs: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
    BATCH_SIZE = 1024

    ae = Autoencoder(
        encoder_hidden_dims=[256, 128, 64, 32, 3],
        decoder_hidden_dims=[16, 32, 64, 128, 256, 512],
        feature_dim=512,
    ).to("cuda")
    ae.load_state_dict(torch.load(args.clip_autoencoder_ckpt_path, map_location="cuda"))
    ae.eval()

    decoded_lfs = []
    with torch.no_grad():
        for i in range(0, lfs.shape[0], BATCH_SIZE):
            batch = lfs[i : min(i + BATCH_SIZE, len(lfs))].to("cuda")
            decoded_lfs.append(ae.decode(batch))
    decoded_lfs = [i.detach().cpu().numpy() for i in decoded_lfs]
    decoded_lfs = np.concatenate(decoded_lfs, axis=0)
    decoded_lfs = decoded_lfs / np.linalg.norm(decoded_lfs, axis=-1, keepdims=True)
    return decoded_lfs


def decode_qwen(lfs: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
    BATCH_SIZE = 1024

    ae = QwenAutoencoder(
        input_dim=3584,
        latent_dim=3,
    ).to("cuda")
    ae.load_state_dict(torch.load(args.qwen_autoencoder_ckpt_path, map_location="cuda"))
    ae.eval()

    decoded_lfs = []
    with torch.no_grad():
        for i in range(0, lfs.shape[0], BATCH_SIZE):
            batch = lfs[i : min(i + BATCH_SIZE, len(lfs))].to("cuda")
            decoded_lfs.append(ae.decode(batch))
    decoded_lfs = [i.detach().cpu().numpy() for i in decoded_lfs]
    decoded_lfs = np.concatenate(decoded_lfs, axis=0)
    return decoded_lfs


def cluster_clip_features(
    gaussians: GaussianModel, clusters: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    # get average language feature weighted by opacity
    weighted_cluster_lfs = []
    n_nodes = len(np.unique(clusters))
    opacities = gaussians.get_opacity.detach().cpu().numpy()
    decoded_lfs = decode_clip(
        gaussians.get_language_feature, args
    )  # decode before aggregation, works slightly better but not much
    for cluster_id in range(n_nodes):
        cluster_mask = clusters == cluster_id
        cluster_opacities = opacities[cluster_mask]
        cluster_lfs = decoded_lfs[cluster_mask]
        cluster_lf = (cluster_lfs * cluster_opacities).sum(
            axis=0
        ) / cluster_opacities.sum()
        cluster_lf = cluster_lf / np.linalg.norm(cluster_lf)
        weighted_cluster_lfs.append(cluster_lf)

    lfs_weighted_centroids = np.stack(weighted_cluster_lfs)

    return lfs_weighted_centroids


def cluster_qwen_features(
    qwen_g: GaussianModel,
    rgb_g: GaussianModel,
    clusters: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """returns list of top cluster lfs (dynamic size)"""
    n_nodes = len(np.unique(clusters))
    opacities = rgb_g.get_opacity.detach().cpu().numpy().squeeze()
    lfs = qwen_g.get_language_feature.detach().cpu().numpy()
    top_cluster_lfs = {}
    full_cluster_lfs = {}
    for cluster_id in range(n_nodes):
        cluster_mask = clusters == cluster_id
        cluster_opacities = opacities[cluster_mask]
        cluster_lfs = lfs[cluster_mask]
        assert cluster_opacities.ndim == 1
        top_indices = np.asarray(torch.topk(torch.as_tensor(cluster_opacities), min(cluster_opacities.size, 100)).indices)
        full_cluster_lfs[cluster_id] = cluster_lfs[top_indices]
    #     lf_cluster_cluster = HDBSCAN(min_cluster_size=10).fit_predict(cluster_lfs)
    #     tmp, unique_indices = np.unique(lf_cluster_cluster, return_index=True)
    #     unique_indices = unique_indices[1:]
    #     print("tmp", tmp)
    #     top_cluster_lf = cluster_lfs[unique_indices]
    #     top_cluster_lfs[cluster_id] = top_cluster_lf
    #     full_cluster_lfs[cluster_id] = cluster_lfs
    # top_cluster_lfs = {k: decode_qwen(torch.tensor(v, dtype=torch.float32), args) for k, v in top_cluster_lfs.items()}
    full_cluster_lfs = {
        k: decode_qwen(torch.tensor(v, dtype=torch.float32), args)
        for k, v in full_cluster_lfs.items()
    }
    return top_cluster_lfs, full_cluster_lfs


def properties_through_time(positions_through_time, clusters):
    """Compute spatial cluster properties through time.

    Args:
        positions_through_time (np.ndarray): Positions through time. (T, N, 3)
        clusters (np.ndarray): Cluster ids through time. (T, N)

    Returns:
        np.ndarray: Centroid through time. (T, C, 3)
        np.ndarray: Center through time. (T, C, 3)
        np.ndarray: Extent through time. (T, C, 3)
    """
    cluster_ids = np.unique(clusters)

    centroid = np.empty((len(positions_through_time), len(cluster_ids), 3))
    center = np.empty((len(positions_through_time), len(cluster_ids), 3))
    extent = np.empty((len(positions_through_time), len(cluster_ids), 3))
    for t in range(len(positions_through_time)):
        for i in range(len(cluster_ids)):
            pos = positions_through_time[t][clusters == cluster_ids[i]]
            centroid[t, i] = pos.mean(0)
            center[t, i] = (pos.max(0) + pos.min(0)) / 2
            extent[t, i] = pos.max(0) - pos.min(0)

    return centroid, center, extent


def timestep_graph(positions, clusters):
    n_nodes = len(np.unique(clusters))
    means = np.stack([positions[clusters == i].mean(0) for i in range(n_nodes)])
    covs = np.stack([np.cov(positions[clusters == i].T) for i in range(n_nodes)])

    distances = np.empty((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distances[i, j] = bhattacharyya_coefficient(
                means[i], covs[i], means[j], covs[j]
            )

    A = np.where(distances >= 0.05, distances, 0)
    return A


def lerf_relevancies(lfs: np.ndarray, queries: List[str], canonical_corpus: List[str]):
    """Compute LERF relevance scores for a set of language features.

    Args:
        lfs (np.ndarray): Language features to compute relevance scores for. (n_lfs, dim)

    Returns:
        np.ndarray: Relevance scores for the language features. (n_queries, n_lfs)
    """
    ocn = OpenCLIPNetwork(device="cuda", canonical_corpus=canonical_corpus)
    ocn.set_positives(queries)

    lfs = torch.tensor(lfs).to("cuda")

    lerf_relevancies = []
    for i in range(len(queries)):
        r = ocn.get_relevancy(lfs, i)
        r = r[:, 0].detach().cpu().numpy()
        lerf_relevancies.append(r)
    lerf_relevancies = np.stack(lerf_relevancies)  # (n_queries, n_lfs)

    return lerf_relevancies


def main():
    # determistic seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # mock render.py config
    args, model_params, pipeline, hyperparam = init_params()

    # construct all objects
    dataset = model_params.extract(args)
    pipe = pipeline.extract(args)
    _ = hyperparam.extract(args)

    # Use refactored loader to get all models; keep using clip for current behavior
    clip_g, clip_scene, rgb_g, rgb_scene, qwen_g, qwen_scene = load_all_models(
        args, model_params, pipeline, hyperparam
    )

    # Normalize qwen features since AE decoder expects unit norm
    qwen_g._language_feature = qwen_g.get_language_feature / qwen_g.get_language_feature.norm(dim=-1, keepdim=True)

    # gaussian filtering
    mask = (rgb_g.get_opacity > 0.1).squeeze()
    filter_gaussians(rgb_g, mask)
    filter_gaussians(qwen_g, mask)
    filter_gaussians(clip_g, mask)

    # cluster, filter clusters, filter gaussians that are not in a cluster
    clusters = cluster_gaussians(clip_g, timestep=0.0, scene=clip_scene)
    filter_clusters(clusters, clip_g, clip_scene)
    cluster_mask = clusters >= 0
    filter_gaussians(rgb_g, cluster_mask)
    filter_gaussians(qwen_g, cluster_mask)
    filter_gaussians(clip_g, cluster_mask)
    clusters = clusters[cluster_mask]
    palette = set_cluster_colors(clip_g, clusters)

    # cluster features
    timesteps = np.linspace(0, 1, 20)
    # clip_features = cluster_clip_features(gaussians, clusters, args)
    qwen_features, full_qwen_features = cluster_qwen_features(
        qwen_g, rgb_g, clusters, args
    )
    pos_through_time = np.stack(
        [positions_at_timestep(rgb_g, t, rgb_scene) for t in timesteps]
    )
    (
        cluster_pos_through_time,
        cluster_center_through_time,
        cluster_extent_through_time,
    ) = properties_through_time(pos_through_time, clusters)

    # graph
    graphs = np.stack(
        [timestep_graph(pos_through_time[i], clusters) for i in range(len(timesteps))]
    )

    # query correspondences
    # gaussian_lfs = decode_lfs(gaussians.get_language_feature, args)
    # # queries = ["hand", "egg"]
    # # canonical_corpus = ["object", "things", "stuff", "texture"]
    # queries = ["gallbladder", "liver"]
    # canonical_corpus = ["object", "things", "stuff", "texture", "surgery", "body", "anatomy", "medical"]
    # gaussian_scores = lerf_relevancies(gaussian_lfs, queries, canonical_corpus)
    # cluster_scores = lerf_relevancies(clip_features, queries, canonical_corpus)

    # render and save everything
    out = Path(args.rgb_model_path) / "graph"
    out.mkdir(parents=True, exist_ok=True)
    render_and_save_all(clip_g, pipe, clip_scene, args, dataset, out)
    store_palette(palette, out / "cluster_palette.png")
    for k, v in qwen_features.items():
        np.save(out / f"cluster_qwen_features_{k}.npy", v)
    for k, v in full_qwen_features.items():
        np.save(out / f"cluster_qwen_features_full_{k}.npy", v)
    np.save(out / "clusters.npy", clusters)
    np.save(out / 'latent_qwen_filtered.npy', qwen_g.get_language_feature.detach().cpu().numpy())
    # gaussians.save_ply(out / "clustered_gaussians.ply")
    # np.save(out / "cluster_clip_features.npy", clip_features)
    # np.save(out / "cluster_ids.npy", clusters)
    # np.save(out / "cluster_centroids_per_timestep.npy", cluster_pos_through_time)
    # np.save(out / "cluster_centers_per_timestep.npy", cluster_center_through_time)
    # np.save(out / "cluster_extents_per_timestep.npy", cluster_extent_through_time)
    # np.save(out / "adjacency_matrices_per_timestep.npy", graphs)

    # visualize to rerun
    rr.init("clusters")
    # rr.log("/", rr.ViewCoordinates.RDF)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")  # log to web viewer if running
    rr.save(out / "graph_visualization.rrd")  # save to file for offline viewing

    log_cluster_pointcloud_through_time(
        gaussians_rgb=clip_g,
        gaussians_qwen=qwen_g,
        clusters=clusters,
        timesteps=timesteps,
        pos_through_time=pos_through_time,
        cluster_pos_through_time=cluster_pos_through_time,
        # text_queries=queries,
        # cluster_correspondences=cluster_scores,
        text_queries=None,
        cluster_correspondences=None,
    )
    log_graph_structure_through_time(
        cluster_pos_through_time=cluster_pos_through_time,
        graphs_through_time=graphs,
    )
    # log_correspondences_static(
    #     positions=gaussians.get_xyz.detach().cpu().numpy(),
    #     clusters=clusters,
    #     text_queries=queries,
    #     correspondences=gaussian_scores,
    #     corr_min=0.0,
    #     corr_max=1.0,
    # )


if __name__ == "__main__":
    main()
