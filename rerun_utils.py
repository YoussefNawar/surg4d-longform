import numpy as np
import rerun as rr
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def log_cluster_pointcloud_through_time(
    gaussians_rgb,
    gaussians_qwen,
    clusters: np.ndarray,
    timesteps: np.ndarray,
    pos_through_time: np.ndarray,
    cluster_pos_through_time: np.ndarray,
    text_queries: list[str],
    cluster_correspondences: np.ndarray,
):
    """Log cluster pointclouds (points + cluster means) over time to Rerun.

    Args:
        gaussians: Gaussian model containing color features used for point coloring.
        clusters: Array of cluster ids per gaussian (length = num_gaussians).
        timesteps: Array of timesteps corresponding to positions.
        pos_through_time: Positions per timestep; shape (T, N, 3).
        cluster_pos_through_time: Cluster mean positions per timestep; shape (T, C, 3).
        text_queries: List of text queries used for cluster correspondences.
        cluster_correspondences: Cluster correspondences of shape (C, n_queries).
    """
    cluster_ids = np.unique(clusters)

    cols = gaussians_rgb._features_dc.detach().cpu().numpy() * 255
    lfs_qwen = np.clip(gaussians_qwen.get_language_feature.detach().cpu().numpy() * 10, a_min=0.0, a_max=1.0) * 255

    for i in range(len(timesteps)):
        rr.set_time("timestep", sequence=i)
        pos = pos_through_time[i]
        cluster_means = cluster_pos_through_time[i]

        # Log individual cluster points
        for c in cluster_ids:
            pc = rr.Points3D(
                positions=pos[clusters == c],
                colors=cols[clusters == c],
                radii=0.02,
            )
            rr.log(f"clusters/points/cluster/{c}", pc)
            qwen_pc = rr.Points3D(
                positions=pos[clusters == c],
                colors=lfs_qwen[clusters == c],
                radii=0.02,
            )
            rr.log(f"clusters/points/qwen/{c}", qwen_pc)

        # Log cluster means
        mean_colors = np.stack([cols[clusters == c][0] for c in cluster_ids])
        mean_labels = None
        if text_queries is not None and cluster_correspondences is not None:
            mean_labels = []
            for c in cluster_ids:
                mean_labels.append("\n".join([f"{text_queries[i]}\t\t{cluster_correspondences[i, c]:.2f}" for i in range(len(text_queries))]))
        means_viz = rr.Points3D(
            positions=cluster_means,
            colors=mean_colors,
            radii=0.2,
            labels=mean_labels,
            show_labels=False,
        )
        rr.log("clusters/means", means_viz)


def log_graph_structure_through_time(
    cluster_pos_through_time: np.ndarray,
    graphs_through_time: np.ndarray,
):
    """Log graph edges to rerun.
    Edge thickness is proportional to the weight.

    Args:
        cluster_pos_through_time: Cluster mean positions per timestep; shape (T, C, 3).
        graphs_through_time: Adjacency matrices per timestep; shape (T, C, C).
    """
    num_timesteps = len(graphs_through_time)
    for i in range(num_timesteps):
        rr.set_time("timestep", sequence=i)
        cluster_means = cluster_pos_through_time[i]
        A = graphs_through_time[i]

        if A.shape[0] == 0:
            continue

        # Clear previously logged edges for this timestep so stale edges don't persist
        rr.log("clusters/edges", rr.Clear(recursive=True))

        edge_indices = np.where(A > 0)
        if len(edge_indices[0]) == 0:
            continue

        edge_weights = A[edge_indices]

        # Normalize weights for visualization
        if len(edge_weights) > 1:
            min_weight = edge_weights.min()
            max_weight = edge_weights.max()
            if max_weight > min_weight:
                normalized_weights = (edge_weights - min_weight) / (max_weight - min_weight)
            else:
                normalized_weights = np.ones_like(edge_weights)
        else:
            normalized_weights = np.ones_like(edge_weights)

        # Create and log edges as line strips
        for idx, (u, v) in enumerate(zip(edge_indices[0], edge_indices[1])):
            if u < v:  # Avoid duplicate edges for symmetric adjacency
                start_pos = cluster_means[u]
                end_pos = cluster_means[v]
                weight = normalized_weights[idx]

                color = [0, 0, 0]
                thickness = 0.05 + 0.15 * weight

                edge_line = rr.LineStrips3D(
                    strips=[[start_pos, end_pos]],
                    colors=[color],
                    radii=[thickness],
                )
                rr.log(f"clusters/edges/edge_{idx}", edge_line)

def log_correspondences_static(
    positions,
    clusters,
    text_queries,
    correspondences,
    corr_min,
    corr_max,
):
    """Log static correspondence heatmaps.

    Args:
        positions: Positions of shape (N, 3).
        text_queries: List of text queries used for cluster correspondences.
        clusters: Array of cluster ids per gaussian (length = num_gaussians).
        correspondences: Correspondences of shape (n_texts, N).
        corr_min: Minimum value for the color map.
        corr_max: Maximum value for the color map.
    """
    mask = clusters >= 0
    positions = positions[mask]
    correspondences = correspondences[:, mask]

    norm = mcolors.Normalize(vmin=corr_min, vmax=corr_max, clip=True)
    cmap = cm.get_cmap('seismic')

    for i, query in enumerate(text_queries):
        corr = correspondences[i]
        rgba = cmap(norm(corr))
        rgb = (rgba[:, :3] * 255.0).astype(np.uint8)
        points = rr.Points3D(
            positions=positions,
            colors=rgb,
            labels=[str(i) for i in corr],
            show_labels=False,
        )
        rr.log(f"correspondences/{query}", points)