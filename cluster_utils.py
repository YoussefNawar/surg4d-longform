"""
Spectral clustering implementations.
Naming follows:
https://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf
"""

import numpy as np
import seaborn as sns
from scipy.linalg import eig, eigh
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt


def laplacian(A):
    # assert np.allclose(A.T, A), "A is not symmetric"
    if sp.isspmatrix(A):
        D_diag = np.asarray(A.sum(axis=0)).ravel()
        D = sp.diags(D_diag, format="csr")
        L = D - A
    else:
        D_diag = np.sum(A, axis=0)
        D = np.diag(D_diag)
        L = D - A
    # assert np.allclose(L, L.T), "L is not symmetric"
    return L


def laplacian_rw(A):
    # assert np.allclose(A.T, A), "A is not symmetric"
    if sp.isspmatrix(A):
        D_diag = np.asarray(A.sum(axis=0)).ravel()
        with np.errstate(divide="ignore"):
            D_inv_diag = np.zeros_like(D_diag)
            nonzero = D_diag != 0
            D_inv_diag[nonzero] = 1.0 / D_diag[nonzero]
        D_inv = sp.diags(D_inv_diag, format="csr")
        I = sp.eye(A.shape[0], format="csr")
        L_rw = I - D_inv @ A
    else:
        D_diag = np.sum(A, axis=0)
        with np.errstate(divide="ignore"):
            D_inv_diag = np.zeros_like(D_diag)
            nonzero = D_diag != 0
            D_inv_diag[nonzero] = 1.0 / D_diag[nonzero]
        D_inv = np.diag(D_inv_diag)
        L_rw = np.eye(A.shape[0]) - D_inv @ A
    # not supposed to be symmetric
    return L_rw


def laplacian_sym(A):
    # assert np.allclose(A.T, A), "A is not symmetric"
    if sp.isspmatrix(A):
        D_diag = np.asarray(A.sum(axis=0)).ravel()
        D_pow_neg_half_diag = np.zeros_like(D_diag)
        nonzero = D_diag != 0
        D_pow_neg_half_diag[nonzero] = D_diag[nonzero] ** -0.5
        D_pow_neg_half = sp.diags(D_pow_neg_half_diag, format="csr")
        normalized_A = D_pow_neg_half @ A @ D_pow_neg_half
        normalized_L = sp.eye(A.shape[0], format="csr") - normalized_A
    else:
        D_diag = np.sum(A, axis=0)
        with np.errstate(divide="ignore"):
            D_pow_neg_half_diag = np.zeros_like(D_diag)
            nonzero = D_diag != 0
            D_pow_neg_half_diag[nonzero] = D_diag[nonzero] ** -0.5
        D_pow_neg_half = np.diag(D_pow_neg_half_diag)
        normalized_A = D_pow_neg_half @ A @ D_pow_neg_half
        normalized_L = np.eye(A.shape[0]) - normalized_A
    # assert np.allclose(normalized_L, normalized_L.T), "L is not symmetric"
    return normalized_L


def spectral_embeddings(L, d, normalize_rows: bool, use_symmetric_eigensolver: bool):
    """Spectral embeddings.

    Args:
        L: Laplacian matrix (n x n)
        d: dimension of embeddings (number of eigenvectors to keep), if set to None, cutoff at largest eigengap
        normalize_rows: Normalize rows of embedding
        use_symmetric_eigensolver: Use symmetric eigensolver

    Returns:
        _type_: _description_
    """
    if sp.isspmatrix(L):
        if use_symmetric_eigensolver:
            eigenvalues, U = eigsh(L, k=d, which="SA")
        else:
            eigenvalues, U = eigs(L, k=d, which="SR")
    else:
        if use_symmetric_eigensolver:
            eigenvalues, U = eigh(L, subset_by_index=[0, d - 1])
        else:
            eigenvalues, eigenvectors = eig(L)
            U = eigenvectors[:, :d]
        assert np.allclose(eigenvalues[0], 0), (
            f"smallest eigenvalue is not 0 but {np.array(eigenvalues[0]).round(6)} with eigenvector {np.array(eigenvectors[0]).round(6)}"
        )
        assert len(eigenvalues) < 2 or not np.allclose(eigenvalues[1], 0), (
            "second smallest eigenvalue is 0 -> graph is not connected"
        )

    # normalize rows if required
    if normalize_rows:
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        T = U / norms
    else:
        T = U

    return T


def unnormalized_spectral_clustering(A, min_cluster_size=1000, seed=42):
    """Unnormalized spectral clustering. Approximates ratio cut.

    Args:
        A: Adjacency matrix (n x n)
        min_cluster_size: Minimum cluster size
        seed: Random seed

    Returns:
        Clusters (n,)
    """
    L = laplacian(A)
    T = spectral_embeddings(
        L, d=None, normalize_rows=False, use_symmetric_eigensolver=True
    )
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", seed=seed)
    clusters = hdbscan.fit_predict(T)
    return clusters


def shi_malik_spectral_clustering(A, min_cluster_size=1000, seed=42):
    """Spectral clustering with random walk Laplacian.
    Approximates normalized cut but uses non-hermitian laplacian and eigensolver,
    so might be slower than other approaches.

    Args:
        A: Adjacency matrix (n x n)
        min_cluster_size: Minimum cluster size
        seed: Random seed

    Returns:
        Clusters (n,)
    """
    L = laplacian_rw(A)
    T = spectral_embeddings(
        L, d=None, normalize_rows=False, use_symmetric_eigensolver=False
    )
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", seed=seed)
    clusters = hdbscan.fit_predict(T)
    return clusters


def ng_jordan_weiss_spectral_clustering(A, spectral_embedding_dim: int, hdbscan_args: dict):
    """Spectral clustering with symmetric, normalized Laplacian.
    Heuristically approximates normalized cut (not principled like Shi-Malik),
    but is faster because it uses a symmetric eigensolver.
    
    Handles disconnected graphs by clustering each connected component separately.

    Args:
        A: Adjacency matrix (n x n)
        spectral_embedding_dim: Number of eigenvectors for embedding
        hdbscan_args: Arguments for HDBSCAN clustering

    Returns:
        Clusters (n,)
    """
    n_samples = A.shape[0]
    n_components, component_labels = connected_components(A, directed=False)
    

    # plain spectral clustering if graph is connected
    if n_components == 1:
        L = laplacian_sym(A)
        T = spectral_embeddings(
            L, d=spectral_embedding_dim, normalize_rows=True, use_symmetric_eigensolver=True
        )
        clusters = HDBSCAN(**hdbscan_args).fit_predict(T)
        return clusters
    
    # cluster separately if disconnected graph
    clusters = np.full(n_samples, -1, dtype=np.int32)
    cluster_offset = 0
    for comp_id in range(n_components):
        comp_mask = component_labels == comp_id
        comp_size = comp_mask.sum()
        
        # throw away component if smaller than min cluster size
        if comp_size < hdbscan_args["min_cluster_size"] or comp_size <= spectral_embedding_dim + 1:
            continue
        
        # subgraph
        comp_indices = np.where(comp_mask)[0]
        A_sub = A[comp_indices][:, comp_indices]
        
        # cluster
        L_sub = laplacian_sym(A_sub)
        d = min(spectral_embedding_dim, comp_size - 1)
        T_sub = spectral_embeddings(
            L_sub, d=d, normalize_rows=True, use_symmetric_eigensolver=True
        )
        comp_clusters = HDBSCAN(**hdbscan_args).fit_predict(T_sub)
        
        # Map back to original indices, offsetting cluster IDs
        valid_mask = comp_clusters >= 0
        clusters[comp_indices[valid_mask]] = comp_clusters[valid_mask] + cluster_offset
        cluster_offset += comp_clusters.max() + 1 if valid_mask.any() else 0
    
    return clusters


def clusters_to_rgb(clusters):
    """Convert cluster ids to RGB colors from HUSL palette.

    Args:
        clusters: Cluster ids (does NOT have to be unique) (n,)

    Returns:
        RGB colors (n,3) in range(0,1), palette (K,3) in range(0,1)
    """
    unique = np.unique(clusters)
    assert np.all(unique == np.arange(len(unique))), "Cluster ids must be contiguous"
    # husl = np.array(sns.color_palette("husl", len(unique)))
    # n_steps, min_brightness = 6, 0.4
    # brightness = (
    #     min_brightness
    #     + (1 - min_brightness)
    #     * np.mod(np.linspace(0, 1, husl.shape[0]), 1 / n_steps)
    #     * n_steps
    # )
    # husl_adjusted = husl * brightness[:, np.newaxis]
    # return husl_adjusted[clusters]
    pal = np.random.rand(len(unique), 3)
    return pal[clusters], pal


def build_graph(
    positions: np.ndarray,
    normalized_lf: np.ndarray,
    k: int,
    weight_pos: float,
    weight_lf: float,
):
    # knn position graph (L2)
    knn_position = kneighbors_graph(
        positions, n_neighbors=k, mode="distance"
    )  # scipy sparse csr (n, n)
    knn_position /= knn_position.max()
    knn_position.data = 1 - knn_position.data # we need similarity graph not distance graph

    # knn language graph (cos dist)
    n_samples = knn_position.shape[0]
    indptr = knn_position.indptr
    indices = knn_position.indices
    data_lf = np.empty(len(indices), dtype=normalized_lf.dtype)
    for i in range(n_samples):
        start, end = indptr[i], indptr[i + 1]
        neighbors = indices[start:end]
        if neighbors.size > 0:
            cos_sim = normalized_lf[i] @ normalized_lf[neighbors].T
            data_lf[start:end] = 1 - cos_sim
    knn_language = sp.csr_matrix(
        (data_lf, indices, indptr), shape=(n_samples, n_samples)
    )
    knn_language /= knn_language.max()
    knn_language.data = 1 - knn_language.data # we need similarity graph not distance graph

    graph = weight_pos * knn_position + weight_lf * knn_language
    graph = graph.minimum(graph.T) # needs to be symmetric
    return graph


def store_palette(palette, path):
    """Store color palette as a PNG image."""
    sns_palette = sns.palettes._ColorPalette(palette)
    fig, ax = plt.subplots(figsize=(8, 2))
    for i, color in enumerate(sns_palette):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, len(palette))
    ax.set_ylim(0, 1)
    ax.axis("off")  # Hide axes
    plt.savefig(path, bbox_inches="tight", dpi=300)
