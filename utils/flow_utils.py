"""
Utilities for optical flow computation and Gaussian flow loss.
"""
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.utils import flow_to_image
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple


def compute_raft_flow(
    image1: torch.Tensor,
    image2: torch.Tensor,
    raft_model: torch.nn.Module,
) -> torch.Tensor:
    """
    Compute optical flow from image1 to image2 using RAFT.
    
    Args:
        image1: First image tensor (C, H, W) in range [0, 1]
        image2: Second image tensor (C, H, W) in range [0, 1]
        raft_model: Pre-loaded RAFT model (must be provided).
    
    Returns:
        Optical flow tensor (2, H, W) where flow[0] is x-direction and flow[1] is y-direction
    """
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    img1_batch = image1.unsqueeze(0)
    img2_batch = image2.unsqueeze(0)

    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            # T.Resize(size=(520, 960)),
        ]
    )
    img1_batch = transforms(img1_batch)
    img2_batch = transforms(img2_batch)
    
    # Ensure images are RGB (3 channels)
    if img1_batch.shape[1] == 1:
        img1_batch = img1_batch.repeat(1, 3, 1, 1)
        img2_batch = img2_batch.repeat(1, 3, 1, 1)
    elif img1_batch.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {img1_batch.shape[1]}")
    
    with torch.no_grad():
        # RAFT returns a list of flow predictions at different scales
        # We use the final prediction
        list_of_flows = raft_model(img1_batch, img2_batch)
        predicted_flows = list_of_flows[-1]  # (B, 2, H, W)
        flow = predicted_flows[0]  # (2, H, W)
    
    return flow


def compute_gaussian_flow(
    render_t_1: Dict[str, torch.Tensor],
    render_t_2: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    
    This implements the full formulation of GaussianFlow as described for GaussianFlow
    
    Args:
        render_t_1: Render outputs from time t-1, must contain:
            - proj_2D: (N, 2) 2D projected positions
            - conic_2D: (N, 3) conic matrix (inverse covariance)
            - gs_per_pixel: (K, H, W) Gaussian indices per pixel (top-K)
            - weight_per_gs_pixel: (K, H, W) weights per Gaussian per pixel
            - x_mu: (K, 2, H, W) pixel-to-Gaussian-center offsets
        render_t_2: Render outputs from time t, must contain:
            - proj_2D: (N, 2) 2D projected positions
            - conic_2D: (N, 3) conic matrix (inverse covariance)
            - conic_2D_inv: (N, 3) inverse conic matrix (covariance)
    
    Returns:
        Scalar loss tensor
    """
    # Extract variables from t_1 (detach as per snippet)
    proj_2D_t_1 = render_t_1["proj_2D"].detach()
    gs_per_pixel = render_t_1["gs_per_pixel"].long()
    weight_per_gs_pixel = render_t_1["weight_per_gs_pixel"].detach()
    x_mu = render_t_1["x_mu"].detach()
    cov2D_inv_t_1 = render_t_1["conic_2D"].detach()
    
    # Extract variables from t_2
    proj_2D_t_2 = render_t_2["proj_2D"]
    cov2D_inv_t_2 = render_t_2["conic_2D"]
    cov2D_t_2 = render_t_2["conic_2D_inv"]
    
    # Get spatial dimensions
    K, H, W = gs_per_pixel.shape
    N = proj_2D_t_1.shape[0]
    
    # Build covariance matrices from flattened representations
    # cov2D_t_2: (N, 3) -> (N, 2, 2)
    cov2D_t_2_mtx = torch.zeros([N, 2, 2], device=cov2D_t_2.device, dtype=cov2D_t_2.dtype)
    cov2D_t_2_mtx[:, 0, 0] = cov2D_t_2[:, 0]
    cov2D_t_2_mtx[:, 0, 1] = cov2D_t_2[:, 1]
    cov2D_t_2_mtx[:, 1, 0] = cov2D_t_2[:, 1]
    cov2D_t_2_mtx[:, 1, 1] = cov2D_t_2[:, 2]
    
    # cov2D_inv_t_1: (N, 3) -> (N, 2, 2)
    cov2D_inv_t_1_mtx = torch.zeros([N, 2, 2], device=cov2D_inv_t_1.device, dtype=cov2D_inv_t_1.dtype)
    cov2D_inv_t_1_mtx[:, 0, 0] = cov2D_inv_t_1[:, 0]
    cov2D_inv_t_1_mtx[:, 0, 1] = cov2D_inv_t_1[:, 1]
    cov2D_inv_t_1_mtx[:, 1, 0] = cov2D_inv_t_1[:, 1]
    cov2D_inv_t_1_mtx[:, 1, 1] = cov2D_inv_t_1[:, 2]
    
    # Compute B_t_2 = U * sqrt(S) * V^T via SVD
    U_t_2, S_t_2, V_t_2 = torch.svd(cov2D_t_2_mtx)
    B_t_2 = torch.bmm(
        torch.bmm(U_t_2, torch.diag_embed(S_t_2 ** 0.5)),
        V_t_2.transpose(1, 2)
    )
    
    # Compute B_inv_t_1 = U * sqrt(S) * V^T via SVD
    U_inv_t_1, S_inv_t_1, V_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)
    B_inv_t_1 = torch.bmm(
        torch.bmm(U_inv_t_1, torch.diag_embed(S_inv_t_1 ** 0.5)),
        V_inv_t_1.transpose(1, 2)
    )
    
    # Compute B_t_2 * B_inv_t_1
    B_t_2_B_inv_t_1 = torch.bmm(B_t_2, B_inv_t_1)
    
    # Reshape x_mu: (K, 2, H, W) -> (K, H, W, 2)
    x_mu_reshaped = x_mu.permute(0, 2, 3, 1)  # (K, H, W, 2)
    
    # Index B_t_2_B_inv_t_1 with gs_per_pixel: (K, H, W) -> (K, H, W, 2, 2)
    B_t_2_B_inv_t_1_indexed = B_t_2_B_inv_t_1[gs_per_pixel]  # (K, H, W, 2, 2)
    
    # Compute cov_multi = B_t_2_B_inv_t_1[gs_per_pixel] @ x_mu
    # x_mu_reshaped: (K, H, W, 2) -> (K, H, W, 2, 1) for matrix multiplication
    x_mu_expanded = x_mu_reshaped.unsqueeze(-1)  # (K, H, W, 2, 1)
    cov_multi = torch.bmm(
        B_t_2_B_inv_t_1_indexed.reshape(-1, 2, 2),
        x_mu_expanded.reshape(-1, 2, 1)
    ).squeeze(-1).reshape(K, H, W, 2)  # (K, H, W, 2)
    
    # Index proj_2D with gs_per_pixel
    proj_2D_t_1_indexed = proj_2D_t_1[gs_per_pixel]  # (K, H, W, 2)
    proj_2D_t_2_indexed = proj_2D_t_2[gs_per_pixel]  # (K, H, W, 2)
    
    # Full formulation of GaussianFlow
    # predicted_flow_by_gs = (cov_multi + proj_2D_t_2 - proj_2D_t_1 - x_mu) * weight_per_gs_pixel
    predicted_flow_by_gs = (
        cov_multi + 
        proj_2D_t_2_indexed - 
        proj_2D_t_1_indexed - 
        x_mu_reshaped
    ) * weight_per_gs_pixel.unsqueeze(-1)  # (K, H, W, 2)
    
    # Sum over K dimension to get per-pixel flow: (H, W, 2)
    predicted_flow = predicted_flow_by_gs.sum(dim=0)  # (H, W, 2)
    return predicted_flow


def compute_flow_loss(
    gt_flow: torch.Tensor,
    predicted_flow: torch.Tensor,
    flow_thresh: float,
) -> torch.Tensor:
    """
    Compute flow loss between ground truth flow and predicted flow.
    
    Args:
        gt_flow: Ground truth flow tensor (2, H, W)
        predicted_flow: Predicted flow tensor (H, W, 2)
        flow_thresh: Threshold for large motion
    """
    gt_flow_reshaped = gt_flow.permute(1, 2, 0)  # (H, W, 2)
    flow_norm = torch.norm(gt_flow_reshaped, p=2, dim=-1)  # (H, W)
    large_motion_msk = flow_norm >= flow_thresh
    if large_motion_msk.sum() > 0:
        flow_diff = predicted_flow - gt_flow_reshaped  # (H, W, 2)
        flow_loss = torch.norm(flow_diff[large_motion_msk], p=2, dim=-1).mean()
    else:
        flow_loss = torch.tensor(0.0, device=gt_flow.device, requires_grad=True)
    return flow_loss


def visualize_flow(
    flow: torch.Tensor,
    image1: torch.Tensor,
    image2: torch.Tensor,
    save_path: str,
) -> None:
    """
    Visualize flow.
    """
    flow = flow.unsqueeze(0)  # (1, 2, H, W)
    flow_img = flow_to_image(flow)  # (1, 3, H, W)

    # Both flow images have to be scaled from -1, 1 to 0, 1
    flow_img = (flow_img + 1) / 2
    
    img_batch1 = image1.unsqueeze(0)  # (1, C, H, W)
    img_batch2 = image2.unsqueeze(0)  # (1, C, H, W)
    
    # Concatenate horizontally
    grid = torch.cat([img_batch1, img_batch2, flow_img], dim=3)  # (1, 3, H, 3*W)
    
    # Convert to PIL and save
    grid_pil = Image.fromarray((grid[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    grid_pil.save(save_path)

    
