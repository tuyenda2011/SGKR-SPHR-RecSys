"""Shared geometry utilities for computing alignment and uniformity metrics on the hypersphere.

References:
    Wang & Isola (ICML 2020) - Understanding Contrastive Learning via Uniformity on the Hypersphere
"""

from typing import Tuple
import torch
import torch.nn.functional as F


def compute_alignment(
    user_embeds: torch.Tensor,
    item_embeds: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float = 2.0,
) -> float:
    """Compute Alignment metric on the unit hypersphere (Wang & Isola, ICML 2020).

    Alignment measures the expected distance between normalized representations of positive pairs:
        L_align = E_{(u, i) ~ p_pos} [ || f(u) - f(i) ||_2^alpha ]

    Args:
        user_embeds: Tensor of user embeddings of shape (num_users, dim)
        item_embeds: Tensor of item embeddings of shape (num_items, dim)
        edge_index: Tensor of positive user-item interaction pairs of shape (2, num_edges)
        alpha: Power parameter (default: 2.0)

    Returns:
        float: Alignment score (lower indicates better alignment)
    """
    u_idx = edge_index[0]
    i_idx = edge_index[1]

    # Normalize to unit sphere
    u_norm = F.normalize(user_embeds[u_idx], dim=-1)
    i_norm = F.normalize(item_embeds[i_idx], dim=-1)

    diff = (u_norm - i_norm).norm(p=2, dim=-1)
    alignment_loss = (diff ** alpha).mean().item()
    return float(alignment_loss)


def compute_uniformity(
    embeds: torch.Tensor,
    t: float = 2.0,
    sample_size: int = 5000,
) -> float:
    """Compute Uniformity metric on the unit hypersphere (Wang & Isola, ICML 2020).

    Uniformity measures how well representations are uniformly distributed over the unit sphere:
        L_uniform = log E_{x, y ~ p_data} [ exp(-t * || f(x) - f(y) ||_2^2) ]

    Args:
        embeds: Tensor of representations of shape (num_nodes, dim)
        t: Temperature scale factor (default: 2.0)
        sample_size: Number of nodes to randomly sample for efficient estimation

    Returns:
        float: Uniformity score (lower indicates more uniform distribution)
    """
    num_nodes = embeds.size(0)
    if num_nodes == 0:
        return 0.0

    if num_nodes > sample_size:
        perm = torch.randperm(num_nodes)[:sample_size]
        sample_embeds = embeds[perm]
    else:
        sample_embeds = embeds

    # Normalize representations to unit sphere
    norm_embeds = F.normalize(sample_embeds, dim=-1)

    # Pairwise squared Euclidean distance: ||x - y||^2 = 2 - 2 * <x, y>
    sim_matrix = torch.matmul(norm_embeds, norm_embeds.T)
    dist_sq = 2.0 - 2.0 * sim_matrix.clamp(min=-1.0, max=1.0)

    # Exclude diagonal self-distances
    mask = ~torch.eye(norm_embeds.size(0), dtype=torch.bool, device=embeds.device)
    exp_dist = torch.exp(-t * dist_sq[mask])

    uniformity = torch.log(exp_dist.mean() + 1e-12).item()
    return float(uniformity)


def batch_pairwise_uniformity(
    embeds: torch.Tensor,
    t: float = 2.0,
) -> torch.Tensor:
    """Compute uniformity loss for a batch of embeddings (used in DirectAU loss).

    Optimized version for mini-batch computations without sampling.

    Args:
        embeds: Tensor of batch embeddings of shape (batch_size, dim)
        t: Temperature scale factor

    Returns:
        Differentiable scalar uniformity loss.
    """
    batch_size = embeds.size(0)
    if batch_size <= 1:
        return embeds.sum() * 0.0

    norm_embeds = F.normalize(embeds, dim=-1)
    sim_matrix = torch.matmul(norm_embeds, norm_embeds.T)
    dist_sq = 2.0 - 2.0 * sim_matrix.clamp(min=-1.0, max=1.0)

    # Exclude self-distances
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeds.device)
    exp_dist = torch.exp(-t * dist_sq[mask])

    return torch.log(exp_dist.mean() + 1e-12)
