from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
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
    seed: int = 42,
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
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(num_nodes, generator=generator)[:sample_size]
        perm = perm.to(embeds.device)
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


def compute_alignment_and_uniformity(
    user_embeds: torch.Tensor,
    item_embeds: torch.Tensor,
    edge_index: torch.Tensor,
    sample_size: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute comprehensive alignment and uniformity metrics for both user and item embeddings.

    Returns:
        Dict containing alignment, user_uniformity, item_uniformity, and mean_uniformity.
    """
    align = compute_alignment(user_embeds, item_embeds, edge_index)
    u_unif = compute_uniformity(user_embeds, sample_size=sample_size, seed=seed)
    i_unif = compute_uniformity(item_embeds, sample_size=sample_size, seed=seed)
    mean_unif = (u_unif + i_unif) / 2.0

    return {
        "alignment": align,
        "user_uniformity": u_unif,
        "item_uniformity": i_unif,
        "mean_uniformity": mean_unif,
    }


def compute_svd_spectrum(
    embeddings: torch.Tensor,
    top_k: int = 30,
) -> Dict[str, Any]:
    """Compute Singular Value Spectrum (SVD) and Effective Rank for embedding matrices.

    Used to detect Dimensional Collapse and Representation Degeneration.

    Args:
        embeddings: Tensor of node embeddings (num_nodes, dim)
        top_k: Number of leading singular values to extract

    Returns:
        Dict containing singular_values, normalized_singular_values, cumulative_energy, and effective_rank.
    """
    # Center embeddings
    centered_emb = embeddings - embeddings.mean(dim=0, keepdim=True)

    # Compute SVD singular values with proper error handling
    s_np = _compute_svd_singular_values(centered_emb)

    total_energy = np.sum(s_np ** 2) + 1e-12
    norm_s = s_np / (np.sum(s_np) + 1e-12)

    # Shannon Entropy / Effective Rank (Roy & Vetterli, 2007)
    # Higher entropy = higher dimensional utilization / less collapse
    entropy = -np.sum(norm_s * np.log(norm_s + 1e-12))
    effective_rank = float(np.exp(entropy))

    cum_energy = np.cumsum(s_np ** 2) / total_energy

    # Compute spectral decay rate with safe indexing
    s_len = len(s_np)
    if s_len > 1:
        spectral_decay = float(s_np[0] / (s_np[min(s_len - 1, 9)] + 1e-12))
    else:
        spectral_decay = 1.0

    return {
        "singular_values": s_np[:top_k].tolist(),
        "normalized_singular_values": norm_s[:top_k].tolist(),
        "cumulative_energy": cum_energy[:top_k].tolist(),
        "effective_rank": effective_rank,
        "spectral_decay_rate": spectral_decay,
    }


def _compute_svd_singular_values(centered_emb: torch.Tensor) -> np.ndarray:
    """Compute SVD singular values with fallback handling.

    Args:
        centered_emb: Centered embedding tensor

    Returns:
        NumPy array of singular values

    Raises:
        RuntimeError: If both PyTorch and NumPy SVD fail
    """
    # Try PyTorch's optimized SVD first
    try:
        singular_vals = torch.linalg.svdvals(centered_emb)
        return singular_vals.detach().cpu().numpy()
    except (RuntimeError, AttributeError) as pytorch_err:
        # Exception aliases are cleared after except; retain only the message.
        pytorch_error_message = str(pytorch_err)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"PyTorch SVD failed ({pytorch_err}), falling back to NumPy SVD")

    # Fallback to NumPy SVD
    try:
        _, s_np, _ = np.linalg.svd(
            centered_emb.detach().cpu().numpy(),
            full_matrices=False
        )
        return s_np
    except Exception as numpy_err:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Both PyTorch and NumPy SVD failed: {numpy_err}")
        raise RuntimeError(
            f"Failed to compute SVD: PyTorch error={pytorch_error_message}, NumPy error={numpy_err}"
        ) from numpy_err


def compute_oversmoothing_analysis(
    model: torch.nn.Module,
    norm_adj: torch.Tensor,
    max_layers: int = 5,
    sample_nodes: int = 2000,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Analyze over-smoothing dynamics across increasing GNN layer propagation depths.

    Computes Dirichlet Energy / Mean Cosine Distance across layers l in [0, max_layers].

    Args:
        model: Trained recommendation model (LightGCN, XSimGCL, DirectAU, or RecFlow)
        norm_adj: Normalized bipartite adjacency sparse tensor
        max_layers: Maximum layer depth to evaluate
        sample_nodes: Number of nodes sampled to compute pairwise cosine distance

    Returns:
        Dict with 'layer_depths' and 'mean_pairwise_distances'
    """
    device = next(model.parameters()).device
    ego_embeddings = torch.cat(
        [model.user_embedding.weight, model.item_embedding.weight], dim=0
    )

    num_total_nodes = ego_embeddings.size(0)
    generator = torch.Generator().manual_seed(seed)
    sample_idx = torch.randperm(num_total_nodes, generator=generator)[
        : min(sample_nodes, num_total_nodes)
    ].to(device)

    layer_distances = []
    layer_depths = list(range(max_layers + 1))

    current_emb = ego_embeddings
    all_layers = [current_emb]

    with torch.no_grad():
        for _ in range(max_layers):
            current_emb = torch.sparse.mm(norm_adj, current_emb)
            all_layers.append(current_emb)

        for l_emb in all_layers:
            sampled = l_emb[sample_idx]
            normed = F.normalize(sampled, dim=-1)
            # Pairwise cosine similarity
            sim = torch.matmul(normed, normed.T)
            # Mean distance = 1 - mean similarity (excluding self)
            mask = ~torch.eye(normed.size(0), dtype=torch.bool, device=device)
            mean_dist = (1.0 - sim[mask]).mean().item()
            layer_distances.append(float(mean_dist))

    return {
        "layer_depths": layer_depths,
        "mean_pairwise_distances": layer_distances,
    }
