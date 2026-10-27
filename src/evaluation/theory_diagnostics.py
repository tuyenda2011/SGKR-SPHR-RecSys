"""Train-only bounded diagnostics; none of these quantities establishes causality."""
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def geometry_diagnostics(vectors, limit=256):
    if limit < 1:
        raise ValueError("Geometry probe limit must be positive")
    sample = vectors[:limit].double()
    if not len(sample):
        return {"geometry_probe_rows": 0, "geometry_effective_rank": 0.}
    singular = torch.linalg.svdvals(sample)
    probabilities = singular/singular.sum().clamp_min(1e-30)
    rank = (-(probabilities*probabilities.clamp_min(1e-30).log()).sum()).exp()
    if singular.sum() == 0:
        rank = singular.sum()
    normalized = F.normalize(sample, dim=1)
    cosine = normalized@normalized.T
    off = ~torch.eye(len(sample), dtype=torch.bool, device=sample.device)
    return {"geometry_probe_rows": len(sample), "geometry_effective_rank": float(rank),
            "geometry_norm_mean": float(sample.norm(dim=1).mean()),
            "geometry_cosine_offdiag_mean": float(cosine[off].mean()) if len(sample) > 1 else None}


def sgkr_graph_diagnostics(model):
    """Principal-block energy and normalized boundary MASS, not boundary energy."""
    diagonal, energy, boundary_mass, pure_ridge, internal_edges = 0., 0., 0., 0, 0
    inside_energy, boundary_energy = 0., 0.
    split_available = hasattr(model, "graph_precision_parts")
    for item, neighbors in enumerate(model.neighborhoods):
        if not len(neighbors):
            continue
        block = model.laplacian[neighbors][:, neighbors]
        beta = model.linear_coefficients[neighbors, item].toarray().ravel()
        diagonal += float(block.diagonal()@beta**2)
        energy += float(beta@(block@beta))
        if split_available:
            inside, boundary = model.graph_precision_parts(item)
            inside_energy += float(beta@(inside@beta))
            boundary_energy += float(beta@(boundary@beta))
        # SciPy returns a NumPy scalar for nnz when CSR indices are int64.
        edges = int(block.nnz)-int(np.count_nonzero(block.diagonal()))
        internal_edges += edges
        pure_ridge += int(edges == 0)
        boundary_mass += float(block.sum()-model.laplacian[neighbors].sum())
    active = np.flatnonzero(model.counts > 0)
    multipliers = model.support_multipliers(active)
    boundary_weight = float(getattr(getattr(model, "config", None), "boundary_weight", 1.))
    return {"targets_with_nonempty_support": sum(bool(len(n)) for n in model.neighborhoods),
            "pure_diagonal_principal_blocks": pure_ridge, "internal_directed_edges_total": internal_edges,
            "coefficient_laplacian_energy": energy, "coefficient_diagonal_energy": diagonal,
            "coefficient_cross_energy": energy-diagonal,
            "coefficient_internal_energy": inside_energy if split_available else None,
            "coefficient_boundary_energy": boundary_energy if split_available else None,
            "boundary_weight": boundary_weight,
            "coefficient_effective_graph_energy": inside_energy+boundary_weight*boundary_energy if split_available else None,
            "normalized_adjacency_boundary_mass": max(0., boundary_mass),
            "active_support_multiplier_quantiles": np.quantile(multipliers, [0, .5, .9, 1]).tolist(),
            "scope": "train-derived structure; boundary mass is neither probability nor causal ranking contribution"}
