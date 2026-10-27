import logging
from typing import Tuple
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
logger = logging.getLogger(__name__)


def build_bipartite_adj_matrix(
    train_df: pd.DataFrame, num_users: int, num_items: int
) -> sp.csr_matrix:
    """Construct SciPy CSR bipartite adjacency matrix A = [[0, R], [R^T, 0]]."""
    user_indices = train_df["u_idx"].values
    item_indices = train_df["i_idx"].values

    # Interaction matrix R (num_users x num_items)
    R = sp.csr_matrix(
        (np.ones(len(train_df), dtype=np.float32), (user_indices, item_indices)),
        shape=(num_users, num_items),
    )

    # Upper right: R (num_users x num_items)
    # Lower left: R^T (num_items x num_users)
    adj_mat = sp.bmat(
        [[None, R], [R.T, None]], format="csr", dtype=np.float32
    )
    return adj_mat


def normalize_adj_matrix(adj_mat: sp.csr_matrix) -> sp.csr_matrix:
    """Compute symmetric D^(-1/2) * A * D^(-1/2)."""
    rowsum = np.array(adj_mat.sum(axis=1), dtype=np.float32).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    mask = rowsum > 0
    d_inv_sqrt[mask] = np.power(rowsum[mask], -0.5)
    d_mat = sp.diags(d_inv_sqrt)

    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    return norm_adj.tocsr()


def scipy_to_torch_sparse(mat: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert SciPy CSR matrix to PyTorch sparse COO tensor on device."""
    coo = mat.tocoo()
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64)
    )
    values = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)

    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, shape, device=device, is_coalesced=True, check_invariants=False
    )
    return sparse_tensor


class _SparseMatmul(torch.autograd.Function):
    """A @ X with dL/dX = A^T G computed from a supplied A^T (no per-call transpose)."""

    @staticmethod
    def forward(ctx, matrix, transpose, dense):
        ctx.transpose = transpose
        return torch.sparse.mm(matrix, dense)

    @staticmethod
    def backward(ctx, grad):
        return None, None, torch.sparse.mm(ctx.transpose, grad)


def sparse_matmul(matrix: torch.Tensor, dense: torch.Tensor, transpose: torch.Tensor | None = None) -> torch.Tensor:
    """Sparse @ dense, differentiable in ``dense`` only; ``transpose`` defaults to ``matrix`` (symmetric).

    COO backward in torch transposes and coalesces the matrix on every call, which
    dominated training time; CSR operands with a precomputed transpose avoid it.
    """
    return _SparseMatmul.apply(matrix, matrix if transpose is None else transpose, dense)


def to_csr(matrix: torch.Tensor) -> torch.Tensor:
    """Same values, CSR layout (faster cuSPARSE SpMM than COO)."""
    return matrix if matrix.layout == torch.sparse_csr else matrix.coalesce().to_sparse_csr()


def get_norm_adj_tensor(
    train_df: pd.DataFrame, num_users: int, num_items: int, device: torch.device
) -> torch.Tensor:
    """Build and normalize bipartite graph tensor directly from train dataframe."""
    adj_mat = build_bipartite_adj_matrix(train_df, num_users, num_items)
    norm_adj = normalize_adj_matrix(adj_mat)
    return scipy_to_torch_sparse(norm_adj, device)


def build_time_weighted_bipartite_adj_matrix(
    train_df: pd.DataFrame, num_users: int, num_items: int, beta: float = 0.5
) -> sp.csr_matrix:
    """Construct time-decay weighted bipartite adjacency matrix.

    W_ui = exp(-beta * (t_max - t_ui) / (t_max - t_min + 1e-5))
    """
    user_indices = train_df["u_idx"].values
    item_indices = train_df["i_idx"].values

    if "timestamp" in train_df.columns:
        timestamps = train_df["timestamp"].values.astype(np.float64)
        t_min = timestamps.min()
        t_max = timestamps.max()
        time_range = max(1.0, t_max - t_min)
        weights = np.exp(-beta * (t_max - timestamps) / time_range).astype(np.float32)
    else:
        weights = np.ones(len(train_df), dtype=np.float32)

    R = sp.csr_matrix((weights, (user_indices, item_indices)), shape=(num_users, num_items))
    adj_mat = sp.bmat([[None, R], [R.T, None]], format="csr", dtype=np.float32)
    return adj_mat


def get_time_weighted_norm_adj_tensor(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    device: torch.device,
    beta: float = 0.5,
) -> torch.Tensor:
    """Build and symmetrically normalize time-decay weighted bipartite graph tensor."""
    adj_mat = build_time_weighted_bipartite_adj_matrix(train_df, num_users, num_items, beta=beta)
    norm_adj = normalize_adj_matrix(adj_mat)
    return scipy_to_torch_sparse(norm_adj, device)


def create_edge_dropout_views(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    drop_ratio: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two augmented sparse graph tensors for SGL by applying edge dropout."""
    num_edges = len(train_df)
    keep_ratio = 1.0 - drop_ratio

    # View 1
    mask1 = np.random.rand(num_edges) < keep_ratio
    df_view1 = train_df.iloc[mask1]
    norm_adj1 = get_norm_adj_tensor(df_view1, num_users, num_items, device)

    # View 2
    mask2 = np.random.rand(num_edges) < keep_ratio
    df_view2 = train_df.iloc[mask2]
    norm_adj2 = get_norm_adj_tensor(df_view2, num_users, num_items, device)

    return norm_adj1, norm_adj2

