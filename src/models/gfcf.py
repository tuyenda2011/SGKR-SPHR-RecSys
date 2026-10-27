"""GF-CF: training-free graph filter for collaborative filtering (Shen et al., CIKM 2021).

score(r_u) = r_u R~^T R~  +  ideal_weight * r_u D_I^-1/2 V_k^T V_k D_I^1/2,
R~ = D_U^-1/2 R D_I^-1/2 (degree-normalised interactions), V_k = top-k right singular vectors of R~.
The first term is the linear low-pass filter, the second the ideal low-pass filter; r_u is the raw
(unnormalised) interaction row, as in the authors' reference implementation. No gradient training,
no dense user-by-item allocation, and scoring applies both terms batch by batch.
V_k is a top-k SVD converged to tolerance, as in the authors' code (sparsesvd/SVDLIBC Lanczos). sparsesvd
no longer builds (its setup.py needs numpy.distutils, removed in numpy 2), so the default backend is block
subspace iteration on R~^T R~ with Rayleigh-Ritz, stopped on a residual tolerance (fit_report records it).
"dense_eigh" (LAPACK on the dense Gram, ~8 GB, slow) and "arpack" (too slow for k=256) are for checks.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import svds

BACKENDS = ("subspace", "dense_eigh", "sparsesvd", "arpack")


def subspace_singular_vectors(matrix, k, *, oversample=128, tol=1e-8, max_iterations=2000, check_every=5, seed=0):
    """Top-k right singular vectors of a sparse A by block subspace iteration on A^T A with Rayleigh-Ritz.

    Iterates until every kept Ritz pair has relative residual ||A^T A v - theta v|| / theta_1 < tol, the
    same converge-to-tolerance contract as Lanczos (SVDLIBC, ARPACK). Only sparse products and thin QRs:
    no dense item-by-item matrix.
    """
    a = sparse.csr_matrix(matrix, dtype=np.float64)
    at = a.T.tocsr()
    n = a.shape[1]
    width = min(n, k + oversample)
    basis, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, width)))
    residual = np.inf
    for iteration in range(1, max_iterations + 1):
        basis, _ = np.linalg.qr(at @ (a @ basis))
        if iteration % check_every and iteration != max_iterations:
            continue
        projected = a @ basis
        values, vectors = np.linalg.eigh(projected.T @ projected)
        order = np.argsort(values)[::-1][:k]
        values, ritz = values[order], basis @ vectors[:, order]
        residual = float(np.max(np.linalg.norm(at @ (a @ ritz) - ritz * values, axis=0)) / max(values[0], 1e-300))
        if residual < tol:
            break
    else:
        raise RuntimeError(f"Subspace iteration did not converge: residual {residual:.2e} after {max_iterations} iterations")
    return np.sqrt(np.clip(values, 0., None)), ritz.T, {"iterations": iteration, "relative_residual": residual,
                                                         "tolerance": tol, "oversample": oversample}


def dense_eigh_memory_required(num_items):
    # One float32 item-by-item Gram matrix reduced in place, plus a 1 GiB reserve.
    return num_items * num_items * np.dtype("float32").itemsize + 2**30


def top_right_singular_vectors(matrix, k, *, available_bytes=None):
    """Exact top-k right singular vectors of a sparse matrix A via the k largest eigenpairs of A^T A (LAPACK evr)."""
    import psutil
    n = matrix.shape[1]
    available = psutil.virtual_memory().available if available_bytes is None else available_bytes
    if available < dense_eigh_memory_required(n):
        raise MemoryError(f"GF-CF dense_eigh needs {dense_eigh_memory_required(n)} free bytes; available {available}")
    single = matrix.astype(np.float32)
    gram = (single.T @ single).toarray(order="F")  # symmetric, so Fortran order avoids a copy inside LAPACK
    eigenvalues, eigenvectors = linalg.eigh(gram, subset_by_index=[n - k, n - 1], driver="evr",
                                            overwrite_a=True, check_finite=False)
    del gram
    order = np.argsort(eigenvalues)[::-1]
    return np.sqrt(np.clip(eigenvalues[order], 0., None)), eigenvectors[:, order].T


@dataclass(frozen=True)
class GFCFConfig:
    singular_vectors: int = 256   # k of the ideal low-pass filter (paper default)
    ideal_weight: float = 0.3     # weight of the ideal low-pass term (paper default)
    svd_seed: int = 0             # ARPACK start vector; V_k^T V_k does not depend on singular-vector signs
    svd_backend: str = "subspace"  # converged subspace iteration; "dense_eigh" (LAPACK, ~8 GB), "sparsesvd", "arpack"

    def __post_init__(self):
        if self.svd_backend not in BACKENDS:
            raise ValueError(f"svd_backend must be one of {BACKENDS}")
        if isinstance(self.singular_vectors, bool) or not isinstance(self.singular_vectors, int) or self.singular_vectors < 0:
            raise ValueError("singular_vectors must be a nonnegative integer")
        if isinstance(self.ideal_weight, bool) or not np.isfinite(self.ideal_weight) or self.ideal_weight < 0:
            raise ValueError("ideal_weight must be finite and nonnegative")


class GFCF:
    def __init__(self, config: GFCFConfig | None = None):
        self.config = config or GFCFConfig()
        self.fitted = False

    def _normalise(self, interactions):
        self.interactions = sparse.csr_matrix(interactions, dtype=np.float64)
        users = np.asarray(self.interactions.sum(1)).ravel()
        items = np.asarray(self.interactions.sum(0)).ravel()
        user_scale = np.divide(1., np.sqrt(users), out=np.zeros_like(users), where=users > 0)
        # Items absent from this training graph get 0 on both sides (the reference code divides by 0).
        self.item_scale = np.divide(1., np.sqrt(items), out=np.zeros_like(items), where=items > 0)
        self.item_unscale = np.sqrt(items)
        self.normalized = (sparse.diags(user_scale) @ self.interactions @ sparse.diags(self.item_scale)).tocsr()

    def fit(self, interactions):
        started = time.perf_counter()
        self._normalise(interactions)
        k = self.config.singular_vectors
        convergence = None
        if k:
            if k >= min(self.normalized.shape):
                raise ValueError("singular_vectors must be smaller than both matrix dimensions")
            if self.config.svd_backend == "subspace":
                singular_values, vt, convergence = subspace_singular_vectors(self.normalized, k, seed=self.config.svd_seed)
            elif self.config.svd_backend == "dense_eigh":
                singular_values, vt = top_right_singular_vectors(self.normalized, k)
            elif self.config.svd_backend == "sparsesvd":
                from sparsesvd import sparsesvd  # pip install sparsesvd (builds from source; Linux/Kaggle)
                _, singular_values, vt = sparsesvd(self.normalized.tocsc(), k)
                if len(singular_values) != k:
                    raise ValueError(f"sparsesvd returned {len(singular_values)} singular vectors, expected {k}")
            else:
                start = np.random.default_rng(self.config.svd_seed).standard_normal(min(self.normalized.shape))
                _, singular_values, vt = svds(self.normalized, k=k, v0=start)
            self.vt = vt.astype(np.float32)
            self.singular_values = np.sort(singular_values)[::-1]
        else:
            self.vt = np.zeros((0, self.interactions.shape[1]), dtype=np.float32)
            self.singular_values = np.zeros(0)
        self.fit_report = {"seconds": time.perf_counter() - started, "singular_vectors": k,
                           "svd_backend": self.config.svd_backend, "svd_convergence": convergence,
                           "largest_singular_value": float(self.singular_values[0]) if k else None,
                           "smallest_kept_singular_value": float(self.singular_values[-1]) if k else None}
        self.fitted = True
        return self

    def score_rows(self, rows):
        """Scores of raw interaction rows (users x items, sparse or dense) over every item."""
        rows = sparse.csr_matrix(rows, dtype=np.float64)
        linear = ((rows @ self.normalized.T) @ self.normalized).toarray()
        if self.config.ideal_weight and len(self.vt):
            latent = (rows @ sparse.diags(self.item_scale)) @ self.vt.T.astype(np.float64)
            linear += self.config.ideal_weight * (latent @ self.vt.astype(np.float64)) * self.item_unscale[None, :]
        return linear

    def score(self, users):
        return self.score_rows(self.interactions[np.asarray(users)])

    def get_user_rating_scores(self, users, all_users=None, all_items=None):
        """Adapter for the project's masked full-ranking Evaluator."""
        import torch
        return torch.from_numpy(self.score(users.detach().cpu().numpy())).to(users.device)

    def save(self, path):
        """Portable numeric checkpoint (training interactions + V_k), no pickle."""
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted model")
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, schema=np.array(1), config=np.array(json.dumps(asdict(self.config))),
                                report=np.array(json.dumps(self.fit_report)), shape=np.array(self.interactions.shape),
                                train_indptr=self.interactions.indptr, train_indices=self.interactions.indices,
                                vt=self.vt, singular_values=self.singular_values)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            if int(state["schema"]) != 1:
                raise ValueError("Unsupported GF-CF checkpoint schema")
            model = cls(GFCFConfig(**json.loads(str(state["config"]))))
            shape = tuple(int(n) for n in state["shape"])
            model._normalise(sparse.csr_matrix((np.ones(len(state["train_indices"])), state["train_indices"],
                                                state["train_indptr"]), shape=shape))
            model.vt, model.singular_values = state["vt"], state["singular_values"]
            model.fit_report = json.loads(str(state["report"]))
        model.fitted = True
        return model
