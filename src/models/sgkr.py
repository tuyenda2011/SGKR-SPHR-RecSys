"""SGKR: item-item graph-regularised ridge regression with exact target-excluded fitting.

No trainable user IDs or gradient optimizer. Every statistic is built from the supplied
retained train. The defaults are the locked configuration (configs/sgkr.yaml).
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)


# Options of earlier experiments that the locked model no longer has. A schema-1 checkpoint may
# carry them only with the values below, which make them inert (no kernel branch, excess rule).
RETIRED_OPTIONS = {"kernel_dim": 0, "neighbor_rule": "excess"}
IGNORED_OPTIONS = ("bandwidth", "eta_kernel", "feature_map", "seed")


@dataclass(frozen=True)
class SGKRConfig:
    neighbors: int = 128
    alpha: float = 10.0
    positive_weight: float = 10.0
    eta_linear: float = 1e-5
    eta_bias: float = 0.001
    graph_weight: float = 0.001
    adaptive_regularization: bool = True
    adaptive_power: float = 2.0
    adaptive_cap: float | None = 10.0
    correction_batch_size: int = 1024
    threads: int = 1
    max_cooccurrence_entries: int = 50_000_000
    boundary_weight: float = 0.5
    support_alpha: float | None = 2.0
    # Inference-only ranking tilt s * (1 + c_i)^popularity_tilt, c_i = retained train count.
    # Fitting uses the untilted SGKR.score; 0 keeps the original ranking.
    popularity_tilt: float = -0.1

    @classmethod
    def from_dict(cls, values: dict):
        """Accept older configs/checkpoints only when their retired options are inert."""
        values = dict(values)
        # Every pre-removal config stored kernel_dim; among those, a missing neighbor_rule predates
        # the option and therefore means lift.
        legacy = "kernel_dim" in values
        for key, required in RETIRED_OPTIONS.items():
            found = values.pop(key, "lift" if legacy and key == "neighbor_rule" else required)
            if found != required:
                raise ValueError(f"{key}={found!r} belongs to a retired SGKR variant; only {required!r} is supported")
        for key in IGNORED_OPTIONS:
            values.pop(key, None)
        return cls(**values)

    def __post_init__(self):
        if isinstance(self.neighbors, bool) or not isinstance(self.neighbors, int) or self.neighbors < 0:
            raise ValueError("neighbors must be a nonnegative integer")
        for key in ("correction_batch_size", "threads", "max_cooccurrence_entries"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("eta_linear", "eta_bias"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("alpha", "graph_weight", "adaptive_power"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not math.isfinite(self.positive_weight) or self.positive_weight < 1:
            raise ValueError("positive_weight must be finite and >= 1")
        if not isinstance(self.adaptive_regularization, bool):
            raise TypeError("adaptive_regularization must be boolean")
        if (isinstance(self.boundary_weight, bool) or not math.isfinite(self.boundary_weight)
                or not 0 <= self.boundary_weight <= 1):
            raise ValueError("boundary_weight must be finite and in [0, 1]")
        if self.adaptive_cap is not None and (isinstance(self.adaptive_cap, bool)
                or not math.isfinite(self.adaptive_cap) or self.adaptive_cap < 1):
            raise ValueError("adaptive_cap must be finite and >= 1, or None")
        if isinstance(self.popularity_tilt, bool) or not math.isfinite(self.popularity_tilt):
            raise ValueError("popularity_tilt must be finite")
        if self.support_alpha is not None and (isinstance(self.support_alpha, bool)
                or not math.isfinite(self.support_alpha) or self.support_alpha < 0):
            raise ValueError("support_alpha must be finite and nonnegative, or None")


class SGKR:
    """Sparse item-local linear coefficients plus an item bias.

    CPU float64 fitting. No dense user-by-item or item-by-item allocation.
    Inference also excludes a candidate from its own context if it was seen.
    """

    def __init__(self, config: SGKRConfig | None = None):
        self.config = config or SGKRConfig()
        self.fitted = False

    @staticmethod
    def interaction_matrix(users, items, num_users: int, num_items: int):
        users, items = np.asarray(users), np.asarray(items)
        if num_users < 1 or num_items < 1 or users.ndim != 1 or users.shape != items.shape:
            raise ValueError("Invalid interaction dimensions")
        if not len(users) or users.dtype.kind not in "iu" or items.dtype.kind not in "iu":
            raise ValueError("Interactions must be nonempty integer ID arrays")
        if users.min() < 0 or users.max() >= num_users or items.min() < 0 or items.max() >= num_items:
            raise ValueError("Interaction ID out of bounds")
        result = sparse.csr_matrix((np.ones(len(users)), (users, items)), shape=(num_users, num_items))
        if np.any(result.data != 1):
            raise ValueError("Duplicate interactions are not allowed")
        result.sort_indices()
        return result

    def _initialize(self, interactions):
        """Build train-only neighborhoods and the item graph.

        Neighbors are the positive-lift sources ranked by excess cooccurrence; directed
        edge weights are lift, then symmetrized and degree-normalized.
        """
        self.fitted = False
        self.interactions = interactions
        self.num_users, self.num_items = interactions.shape
        self.degrees = np.asarray(interactions.sum(axis=1)).ravel()
        self.counts = np.asarray(interactions.sum(axis=0)).ravel()
        self.prior_probability = (self.counts + 1) / (self.num_users + 2)
        support_alpha = self.config.alpha if self.config.support_alpha is None else self.config.support_alpha
        self.support = self.counts / np.maximum(self.counts + support_alpha, 1)
        normalizer = 1 / np.sqrt(np.maximum(self.degrees, 1))
        self.history = interactions.multiply(normalizer[:, None]).tocsr()
        bound = int(np.dot(self.degrees, self.degrees))
        if bound > self.config.max_cooccurrence_entries:
            raise MemoryError(f"Cooccurrence upper bound {bound:,} exceeds configured limit")
        cooccurrence = (interactions.T @ interactions).tocsr()
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()
        self.neighborhoods = []
        rows, columns, weights = [], [], []
        for item in range(self.num_items):
            start, end = cooccurrence.indptr[item:item + 2]
            sources = cooccurrence.indices[start:end]
            # Target i conditioned on source j; diagonal is never a feature.
            values = cooccurrence.data[start:end]
            excess = values - self.counts[sources] * self.prior_probability[item]
            lift = excess / (self.counts[sources] + self.config.alpha)
            keep = np.flatnonzero(lift > 0)
            order = np.lexsort((sources[keep], -excess[keep]))[:self.config.neighbors]
            chosen = keep[order]
            selected = sources[chosen].copy()
            self.neighborhoods.append(selected)
            rows.extend([item] * len(selected))
            columns.extend(selected.tolist())
            weights.extend(lift[chosen].tolist())
        directed = sparse.csr_matrix((weights, (rows, columns)), shape=(self.num_items, self.num_items))
        graph = directed.maximum(directed.T)
        graph_degree = np.asarray(graph.sum(axis=1)).ravel()
        self.graph_degree = graph_degree
        inverse = np.zeros_like(graph_degree)
        active = graph_degree > 0
        inverse[active] = 1 / np.sqrt(graph_degree[active])
        diagonal = sparse.diags(inverse)
        self.laplacian = (sparse.diags(active.astype(float)) - diagonal @ graph @ diagonal).tocsr()

    def support_multipliers(self, items):
        """Penalty multipliers only: changing these never changes graph selection.

        Zero-support items use the explicit prior fallback and a neutral scale.
        The optional cap bounds positive-support penalties; legacy defaults retain
        exactly support**-2. These controls do not assert a ranking improvement.
        """
        items = np.asarray(items, dtype=np.int64)
        scales = np.ones(items.shape, dtype=np.float64)
        if self.config.adaptive_regularization:
            active = self.counts[items] > 0
            support = self.support[items][active]
            if self.config.adaptive_cap is None:
                scales[active] = support ** -self.config.adaptive_power
            else:
                log_scale = np.minimum(-self.config.adaptive_power * np.log(support),
                                       math.log(self.config.adaptive_cap))
                scales[active] = np.exp(log_scale)
        if not np.isfinite(scales).all():
            raise FloatingPointError("Nonfinite support penalty; use a finite adaptive_cap")
        return scales

    def graph_precision_parts(self, item: int):
        """Internal and boundary normalized energies, using GLOBAL graph degrees."""
        neighbors = self.neighborhoods[item]
        principal = self.laplacian[neighbors][:, neighbors].toarray()
        root_degree = np.sqrt(self.graph_degree[neighbors])
        adjacency = np.diag(principal.diagonal())-principal
        inside = np.divide(adjacency@root_degree, root_degree, out=np.zeros_like(root_degree), where=root_degree > 0)
        ratio = principal.diagonal()-inside
        boundary = np.diag(np.clip(ratio, 0., 1.))
        return principal-boundary, boundary

    def precision(self, item: int):
        """M-scaled positive definite penalty; graph principal block includes boundary."""
        neighbors = self.neighborhoods[item]
        local = len(neighbors)
        matrix = np.zeros((local + 1, local + 1))
        source_scale = self.support_multipliers(neighbors)
        matrix[:local, :local] = np.diag(self.config.eta_linear * source_scale)
        if self.config.graph_weight and local:
            graph_block = self.laplacian[neighbors][:, neighbors].toarray()
            if self.config.boundary_weight != 1:
                _, boundary = self.graph_precision_parts(item)
                graph_block -= (1-self.config.boundary_weight)*boundary
            matrix[:local, :local] += self.config.graph_weight * graph_block
        matrix[-1, -1] = self.config.eta_bias
        return matrix * self.num_users

    def prior(self, item: int):
        theta = np.zeros(len(self.neighborhoods[item]) + 1)
        probability, weight = self.prior_probability[item], self.config.positive_weight
        theta[-1] = weight * probability / (1 + (weight - 1) * probability)
        return theta

    def design_rows(self, users, item: int, *, exclude_target: bool = True):
        """Explicit rows for exact positive corrections and independent audits."""
        users = np.asarray(users, dtype=np.int64)
        raw = self.interactions[users]
        seen = raw[:, item].toarray().ravel() if exclude_target else np.zeros(len(users))
        divisor = np.sqrt(np.maximum(self.degrees[users] - seen, 1))
        local = raw[:, self.neighborhoods[item]].toarray() / divisor[:, None]
        return np.column_stack((local, np.ones(len(users))))

    def fit(self, users, items, num_users: int, num_items: int,
            progress: Callable[[dict], None] | None = None):
        """Solve each item, using cached Gram blocks and exact positive-row replacement."""
        with threadpool_limits(limits=self.config.threads):
            return self._fit(users, items, num_users, num_items, progress)

    def _fit(self, users, items, num_users, num_items, progress):
        started = time.perf_counter()
        self._initialize(self.interaction_matrix(users, items, num_users, num_items))
        if progress:
            progress({"stage": "features_ready", "items": num_items, "seconds": time.perf_counter() - started})
        gram_linear = (self.history.T @ self.history).tocsr()
        sum_linear = np.asarray(self.history.sum(axis=0)).ravel()
        bias_column = np.ones((num_users, 1))
        gram_bias = bias_column.T @ bias_column
        positives = self.interactions.tocsc()
        self.bias = np.zeros(num_items)
        rows, cols, values = [], [], []
        residuals, objective = [], 0.
        for item, neighbors in enumerate(self.neighborhoods):
            local = len(neighbors)
            gram = np.zeros((local + 1,) * 2)
            gram[:local, :local] = gram_linear[neighbors][:, neighbors].toarray()
            block = sum_linear[neighbors][:, None]
            gram[:local, local:] = block
            gram[local:, :local] = block.T
            gram[local:, local:] = gram_bias
            data_rhs = np.zeros(len(gram))
            positive_users = positives.indices[positives.indptr[item]:positives.indptr[item + 1]]
            weight = self.config.positive_weight
            for offset in range(0, len(positive_users), self.config.correction_batch_size):
                batch = positive_users[offset:offset + self.config.correction_batch_size]
                original = self.design_rows(batch, item, exclude_target=False)
                excluded = self.design_rows(batch, item)
                gram += weight * (excluded.T @ excluded) - original.T @ original
                data_rhs += weight * excluded.sum(axis=0)
            precision, prior = self.precision(item), self.prior(item)
            hessian = (gram + gram.T) / 2 + precision
            rhs = data_rhs + precision @ prior
            # Unobserved items are prior-only fallbacks, as specified in the design.
            theta = cho_solve(cho_factor(hessian, lower=True, check_finite=True), rhs) if len(positive_users) else prior
            if not np.isfinite(theta).all():
                raise FloatingPointError(f"Nonfinite coefficients for item {item}")
            if len(positive_users):
                residual = np.linalg.norm(hessian @ theta - rhs) / max(np.linalg.norm(rhs), 1e-15)
                if residual > 1e-8:
                    raise FloatingPointError(f"Normal-equation residual {residual} for item {item}")
                residuals.append(float(residual))
                objective += .5 * (weight * len(positive_users) - 2 * theta @ data_rhs + theta @ gram @ theta
                                    + (theta - prior) @ precision @ (theta - prior))
            rows.extend(neighbors.tolist())
            cols.extend([item] * local)
            values.extend(theta[:local].tolist())
            self.bias[item] = theta[-1]
            if progress and ((item + 1) % 1000 == 0 or item + 1 == num_items):
                progress({"stage": "solving", "completed": item + 1, "items": num_items,
                          "seconds": time.perf_counter() - started})
        self.linear_coefficients = sparse.csr_matrix((values, (rows, cols)), shape=(num_items, num_items))
        self.fitted = True
        self.fit_report = {"solver": "cholesky_float64_cpu", "seconds": time.perf_counter() - started,
                           "max_relative_residual": max(residuals, default=0.),
                           "objective_observed_items": float(objective), "retained_edges": int(self.interactions.nnz),
                           "active_items": int(np.count_nonzero(self.counts)),
                           "prior_fallback_items": int(np.count_nonzero(self.counts == 0)),
                           "zero_support_mode": "prior_only_not_ridge_minimizer",
                           "adaptive_multiplier_max_active": float(self.support_multipliers(
                               np.flatnonzero(self.counts > 0)).max(initial=1.)),
                           "learned_coefficients": int(len(values) + num_items)}
        return self

    def score_parts(self, users, candidates, *, exclude_seen: bool = True):
        """Bounded Cartesian user/candidate diagnostics, with exact seen exclusion.

        Allocate only len(users)*len(candidates) outputs, never an extra U*I
        score matrix. Branches sum to score() on the same candidate entries.
        """
        if not self.fitted:
            raise RuntimeError("Fit or load SGKR before scoring")
        users, candidates = np.asarray(users), np.asarray(candidates)
        for indices, size in ((users, self.num_users), (candidates, self.num_items)):
            if indices.ndim != 1 or indices.dtype.kind not in "iu" or (
                    len(indices) and (indices.min() < 0 or indices.max() >= size)):
                raise ValueError("Invalid diagnostic indices")
        parts = {key: np.empty((len(users), len(candidates))) for key in ("linear", "bias")}
        for column, item in enumerate(candidates):
            neighbors = self.neighborhoods[item]
            local = len(neighbors)
            design = self.design_rows(users, int(item), exclude_target=exclude_seen)
            coefficients = self.linear_coefficients[neighbors, item].toarray().ravel()
            parts["linear"][:, column] = design[:, :local] @ coefficients
            parts["bias"][:, column] = self.bias[item]
        if any(not np.isfinite(value).all() for value in parts.values()):
            raise FloatingPointError("Nonfinite SGKR score parts")
        return parts

    def score(self, users, *, exclude_seen: bool = True):
        if not self.fitted:
            raise RuntimeError("Fit or load SGKR before scoring")
        users = np.asarray(users)
        if users.ndim != 1 or users.dtype.kind not in "iu" or (len(users) and (users.min() < 0 or users.max() >= self.num_users)):
            raise ValueError("Invalid user indices")
        scores = (self.history[users] @ self.linear_coefficients).toarray()
        scores += self.bias
        # Usually these entries are masked by Evaluator, but preserve exact score semantics.
        row, item = self.interactions[users].nonzero() if exclude_seen else (np.array([], dtype=int), np.array([], dtype=int))
        for candidate in np.unique(item):
            selected = row[item == candidate]
            theta = np.concatenate((self.linear_coefficients[self.neighborhoods[candidate], candidate].toarray().ravel(),
                                    [self.bias[candidate]]))
            scores[selected, candidate] = self.design_rows(users[selected], int(candidate)) @ theta
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite SGKR scores")
        return scores

    def get_user_rating_scores(self, users, all_users=None, all_items=None):
        """Adapter for the project's existing masked full-ranking Evaluator."""
        import torch

        with threadpool_limits(limits=self.config.threads):
            # Evaluator masks every retained positive (and the full train history).
            scores = self.score(users.detach().cpu().numpy(), exclude_seen=False)
        if self.config.popularity_tilt:
            counts = np.asarray(self.interactions.sum(0)).ravel()
            scores *= (1. + counts) ** self.config.popularity_tilt
        return torch.from_numpy(scores).to(users.device)

    def save(self, path):
        """Portable numeric checkpoint, no pickle or executable objects."""
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted model")
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, schema=np.array(2), config=np.array(json.dumps(asdict(self.config))),
                                report=np.array(json.dumps(self.fit_report)), shape=np.array(self.interactions.shape),
                                train_indptr=self.interactions.indptr, train_indices=self.interactions.indices,
                                linear_indptr=self.linear_coefficients.indptr, linear_indices=self.linear_coefficients.indices,
                                linear_data=self.linear_coefficients.data, bias=self.bias)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            schema = int(state["schema"])
            if schema not in (1, 2):
                raise ValueError("Unsupported SGKR checkpoint schema")
            # Schema 1 also stored a kernel block; only an empty one is compatible with this model.
            if schema == 1 and state["kernel"].size:
                raise ValueError("Checkpoint uses the retired SGKR kernel branch")
            model = cls(SGKRConfig.from_dict(json.loads(str(state["config"]))))
            shape = tuple(int(n) for n in state["shape"])
            interactions = sparse.csr_matrix((np.ones(len(state["train_indices"])), state["train_indices"],
                                              state["train_indptr"]), shape=shape)
            with threadpool_limits(limits=model.config.threads):
                model._initialize(interactions)
            model.linear_coefficients = sparse.csr_matrix((state["linear_data"].copy(), state["linear_indices"].copy(),
                                                           state["linear_indptr"].copy()), shape=(shape[1], shape[1]))
            model.bias = state["bias"].copy()
            model.fit_report = json.loads(str(state["report"]))
            model.fitted = True
            return model
