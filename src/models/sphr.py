"""SPHR: users are their normalized retained history over diffusion-filtered item tables.

score(u, i) = q_u . V~_i + b_i, with q_u = n_u^(-1/2) sum_{j in H_u} S~_j and X~ = (I/2 + T/4 + T^2/4) X,
T = D_I^-1 X^T D_U^-1 X. Trained with Adam on a full-catalog softmax over leave-one-out contexts.
The defaults are the locked configuration (configs/sphr.yaml, arm D2).
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.data.graph import sparse_matmul, to_csr
from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)

# Options of earlier experimental arms. An older config or checkpoint is accepted only when these
# hold the values that make it the locked model; options that were inert there are dropped.
RETIRED_OPTIONS = {"user_residual": False, "history_weight": 1.0, "context_mode": "shared-loo", "diffusion_mix": 1.0,
                   "history_tying": "untied", "item_regularization_mass": "positive-exposure",
                   "item_parameterization": "filtered", "graph_regularization": 0.0, "tying_regularization": 0.0,
                   "ranking_loss": "full-softmax", "item_bias": True, "uniformity_weight": 0.0, "history_dropout": 0.0}
IGNORED_OPTIONS = ("tau", "residual_mode", "user_regularization", "user_radius", "user_regularization_space",
                   "user_radius_space", "ranking_topk", "tail_smoothing", "uniformity_temperature")


@dataclass(frozen=True)
class SPHRConfig:
    embedding_dim: int = 64
    learning_rate: float = 0.0003
    # Ridge 0.5*lambda*sum_i p_i (|V_i|^2 + |S_i|^2)/2 with positive-exposure mass p (sum p = 1).
    item_regularization: float = 0.1
    item_bias_regularization: float = 1e-4
    item_radius: float = 100.0
    history_normalization: float = 0.5
    singleton_history_weight: float = 0.0
    ranking_temperature: float = 1.0

    @classmethod
    def from_dict(cls, values: dict):
        """Accept older configs/checkpoints only when their retired options give the locked model."""
        values = dict(values)
        for key, required in RETIRED_OPTIONS.items():
            if key in values and values.pop(key) != required:
                raise ValueError(f"{key} belongs to a retired SPHR variant; only {required!r} is supported")
        for key in IGNORED_OPTIONS:
            values.pop(key, None)
        return cls(**values)

    def __post_init__(self):
        if isinstance(self.embedding_dim, bool) or not isinstance(self.embedding_dim, int) or self.embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        for name in ("learning_rate", "item_radius", "ranking_temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("item_regularization", "item_bias_regularization", "singleton_history_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if (isinstance(self.history_normalization, bool) or not math.isfinite(self.history_normalization)
                or not 0 <= self.history_normalization <= 1):
            raise ValueError("history_normalization must be finite and in [0, 1]")


class SPHR(nn.Module):
    scoring_metric = "sphr_context_dot_product"

    def __init__(self, num_users: int, num_items: int, config: SPHRConfig | None = None):
        super().__init__()
        if num_users < 1 or num_items < 1:
            raise ValueError("Catalog dimensions must be positive")
        self.num_users, self.num_items = num_users, num_items
        self.config = config or SPHRConfig()
        # Target table V and source (history) table S, both filtered before use; item bias b.
        self.item_embedding = nn.Embedding(num_items, self.config.embedding_dim)
        nn.init.normal_(self.item_embedding.weight, std=1 / math.sqrt(self.config.embedding_dim))
        self.history_embedding = nn.Embedding.from_pretrained(self.item_embedding.weight.detach().clone(), freeze=False)
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.register_buffer("user_degree", torch.zeros(num_users))
        self.register_buffer("item_degree", torch.zeros(num_items))
        self.register_buffer("positive_exposure", torch.zeros(num_items), persistent=False)
        self.register_buffer("history_offsets", torch.zeros(num_users + 1, dtype=torch.long))
        self.register_buffer("history_items", torch.empty(0, dtype=torch.long))
        self.register_buffer("edge_keys", torch.empty(0, dtype=torch.long))
        for name, shape in (("interaction", (num_users, num_items)), ("reverse_interaction", (num_items, num_users))):
            self.register_buffer(name, torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), shape).coalesce())
        self.project_parameters()

    def set_history(self, users, items):
        device = self.item_embedding.weight.device
        users = users.to(device) if isinstance(users, torch.Tensor) else torch.tensor(users, device=device)
        items = items.to(device) if isinstance(items, torch.Tensor) else torch.tensor(items, device=device)
        if users.ndim != 1 or users.shape != items.shape or not users.numel():
            raise ValueError("History must contain matching nonempty ID arrays")
        if users.dtype not in (torch.int32, torch.int64) or items.dtype not in (torch.int32, torch.int64):
            raise ValueError("History IDs must be integers")
        if users.min() < 0 or users.max() >= self.num_users or items.min() < 0 or items.max() >= self.num_items:
            raise ValueError("History IDs out of bounds")
        keys = (users.long() * self.num_items + items.long()).sort().values
        if keys.unique().numel() != keys.numel():
            raise ValueError("Duplicate retained edges")
        self.edge_keys = keys
        users, items = keys // self.num_items, keys % self.num_items
        self.user_degree = torch.bincount(users, minlength=self.num_users).to(self.item_embedding.weight.dtype)
        self.item_degree = torch.bincount(items, minlength=self.num_items).to(self.item_embedding.weight.dtype)
        counts = self.user_degree.long()
        self.history_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        self.history_items = items
        self.interaction = torch.sparse_coo_tensor(torch.stack((users, items)), self.user_degree.new_ones(len(keys)),
                                                  (self.num_users, self.num_items), device=device).coalesce()
        self.reverse_interaction = self.interaction.transpose(0, 1).coalesce()
        self._refresh_exposure()

    def _refresh_exposure(self):
        self.positive_exposure = torch.zeros_like(self.item_degree)
        users, items = self.edge_keys // self.num_items, self.edge_keys % self.num_items
        self.positive_exposure.scatter_add_(0, items, self.user_degree[users].reciprocal())
        self.positive_exposure /= (self.user_degree > 0).sum().clamp_min(1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Edge count is data-dependent, while catalog/embedding dimensions are fixed.
        for name in ("history_items", "edge_keys"):
            if prefix + name in state_dict:
                setattr(self, name, torch.empty_like(state_dict[prefix + name], device=self.user_degree.device))
        # Checkpoints of the earlier code carry an unused user table.
        state_dict.pop(prefix + "user_embedding.weight", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._csr_cache = None  # loading may copy into the existing buffers in place
        self._refresh_exposure()

    def _csr_interactions(self):
        """CSR copies of the (persistent, COO) interaction buffers, rebuilt whenever a buffer is replaced."""
        source = (self.interaction, self.reverse_interaction)
        cached = getattr(self, "_csr_cache", None)
        if cached is None or cached[0][0] is not source[0] or cached[0][1] is not source[1]:
            cached = (source, (to_csr(source[0]), to_csr(source[1])))
            self._csr_cache = cached
        return cached[1]

    def apply_transition(self, item_vectors):
        forward, reverse = self._csr_interactions()
        means = sparse_matmul(forward, item_vectors, reverse) / self.user_degree.clamp_min(1)[:, None]
        output = sparse_matmul(reverse, means, forward) / self.item_degree.clamp_min(1)[:, None]
        # Isolated catalog items have identity fallback and are excluded from warm ranking.
        return torch.where(self.item_degree[:, None] > 0, output, item_vectors)

    def _filter(self, base):
        first = self.apply_transition(base)
        return .5 * base + .25 * first + .25 * self.apply_transition(first)

    def filtered_items(self):
        if not self.edge_keys.numel():
            raise RuntimeError("Set retained history before forwarding SPHR")
        return self._filter(self.item_embedding.weight)

    def history_vectors(self):
        return self._filter(self.history_embedding.weight)

    def history_scale(self, users):
        # Always use full retained degree, including during positive exclusion.
        return torch.where(self.user_degree[users] == 1, self.config.singleton_history_weight, 1.0)

    def _history_denominator(self, counts):
        counts = counts.to(dtype=self.item_embedding.weight.dtype)
        return counts.clamp_min(1).pow(self.config.history_normalization)[:, None]

    def _retained_history_sum(self, users, source):
        selected, inverse = users.unique(return_inverse=True)
        counts = self.user_degree[selected].long()
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        rows = torch.repeat_interleave(torch.arange(len(selected), device=users.device), counts)
        positions = torch.arange(rows.numel(), device=users.device)-offsets[rows]+self.history_offsets[selected[rows]]
        pooled = F.embedding_bag(self.history_items[positions], source, offsets, mode="sum", include_last_offset=True)
        return pooled[inverse]

    def context_embeddings(self, users, positives):
        """Leave-one-out training context: the user's retained history without the positive."""
        if users.ndim != 1 or positives.shape != users.shape:
            raise ValueError("Context IDs must be matching vectors")
        if users.dtype not in (torch.int32, torch.int64) or positives.dtype not in (torch.int32, torch.int64):
            raise ValueError("Context IDs must be integers")
        if users.device != self.edge_keys.device or positives.device != users.device or bool(
                ((users < 0) | (users >= self.num_users) | (positives < 0) | (positives >= self.num_items)).any()):
            raise ValueError("Context IDs out of bounds or on the wrong device")
        if not self.edge_keys.numel():
            raise RuntimeError("Set retained history before constructing contexts")
        keys = users.long() * self.num_items + positives.long()
        positions = torch.searchsorted(self.edge_keys, keys)
        if not bool((self.edge_keys[positions.clamp_max(len(self.edge_keys) - 1)] == keys).all()):
            raise ValueError("Positive must belong to retained history")
        source = self.history_vectors()
        degree = self.user_degree[users]
        history = (self._retained_history_sum(users, source)-source[positives])/self._history_denominator(degree-1)
        return self.history_scale(users)[:, None] * history

    def forward(self):
        """Inference representations: full retained history per user, filtered target items."""
        items = self.filtered_items()
        users = torch.arange(self.num_users, device=items.device)
        forward, reverse = self._csr_interactions()
        pooled = sparse_matmul(forward, self.history_vectors(), reverse)/self._history_denominator(self.user_degree)
        return self.history_scale(users)[:, None] * pooled, items

    def regularization_components(self):
        squared = (self.item_embedding.weight.square().sum(-1)+self.history_embedding.weight.square().sum(-1))/2
        item = .5 * self.config.item_regularization * (self.positive_exposure * squared).sum()
        bias = .5 * self.config.item_bias_regularization * self.item_bias.square().sum()
        return {"item_ridge": item, "bias_penalty": bias}

    def full_softmax_values(self, users, positives):
        """Per-query cross-entropy over the active catalog, -log softmax_{i+}(s(u, .)/T).

        The context is the leave-one-out context of the positive. The user's other retained
        positives and items without retained edges are excluded from the denominator,
        matching the Evaluator's train mask and candidate set.
        """
        items = self.filtered_items()
        context = self.context_embeddings(users, positives)
        logits = context @ items.T
        logits = logits + self.item_bias
        logits = logits / self.config.ranking_temperature
        counts = self.user_degree[users].long()
        rows = torch.repeat_interleave(torch.arange(len(users), device=users.device), counts)
        starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
        positions = torch.arange(rows.numel(), device=users.device) - starts[rows] + self.history_offsets[users[rows]]
        seen = self.history_items[positions]
        keep = seen != positives[rows]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[rows[keep], seen[keep]] = True
        mask[:, self.item_degree == 0] = True
        return F.cross_entropy(logits.masked_fill(mask, float("-inf")), positives, reduction="none")

    def ranking_loss(self, users, positives):
        ranking = self.full_softmax_values(users, positives).mean()
        parts = self.regularization_components()
        penalty = parts["item_ridge"]+parts["bias_penalty"]
        return ranking+penalty, {**parts, "ranking": ranking, "penalty": penalty}

    def constrained_tables(self):
        return self.item_embedding.weight, self.history_embedding.weight

    def parameter_radii(self):
        return self.config.item_radius, self.config.item_radius

    @torch.no_grad()
    def project_parameters(self):
        fractions = []
        for table, radius in zip(self.constrained_tables(), self.parameter_radii()):
            lengths = table.norm(dim=1, keepdim=True)
            fractions.append(float((lengths > radius).float().mean()))
            table.mul_((radius / lengths.clamp_min(1e-12)).clamp_max(1))
        return fractions

    @torch.no_grad()
    def diagnostics(self, item_vectors):
        """Once-per-epoch geometry of the filtered item table."""
        from src.evaluation.theory_diagnostics import geometry_diagnostics
        return geometry_diagnostics(item_vectors)

    def get_user_rating_scores(self, users, all_users, all_items):
        # This adapter is for unseen candidates; Evaluator masks all known positives.
        return all_users[users] @ all_items.T + self.item_bias
