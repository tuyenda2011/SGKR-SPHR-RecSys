"""Scoring back end of the demo app: load the trained models of one density and rank items.

Reads only train + validation (scripts/validation_io.py) and the item metadata; never the test split.
Rankings follow the Evaluator: candidates are the items of the density's training graph and the user's
full-train history is masked, so a demo top-20 equals the run's stored val_predictions.npy row
(except EASE, whose checkpoint keeps only the top-128 weights per item of the full matrix).
SGKR, SPHR, EASE and GF-CF score any item history directly (no user table), which is what the "new user" tab shows.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401 -- load Arrow DLLs before Torch on Windows
import torch

ROOT = Path(__file__).resolve().parents[1]

MODELS = ("sgkr", "sphr", "lightgcn", "xsimgcl", "directau", "ease", "gfcf")
PROPOSED = ("sgkr", "sphr")
LABELS = {"sgkr": "SGKR", "sphr": "SPHR", "lightgcn": "LightGCN", "xsimgcl": "XSimGCL", "directau": "DirectAU",
          "ease": "EASE (top-128)", "gfcf": "GF-CF"}
COLORS = {"sgkr": "#2a78d6", "sphr": "#eb6834", "lightgcn": "#1baf7a", "xsimgcl": "#eda100", "directau": "#e87ba4",
          "ease": "#7a5cc7", "gfcf": "#8c6d46"}
# Display rows of the demo: (heading, card subtitle, models).
GROUPS = (("Mô hình đề xuất", "Đề xuất", ("sgkr", "sphr")),
          ("Baseline đồ thị · học embedding", "Baseline · GNN", ("lightgcn", "xsimgcl", "directau")),
          ("Baseline closed-form · cùng họ SGKR", "Baseline · closed-form", ("ease", "gfcf")))
KINDS = {model: kind for _, kind, models in GROUPS for model in models}
# Accent per family (badges, table text) with a light tint for backgrounds.
FAMILY_COLORS = {"Đề xuất": ("#1d63c2", "#e8f0fc"), "Baseline · GNN": ("#13865d", "#e6f5ee"),
                 "Baseline · closed-form": ("#6a4bbd", "#f0ecfa")}
DENSITIES = (1.0, 0.75, 0.5, 0.25)
SEED = 42
# DEMO_RESULTS points the demo at another results tree, e.g. results\preview (1-epoch checkpoints).
PREVIEW = os.environ.get("DEMO_RESULTS")
RESULTS = Path(PREVIEW) if PREVIEW else ROOT / "results"


def run_dir(model, density, results=None, seed=SEED):
    return Path(results or RESULTS) / model / f"d{round(density * 100)}" / f"seed{seed}"


def checkpoint_ndcg(model, density, results=None):
    """Validation NDCG@20 of the checkpoint the demo loads (EASE: its top-128 version), or None."""
    path = run_dir(model, density, results) / "metrics.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))
    return (run.get("checkpoint_val_metrics") or run["val_metrics"])["NDCG@20"]


@dataclass
class DemoData:
    full_train: pd.DataFrame
    validation: pd.DataFrame
    num_users: int
    num_items: int
    metadata: pd.DataFrame  # indexed by i_idx: title, brand, leaf, has_title, labeled
    history: dict  # user -> sorted array of full-train items (the Evaluator's mask)
    target: dict  # user -> validation item


def load_metadata(path, num_items):
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=["i_idx", "title", "brand", "category",
                                                             "has_title", "has_brand"])
    frame = frame.set_index("i_idx").reindex(range(num_items))
    frame["has_title"] = frame.has_title.fillna(False).astype(bool)
    frame["has_brand"] = frame.has_brand.fillna(False).astype(bool)
    frame["labeled"] = frame.has_title & frame.has_brand
    frame["leaf"] = frame.category.fillna("").str.split(" > ").str[-1]
    return frame


def load_data(root=ROOT):
    from scripts.validation_io import load_validation_inputs
    _, full_train, validation, stats = load_validation_inputs(None, Path(root))
    metadata = load_metadata(Path(root) / "data/processed/csv/metadata.csv", stats["num_items"])
    return make_data(full_train, validation, stats["num_users"], stats["num_items"], metadata)


def make_data(full_train, validation, num_users, num_items, metadata):
    history = {int(u): np.sort(g.to_numpy()) for u, g in full_train.groupby("u_idx").i_idx}
    target = dict(zip(validation.u_idx.astype(int), validation.i_idx.astype(int)))
    return DemoData(full_train, validation, num_users, num_items, metadata, history, target)


def sample_users(data, per_group=3, seed=SEED):
    """Demo users by a fixed public rule: fully labeled history and target, a few per history-length group."""
    labeled = data.metadata.labeled.to_numpy()
    rng = np.random.default_rng(seed)
    groups = {"3–4": (3, 4), "5–9": (5, 9), "≥10": (10, 10 ** 9)}
    chosen = []
    eligible = [u for u, items in data.history.items()
                if u in data.target and labeled[items].all() and labeled[data.target[u]]]
    for name, (low, high) in groups.items():
        members = sorted(u for u in eligible if low <= len(data.history[u]) <= high)
        if members:
            chosen += [(name, int(u)) for u in rng.choice(members, min(per_group, len(members)), replace=False)]
    return chosen, eligible


class Ranker:
    """One trained model at one density: scores existing users and, for SGKR/SPHR, any item history."""

    def __init__(self, model, candidates, score_users, score_history=None, explain=None, train_items=None):
        self.model, self.candidates = model, candidates
        self._score_users, self._score_history, self._explain = score_users, score_history, explain
        # The user's history inside this density's training graph (what SGKR's features see).
        self.train_items = train_items

    @property
    def supports_new_users(self):
        return self._score_history is not None

    def score_users(self, users):
        return self._score_users(np.asarray(users, dtype=np.int64))

    def score_history(self, items):
        if self._score_history is None:
            raise NotImplementedError(f"{LABELS[self.model]} needs a learned user vector")
        return self._score_history(np.unique(np.asarray(items, dtype=np.int64)))

    def explain(self, items, target, k=3):
        """SGKR only: the history items with the largest contribution x_j * W[j, target]."""
        if self._explain is None:
            return []
        return self._explain(np.unique(np.asarray(items, dtype=np.int64)), int(target))[:k]

    def top_k(self, scores, exclude, k=10, allowed=None):
        """Evaluator semantics (candidate mask, history mask, torch.topk) plus an optional display filter."""
        scores = torch.as_tensor(np.asarray(scores, dtype=np.float64)).clone()
        scores[torch.from_numpy(~self.candidates)] = float("-inf")
        if len(exclude):
            scores[torch.as_tensor(np.asarray(exclude, dtype=np.int64))] = float("-inf")
        if allowed is not None:
            scores[torch.from_numpy(~np.asarray(allowed, dtype=bool))] = float("-inf")
        values, items = torch.topk(scores, k)
        return items[torch.isfinite(values)].numpy()


def sgkr_ranker(model, candidates):
    weights = model.linear_coefficients.tocsc()
    tilt = (1. + model.counts) ** model.config.popularity_tilt if model.config.popularity_tilt else None

    def users(ids):
        return model.get_user_rating_scores(torch.from_numpy(ids)).numpy()

    def history(items):
        # Same features as a training row: each history item weighted 1/sqrt(|history|).
        x = np.zeros(model.num_items)
        x[items] = 1 / np.sqrt(len(items))
        scores = weights.T @ x + model.bias
        return scores * tilt if tilt is not None else scores

    def explain(items, target):
        column = weights[:, target].toarray().ravel()
        contribution = column[items] / np.sqrt(len(items))
        order = np.argsort(-contribution, kind="stable")
        return [(int(items[i]), float(contribution[i])) for i in order if contribution[i] > 0]

    def train_items(user):
        return model.interactions[user].indices.astype(np.int64)

    return Ranker("sgkr", candidates, users, history, explain, train_items)


@torch.no_grad()
def sphr_ranker(model, candidates):
    user_table, items = model()
    sources, bias = model.history_vectors(), model.item_bias.detach()

    def users(ids):
        return (user_table[torch.from_numpy(ids)] @ items.T + bias).double().numpy()

    def history(ids):
        # forward(): pooled filtered history / |H|^normalization, singleton weight for one item.
        pooled = sources[torch.from_numpy(ids)].sum(0) / len(ids) ** model.config.history_normalization
        scale = model.config.singleton_history_weight if len(ids) == 1 else 1.
        return ((scale * pooled) @ items.T + bias).double().numpy()

    return Ranker("sphr", candidates, users, history)


def ease_ranker(weights, interactions, candidates):
    """x @ B with the pruned EASE matrix; x is the user's history in this density's training graph."""
    weights = weights.tocsr()

    def users(ids):
        return (interactions[ids] @ weights).toarray()

    def history(items):
        x = np.zeros(weights.shape[0])
        x[items] = 1.
        return weights.T @ x

    def train_items(user):
        return interactions[user].indices.astype(np.int64)

    return Ranker("ease", candidates, users, history, train_items=train_items)


def gfcf_ranker(model, candidates):
    def users(ids):
        return model.score(ids)

    def history(items):
        x = np.zeros((1, model.interactions.shape[1]))
        x[0, items] = 1.
        return model.score_rows(x)[0]

    def train_items(user):
        return model.interactions[user].indices.astype(np.int64)

    return Ranker("gfcf", candidates, users, history, train_items=train_items)


@torch.no_grad()
def baseline_ranker(name, model, adjacency, candidates):
    user_table, item_table = model(adjacency)

    @torch.no_grad()
    def users(ids):
        return model.get_user_rating_scores(torch.from_numpy(ids), user_table, item_table).double().numpy()

    return Ranker(name, candidates, users)


def candidate_mask(train, num_items):
    mask = np.zeros(num_items, dtype=bool)
    mask[train.i_idx.unique()] = True
    return mask


def load_ranker(model, density, data, results=None):
    """The run's checkpoint at this density, or None when it has not been trained yet."""
    from src.data.sparsity import create_sparse_train_set
    folder = run_dir(model, density, results)
    checkpoint = folder / ("checkpoint.npz" if model in ("sgkr", "ease", "gfcf") else "checkpoint.pt")
    if not checkpoint.exists():
        return None
    train = create_sparse_train_set(data.full_train, density, SEED)
    candidates = candidate_mask(train, data.num_items)
    if model == "gfcf":
        from src.models.gfcf import GFCF
        return gfcf_ranker(GFCF.load(checkpoint), candidates)
    if model == "ease":
        from scipy import sparse
        from src.models.sgkr import SGKR
        interactions = SGKR.interaction_matrix(train.u_idx.to_numpy(), train.i_idx.to_numpy(),
                                               data.num_users, data.num_items)
        return ease_ranker(sparse.load_npz(checkpoint), interactions, candidates)
    if model == "sgkr":
        from src.models.sgkr import SGKR
        return sgkr_ranker(SGKR.load(checkpoint), candidates)
    if model == "sphr":
        from src.models.sphr import SPHR, SPHRConfig
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        network = SPHR(data.num_users, data.num_items, SPHRConfig.from_dict(state["config"]))
        network.set_history(train.u_idx.to_numpy(), train.i_idx.to_numpy())
        network.load_state_dict(state["model_state_dict"])
        return sphr_ranker(network.eval(), candidates)
    from scripts.rescore_neural_checkpoints import build_model
    from src.data.graph import get_norm_adj_tensor
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    network = build_model(model, state["config"], data.num_users, data.num_items)
    network.load_state_dict(state["model_state_dict"])
    adjacency = get_norm_adj_tensor(train, data.num_users, data.num_items, torch.device("cpu"))
    return baseline_ranker(model, network.eval(), adjacency, candidates)


def recommend(ranker, data, user=None, items=None, k=10, hide_untitled=True):
    """Top-k for an existing user (mask = full-train history) or a new history; returns items and ms."""
    started = time.perf_counter()
    if user is not None:
        scores, exclude = ranker.score_users([user])[0], data.history.get(user, np.array([], dtype=np.int64))
    else:
        scores, exclude = ranker.score_history(items), np.asarray(items, dtype=np.int64)
    allowed = data.metadata.has_title.to_numpy() if hide_untitled else None
    top = ranker.top_k(scores, exclude, k, allowed)
    return top, (time.perf_counter() - started) * 1000
