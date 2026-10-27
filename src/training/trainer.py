import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Force UTF-8 encoding for Windows Command Prompt/PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.data.graph import get_norm_adj_tensor, to_csr
from src.evaluation.evaluator import Evaluator
from src.evaluation.representation import (
    compute_alignment_and_uniformity,
    compute_svd_spectrum,
)
from src.evaluation.subgroup import evaluate_degree_subgroups
from src.models.base import BaseRecommender
from src.training.early_stopping import (
    EarlyStopping,
    load_checkpoint,
    restore_training_rng,
    save_checkpoint,
)
from src.training.loss_strategies import get_loss_strategy
from src.training.validation_lr import build_validation_scheduler
from src.utils.checkpoints import get_model_output_dir

logger = logging.getLogger(__name__)

# Prefer the current AMP API while retaining compatibility with older PyTorch.
try:
    from torch.amp import GradScaler
    MODERN_AMP_API = True
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler
        MODERN_AMP_API = False
        AMP_AVAILABLE = True
    except ImportError:
        MODERN_AMP_API = False
        AMP_AVAILABLE = False


# =============================================================================
# Constants for training configuration
# =============================================================================
MASK_VALUE: float = -1e9  # Mask value for filtering seen items
GRADIENT_CLIP_VALUE: float = 1.0  # Gradient clipping max norm
SYNC_CUDA: bool = False  # Whether to synchronize CUDA after each epoch (for accurate timing)

LOSS_COMPONENT_KEYS = (
    "path_fit_raw", "path_trajectory_raw", "path_aux_weighted",
    "graph_aux_fit_raw", "graph_aux_trajectory_raw", "graph_aux_weighted",
    "flow_raw", "flow_weighted", "propagated_l2_raw", "propagated_l2_weighted",
    "flow_rank_raw", "flow_rank_weighted", "residual_anchor_raw", "residual_anchor_weighted",
    "bpr_raw", "id_l2_raw", "id_l2_weighted",
    "ssl_raw", "ssl_weight", "ssl_weighted",
    "hard_penalty_raw", "hard_margin_raw", "hard_penalty_weighted",
    "dirichlet_raw", "dirichlet_weighted",
    "alignment_raw", "alignment_weighted",
    "uniformity_raw", "uniformity_weight", "uniformity_weighted",
)


def sample_negative_items(
    users: np.ndarray,
    num_items: int,
    train_history: Dict[int, Set[int]],
    max_attempts: int = 100,
) -> np.ndarray:
    """Fast uniform random negative sampling with vectorized collision rejection.

    Optimized implementation that batches operations and minimizes Python loops.

    Args:
        users: Array of user indices
        num_items: Total number of items
        train_history: Dictionary mapping user to set of positive items
        max_attempts: Maximum retry attempts per negative sample to prevent infinite loops

    Returns:
        Array of negative item indices
    """
    if num_items <= 0 or max_attempts < 0:
        raise ValueError("num_items must be positive and max_attempts nonnegative")
    n = len(users)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    neg_items = np.random.randint(0, num_items, size=n)

    # Pre-compute history sets for faster lookup
    user_histories = [train_history.get(u, set()) for u in users]

    # Vectorized collision detection using numpy
    collisions = np.array([
        neg_items[i] in user_histories[i]
        for i in range(n)
    ], dtype=bool)

    # Retry collisions with limit
    attempts: np.ndarray = np.zeros(n, dtype=np.int32)
    while True:
        retry_mask = collisions & (attempts < max_attempts)
        if not retry_mask.any():
            break
        neg_items[retry_mask] = np.random.randint(0, num_items, size=retry_mask.sum())

        # Update collision status
        for i in np.where(retry_mask)[0]:
            attempts[i] += 1
            collisions[i] = neg_items[i] in user_histories[i]

    # Resolve only exhausted collisions, sharing the complement per user.
    for user in np.unique(users[collisions]):
        remaining = np.flatnonzero(collisions & (users == user))
        seen = np.fromiter(train_history.get(int(user), set()), dtype=np.int64)
        candidates = np.setdiff1d(np.arange(num_items), seen, assume_unique=True)
        if candidates.size == 0:
            raise ValueError(f"User {int(user)} has no valid negative items")
        neg_items[remaining] = np.random.choice(candidates, size=remaining.size)

    return neg_items


def sample_hard_negative_items(
    users: np.ndarray,
    user_disliked_items: Dict[int, List[int]],
    train_history: Dict[int, Set[int]],
    num_items: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample one valid explicit dislike per user when available."""
    if len(users) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)

    selected_by_user: np.ndarray = np.full(int(users.max()) + 1, -1, dtype=np.int64)
    for user in np.unique(users):
        positives = train_history.get(int(user), set())
        candidates = [
            item
            for item in user_disliked_items.get(int(user), [])
            if 0 <= item < num_items and item not in positives
        ]
        if candidates:
            selected_by_user[int(user)] = candidates[np.random.randint(len(candidates))]

    hard_items = selected_by_user[users.astype(np.int64)]
    available = hard_items >= 0
    return hard_items, available


class Trainer:
    """Trainer for the four recommendation models used in this project."""

    def __init__(
        self,
        model: BaseRecommender,
        train_df: pd.DataFrame,
        val_evaluator: Evaluator,
        test_evaluator: Evaluator | None,
        config: Dict[str, Any],
        device: torch.device,
        user_disliked_items: Optional[Dict[int, List[int]]] = None,
        subgroup_reference_df: Optional[pd.DataFrame] = None,
    ):
        self.model = model.to(device)
        self.train_df = train_df
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        # Validation, final test, and subgroups must share the model's scorer.
        self.val_evaluator.score_fn = self.model.get_user_rating_scores
        if self.test_evaluator is not None:
            self.test_evaluator.score_fn = self.model.get_user_rating_scores
        self.config = config
        self.device = device
        self.user_disliked_items = user_disliked_items or {}
        self.subgroup_reference_df = (
            subgroup_reference_df if subgroup_reference_df is not None else train_df
        )

        self.model_name = config["model_name"]
        if self.test_evaluator is None and not config.get("validation_only", False):
            raise ValueError("A test evaluator is required unless validation_only is enabled")
        self.num_users = model.num_users
        self.num_items = model.num_items

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.monitor = config.get("evaluation", {}).get("monitor", "NDCG@10")
        if self.monitor not in {f"NDCG@{k}" for k in self.val_evaluator.k_list}:
            raise ValueError("Monitor must be NDCG at an evaluated cutoff")
        self.batch_size = train_cfg["batch_size"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        eval_cfg = config.get("evaluation", {})
        self.diagnostics_enabled = bool(
            eval_cfg.get("model_diagnostics", False)
            and hasattr(self.model, "collect_diagnostics")
        )
        diagnostics_sample_size = max(1, int(eval_cfg.get("diagnostics_sample_size", 4096)))
        self.diagnostic_item_indices = torch.linspace(
            0, max(0, self.num_items - 1), steps=min(self.num_items, diagnostics_sample_size),
            device=device, dtype=torch.long,
        ).unique()
        self.diagnostic_user_indices = torch.linspace(
            0, max(0, self.num_users - 1), steps=min(self.num_users, diagnostics_sample_size),
            device=device, dtype=torch.long,
        ).unique()

        parameters = (parameter for parameter in self.model.parameters() if parameter.requires_grad)
        if self.model_name == "directau" and config.get("directau", {}).get("profile", "project_cosine") != model.profile:
            raise ValueError("DirectAU model and config profiles differ")
        reference_directau = self.model_name == "directau" and getattr(model, "profile", "project_cosine") == "reference_lgcn"
        optimizer_decay = config.get("directau", {}).get("optimizer_weight_decay", 1e-6) if reference_directau else 0.0
        self.optimizer_weight_decay = float(optimizer_decay)
        self.optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=optimizer_decay)
        self.validation_lr_scheduler = build_validation_scheduler(
            self.optimizer, config.get("validation_lr_scheduler")
        )

        self.loss_strategy = get_loss_strategy(self.model_name, config)

        # Precompute train history mapping for negative sampling
        self.train_history: Dict[int, Set[int]] = (
            train_df.groupby("u_idx")["i_idx"].apply(set).to_dict()
        )

        # Precompute base graph normalized adjacency tensor (CSR: symmetric, so it is its own transpose)
        self.norm_adj = to_csr(get_norm_adj_tensor(
            train_df, self.num_users, self.num_items, device
        ))

        # Training positive edge index for Alignment/Uniformity computation
        self.train_edge_index = torch.tensor(
            np.stack([train_df["u_idx"].values, train_df["i_idx"].values], axis=0),
            dtype=torch.long,
            device=device,
        )

        # Early stopping handler
        self.early_stopping = EarlyStopping(
            patience=train_cfg["early_stopping_patience"],
            monitor=self.monitor,
            mode="max",
        )

        # Mixed precision training (AMP)
        self.use_amp = AMP_AVAILABLE and device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler("cuda") if MODERN_AMP_API else GradScaler()
            logger.info("AMP gradient scaling enabled for CUDA device")
        else:
            self.scaler = None

    def _clip_gradients(self):
        groups = self.model.gradient_parameter_groups() if hasattr(self.model, "gradient_parameter_groups") else [self.model.parameters()]
        norms = [torch.nn.utils.clip_grad_norm_(group, max_norm=GRADIENT_CLIP_VALUE, error_if_nonfinite=True)
                 for group in groups]
        # For grouped clipping, report the largest group norm: >limit means
        # at least one group was clipped, preserving clip-fraction semantics.
        return max(float(norm) for norm in norms)

    def _backward_and_step(self, total_loss: torch.Tensor) -> None:
        """Backpropagate, clip gradients, and update model parameters."""
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite training loss; refusing to update parameters")
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            # Gradients must be unscaled before their norm is clipped.
            self.scaler.unscale_(self.optimizer)
            grad_norm = self._clip_gradients()
            self.last_gradient_norm = float(grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_norm = self._clip_gradients()
            self.last_gradient_norm = float(grad_norm)
            self.optimizer.step()

        if hasattr(self.model, "after_optimizer_step"):
            self.model.after_optimizer_step()

    def _scoring_users(self, users, indices):
        if hasattr(self.model, "refine_users"):
            return self.model.refine_users(indices, users)
        return users[indices]

    def train(self, checkpoint_path: str, resume: bool = False) -> Dict[str, Any]:
        """Train with one metric summary per epoch and validation early stopping."""
        logger.info(
            f"Starting training for {self.model_name.upper()} on device {self.device}..."
        )

        latest_checkpoint_path = checkpoint_path.replace(".pt", "_latest.pt")
        start_epoch = 1
        best_val_metrics: Dict[str, float] = {}
        committed_state = {}
        resumed_checkpoint = None

        if resume and os.path.exists(latest_checkpoint_path):
            try:
                loaded_epoch, best_sc, ckpt = load_checkpoint(
                    latest_checkpoint_path,
                    self.model,
                    self.optimizer,
                    device=self.device,
                    expected_fingerprint=self.config.get("experiment_fingerprint"),
                )
                start_epoch = loaded_epoch + 1
                self.early_stopping.best_score = best_sc
                training_state = ckpt.get("training_state", {})
                if "history" not in training_state or "rng_state" not in ckpt:
                    raise ValueError("Checkpoint lacks complete resume state; start a fresh run")
                if ckpt.get("config", {}).get("evaluation", {}).get("monitor", "NDCG@10") != self.monitor:
                    raise ValueError("Resume monitor differs from checkpoint")
                if ckpt.get("config", {}).get("validation_lr_scheduler") != self.config.get("validation_lr_scheduler"):
                    raise ValueError("Resume learning-rate controller differs from checkpoint")
                scheduler_state = training_state.get("validation_lr_scheduler_state")
                if self.validation_lr_scheduler is not None:
                    if scheduler_state is None:
                        raise ValueError("Checkpoint lacks learning-rate controller state")
                    self.validation_lr_scheduler.load_state_dict(scheduler_state)
                elif scheduler_state is not None:
                    raise ValueError("Scheduler state supplied for disabled controller")
                committed_state = training_state
                resumed_checkpoint = ckpt
                if self.epochs < loaded_epoch:
                    raise ValueError("Requested epochs precede the checkpoint epoch")
                self.early_stopping.best_epoch = int(
                    training_state.get("best_epoch", loaded_epoch)
                )
                self.early_stopping.counter = int(
                    training_state.get("early_stopping_counter", 0)
                )
                self.early_stopping.early_stop = (
                    self.early_stopping.counter >= self.early_stopping.patience
                )
                best_val_metrics = dict(
                    training_state.get("best_val_metrics", ckpt.get("val_metrics", {}))
                )
                print(
                    f"[RESUME CHECKPOINT] Da khoi phuc trong so. Tiep tuc train tu Epoch {start_epoch:02d}/{self.epochs:02d} (Val {self.monitor} dinh hien tai: {best_sc:.4f})\n",
                    flush=True,
                )
            except Exception as ex:
                raise RuntimeError("Cannot resume incompatible checkpoint; use a fresh run") from ex

        user_array = self.train_df["u_idx"].values
        pos_item_array = self.train_df["i_idx"].values
        num_samples = len(user_array)

        start_train_time = time.perf_counter()

        best_epoch = self.early_stopping.best_epoch

        history_dir = self.config.get("history_dir") or get_model_output_dir("history", self.model_name)
        os.makedirs(history_dir, exist_ok=True)
        history_csv_name = os.path.basename(checkpoint_path).replace(".pt", "_history.csv")
        history_csv_path = os.path.join(history_dir, history_csv_name)

        history_records = list(committed_state.get("history", []))
        initial_diagnostics: Dict[str, float] = dict(
            committed_state.get("diagnostics_epoch0", {})
        )
        previous_train_time = float(committed_state.get("total_train_time", 0.0))
        previous_validation_time = float(committed_state.get("validation_time", 0.0))
        cumulative_train_time = previous_train_time
        best_model_state = committed_state.get("best_model_state")
        validation_time = 0.0
        if resumed_checkpoint is not None:
            if [row["epoch"] for row in history_records] != list(range(1, start_epoch)):
                raise ValueError("Checkpoint history does not match committed epochs")
            pd.DataFrame(history_records).to_csv(history_csv_path, index=False)

            if best_model_state is None:
                raise ValueError("Checkpoint lacks committed best model state")
            # Rebuild the best artifact from the same committed transaction.
            best_artifact = dict(resumed_checkpoint)
            best_artifact.update(model_state_dict=best_model_state, epoch=best_epoch,
                                 val_metrics=best_val_metrics)
            # Recovered best weights are for inference; resume uses latest only.
            for key in ("optimizer_state_dict", "rng_state", "scaler_state", "training_state"):
                best_artifact.pop(key, None)
            torch.save(best_artifact, checkpoint_path + ".tmp")
            os.replace(checkpoint_path + ".tmp", checkpoint_path)
            restore_training_rng(resumed_checkpoint, self.scaler)
        elif resume:
            raise FileNotFoundError(f"Resume checkpoint missing: {latest_checkpoint_path}")

        # Optional epoch-0 snapshot.  It is kept outside ``history`` so epoch
        # numbering and early-stopping semantics remain unchanged.
        if self.diagnostics_enabled and start_epoch == 1:
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                initial_user, initial_item = self.model(self.norm_adj)
                initial_diagnostics = self.model.collect_diagnostics(
                    self.diagnostic_item_indices, self.diagnostic_user_indices
                )
                initial_diagnostics["sample_user_effective_rank"] = float(
                    compute_svd_spectrum(self._scoring_users(initial_user, self.diagnostic_user_indices))["effective_rank"]
                )
                initial_diagnostics["sample_item_effective_rank"] = float(
                    compute_svd_spectrum(initial_item[self.diagnostic_item_indices])["effective_rank"]
                )
                initial_diagnostics["sample_user_count"] = float(self.diagnostic_user_indices.numel())
                initial_diagnostics["sample_item_count"] = float(self.diagnostic_item_indices.numel())
            if was_training:
                self.model.train()

        for epoch in range(start_epoch, self.epochs + 1):
            if self.early_stopping.early_stop:
                logger.info("Resume state had already reached early stopping; skipping training")
                break
            epoch_start = time.perf_counter()
            epoch_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            gradient_norms = []
            self.model.train()

            if hasattr(self.model, "on_epoch_start"):
                self.model.on_epoch_start(self.norm_adj)

            # 1. Random negative sampling per epoch (only needed if model uses negative sampling)
            if self.model_name != "directau":
                neg_item_array = sample_negative_items(
                    user_array, self.num_items, self.train_history
                )
            else:
                neg_item_array = None

            # Shuffle mini-batches
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            total_loss_accum = 0.0
            bpr_loss_accum = 0.0
            cl_loss_accum = 0.0
            num_batches = 0
            loss_component_accum = {key: 0.0 for key in LOSS_COMPONENT_KEYS}
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            for i in range(0, num_samples, self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                u_batch = torch.tensor(
                    user_array[batch_idx], dtype=torch.long, device=self.device
                )
                pos_batch = torch.tensor(
                    pos_item_array[batch_idx], dtype=torch.long, device=self.device
                )

                self.optimizer.zero_grad()

                neg_batch = None if neg_item_array is None else torch.tensor(
                    neg_item_array[batch_idx], dtype=torch.long, device=self.device
                )
                losses = self.loss_strategy.compute_loss(
                    self.model, self.norm_adj, u_batch, pos_batch, neg_batch, self.config
                )
                total_loss = losses.total_loss
                bpr_loss_accum += losses.bpr_loss
                cl_loss_accum += losses.cl_loss
                for key, value in losses.extra_losses.items():
                    if key in loss_component_accum:
                        loss_component_accum[key] += float(
                            value.detach().item() if isinstance(value, torch.Tensor) else value
                        )

                self._backward_and_step(total_loss)
                gradient_norms.append(self.last_gradient_norm)

                total_loss_accum += total_loss.item()
                num_batches += 1

            if SYNC_CUDA and self.device.type == "cuda":
                torch.cuda.synchronize()

            epoch_time = time.perf_counter() - epoch_start

            validation_start = time.perf_counter()
            # Evaluation on Validation set every epoch
            self.model.eval()
            with torch.no_grad():
                val_u_embeds, val_i_embeds = self.model(self.norm_adj)
                val_metrics, _ = self.val_evaluator.evaluate(
                    val_u_embeds, val_i_embeds, self.device, include_beyond_accuracy=False
                )
                diagnostics = (
                    self.model.collect_diagnostics(
                        self.diagnostic_item_indices, self.diagnostic_user_indices
                    )
                    if self.diagnostics_enabled
                    else {}
                )
                if self.diagnostics_enabled:
                    sampled_user_svd = compute_svd_spectrum(
                        self._scoring_users(val_u_embeds, self.diagnostic_user_indices)
                    )
                    sampled_item_svd = compute_svd_spectrum(
                        val_i_embeds[self.diagnostic_item_indices]
                    )
                    diagnostics.update({
                        "sample_user_effective_rank": float(sampled_user_svd["effective_rank"]),
                        "sample_item_effective_rank": float(sampled_item_svd["effective_rank"]),
                        "sample_user_count": float(self.diagnostic_user_indices.numel()),
                        "sample_item_count": float(self.diagnostic_item_indices.numel()),
                    })

            validation_time += time.perf_counter() - validation_start
            val_ndcg10 = val_metrics["NDCG@10"]
            val_rec10 = val_metrics["Recall@10"]
            avg_loss = total_loss_accum / max(1, num_batches)

            is_improved = self.early_stopping(
                val_metrics[self.monitor],
                epoch,
                self.model,
                checkpoint_path,
                optimizer=self.optimizer,
                val_metrics=val_metrics,
                config=self.config,
            )
            if is_improved:
                best_val_metrics = val_metrics
                best_epoch = epoch
                best_model_state = {key: value.detach().cpu().clone()
                                    for key, value in self.model.state_dict().items()}

            if self.validation_lr_scheduler is not None:
                self.validation_lr_scheduler.step(float(val_metrics[self.monitor]))

            best_tag = " [BEST]" if is_improved else ""

            epoch_summary = f"Epoch {epoch:02d}/{self.epochs:02d} [{epoch_time:4.1f}s] | Loss: {avg_loss:.4f} | Val Recall@10: {val_rec10:.4f} | Val NDCG@10: {val_ndcg10:.4f} | Monitor {self.monitor}: {val_metrics[self.monitor]:.4f}{best_tag}"
            print(epoch_summary, flush=True)

            # Record epoch training history
            history_record = {
                "epoch": epoch,
                "train_loss": round(total_loss_accum / max(1, num_batches), 4),
                "loss_total": total_loss_accum / max(1, num_batches),
                "bpr_loss": round(bpr_loss_accum / max(1, num_batches), 4),
                "cl_loss": round(cl_loss_accum / max(1, num_batches), 4),
                "val_ndcg_10": round(val_ndcg10, 4),
                "val_recall_10": round(val_metrics.get("Recall@10", 0.0), 4),
                "val_mrr_10": round(val_metrics.get("MRR@10", 0.0), 4),
                "epoch_time_sec": epoch_time,
                "val_ndcg_20": val_metrics.get("NDCG@20"),
                "val_recall_20": val_metrics.get("Recall@20"),
                "monitor": self.monitor,
                "monitor_value": val_metrics[self.monitor],
                "alignment_loss": bpr_loss_accum / max(1, num_batches) if self.model_name == "directau" else None,
                "uniformity_loss": cl_loss_accum / max(1, num_batches) if self.model_name == "directau" else None,
                "mean_gradient_norm": float(np.mean(gradient_norms)),
                "gradient_clip_fraction": float(np.mean(np.asarray(gradient_norms) > GRADIENT_CLIP_VALUE)),
                "mean_valid_semantic_pairs": None,
                "hard_pair_count": None,
                "hard_pair_fraction": None,
                "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if self.device.type == "cuda" else None,
                "is_best": bool(is_improved),
            }
            for key in LOSS_COMPONENT_KEYS:
                # Keep component values at full precision; only the legacy
                # display aliases above are rounded for backwards compatibility.
                history_record[f"loss_{key}"] = loss_component_accum[key] / max(1, num_batches)
            history_record.update({f"diagnostic_{key}": value for key, value in diagnostics.items()})
            history_record.update({
                f"lr_group_{index}": epoch_lrs[index]
                for index, group in enumerate(self.optimizer.param_groups)
            })
            history_record.update({
                f"lr_next_group_{index}": float(group["lr"])
                for index, group in enumerate(self.optimizer.param_groups)
            })
            history_records.append(history_record)

            # Save epoch history CSV
            history_dir = self.config.get("history_dir") or get_model_output_dir("history", self.model_name)
            os.makedirs(history_dir, exist_ok=True)
            history_csv_name = os.path.basename(checkpoint_path).replace(".pt", "_history.csv")
            history_csv_path = os.path.join(history_dir, history_csv_name)
            cumulative_train_time = previous_train_time + time.perf_counter() - start_train_time
            # Commit the authoritative history with the latest checkpoint.
            save_checkpoint(
                latest_checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_score=self.early_stopping.best_score,
                val_metrics=val_metrics,
                config=self.config,
                training_state={
                    "best_epoch": self.early_stopping.best_epoch,
                    "early_stopping_counter": self.early_stopping.counter,
                    "best_val_metrics": best_val_metrics,
                    "history": history_records,
                    "total_train_time": cumulative_train_time,
                    "best_model_state": best_model_state,
                    "validation_time": previous_validation_time + validation_time,
                    "diagnostics_epoch0": initial_diagnostics,
                    "validation_lr_scheduler_state": (
                        self.validation_lr_scheduler.state_dict()
                        if self.validation_lr_scheduler is not None else None
                    ),
                },
                scaler=self.scaler,
            )
            pd.DataFrame(history_records).to_csv(history_csv_path, index=False)

            if epoch in self.config.get("diagnostic_checkpoint_epochs", []):
                snapshot_path = checkpoint_path.replace(".pt", f"_epoch{epoch:03d}.pt")
                temporary_snapshot = snapshot_path + ".tmp"
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                            "val_metrics": val_metrics, "config": self.config,
                            "checkpoint_kind": "inference_only"}, temporary_snapshot)
                os.replace(temporary_snapshot, snapshot_path)

            if self.config.get('graph_embedding_snapshots', False):
                snapshot_path = checkpoint_path.replace('.pt', f'_graph_epoch{epoch:03d}.pt')
                torch.save({'epoch': epoch, 'user_embedding': self.model.user_embedding.weight.detach().cpu(),
                            'item_embedding': self.model.item_embedding.weight.detach().cpu(),
                            'val_metrics': val_metrics, 'experiment_fingerprint': self.config.get('experiment_fingerprint'),
                            'checkpoint_kind': 'graph_tables_only'}, snapshot_path + '.tmp')
                os.replace(snapshot_path + '.tmp', snapshot_path)

            if self.early_stopping.early_stop:
                logger.info(f"Stopping early at epoch {epoch}")
                break

        total_train_time = cumulative_train_time
        all_epoch_times = [float(row["epoch_time_sec"]) for row in history_records]
        avg_epoch_time = float(np.mean(all_epoch_times)) if all_epoch_times else 0.0
        timing = {"training_time": sum(all_epoch_times),
                  "validation_time": previous_validation_time + validation_time}

        # Load best checkpoint for final evaluation
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint_path}")
        load_checkpoint(checkpoint_path, self.model, device=self.device,
                        expected_fingerprint=self.config.get("experiment_fingerprint"))
        logger.info(f"Loaded best checkpoint for final evaluation: {checkpoint_path}")

        self.model.eval()
        last_loss_components = {
            key: history_records[-1].get(f"loss_{key}")
            for key in LOSS_COMPONENT_KEYS
        } if history_records else {}
        last_diagnostics = {
            key.removeprefix("diagnostic_"): value
            for key, value in (history_records[-1].items() if history_records else [])
            if key.startswith("diagnostic_")
        }
        if self.config.get("validation_only", False):
            return {"model_name": self.model_name, "architecture": getattr(self.model, "architecture", self.model_name), "best_epoch": best_epoch,
                    "total_epochs": len(history_records), "total_train_time": total_train_time,
                    "avg_epoch_time": avg_epoch_time, "val_metrics": best_val_metrics,
                    "validation_only": True, "history": history_records,
                    "monitor": self.monitor, "loss_schema_version": 2,
                    "loss_components_last": last_loss_components,
                    "diagnostics_last": last_diagnostics,
                    "diagnostics_epoch0": initial_diagnostics,
                    "optimizer_weight_decay": self.optimizer_weight_decay,
                    "gradient_clip_scope": getattr(self.model, "gradient_clip_scope", "all_parameters"),
                    "mlp_optimizer_weight_decay": float(0.0),
                    **timing}
        final_evaluation_start = time.perf_counter()
        with torch.no_grad():
            final_u_embeds, final_i_embeds = self.model(self.norm_adj)

            # 1. Full ranking evaluation including beyond-accuracy metrics with progress bar
            assert self.test_evaluator is not None
            test_metrics, avg_user_latency_ms = self.test_evaluator.evaluate(
                final_u_embeds, final_i_embeds, self.device, include_beyond_accuracy=True, show_progress=True
            )

            # 2. Representation Geometry (Alignment & Uniformity)
            geometry_users = final_u_embeds
            if hasattr(self.model, "refine_users"):
                geometry_users = torch.cat([
                    self._scoring_users(final_u_embeds, torch.arange(start, min(start + 1024, self.num_users), device=self.device))
                    for start in range(0, self.num_users, 1024)
                ])
            rep_metrics = compute_alignment_and_uniformity(
                geometry_users,
                final_i_embeds,
                self.train_edge_index,
                seed=int(self.config["training"].get("seed", 42)),
            )

            # 3. SVD Spectrum & Dimensional Collapse
            svd_user = compute_svd_spectrum(geometry_users)
            svd_item = compute_svd_spectrum(final_i_embeds)

            # 4. Degree-Stratified Subgroup Analysis (Head / Torso / Tail)
            subgroups = evaluate_degree_subgroups(
                self.test_evaluator,
                final_u_embeds,
                final_i_embeds,
                self.device,
                self.subgroup_reference_df,
                k_list=getattr(self.test_evaluator, "k_list", [10, 20]),
            )

        throughput_users_per_sec = 1000.0 / avg_user_latency_ms if avg_user_latency_ms > 0 else 0.0

        summary_results = {
            "model_name": self.model_name,
            "architecture": getattr(self.model, "architecture", self.model_name),
            "monitor": self.monitor,
            **timing,
            "final_evaluation_time": time.perf_counter() - final_evaluation_start,
            "best_epoch": best_epoch,
            "total_epochs": len(history_records),
            "total_train_time": total_train_time,
            "avg_epoch_time": avg_epoch_time,
            "inference_latency_ms_per_user": avg_user_latency_ms,
            "throughput_users_per_sec": throughput_users_per_sec,
            "val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "representation_metrics": rep_metrics,
            "svd_metrics": {
                "user_effective_rank": svd_user["effective_rank"],
                "item_effective_rank": svd_item["effective_rank"],
                "user_singular_values": svd_user["singular_values"][:15],
                "item_singular_values": svd_item["singular_values"][:15],
            },
            "subgroup_metrics": subgroups,
            "loss_schema_version": 2,
            "loss_components_last": last_loss_components,
            "diagnostics_last": last_diagnostics,
            "diagnostics_epoch0": initial_diagnostics,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "gradient_clip_scope": getattr(self.model, "gradient_clip_scope", "all_parameters"),
            "mlp_optimizer_weight_decay": float(0.0),
        }

        logger.info(f"Training completed for {self.model_name.upper()}.")
        logger.info(
            f"Final Test Results -> Recall@10: {test_metrics.get('Recall@10', 0):.4f} | NDCG@10: {test_metrics.get('NDCG@10', 0):.4f} | Diversity@10: {test_metrics.get('Diversity@10', 0):.4f} | Novelty@10: {test_metrics.get('Novelty@10', 0):.4f}"
        )
        logger.info(
            f"Representation -> Alignment: {rep_metrics['alignment']:.4f} | Mean Uniformity: {rep_metrics['mean_uniformity']:.4f} | User Eff Rank: {svd_user['effective_rank']:.2f}"
        )

        return summary_results
