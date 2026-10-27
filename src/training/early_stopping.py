import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_score: float = 0.0,
    val_metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    training_state: Optional[Dict[str, Any]] = None,
    scaler=None,
) -> None:
    """Save full training checkpoint with model weights, optimizer state, metrics, and config.

    Uses atomic write (temp file + rename) to prevent corruption on interruption.
    """
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    state = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "val_metrics": val_metrics or {},
        "config": config or {},
        "training_state": training_state or {},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }

    # Atomic write: save to temp file first, then rename
    temp_path = checkpoint_path + ".tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, checkpoint_path)  # Atomic on POSIX, semi-atomic on Windows
    logger.info(f"Saved complete model checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    expected_fingerprint: Optional[str] = None,
) -> Tuple[int, float, Dict[str, Any]]:
    """Load model weights and optimizer state from checkpoint with validation."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # State loaders move parameter tensors to the model/optimizer devices.
    # Keep Adam's scalar step counters and committed best snapshots on CPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Validate checkpoint integrity
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format at {checkpoint_path}: missing 'model_state_dict' key")

    stored_fingerprint = checkpoint.get("config", {}).get("experiment_fingerprint")
    if expected_fingerprint and stored_fingerprint != expected_fingerprint:
        raise ValueError(
            "Checkpoint fingerprint does not match the current data/config/code"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, best score: {best_score:.4f})")

    return epoch, best_score, checkpoint


def restore_training_rng(checkpoint, scaler=None):
    """Restore stochastic training state only for an explicit resume."""
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])


class EarlyStopping:
    """Early Stopping handler to track validation metrics and save best model checkpoint."""

    def __init__(self, patience: int = 20, monitor: str = "NDCG@10", mode: str = "max"):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def __call__(
        self,
        score: float,
        epoch: int,
        model: torch.nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        improved = (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

            # Save best checkpoint
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=score,
                val_metrics=val_metrics,
                config=config,
            )
            logger.info(
                f"Epoch {epoch}: Validation {self.monitor} improved to {score:.4f}."
            )
        else:
            self.counter += 1
            logger.info(
                f"Epoch {epoch}: Validation {self.monitor} did not improve ({score:.4f} vs best {self.best_score:.4f}). Counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}!")

        return improved
