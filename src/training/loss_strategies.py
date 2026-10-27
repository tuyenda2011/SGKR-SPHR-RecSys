"""Loss computation strategies for different recommendation models.

Implements the Strategy Pattern to decouple Trainer from model-specific loss computations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional

import torch
import torch.nn as nn

from src.losses.bpr import BPRLoss
from src.losses.contrastive import InfoNCELoss
from src.losses.directau import DirectAULoss


class LossOutput(NamedTuple):
    """Output from loss computation."""
    total_loss: torch.Tensor
    bpr_loss: float
    cl_loss: float
    extra_losses: Dict[str, torch.Tensor]


class LossStrategy(ABC):
    """Abstract base class for loss computation strategies."""

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        norm_adj: torch.Tensor,
        u_batch: torch.Tensor,
        pos_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> LossOutput:
        """Compute loss for the current batch.

        Args:
            model: The recommendation model
            norm_adj: Normalized adjacency matrix
            u_batch: User indices for the batch
            pos_batch: Positive item indices
            neg_batch: Negative item indices (may be None for DirectAU)
            config: Configuration dictionary

        Returns:
            LossOutput containing total loss and components
        """
        pass

    @property
    @abstractmethod
    def requires_negative_samples(self) -> bool:
        """Whether this strategy requires negative sampling."""
        pass


class BPRStrategy(LossStrategy):
    """Standard BPR loss strategy for LightGCN."""

    def __init__(self, weight_decay: float = 1e-4):
        self.bpr_loss_fn = BPRLoss(weight_decay=weight_decay)
        self.weight_decay = weight_decay

    @property
    def requires_negative_samples(self) -> bool:
        return True

    def compute_loss(
        self,
        model: nn.Module,
        norm_adj: torch.Tensor,
        u_batch: torch.Tensor,
        pos_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> LossOutput:
        if neg_batch is None:
            raise ValueError("BPRStrategy requires negative samples")

        u_embeds, i_embeds = model(norm_adj)
        pos_scores = (u_embeds[u_batch] * i_embeds[pos_batch]).sum(dim=-1)
        neg_scores = (u_embeds[u_batch] * i_embeds[neg_batch]).sum(dim=-1)

        u_emb0 = model.user_embedding(u_batch)
        pos_emb0 = model.item_embedding(pos_batch)
        neg_emb0 = model.item_embedding(neg_batch)

        total_loss, bpr_loss = self.bpr_loss_fn(
            pos_scores, neg_scores, u_emb0, pos_emb0, neg_emb0
        )
        id_l2_raw = self.bpr_loss_fn.compute_regularization(
            u_emb0, pos_emb0, neg_emb0, pos_scores.shape[0]
        )

        return LossOutput(
            total_loss=total_loss,
            bpr_loss=bpr_loss.item(),
            cl_loss=0.0,
            extra_losses={
                "bpr_raw": bpr_loss,
                "id_l2_raw": id_l2_raw,
                "id_l2_weighted": self.weight_decay * id_l2_raw,
            },
        )


class XSimGCLStrategy(LossStrategy):
    """BPR + Contrastive SSL strategy for XSimGCL."""

    def __init__(
        self,
        weight_decay: float = 1e-4,
        contrastive_weight: float = 0.1,
        temperature: float = 0.2,
    ):
        self.bpr_loss_fn = BPRLoss(weight_decay=weight_decay, regularization="selfrec")
        self.cl_loss_fn = InfoNCELoss(temperature=temperature)
        self.weight_decay = weight_decay
        self.contrastive_weight = contrastive_weight

    @property
    def requires_negative_samples(self) -> bool:
        return True

    def compute_loss(
        self,
        model: nn.Module,
        norm_adj: torch.Tensor,
        u_batch: torch.Tensor,
        pos_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> LossOutput:
        if neg_batch is None:
            raise ValueError("XSimGCLStrategy requires negative samples")

        u_embeds, i_embeds, cl_u_embeds, cl_i_embeds = model(
            norm_adj, perturbed=True
        )
        pos_scores = (u_embeds[u_batch] * i_embeds[pos_batch]).sum(dim=-1)
        neg_scores = (u_embeds[u_batch] * i_embeds[neg_batch]).sum(dim=-1)

        u_emb0 = u_embeds[u_batch]
        pos_emb0 = i_embeds[pos_batch]
        neg_emb0 = None

        total_loss, bpr_loss = self.bpr_loss_fn(
            pos_scores, neg_scores, u_emb0, pos_emb0, neg_emb0
        )
        id_l2_raw = self.bpr_loss_fn.compute_regularization(
            u_emb0, pos_emb0, neg_emb0, pos_scores.shape[0]
        )

        unique_users = torch.unique(u_batch)
        unique_items = torch.unique(pos_batch)
        cl_loss = (
            self.cl_loss_fn.compute_view_loss(
                u_embeds[unique_users], cl_u_embeds[unique_users]
            )
            + self.cl_loss_fn.compute_view_loss(
                i_embeds[unique_items], cl_i_embeds[unique_items]
            )
        )
        total_loss = total_loss + self.contrastive_weight * cl_loss
        ssl_weighted = self.contrastive_weight * cl_loss

        return LossOutput(
            total_loss=total_loss,
            bpr_loss=bpr_loss.item(),
            cl_loss=cl_loss.item(),
            extra_losses={
                "bpr_raw": bpr_loss,
                "id_l2_raw": id_l2_raw,
                "id_l2_weighted": self.weight_decay * id_l2_raw,
                "ssl_raw": cl_loss,
                "ssl_weight": torch.as_tensor(self.contrastive_weight, device=cl_loss.device),
                "ssl_weighted": ssl_weighted,
                # Backwards-compatible alias used by existing history readers.
                "cl_loss": cl_loss,
            },
        )


class DirectAUStrategy(LossStrategy):
    """DirectAU loss strategy (no negative sampling)."""

    def __init__(self, gamma: float = 1.0, t: float = 2.0, weight_decay: float = 1e-4):
        self.directau_loss_fn = DirectAULoss(gamma=gamma, t=t, weight_decay=weight_decay)
        self.gamma = gamma
        self.t = t
        self.weight_decay = weight_decay

    @property
    def requires_negative_samples(self) -> bool:
        return False

    def compute_loss(
        self,
        model: nn.Module,
        norm_adj: torch.Tensor,
        u_batch: torch.Tensor,
        pos_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> LossOutput:
        u_embeds, i_embeds = model(norm_adj)
        u_emb0 = model.user_embedding(u_batch)
        pos_emb0 = model.item_embedding(pos_batch)

        total_loss, align_loss, unif_loss = self.directau_loss_fn(
            u_embeds[u_batch], i_embeds[pos_batch], u_emb0, pos_emb0
        )
        id_l2_raw = (u_emb0.norm(2).pow(2) + pos_emb0.norm(2).pow(2)) / (
            2.0 * u_embeds[u_batch].size(0)
        )
        align_weighted = align_loss
        uniformity_weighted = self.gamma * unif_loss
        id_l2_weighted = self.weight_decay * id_l2_raw

        return LossOutput(
            total_loss=total_loss,
            bpr_loss=align_loss.item(),
            cl_loss=unif_loss.item(),
            extra_losses={
                "alignment_raw": align_loss,
                "alignment_weighted": align_weighted,
                "uniformity_raw": unif_loss,
                "uniformity_weight": torch.as_tensor(self.gamma, device=unif_loss.device),
                "uniformity_weighted": uniformity_weighted,
                "id_l2_raw": id_l2_raw,
                "id_l2_weighted": id_l2_weighted,
                # Existing aliases retained for old CSV/history consumers.
                "align_loss": align_loss,
                "unif_loss": unif_loss,
            },
        )




def get_loss_strategy(model_name: str, config: Dict[str, Any]) -> LossStrategy:
    """Factory function to get the appropriate loss strategy for a model.

    Args:
        model_name: Name of the model
        config: Configuration dictionary

    Returns:
        Appropriate LossStrategy instance
    """
    train_cfg = config.get("training", {})
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    if model_name == "lightgcn":
        return BPRStrategy(weight_decay=weight_decay)

    elif model_name == "xsimgcl":
        xsim_cfg = config.get("xsimgcl", {})
        return XSimGCLStrategy(
            weight_decay=weight_decay,
            contrastive_weight=xsim_cfg.get("contrastive_weight", 0.1),
            temperature=xsim_cfg.get("temperature", 0.2),
        )

    elif model_name == "directau":
        dau_cfg = config.get("directau", {})
        return DirectAUStrategy(
            gamma=dau_cfg.get("gamma", 1.0),
            t=dau_cfg.get("t", 2.0),
            weight_decay=0.0 if dau_cfg.get("profile") == "reference_lgcn" else weight_decay,
        )

    else:
        raise ValueError(f"Unknown or retired model: {model_name}")
