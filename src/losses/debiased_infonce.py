import math

import torch
import torch.nn.functional as F
from torch import nn


class DebiasedInfoNCELoss(nn.Module):
    """Debiased Contrastive InfoNCE Loss (NeurIPS '20 / RecSys SSL).

    Optional DCL correction with an assumed positive prior, not an estimated
    false-negative rate. At tau_plus=0 this is standard cross-view InfoNCE.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        tau_plus: float = 0.0,
        hard_negative_weight: float = 1.0,
    ):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= tau_plus < 1:
            raise ValueError("tau_plus must be in [0, 1)")
        if not math.isfinite(hard_negative_weight) or hard_negative_weight < 0:
            raise ValueError("hard_negative_weight must be finite and nonnegative")
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.hard_negative_weight = hard_negative_weight

    def compute_debiased_contrastive_loss(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor | None = None,
        hard_negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute debiased InfoNCE loss for a batch of query-positive pairs.

        Args:
            query: Query representations (B, dim)
            positive: Positive key representations (B, dim)
            negatives: Optional negative-only pool (M, dim), excluding known positives.
                If None, other in-batch keys are used, excluding the diagonal.
            hard_negatives: Optional explicit hard negatives (B, dim) from 1-2 star feedback.

        Returns:
            Scalar loss tensor.
        """
        if query.ndim != 2 or query.shape != positive.shape or query.shape[0] == 0:
            raise ValueError(
                "query and positive must have matching nonempty (B, dim) shapes"
            )
        dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        q = F.normalize(query.to(dtype), dim=-1)
        k_pos = F.normalize(positive.to(dtype), dim=-1)
        pos_sim = torch.sum(q * k_pos, dim=-1) / self.temperature

        if negatives is None:
            neg_sim = (q @ k_pos.T) / self.temperature
            diagonal = torch.eye(q.shape[0], dtype=torch.bool, device=q.device)
            neg_sim = neg_sim.masked_fill(diagonal, float("-inf"))
            n_samples = q.shape[0] - 1
        else:
            neg_sim = (
                q @ F.normalize(negatives.to(dtype), dim=-1).T
            ) / self.temperature
            n_samples = negatives.shape[0]

        # Subtract a common row offset before exponentiation. This preserves
        # the DCL estimator and prevents exp(cosine / temperature) overflow.
        shift = pos_sim
        if n_samples:
            shift = torch.maximum(shift, neg_sim.max(dim=-1).values)
        hard_sim = None
        if hard_negatives is not None and self.hard_negative_weight > 0:
            k_hard = F.normalize(hard_negatives.to(dtype), dim=-1)
            hard_sim = (q * k_hard).sum(dim=-1) / self.temperature
            hard_sim = hard_sim + math.log(self.hard_negative_weight)
            shift = torch.maximum(shift, hard_sim)
        shift = shift.detach()
        pos_exp = torch.exp(pos_sim - shift)
        neg_sum = torch.exp(neg_sim - shift.unsqueeze(1)).sum(dim=-1)

        if self.tau_plus > 0 and n_samples:
            corrected = (neg_sum - self.tau_plus * n_samples * pos_exp) / (
                1 - self.tau_plus
            )
            lower_bound = n_samples * torch.exp(-1 / self.temperature - shift)
            neg_sum = torch.maximum(corrected, lower_bound)
        if hard_sim is not None:
            neg_sum = neg_sum + torch.exp(hard_sim - shift)

        return (torch.log(pos_exp + neg_sum) - (pos_sim - shift)).mean()

    def forward(
        self,
        u_view1: torch.Tensor,
        u_view2: torch.Tensor,
        i_view1: torch.Tensor,
        i_view2: torch.Tensor,
        hard_negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dual user and item debiased contrastive loss."""
        user_loss = self.compute_debiased_contrastive_loss(u_view1, u_view2)
        item_loss = self.compute_debiased_contrastive_loss(
            i_view1, i_view2, hard_negatives=hard_negatives
        )
        return user_loss + item_loss
