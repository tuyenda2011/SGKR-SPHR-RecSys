import torch
import torch.nn.functional as F
from torch import nn


class HardNegativeBPRLoss(nn.Module):
    """Bayesian Personalized Ranking (BPR) Loss with Margin-based Explicit Hard Negative Penalty.

    L = -sum ln(sigma(y_ui - y_uj_rand)) + alpha * sum max(0, y_uj_hard - y_ui + margin)
    """

    def __init__(self, alpha: float = 0.2, margin: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.margin = margin

    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        rand_neg_emb: torch.Tensor,
        hard_neg_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute combined BPR and Hard Negative loss.

        Args:
            user_emb: (batch_size, emb_dim)
            pos_item_emb: (batch_size, emb_dim)
            rand_neg_emb: (batch_size, emb_dim)
            hard_neg_emb: Optional (batch_size, emb_dim) explicit disliked item embeddings

        Returns:
            Scalar loss tensor.
        """
        # Standard BPR loss against unobserved random negative items
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        rand_neg_scores = torch.sum(user_emb * rand_neg_emb, dim=-1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - rand_neg_scores))

        # Margin penalty against explicit disliked items (if available)
        if hard_neg_emb is not None and self.alpha > 0.0:
            return bpr_loss + self.compute_hard_penalty(
                user_emb, pos_item_emb, hard_neg_emb
            )

        return bpr_loss

    def compute_hard_penalty(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        hard_neg_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize explicit dislikes that outrank a user's positive item."""
        return self.alpha * self.compute_hard_margin(
            user_emb, pos_item_emb, hard_neg_emb
        )

    def compute_hard_margin(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        hard_neg_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return the unweighted mean margin violation."""
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        hard_neg_scores = torch.sum(user_emb * hard_neg_emb, dim=-1)
        return F.relu(hard_neg_scores - pos_scores + self.margin).mean()
