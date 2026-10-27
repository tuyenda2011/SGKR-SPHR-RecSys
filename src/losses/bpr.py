import torch
import torch.nn.functional as F
from torch import nn


class BPRLoss(nn.Module):
    """BPR with squared L2 or SELFRec's unsquared batch-norm regularization."""

    def __init__(self, weight_decay: float = 1e-4, regularization: str = "squared"):
        super().__init__()
        if regularization not in {"squared", "selfrec"}:
            raise ValueError("regularization must be 'squared' or 'selfrec'")
        self.weight_decay = weight_decay
        self.regularization = regularization

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        u_emb0: torch.Tensor,
        pos_emb0: torch.Tensor,
        neg_emb0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute BPR Loss and L2 Regularization Loss.

        Args:
            pos_scores: Inner product scores for positive user-item pairs (batch_size,)
            neg_scores: Inner product scores for negative user-item pairs (batch_size,)
            u_emb0: Initial user embeddings (batch_size, dim)
            pos_emb0: Initial positive item embeddings (batch_size, dim)
            neg_emb0: Optional negative item embeddings for L2 regularization

        Returns:
            total_loss: BPR loss + L2 regularization loss
            bpr_loss: Pure BPR loss for logging
        """
        # BPR Loss: -log(sigmoid(pos_score - neg_score)) = softplus(-(pos_score - neg_score))
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        reg_loss = self.compute_regularization(u_emb0, pos_emb0, neg_emb0, pos_scores.shape[0])
        total_loss = bpr_loss + self.weight_decay * reg_loss
        return total_loss, bpr_loss

    def compute_regularization(
        self,
        u_emb0: torch.Tensor,
        pos_emb0: torch.Tensor,
        neg_emb0: torch.Tensor | None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Return the raw embedding regularizer before ``weight_decay``.

        Keeping this calculation next to ``forward`` makes the logged loss
        decomposition exactly match the objective used for optimization.
        """
        embeddings = [u_emb0, pos_emb0]
        if neg_emb0 is not None:
            embeddings.append(neg_emb0)
        if self.regularization == "selfrec":
            return sum(embedding.norm(2) / embedding.shape[0] for embedding in embeddings)
        size = int(batch_size or pos_emb0.shape[0])
        return sum(embedding.norm(2).pow(2) for embedding in embeddings) / (2.0 * size)
