import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE Contrastive Loss for graph self-supervised contrastive learning views."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def compute_view_loss(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE contrastive loss between view1 and view2.

        Args:
            view1: Embeddings from first view (N, dim)
            view2: Embeddings from second view (N, dim)
        """
        # Normalize embeddings along feature dimension
        view1_norm = F.normalize(view1, dim=1)
        view2_norm = F.normalize(view2, dim=1)

        # Cosine similarity matrix (N, N)
        sim_matrix = torch.matmul(view1_norm, view2_norm.T) / self.temperature

        # Positive pairs are diagonal elements (i, i)
        labels = torch.arange(view1.shape[0], device=view1.device)

        # Cross entropy loss over rows
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def forward(
        self,
        u_view1: torch.Tensor,
        u_view2: torch.Tensor,
        i_view1: torch.Tensor,
        i_view2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined user and item contrastive loss."""
        user_loss = self.compute_view_loss(u_view1, u_view2)
        item_loss = self.compute_view_loss(i_view1, i_view2)
        return user_loss + item_loss
