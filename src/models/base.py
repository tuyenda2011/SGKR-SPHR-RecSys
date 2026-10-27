from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from src.data.graph import sparse_matmul


class BaseRecommender(nn.Module, ABC):
    """Abstract base class for Graph Collaborative Filtering Models."""

    scoring_metric = "dot_product"

    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # User and Item Embedding tables (E0)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize user and item embeddings using Normal distribution."""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def _mean_propagation(self, norm_adj: torch.Tensor) -> torch.Tensor:
        """Average E0..EL without allocating a stacked layer tensor."""
        current = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        total = current
        for _ in range(self.num_layers):
            current = sparse_matmul(norm_adj, current)
            total = total + current
        return total / (self.num_layers + 1)

    @abstractmethod
    def forward(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform graph convolution propagation and return final (user_embeds, item_embeds)."""
        pass

    def get_user_rating_scores(
        self, u_indices: torch.Tensor, all_users: torch.Tensor, all_items: torch.Tensor
    ) -> torch.Tensor:
        """Compute rating matrix prediction scores for given users against all items."""
        u_embeds = all_users[u_indices] # (batch_users, dim)
        scores = torch.matmul(u_embeds, all_items.T) # (batch_users, num_items)
        return scores
