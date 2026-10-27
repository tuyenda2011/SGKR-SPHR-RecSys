from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseRecommender


class DirectAU(BaseRecommender):
    """DirectAU: Direct Alignment and Uniformity for Collaborative Filtering (KDD '22).

    Optimizes representation alignment of positive pairs and uniformity across hypersphere directly.
    """

    scoring_metric = "cosine"

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        gamma: float = 1.0,
        t: float = 2.0,
        profile: str = "project_cosine",
    ):
        if profile not in {"project_cosine", "reference_lgcn"}:
            raise ValueError("Unknown DirectAU profile")
        self.profile = profile
        super().__init__(num_users, num_items, embedding_dim, num_layers)
        self.scoring_metric = "dot_product" if profile == "reference_lgcn" else "cosine"
        self.gamma = gamma
        self.t = t

    def _init_weights(self):
        if self.profile == "reference_lgcn":
            nn.init.xavier_normal_(self.user_embedding.weight)
            nn.init.xavier_normal_(self.item_embedding.weight)
        else:
            super()._init_weights()

    def forward(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform LightGCN layer aggregation over normalized bipartite graph adjacency matrix."""
        final_embeddings = self._mean_propagation(norm_adj)
        user_final, item_final = torch.split(
            final_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_final, item_final

    def get_user_rating_scores(
        self, u_indices: torch.Tensor, all_users: torch.Tensor, all_items: torch.Tensor
    ) -> torch.Tensor:
        """Score using the explicitly selected reference or project profile."""
        if self.profile == "reference_lgcn":
            return super().get_user_rating_scores(u_indices, all_users, all_items)
        u_embeds = F.normalize(all_users[u_indices], dim=-1)
        i_embeds = F.normalize(all_items, dim=-1)
        scores = torch.matmul(u_embeds, i_embeds.T)
        return scores
