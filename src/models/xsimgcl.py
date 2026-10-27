from typing import Tuple, Union

import torch
import torch.nn.functional as F

from src.data.graph import sparse_matmul
from src.models.base import BaseRecommender

EmbeddingPair = Tuple[torch.Tensor, torch.Tensor]
PerturbedEmbeddingOutput = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class XSimGCL(BaseRecommender):
    """XSimGCL: Extreme Simple Graph Contrastive Learning for Recommendation (TKDE '23).

    Applies sign-aware normalized perturbations during propagation and contrasts
    the aggregated recommendation representation against a selected graph layer.
    """

    def _init_weights(self) -> None:
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        contrastive_weight: float = 0.1,
        temperature: float = 0.2,
        epsilon: float = 0.1,
        contrastive_layer: int = 1,
    ):
        super().__init__(num_users, num_items, embedding_dim, num_layers)
        if not 1 <= contrastive_layer <= num_layers:
            raise ValueError("contrastive_layer must be between 1 and num_layers")
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.epsilon = epsilon
        self.contrastive_layer = contrastive_layer

    def forward(
        self, norm_adj: torch.Tensor, perturbed: bool = False
    ) -> Union[EmbeddingPair, PerturbedEmbeddingOutput]:
        """Run the official XSimGCL perturbed propagation protocol."""
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        total_embeddings = None
        contrastive_embeddings = ego_embeddings

        for layer in range(1, self.num_layers + 1):
            ego_embeddings = sparse_matmul(norm_adj, ego_embeddings)
            if perturbed:
                noise = F.normalize(torch.rand_like(ego_embeddings), dim=-1)
                ego_embeddings = (
                    ego_embeddings
                    + torch.sign(ego_embeddings) * noise * self.epsilon
                )
            total_embeddings = (
                ego_embeddings if total_embeddings is None else total_embeddings + ego_embeddings
            )
            if layer == self.contrastive_layer:
                contrastive_embeddings = ego_embeddings

        # The official protocol averages E1..EL, excluding the input E0.
        final_embeddings = total_embeddings / self.num_layers
        user_final, item_final = torch.split(
            final_embeddings, [self.num_users, self.num_items], dim=0
        )
        if not perturbed:
            return user_final, item_final

        user_cl, item_cl = torch.split(
            contrastive_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_final, item_final, user_cl, item_cl
