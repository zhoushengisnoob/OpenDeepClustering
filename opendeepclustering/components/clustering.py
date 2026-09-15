"""Clustering assignment components and exact mathematical helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Compute the DEC auxiliary target distribution from all assignments."""
    frequencies = q.sum(dim=0).clamp_min(torch.finfo(q.dtype).eps)
    weight = q.square() / frequencies
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(q.dtype).eps
    )


class StudentTClustering(nn.Module):
    """Student-t soft assignment used by DEC-family algorithms."""

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        alpha: float = 1.0,
        initial_centers: np.ndarray | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.empty(n_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.centroids)
        if initial_centers is not None:
            with torch.no_grad():
                self.centroids.copy_(
                    torch.as_tensor(initial_centers, dtype=self.centroids.dtype)
                )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        distances = (embeddings.unsqueeze(1) - self.centroids).square().sum(dim=2)
        numerator = (1.0 + distances / self.alpha).pow(
            -(self.alpha + 1.0) / 2.0
        )
        return numerator / numerator.sum(dim=1, keepdim=True)
