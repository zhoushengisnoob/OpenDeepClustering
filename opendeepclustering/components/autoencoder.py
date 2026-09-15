"""Autoencoder building block shared by early deep clustering methods."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class StackedAutoEncoder(nn.Module):
    """Symmetric MLP autoencoder with addressable layer pairs."""

    def __init__(self, input_dim: int, dims: Iterable[int]):
        super().__init__()
        dims = tuple(dims)
        if not dims:
            raise ValueError("dims must contain at least one latent dimension.")

        layer_dims = (input_dim, *dims)
        self.encoders = nn.ModuleList(
            nn.Linear(layer_dims[index], layer_dims[index + 1])
            for index in range(len(dims))
        )
        self.decoders = nn.ModuleList(
            nn.Linear(layer_dims[index + 1], layer_dims[index])
            for index in range(len(dims))
        )

    def encode_to(self, X: torch.Tensor, layer: int) -> torch.Tensor:
        for index in range(layer + 1):
            X = self.encoders[index](X)
            if index != len(self.encoders) - 1:
                X = torch.relu(X)
        return X

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.encode_to(X, len(self.encoders) - 1)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        output = embedding
        for index in reversed(range(len(self.decoders))):
            output = self.decoders[index](output)
            if index != 0:
                output = torch.relu(output)
        return output

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(X)
        return embedding, self.decode(embedding)
