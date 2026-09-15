"""Small, index-preserving data adapters."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def to_tensor(X: np.ndarray) -> torch.Tensor:
    """Convert a validated NumPy matrix without changing sample order."""
    return torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)


def flatten_samples(X):
    """Explicitly flatten image-like samples into a feature matrix."""
    if isinstance(X, torch.Tensor):
        return X.reshape(len(X), -1)
    return np.asarray(X).reshape(len(X), -1)


class IndexedTensorDataset(Dataset):
    """Tensor dataset that returns stable sample indices with each row."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        return self.data[index], index
