"""Small utilities shared by non-DEC estimator training loops."""

from __future__ import annotations

import numpy as np
import torch


def resolve_device(name: str) -> torch.device:
    """Resolve an estimator device setting and fail clearly for unavailable CUDA."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device: {name!r}.") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def validate_dims(dims, *, name: str = "dims") -> tuple[int, ...]:
    """Validate a non-empty hidden-dimension sequence."""
    if not isinstance(dims, (tuple, list)) or not dims:
        raise ValueError(f"{name} must be a non-empty sequence.")
    if any(not isinstance(dim, (int, np.integer)) or dim <= 0 for dim in dims):
        raise ValueError(f"{name} must contain only positive integers.")
    return tuple(int(dim) for dim in dims)


def empirical_centers(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return a deterministic centroid row for each observed integer label."""
    return np.stack(
        [embeddings[labels == label].mean(axis=0) for label in np.unique(labels)]
    ).astype(np.float32, copy=False)
