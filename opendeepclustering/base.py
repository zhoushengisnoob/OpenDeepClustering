"""Shared estimator conventions for deep clustering algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DeepClusterMixin(ClusterMixin, TransformerMixin, BaseEstimator, ABC):
    """Thin public contract shared by OpenDeepClustering estimators.

    The mixin deliberately does not prescribe a training loop. Multi-stage,
    iterative, generative, and simultaneous algorithms orchestrate their own
    stages while sharing validation and public fitted-attribute conventions.
    """

    taxonomy: str
    supports_predict: bool = True
    supports_soft_assignment: bool = False

    def _validate_X(self, X: Any, *, reset: bool) -> np.ndarray:
        """Validate dense array-like input and return a 2D float32 matrix."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if sparse.issparse(X):
            raise TypeError("Sparse input is not supported; provide a dense array.")

        array = np.asarray(X) if not hasattr(X, "iloc") else X
        if hasattr(array, "ndim") and array.ndim > 2:
            raise ValueError(
                "Expected a 2D feature matrix. Use data.flatten_samples explicitly "
                "for image-like arrays or a method-specific Dataset adapter."
            )

        return self._validate_data(
            array,
            reset=reset,
            accept_sparse=False,
            ensure_2d=True,
            allow_nd=False,
            dtype=np.float32,
            ensure_min_samples=1,
            ensure_min_features=1,
        )

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "model_") and hasattr(self, "labels_")

    def get_embeddings(self, X=None) -> np.ndarray:
        """Return training embeddings or transform new samples."""
        check_is_fitted(self)
        if X is None:
            return self.embedding_
        return self.transform(X)

    @abstractmethod
    def transform(self, X) -> np.ndarray:
        """Encode samples into the learned representation space."""

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": False,
            "requires_y": False,
            "poor_score": True,
        }
