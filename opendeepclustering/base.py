"""Shared estimator conventions for deep clustering algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # scikit-learn < 1.6
    validate_data = None


class DeepClusterMixin(ClusterMixin, TransformerMixin, BaseEstimator, ABC):
    """Thin public contract shared by OpenDeepClustering estimators.

    The mixin deliberately does not prescribe a training loop. Multi-stage,
    iterative, generative, and simultaneous algorithms orchestrate their own
    stages while sharing validation and public fitted-attribute conventions.
    """

    taxonomy: str
    input_modalities: tuple[str, ...] = ("tabular",)
    supports_predict: bool = True
    supports_soft_assignment: bool = False
    supports_sample: bool = False

    def get_capabilities(self) -> dict[str, Any]:
        """Return declared, pre-fit capabilities for discovery and documentation."""
        return {
            "taxonomy": self.taxonomy,
            "input_modalities": self.input_modalities,
            "predict": self.supports_predict,
            "soft_assignment": self.supports_soft_assignment,
            "sample": self.supports_sample,
        }

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

        options = {
            "reset": reset,
            "accept_sparse": False,
            "ensure_2d": True,
            "allow_nd": False,
            "dtype": np.float32,
            "ensure_min_samples": 1,
            "ensure_min_features": 1,
        }
        if validate_data is not None:
            return validate_data(self, array, **options)
        return self._validate_data(array, **options)

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

    def __sklearn_tags__(self):
        """Expose native tags on scikit-learn 1.6+ while retaining 1.2 support."""
        parent = super()
        if not hasattr(parent, "__sklearn_tags__"):
            return self._more_tags()
        tags = parent.__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        return tags
