"""Multi-stage autoencoder plus shallow clustering estimator."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opendeepclustering.base import DeepClusterMixin
from opendeepclustering.components import StackedAutoEncoder, make_kmeans
from opendeepclustering.data import to_tensor
from opendeepclustering.estimators._utils import (
    empirical_centers,
    resolve_device,
    validate_dims,
)
from opendeepclustering.training.callbacks import emit
from opendeepclustering.training.random import SeedManager
from opendeepclustering.training.state import FitState


class AutoencoderKMeans(DeepClusterMixin):
    """Sequentially fit an autoencoder and a cloned shallow clusterer.

    Parameters
    ----------
    n_clusters : int, default=10
        Number of clusters used by the default KMeans stage.
    dims : tuple of int, default=(500, 500, 2000, 10)
        Encoder layer widths, ending with the embedding dimension.
    clusterer : estimator or None, default=None
        Optional scikit-learn-style clusterer. It must expose ``labels_`` after
        ``fit`` or implement ``fit_predict``. New-sample prediction additionally
        requires ``predict``. The estimator is cloned before fitting.
    max_epochs : int, default=50
        Number of full autoencoder training epochs.
    """

    algorithm_name = "AutoencoderKMeans"
    taxonomy = "Multi-stage"
    input_modalities = ("tabular",)

    def __init__(
        self,
        n_clusters: int = 10,
        dims: tuple[int, ...] = (500, 500, 2000, 10),
        clusterer=None,
        max_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_init: int = 20,
        kmeans_max_iter: int = 300,
        kmeans_tol: float = 1e-4,
        device: str = "auto",
        random_state=None,
        deterministic: bool = False,
        callbacks=None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.dims = dims
        self.clusterer = clusterer
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_init = n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol
        self.device = device
        self.random_state = random_state
        self.deterministic = deterministic
        self.callbacks = callbacks
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the representation stage, freeze it, then fit the clusterer."""
        del y
        X = self._validate_X(X, reset=True)
        dims = self._validate_parameters(X)
        self.device_ = resolve_device(self.device)
        self.seed_manager_ = SeedManager(self.random_state)
        self.fit_state_ = FitState(stage="representation")
        emit(self.callbacks, "fit_start", self, self.fit_state_)

        data = to_tensor(X)
        with self.seed_manager_.torch_fork(
            self.device_, deterministic=self.deterministic
        ):
            self.model_ = StackedAutoEncoder(self.n_features_in_, dims).to(
                self.device_
            )
            optimizer = torch.optim.Adam(
                self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            criterion = nn.MSELoss()
            for epoch in range(self.max_epochs):
                self.model_.train()
                total = 0.0
                loader = DataLoader(
                    TensorDataset(data),
                    batch_size=self.batch_size,
                    shuffle=True,
                    generator=self.seed_manager_.torch,
                )
                for (batch,) in loader:
                    batch = batch.to(self.device_)
                    _, reconstruction = self.model_(batch)
                    loss = criterion(reconstruction, batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(batch)
                    self.fit_state_.n_iter += 1
                self.fit_state_.epoch = epoch + 1
                self.fit_state_.record("reconstruction_loss", total / len(data))
                emit(self.callbacks, "epoch_end", self, self.fit_state_)

            self.embedding_ = self._encode(data)

        self.fit_state_.stage = "clustering"
        emit(self.callbacks, "stage_start", self, self.fit_state_)
        self.clusterer_ = self._make_clusterer()
        if hasattr(self.clusterer_, "fit_predict"):
            labels = self.clusterer_.fit_predict(self.embedding_)
        else:
            self.clusterer_.fit(self.embedding_)
            if not hasattr(self.clusterer_, "labels_"):
                raise TypeError("clusterer must implement fit_predict or expose labels_ after fit.")
            labels = self.clusterer_.labels_
        self.labels_ = np.asarray(labels, dtype=np.int64)
        self.cluster_centers_ = empirical_centers(self.embedding_, self.labels_)
        self.n_iter_ = self.fit_state_.n_iter
        self.converged_ = True
        self.stop_reason_ = "completed"
        self.fit_state_.converged = True
        self.fit_state_.stop_reason = self.stop_reason_
        self.history_ = self.fit_state_.history
        self.supports_predict_ = hasattr(self.clusterer_, "predict")
        emit(self.callbacks, "fit_end", self, self.fit_state_)
        return self

    def _validate_parameters(self, X):
        dims = validate_dims(self.dims)
        integer_parameters = {
            "n_clusters": (self.n_clusters, 1),
            "max_epochs": (self.max_epochs, 0),
            "batch_size": (self.batch_size, 1),
            "n_init": (self.n_init, 1),
            "kmeans_max_iter": (self.kmeans_max_iter, 1),
        }
        for name, (value, minimum) in integer_parameters.items():
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.clusterer is None and self.n_clusters > len(X):
            raise ValueError("n_clusters must not exceed the number of samples.")
        if self.lr <= 0 or self.kmeans_tol <= 0:
            raise ValueError("lr and kmeans_tol must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        return dims

    def _make_clusterer(self):
        if self.clusterer is not None:
            return clone(self.clusterer)
        return make_kmeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.kmeans_max_iter,
            tol=self.kmeans_tol,
            random_state=self.seed_manager_.seed,
        )

    def _encode(self, data: torch.Tensor) -> np.ndarray:
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for (batch,) in DataLoader(TensorDataset(data), batch_size=self.batch_size):
                outputs.append(self.model_.encode(batch.to(self.device_)).cpu())
        return torch.cat(outputs).numpy()

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = self._validate_X(X, reset=False)
        return self._encode(to_tensor(X))

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "clusterer_")
        if not hasattr(self.clusterer_, "predict"):
            raise AttributeError("The fitted shallow clusterer does not support predict.")
        return np.asarray(self.clusterer_.predict(self.transform(X)), dtype=np.int64)
