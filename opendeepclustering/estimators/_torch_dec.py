"""Shared PyTorch implementation for DEC-style estimators."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_2d_numpy(X) -> np.ndarray:
    """Convert common array-like inputs to a finite 2D float32 matrix."""
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    elif isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    X = np.asarray(X, dtype=np.float32)
    if X.ndim < 2:
        raise ValueError("X must be at least 2-dimensional.")
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")
    return X


def _set_random_state(random_state: int | None) -> None:
    if random_state is None:
        return
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = q.pow(2) / q.sum(dim=0)
    return (weight.t() / weight.sum(dim=1)).t()


class _StackedAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, dims: Iterable[int]):
        super().__init__()
        dims = list(dims)
        if not dims:
            raise ValueError("dims must contain at least one latent dimension.")

        encoder_layers = []
        previous_dim = input_dim
        for index, dim in enumerate(dims):
            encoder_layers.append(nn.Linear(previous_dim, dim))
            if index != len(dims) - 1:
                encoder_layers.append(nn.ReLU())
            previous_dim = dim

        decoder_layers = []
        reversed_dims = list(reversed(dims[:-1])) + [input_dim]
        previous_dim = dims[-1]
        for index, dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(previous_dim, dim))
            if index != len(reversed_dims) - 1:
                decoder_layers.append(nn.ReLU())
            previous_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(X)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


class _ClusteringLayer(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        alpha: float,
        initial_centers: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.empty(n_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.centroids)
        if initial_centers is not None:
            self.centroids.data.copy_(
                torch.as_tensor(initial_centers, dtype=torch.float32)
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        distances = torch.sum((embeddings.unsqueeze(1) - self.centroids) ** 2, dim=2)
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        return q / q.sum(dim=1, keepdim=True)


class _DeepEmbeddedClusteringModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dims: Iterable[int],
        n_clusters: int,
        alpha: float,
        initial_centers: np.ndarray | None = None,
    ):
        super().__init__()
        self.autoencoder = _StackedAutoEncoder(input_dim, dims)
        self.clustering = _ClusteringLayer(
            n_clusters=n_clusters,
            embedding_dim=list(dims)[-1],
            alpha=alpha,
            initial_centers=initial_centers,
        )

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings, reconstruction = self.autoencoder(X)
        q = self.clustering(embeddings)
        return q, embeddings, reconstruction


@dataclass
class _FitResult:
    labels: np.ndarray
    embeddings: np.ndarray
    centers: np.ndarray


class _BaseDEC(BaseEstimator, ClusterMixin):
    """Base estimator for DEC-family algorithms.

    The class intentionally exposes a small scikit-learn style surface while
    keeping the PyTorch training loop internal.
    """

    algorithm_name = "DEC"
    taxonomy = "Simultaneous"
    reconstruction_weight = 0.0

    def __init__(
        self,
        n_clusters: int = 10,
        dims: tuple[int, ...] = (500, 500, 2000, 10),
        alpha: float = 1.0,
        pretrain_epochs: int = 50,
        max_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_init: int = 20,
        kmeans_max_iter: int = 300,
        tol: float = 1e-3,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.dims = dims
        self.alpha = alpha
        self.pretrain_epochs = pretrain_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_init = n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.tol = tol
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        X = _as_2d_numpy(X)
        self._validate_parameters(X)
        _set_random_state(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.device_ = self._resolve_device()

        data = torch.as_tensor(X, dtype=torch.float32)
        self.model_ = _DeepEmbeddedClusteringModel(
            input_dim=self.n_features_in_,
            dims=self.dims,
            n_clusters=self.n_clusters,
            alpha=self.alpha,
        ).to(self.device_)

        self.history_ = {"pretrain_loss": [], "cluster_loss": []}
        self._pretrain(data)
        fit_result = self._initialize_clusters(data)
        self._replace_clustering_layer(fit_result.centers)
        self._finetune(data)
        final_embeddings = self.transform(X)
        self.embedding_ = final_embeddings
        self.labels_ = self.predict(X)
        self.cluster_centers_ = self.model_.clustering.centroids.detach().cpu().numpy()
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).labels_

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = _as_2d_numpy(X)
        self._check_n_features(X)
        self.model_.eval()
        q_values = []
        with torch.no_grad():
            for (batch,) in self._loader(torch.as_tensor(X, dtype=torch.float32), shuffle=False):
                q, _, _ = self.model_(batch.to(self.device_))
                q_values.append(q.cpu().numpy())
        return np.concatenate(q_values, axis=0)

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = _as_2d_numpy(X)
        self._check_n_features(X)
        self.model_.eval()
        embeddings = []
        with torch.no_grad():
            for (batch,) in self._loader(torch.as_tensor(X, dtype=torch.float32), shuffle=False):
                _, embedding, _ = self.model_(batch.to(self.device_))
                embeddings.append(embedding.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def get_embeddings(self, X=None) -> np.ndarray:
        check_is_fitted(self, "model_")
        if X is None:
            return self.embedding_
        return self.transform(X)

    def _validate_parameters(self, X: np.ndarray) -> None:
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of samples.")
        if len(self.dims) == 0 or min(self.dims) <= 0:
            raise ValueError("dims must contain positive integers.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.pretrain_epochs < 0 or self.max_epochs < 0:
            raise ValueError("pretrain_epochs and max_epochs must be non-negative.")

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _loader(self, data: torch.Tensor, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _pretrain(self, data: torch.Tensor) -> None:
        if self.pretrain_epochs == 0:
            return
        optimizer = torch.optim.Adam(
            self.model_.autoencoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        for epoch in range(self.pretrain_epochs):
            epoch_loss = 0.0
            seen = 0
            self.model_.train()
            for (batch,) in self._loader(data, shuffle=True):
                batch = batch.to(self.device_)
                _, reconstruction = self.model_.autoencoder(batch)
                loss = criterion(reconstruction, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
                seen += batch.size(0)
            self.history_["pretrain_loss"].append(epoch_loss / seen)
            self._log(epoch, "pretrain", self.history_["pretrain_loss"][-1])

    def _initialize_clusters(self, data: torch.Tensor) -> _FitResult:
        embeddings = self.transform(data)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.kmeans_max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        labels = kmeans.fit_predict(embeddings)
        return _FitResult(
            labels=labels,
            embeddings=embeddings,
            centers=kmeans.cluster_centers_,
        )

    def _replace_clustering_layer(self, centers: np.ndarray) -> None:
        self.model_.clustering = _ClusteringLayer(
            n_clusters=self.n_clusters,
            embedding_dim=self.dims[-1],
            alpha=self.alpha,
            initial_centers=centers,
        ).to(self.device_)

    def _finetune(self, data: torch.Tensor) -> None:
        if self.max_epochs == 0:
            return
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        reconstruction_loss = nn.MSELoss()
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            seen = 0
            self.model_.train()
            for (batch,) in self._loader(data, shuffle=True):
                batch = batch.to(self.device_)
                q, _, reconstruction = self.model_(batch)
                with torch.no_grad():
                    p = _target_distribution(q)
                loss = nn.functional.kl_div(q.log(), p, reduction="batchmean")
                if self.reconstruction_weight:
                    loss = loss + self.reconstruction_weight * reconstruction_loss(
                        reconstruction, batch
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
                seen += batch.size(0)
            self.history_["cluster_loss"].append(epoch_loss / seen)
            self._log(epoch, "cluster", self.history_["cluster_loss"][-1])

    def _check_n_features(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this estimator was fitted with "
                f"{self.n_features_in_} features."
            )

    def _log(self, epoch: int, stage: str, loss: float) -> None:
        if self.verbose:
            print(
                f"[{self.algorithm_name}] {stage} epoch={epoch + 1} loss={loss:.6f}"
            )
