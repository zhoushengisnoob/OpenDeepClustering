"""Iterative DeepCluster estimator with tabular and image backbones."""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opendeepclustering.base import DeepClusterMixin
from opendeepclustering.components import make_kmeans
from opendeepclustering.estimators._utils import resolve_device
from opendeepclustering.training.callbacks import emit
from opendeepclustering.training.random import SeedManager
from opendeepclustering.training.state import FitState


class _MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, X):
        return self.network(X)


class _CNNBackbone(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        width = max(8, hidden_dim // 2)
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, X):
        return self.projection(self.features(X).flatten(1))


class DeepCluster(DeepClusterMixin):
    """Alternate KMeans pseudo-labeling and discriminative backbone updates.

    Dense 2D inputs use a compact MLP. Four-dimensional NCHW image inputs use
    a compact CNN and stochastic horizontal-flip/noise augmentation during the
    classifier-update stage. The final KMeans model supplies public labels and
    out-of-sample predictions.
    """

    algorithm_name = "DeepCluster"
    taxonomy = "Iterative"
    input_modalities = ("tabular", "image_nchw")

    def __init__(
        self,
        n_clusters: int = 10,
        hidden_dim: int = 64,
        embedding_dim: int = 16,
        rounds: int = 10,
        epochs_per_round: int = 1,
        batch_size: int = 256,
        lr: float = 5e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
        n_init: int = 20,
        kmeans_max_iter: int = 300,
        kmeans_tol: float = 1e-4,
        augment: bool = True,
        augmentation_noise: float = 0.01,
        device: str = "auto",
        random_state=None,
        deterministic: bool = False,
        callbacks=None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.rounds = rounds
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_init = n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol
        self.augment = augment
        self.augmentation_noise = augmentation_noise
        self.device = device
        self.random_state = random_state
        self.deterministic = deterministic
        self.callbacks = callbacks
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit alternating clustering and pseudo-label classification rounds."""
        del y
        X = self._validate_input(X, reset=True)
        self._validate_parameters(len(X))
        self.device_ = resolve_device(self.device)
        self.seed_manager_ = SeedManager(self.random_state)
        self.fit_state_ = FitState(stage="initializing")
        self.augmentation_calls_ = 0
        emit(self.callbacks, "fit_start", self, self.fit_state_)
        data = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)

        with self.seed_manager_.torch_fork(
            self.device_, deterministic=self.deterministic
        ):
            if self.input_kind_ == "image":
                self.model_ = _CNNBackbone(
                    self.input_shape_[0], self.hidden_dim, self.embedding_dim
                ).to(self.device_)
            else:
                self.model_ = _MLPBackbone(
                    self.n_features_in_, self.hidden_dim, self.embedding_dim
                ).to(self.device_)

            for round_index in range(self.rounds):
                self.fit_state_.stage = "pseudo_labeling"
                embeddings = self._extract(data)
                clusterer = self._make_kmeans(round_index)
                pseudo_labels = clusterer.fit_predict(embeddings).astype(np.int64)
                self.fit_state_.record(
                    "active_clusters", float(len(np.unique(pseudo_labels)))
                )

                self.fit_state_.stage = "classifier_update"
                emit(self.callbacks, "stage_start", self, self.fit_state_)
                self.classifier_ = nn.Linear(self.embedding_dim, self.n_clusters).to(
                    self.device_
                )
                optimizer = torch.optim.SGD(
                    [*self.model_.parameters(), *self.classifier_.parameters()],
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
                labels_tensor = torch.as_tensor(pseudo_labels, dtype=torch.long)
                counts = np.bincount(pseudo_labels, minlength=self.n_clusters)
                sample_weights = torch.as_tensor(
                    1.0 / np.maximum(counts[pseudo_labels], 1), dtype=torch.float32
                )
                for _ in range(self.epochs_per_round):
                    sampled = torch.multinomial(
                        sample_weights,
                        num_samples=len(data),
                        replacement=True,
                        generator=self.seed_manager_.torch,
                    )
                    total = 0.0
                    self.model_.train()
                    self.classifier_.train()
                    for start in range(0, len(data), self.batch_size):
                        indices = sampled[start : start + self.batch_size]
                        batch = data[indices]
                        if self.augment:
                            batch = self._augment(batch)
                        batch = batch.to(self.device_)
                        targets = labels_tensor[indices].to(self.device_)
                        logits = self.classifier_(self.model_(batch))
                        loss = nn.functional.cross_entropy(logits, targets)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total += loss.item() * len(batch)
                        self.fit_state_.n_iter += 1
                    self.fit_state_.record("classification_loss", total / len(data))
                self.fit_state_.epoch = round_index + 1
                emit(self.callbacks, "round_end", self, self.fit_state_)

            self.embedding_ = self._extract(data)
            self.clusterer_ = self._make_kmeans(self.rounds)
            self.labels_ = self.clusterer_.fit_predict(self.embedding_).astype(np.int64)
            self.cluster_centers_ = self.clusterer_.cluster_centers_.astype(
                np.float32, copy=True
            )

        self.n_iter_ = self.fit_state_.n_iter
        self.converged_ = True
        self.stop_reason_ = "completed"
        self.fit_state_.stage = "completed"
        self.fit_state_.converged = True
        self.fit_state_.stop_reason = self.stop_reason_
        self.history_ = self.fit_state_.history
        emit(self.callbacks, "fit_end", self, self.fit_state_)
        return self

    def _validate_input(self, X, *, reset: bool):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if sparse.issparse(X):
            return self._validate_X(X, reset=reset)
        array = np.asarray(X)
        if array.ndim <= 2:
            validated = self._validate_X(array, reset=reset)
            if reset:
                self.input_kind_ = "tabular"
                self.input_shape_ = (validated.shape[1],)
            elif self.input_kind_ != "tabular":
                raise ValueError("Expected NCHW image input matching the fitted model.")
            return validated
        if array.ndim != 4:
            raise ValueError("DeepCluster expects a 2D matrix or a 4D NCHW image array.")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("Image input must be numeric.")
        array = np.asarray(array, dtype=np.float32)
        if len(array) == 0 or any(size == 0 for size in array.shape[1:]):
            raise ValueError("Image input dimensions must be non-empty.")
        if not np.isfinite(array).all():
            raise ValueError("Image input must contain only finite values.")
        shape = tuple(array.shape[1:])
        if reset:
            self.input_kind_ = "image"
            self.input_shape_ = shape
            self.n_features_in_ = int(np.prod(shape))
        elif self.input_kind_ != "image" or shape != self.input_shape_:
            raise ValueError(f"Expected image shape {self.input_shape_}, got {shape}.")
        return array

    def _validate_parameters(self, n_samples: int) -> None:
        integer_parameters = {
            "n_clusters": (self.n_clusters, 1),
            "hidden_dim": (self.hidden_dim, 1),
            "embedding_dim": (self.embedding_dim, 1),
            "rounds": (self.rounds, 0),
            "epochs_per_round": (self.epochs_per_round, 0),
            "batch_size": (self.batch_size, 1),
            "n_init": (self.n_init, 1),
            "kmeans_max_iter": (self.kmeans_max_iter, 1),
        }
        for name, (value, minimum) in integer_parameters.items():
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters must not exceed the number of samples.")
        if self.lr <= 0 or self.kmeans_tol <= 0:
            raise ValueError("lr and kmeans_tol must be positive.")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1).")
        if self.weight_decay < 0 or self.augmentation_noise < 0:
            raise ValueError("weight_decay and augmentation_noise must be non-negative.")

    def _make_kmeans(self, round_index: int):
        return make_kmeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.kmeans_max_iter,
            tol=self.kmeans_tol,
            random_state=self.seed_manager_.seed + round_index,
        )

    def _extract(self, data: torch.Tensor) -> np.ndarray:
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for (batch,) in DataLoader(TensorDataset(data), batch_size=self.batch_size):
                outputs.append(self.model_(batch.to(self.device_)).cpu())
        return torch.cat(outputs).numpy()

    def _augment(self, batch: torch.Tensor) -> torch.Tensor:
        self.augmentation_calls_ += 1
        augmented = batch.clone()
        if self.input_kind_ == "image":
            flip = torch.rand(len(batch), generator=self.seed_manager_.torch) < 0.5
            augmented[flip] = torch.flip(augmented[flip], dims=(-1,))
        if self.augmentation_noise:
            noise = torch.randn(
                augmented.shape,
                dtype=augmented.dtype,
                generator=self.seed_manager_.torch,
            )
            augmented = augmented + self.augmentation_noise * noise
        return augmented

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = self._validate_input(X, reset=False)
        data = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
        return self._extract(data)

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "clusterer_")
        return self.clusterer_.predict(self.transform(X)).astype(np.int64)
