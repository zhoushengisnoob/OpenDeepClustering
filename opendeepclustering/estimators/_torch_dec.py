"""Paper-traceable PyTorch core for DEC-family estimators."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opendeepclustering.base import DeepClusterMixin
from opendeepclustering.components import (
    StackedAutoEncoder,
    StudentTClustering,
    make_kmeans,
    target_distribution,
)
from opendeepclustering.data import IndexedTensorDataset, to_tensor
from opendeepclustering.training.callbacks import emit
from opendeepclustering.training.random import SeedManager
from opendeepclustering.training.state import FitState


class _DECModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dims: Iterable[int],
        n_clusters: int,
        alpha: float,
    ):
        super().__init__()
        dims = tuple(dims)
        self.autoencoder = StackedAutoEncoder(input_dim, dims)
        self.clustering = StudentTClustering(n_clusters, dims[-1], alpha)

    def forward(self, X: torch.Tensor):
        embedding, reconstruction = self.autoencoder(X)
        assignment = self.clustering(embedding)
        return assignment, embedding, reconstruction


class _BaseDEC(DeepClusterMixin):
    """Shared implementation for DEC and IDEC.

    The clustering target is computed over the complete training set and
    indexed back into shuffled mini-batches, matching the DEC-family papers.
    """

    algorithm_name = "DEC"
    taxonomy = "Simultaneous"
    supports_soft_assignment = True

    def __init__(
        self,
        n_clusters: int = 10,
        dims: tuple[int, ...] = (500, 500, 2000, 10),
        alpha: float = 1.0,
        pretrain_epochs: int = 50,
        layerwise_pretrain_epochs: int | None = None,
        pretrain_method: str = "joint",
        corruption: float = 0.2,
        max_epochs: int = 100,
        update_interval: int = 140,
        batch_size: int = 256,
        num_workers: int = 0,
        lr: float = 1e-3,
        pretrain_lr: float | None = None,
        optimizer: str = "adam",
        pretrain_optimizer: str | None = None,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        n_init: int = 20,
        kmeans_max_iter: int = 300,
        kmeans_tol: float = 1e-4,
        tol: float = 1e-3,
        device: str = "auto",
        random_state=None,
        deterministic: bool = False,
        resume_from: str | None = None,
        callbacks=None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.dims = dims
        self.alpha = alpha
        self.pretrain_epochs = pretrain_epochs
        self.layerwise_pretrain_epochs = layerwise_pretrain_epochs
        self.pretrain_method = pretrain_method
        self.corruption = corruption
        self.max_epochs = max_epochs
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.pretrain_lr = pretrain_lr
        self.optimizer = optimizer
        self.pretrain_optimizer = pretrain_optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_init = n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol
        self.tol = tol
        self.device = device
        self.random_state = random_state
        self.deterministic = deterministic
        self.resume_from = resume_from
        self.callbacks = callbacks
        self.verbose = verbose

    def fit(self, X, y=None):
        X = self._validate_X(X, reset=True)
        self._validate_parameters(X)
        self.device_ = self._resolve_device()
        self.seed_manager_ = SeedManager(self.random_state)
        self.fit_state_ = FitState()
        emit(self.callbacks, "fit_start", self, self.fit_state_)

        data = to_tensor(X)
        with self.seed_manager_.torch_fork(
            self.device_, deterministic=self.deterministic
        ):
            self.model_ = _DECModel(
                input_dim=self.n_features_in_,
                dims=self.dims,
                n_clusters=self.n_clusters,
                alpha=self.alpha,
            ).to(self.device_)
            if self.resume_from is not None:
                initial_labels = self._restore_checkpoint(data)
            else:
                if self.pretrain_method == "greedy":
                    self._initialize_reference_weights()
                self._pretrain(data)
                initial_labels, centers = self._initialize_clusters(data)
                self.model_.clustering = StudentTClustering(
                    self.n_clusters,
                    self.dims[-1],
                    self.alpha,
                    centers,
                ).to(self.device_)
            self._finetune(data, initial_labels)

        assignments, embeddings, _ = self._all_outputs(data)
        self.labels_ = assignments.argmax(dim=1).numpy().astype(np.int64)
        self.embedding_ = embeddings.numpy()
        self.cluster_centers_ = (
            self.model_.clustering.centroids.detach().cpu().numpy().copy()
        )
        self.n_iter_ = self.fit_state_.n_iter
        self.converged_ = self.fit_state_.converged
        if self.fit_state_.stop_reason is None:
            self.fit_state_.stop_reason = "max_epochs"
        self.stop_reason_ = self.fit_state_.stop_reason
        self.history_ = self.fit_state_.history
        emit(self.callbacks, "fit_end", self, self.fit_state_)
        return self

    def save_checkpoint(self, path) -> None:
        """Save explicit resumable state; fitting itself never writes files."""
        check_is_fitted(self, "model_")
        torch.save(
            {
                "model_state": self.model_.state_dict(),
                "optimizer_state": getattr(self, "optimizer_state_", None),
                "fit_state": self.fit_state_.as_dict(),
                "n_features_in": self.n_features_in_,
            },
            path,
        )

    def _restore_checkpoint(self, data: torch.Tensor) -> np.ndarray:
        checkpoint = torch.load(
            self.resume_from, map_location=self.device_, weights_only=True
        )
        if checkpoint.get("n_features_in") != self.n_features_in_:
            raise ValueError("Checkpoint feature count does not match X.")
        self.model_.load_state_dict(checkpoint["model_state"])
        state = checkpoint.get("fit_state", {})
        self.fit_state_ = FitState(**state)
        self.fit_state_.converged = False
        self.fit_state_.stop_requested = False
        self.fit_state_.stop_reason = None
        self._resume_optimizer_state = checkpoint.get("optimizer_state")
        self._just_resumed = True
        return self._soft_assign_tensor(data).argmax(dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.soft_assign(X).argmax(axis=1).astype(np.int64)

    def soft_assign(self, X) -> np.ndarray:
        """Return Student-t cluster assignments for each sample."""
        check_is_fitted(self, "model_")
        X = self._validate_X(X, reset=False)
        assignments, _, _ = self._all_outputs(to_tensor(X))
        return assignments.numpy()

    def predict_proba(self, X) -> np.ndarray:
        """Alias for soft_assign; outputs are not calibrated probabilities."""
        return self.soft_assign(X)

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = self._validate_X(X, reset=False)
        _, embeddings, _ = self._all_outputs(to_tensor(X))
        return embeddings.numpy()

    def _validate_parameters(self, X: np.ndarray) -> None:
        integer_parameters = {
            "n_clusters": self.n_clusters,
            "pretrain_epochs": self.pretrain_epochs,
            "max_epochs": self.max_epochs,
            "update_interval": self.update_interval,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "n_init": self.n_init,
            "kmeans_max_iter": self.kmeans_max_iter,
        }
        for name, value in integer_parameters.items():
            minimum = (
                0
                if name in {"pretrain_epochs", "max_epochs", "num_workers"}
                else 1
            )
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )
        if not isinstance(self.dims, (tuple, list)) or not self.dims:
            raise ValueError("dims must be a non-empty sequence of positive integers.")
        if any(not isinstance(dim, int) or dim <= 0 for dim in self.dims):
            raise ValueError("dims must contain only positive integers.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not 0 <= self.corruption < 1:
            raise ValueError("corruption must be in [0, 1).")
        if self.pretrain_method not in {"joint", "greedy"}:
            raise ValueError("pretrain_method must be 'joint' or 'greedy'.")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'.")
        if self.pretrain_optimizer not in {None, "adam", "sgd"}:
            raise ValueError("pretrain_optimizer must be None, 'adam', or 'sgd'.")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1).")
        if self.layerwise_pretrain_epochs is not None and (
            not isinstance(self.layerwise_pretrain_epochs, int)
            or self.layerwise_pretrain_epochs < 0
        ):
            raise ValueError("layerwise_pretrain_epochs must be None or >= 0.")
        for name in ("lr", "kmeans_tol", "tol"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.pretrain_lr is not None and self.pretrain_lr <= 0:
            raise ValueError("pretrain_lr must be None or positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            resolved = torch.device(self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid device: {self.device!r}.") from exc
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return resolved

    def _loader(self, data: torch.Tensor, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            IndexedTensorDataset(data),
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=self.seed_manager_.torch if shuffle else None,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_manager_.seed_worker if self.num_workers else None,
        )

    def _initialize_reference_weights(self) -> None:
        for module in self.model_.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def _pretrain(self, data: torch.Tensor) -> None:
        self.fit_state_.stage = "pretraining"
        emit(self.callbacks, "stage_start", self, self.fit_state_)
        if self.pretrain_epochs == 0:
            emit(self.callbacks, "stage_end", self, self.fit_state_)
            return
        if self.pretrain_method == "greedy":
            self._greedy_layerwise_pretrain(data)
        self._joint_autoencoder_pretrain(data)
        emit(self.callbacks, "stage_end", self, self.fit_state_)

    def _greedy_layerwise_pretrain(self, data: torch.Tensor) -> None:
        epochs = (
            self.pretrain_epochs
            if self.layerwise_pretrain_epochs is None
            else self.layerwise_pretrain_epochs
        )
        if epochs == 0:
            return
        current = data
        autoencoder = self.model_.autoencoder
        criterion = nn.MSELoss()
        for layer_index, (encoder, decoder) in enumerate(
            zip(autoencoder.encoders, autoencoder.decoders)
        ):
            optimizer = self._make_optimizer(
                [*encoder.parameters(), *decoder.parameters()],
                lr=self.pretrain_lr or self.lr,
                name=self.pretrain_optimizer or "sgd",
            )
            for _ in range(epochs):
                total_loss = 0.0
                seen = 0
                loader = DataLoader(
                    TensorDataset(current),
                    batch_size=self.batch_size,
                    shuffle=True,
                    generator=self.seed_manager_.torch,
                )
                for (batch,) in loader:
                    batch = batch.to(self.device_)
                    corrupted = nn.functional.dropout(
                        batch, p=self.corruption, training=True
                    )
                    hidden = encoder(corrupted)
                    if layer_index != len(autoencoder.encoders) - 1:
                        hidden = torch.relu(hidden)
                    hidden = nn.functional.dropout(
                        hidden, p=self.corruption, training=True
                    )
                    reconstruction = decoder(hidden)
                    if layer_index != 0:
                        reconstruction = torch.relu(reconstruction)
                    loss = criterion(reconstruction, batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch.shape[0]
                    seen += batch.shape[0]
                self.fit_state_.record(
                    f"layerwise_loss_{layer_index}", total_loss / seen
                )
            with torch.no_grad():
                transformed = []
                for (batch,) in DataLoader(
                    TensorDataset(current), batch_size=self.batch_size
                ):
                    batch = encoder(batch.to(self.device_))
                    if layer_index != len(autoencoder.encoders) - 1:
                        batch = torch.relu(batch)
                    transformed.append(batch.cpu())
                current = torch.cat(transformed)

    def _joint_autoencoder_pretrain(self, data: torch.Tensor) -> None:
        optimizer = self._make_optimizer(
            self.model_.autoencoder.parameters(),
            lr=self.pretrain_lr or self.lr,
            name=self.pretrain_optimizer or self.optimizer,
        )
        criterion = nn.MSELoss()
        for epoch in range(self.pretrain_epochs):
            self.fit_state_.epoch = epoch + 1
            total_loss = 0.0
            seen = 0
            self.model_.train()
            for batch, _ in self._loader(data, shuffle=True):
                batch = batch.to(self.device_)
                _, reconstruction = self.model_.autoencoder(batch)
                loss = criterion(reconstruction, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.shape[0]
                seen += batch.shape[0]
            average = total_loss / seen
            self.fit_state_.record("pretrain_loss", average)
            self._log("pretrain", epoch + 1, average)
            emit(self.callbacks, "epoch_end", self, self.fit_state_)

    def _initialize_clusters(self, data: torch.Tensor):
        embeddings = self._encode_tensor(data).numpy()
        kmeans = make_kmeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.kmeans_max_iter,
            tol=self.kmeans_tol,
            random_state=self.random_state,
        )
        labels = kmeans.fit_predict(embeddings).astype(np.int64)
        return labels, kmeans.cluster_centers_.astype(np.float32)

    def _finetune(self, data: torch.Tensor, initial_labels: np.ndarray) -> None:
        self.fit_state_.stage = "clustering"
        emit(self.callbacks, "stage_start", self, self.fit_state_)
        if self.max_epochs == 0:
            emit(self.callbacks, "stage_end", self, self.fit_state_)
            return

        optimizer = self._make_optimizer(
            self._finetune_parameters(),
            lr=self.lr,
            name=self.optimizer,
        )
        if getattr(self, "_resume_optimizer_state", None) is not None:
            optimizer.load_state_dict(self._resume_optimizer_state)
        previous_labels = initial_labels
        targets = None
        should_stop = False
        for epoch in range(self.max_epochs):
            self.fit_state_.epoch = epoch + 1
            epoch_total = epoch_cluster = epoch_reconstruction = 0.0
            seen = 0
            self.model_.train()
            for batch, indices in self._loader(data, shuffle=True):
                if targets is None or self.fit_state_.n_iter % self.update_interval == 0:
                    assignments = self._soft_assign_tensor(data)
                    targets = target_distribution(assignments).cpu()
                    labels = assignments.argmax(dim=1).cpu().numpy()
                    delta = float(np.mean(labels != previous_labels))
                    self.fit_state_.record("delta_label", delta)
                    if (
                        self.fit_state_.n_iter > 0
                        and not getattr(self, "_just_resumed", False)
                        and delta < self.tol
                    ):
                        self.fit_state_.converged = True
                        self.fit_state_.stop_reason = "label_change_below_tol"
                        should_stop = True
                        break
                    previous_labels = labels
                    self._just_resumed = False

                self.model_.train()

                batch = batch.to(self.device_)
                target = targets[indices].to(self.device_)
                assignment, _, reconstruction = self.model_(batch)
                cluster_loss = nn.functional.kl_div(
                    assignment.clamp_min(1e-12).log(),
                    target,
                    reduction="batchmean",
                )
                reconstruction_loss = nn.functional.mse_loss(reconstruction, batch)
                total_loss = self._combine_losses(
                    cluster_loss, reconstruction_loss
                )
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_size = batch.shape[0]
                epoch_total += total_loss.item() * batch_size
                epoch_cluster += cluster_loss.item() * batch_size
                epoch_reconstruction += reconstruction_loss.item() * batch_size
                seen += batch_size
                self.fit_state_.n_iter += 1
                self.fit_state_.step = self.fit_state_.n_iter

            if seen:
                self.fit_state_.record("total_loss", epoch_total / seen)
                self.fit_state_.record("cluster_loss", epoch_cluster / seen)
                self.fit_state_.record(
                    "reconstruction_loss", epoch_reconstruction / seen
                )
                self._log("cluster", epoch + 1, epoch_total / seen)
                self.optimizer_state_ = optimizer.state_dict()
                emit(self.callbacks, "epoch_end", self, self.fit_state_)
                if self.fit_state_.stop_requested:
                    self.fit_state_.stop_reason = (
                        self.fit_state_.stop_reason or "callback"
                    )
                    should_stop = True
            if should_stop:
                break
        if not self.fit_state_.converged and self.fit_state_.stop_reason is None:
            self.fit_state_.stop_reason = "max_epochs"
        self.optimizer_state_ = optimizer.state_dict()
        emit(self.callbacks, "stage_end", self, self.fit_state_)

    def _finetune_parameters(self):
        return [
            *self.model_.autoencoder.encoders.parameters(),
            *self.model_.clustering.parameters(),
        ]

    def _make_optimizer(self, parameters, *, lr: float, name: str):
        if name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        return torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=self.weight_decay,
        )

    def _combine_losses(
        self,
        cluster_loss: torch.Tensor,
        reconstruction_loss: torch.Tensor,
    ) -> torch.Tensor:
        del reconstruction_loss
        return cluster_loss

    def _encode_tensor(self, data: torch.Tensor) -> torch.Tensor:
        outputs = []
        self.model_.eval()
        with torch.no_grad():
            for batch, _ in self._loader(data, shuffle=False):
                outputs.append(
                    self.model_.autoencoder.encode(batch.to(self.device_)).cpu()
                )
        return torch.cat(outputs)

    def _soft_assign_tensor(self, data: torch.Tensor) -> torch.Tensor:
        assignments, _, _ = self._all_outputs(data)
        return assignments

    def _all_outputs(self, data: torch.Tensor):
        assignments = []
        embeddings = []
        reconstructions = []
        self.model_.eval()
        with torch.no_grad():
            for batch, _ in self._loader(data, shuffle=False):
                assignment, embedding, reconstruction = self.model_(
                    batch.to(self.device_)
                )
                assignments.append(assignment.cpu())
                embeddings.append(embedding.cpu())
                reconstructions.append(reconstruction.cpu())
        return (
            torch.cat(assignments),
            torch.cat(embeddings),
            torch.cat(reconstructions),
        )

    def _log(self, stage: str, epoch: int, loss: float) -> None:
        if self.verbose:
            print(f"[{self.algorithm_name}] {stage} epoch={epoch} loss={loss:.6f}")
