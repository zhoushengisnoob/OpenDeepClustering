"""Variational Deep Embedding estimator."""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opendeepclustering.base import DeepClusterMixin
from opendeepclustering.components import GaussianMixturePrior
from opendeepclustering.data import to_tensor
from opendeepclustering.estimators._utils import resolve_device, validate_dims
from opendeepclustering.training.callbacks import emit
from opendeepclustering.training.random import SeedManager
from opendeepclustering.training.state import FitState


class _VaDEModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: tuple[int, ...], latent_dim: int, n_clusters: int
    ):
        super().__init__()
        encoder_layers = []
        previous = input_dim
        for width in hidden_dims:
            encoder_layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        self.encoder = nn.Sequential(*encoder_layers)
        self.posterior_mean = nn.Linear(previous, latent_dim)
        self.posterior_log_variance = nn.Linear(previous, latent_dim)

        decoder_layers = []
        previous = latent_dim
        for width in reversed(hidden_dims):
            decoder_layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.prior = GaussianMixturePrior(n_clusters, latent_dim)

    def encode(self, X):
        hidden = self.encoder(X)
        mean = self.posterior_mean(hidden)
        log_variance = self.posterior_log_variance(hidden).clamp(-12.0, 12.0)
        return mean, log_variance

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, log_variance, generator):
        noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype).to(
            mean.device
        )
        return mean + (0.5 * log_variance).exp() * noise


class VaDE(DeepClusterMixin):
    """Variational autoencoder with a trainable Gaussian-mixture latent prior."""

    algorithm_name = "VaDE"
    taxonomy = "Generative"
    input_modalities = ("tabular",)
    supports_soft_assignment = True
    supports_sample = True

    def __init__(
        self,
        n_clusters: int = 10,
        hidden_dims: tuple[int, ...] = (500, 500, 2000),
        latent_dim: int = 10,
        pretrain_epochs: int = 10,
        max_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 2e-3,
        pretrain_lr: float | None = None,
        reconstruction: str = "gaussian",
        beta: float = 1.0,
        n_init: int = 10,
        reg_covar: float = 1e-4,
        device: str = "auto",
        random_state=None,
        deterministic: bool = False,
        callbacks=None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.pretrain_epochs = pretrain_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pretrain_lr = pretrain_lr
        self.reconstruction = reconstruction
        self.beta = beta
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.device = device
        self.random_state = random_state
        self.deterministic = deterministic
        self.callbacks = callbacks
        self.verbose = verbose

    def fit(self, X, y=None):
        """Warm up the autoencoder, initialize the GMM, then optimize the ELBO."""
        del y
        X = self._validate_X(X, reset=True)
        hidden_dims = self._validate_parameters(X)
        self.device_ = resolve_device(self.device)
        self.seed_manager_ = SeedManager(self.random_state)
        self.fit_state_ = FitState(stage="pretraining")
        emit(self.callbacks, "fit_start", self, self.fit_state_)
        data = to_tensor(X)

        with self.seed_manager_.torch_fork(
            self.device_, deterministic=self.deterministic
        ):
            self.model_ = _VaDEModel(
                self.n_features_in_, hidden_dims, self.latent_dim, self.n_clusters
            ).to(self.device_)
            self._pretrain(data)
            self._initialize_mixture(data)
            self._fit_elbo(data)
            self.embedding_ = self._posterior_means(data)
            assignments = self._responsibilities(data)

        self.labels_ = assignments.argmax(axis=1).astype(np.int64)
        self.cluster_centers_ = self.model_.prior.means.detach().cpu().numpy().copy()
        self.n_iter_ = self.fit_state_.n_iter
        self.converged_ = True
        self.stop_reason_ = "completed"
        self.fit_state_.stage = "completed"
        self.fit_state_.converged = True
        self.fit_state_.stop_reason = self.stop_reason_
        self.history_ = self.fit_state_.history
        emit(self.callbacks, "fit_end", self, self.fit_state_)
        return self

    def _validate_parameters(self, X):
        hidden_dims = validate_dims(self.hidden_dims, name="hidden_dims")
        integer_parameters = {
            "n_clusters": (self.n_clusters, 1),
            "latent_dim": (self.latent_dim, 1),
            "pretrain_epochs": (self.pretrain_epochs, 0),
            "max_epochs": (self.max_epochs, 0),
            "batch_size": (self.batch_size, 1),
            "n_init": (self.n_init, 1),
        }
        for name, (value, minimum) in integer_parameters.items():
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.n_clusters > len(X):
            raise ValueError("n_clusters must not exceed the number of samples.")
        if self.lr <= 0 or (self.pretrain_lr is not None and self.pretrain_lr <= 0):
            raise ValueError("lr and pretrain_lr must be positive.")
        if self.beta < 0 or self.reg_covar <= 0:
            raise ValueError("beta must be non-negative and reg_covar must be positive.")
        if self.reconstruction not in {"gaussian", "bernoulli"}:
            raise ValueError("reconstruction must be 'gaussian' or 'bernoulli'.")
        if self.reconstruction == "bernoulli" and (
            np.min(X) < 0 or np.max(X) > 1
        ):
            raise ValueError("Bernoulli reconstruction requires inputs in [0, 1].")
        return hidden_dims

    def _loader(self, data, *, shuffle):
        return DataLoader(
            TensorDataset(data),
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=self.seed_manager_.torch if shuffle else None,
        )

    def _reconstruction_loss(self, output, target):
        if self.reconstruction == "bernoulli":
            return nn.functional.binary_cross_entropy_with_logits(
                output, target, reduction="none"
            ).sum(dim=1)
        return 0.5 * (
            (output - target).square() + math.log(2 * math.pi)
        ).sum(dim=1)

    def _pretrain(self, data):
        if self.pretrain_epochs == 0:
            return
        parameters = [
            *self.model_.encoder.parameters(),
            *self.model_.posterior_mean.parameters(),
            *self.model_.decoder.parameters(),
        ]
        optimizer = torch.optim.Adam(parameters, lr=self.pretrain_lr or self.lr)
        for epoch in range(self.pretrain_epochs):
            total = 0.0
            for (batch,) in self._loader(data, shuffle=True):
                batch = batch.to(self.device_)
                mean, _ = self.model_.encode(batch)
                reconstruction = self.model_.decode(mean)
                loss = self._reconstruction_loss(reconstruction, batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
                self.fit_state_.n_iter += 1
            self.fit_state_.record("pretrain_loss", total / len(data))
            self.fit_state_.epoch = epoch + 1
            emit(self.callbacks, "epoch_end", self, self.fit_state_)

    def _initialize_mixture(self, data):
        self.fit_state_.stage = "mixture_initialization"
        embeddings = self._posterior_means(data)
        mixture = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="diag",
            n_init=self.n_init,
            reg_covar=self.reg_covar,
            random_state=self.seed_manager_.seed,
        ).fit(embeddings)
        self.model_.prior.initialize(
            mixture.weights_, mixture.means_, mixture.covariances_
        )

    def _fit_elbo(self, data):
        self.fit_state_.stage = "elbo"
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        for epoch in range(self.max_epochs):
            total = total_reconstruction = total_kl = 0.0
            self.model_.train()
            for (batch,) in self._loader(data, shuffle=True):
                batch = batch.to(self.device_)
                mean, log_variance = self.model_.encode(batch)
                z = self.model_.reparameterize(
                    mean, log_variance, self.seed_manager_.torch
                )
                reconstruction = self.model_.decode(z)
                reconstruction_loss = self._reconstruction_loss(
                    reconstruction, batch
                )
                kl, _ = self.model_.prior.expected_kl(mean, log_variance)
                loss = (reconstruction_loss + self.beta * kl).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
                total_reconstruction += reconstruction_loss.sum().item()
                total_kl += kl.sum().item()
                self.fit_state_.n_iter += 1
            self.fit_state_.record("elbo_loss", total / len(data))
            self.fit_state_.record(
                "reconstruction_loss", total_reconstruction / len(data)
            )
            self.fit_state_.record("kl_loss", total_kl / len(data))
            self.fit_state_.epoch = epoch + 1
            emit(self.callbacks, "epoch_end", self, self.fit_state_)

    def _posterior_means(self, data):
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for (batch,) in self._loader(data, shuffle=False):
                mean, _ = self.model_.encode(batch.to(self.device_))
                outputs.append(mean.cpu())
        return torch.cat(outputs).numpy()

    def _responsibilities(self, data):
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for (batch,) in self._loader(data, shuffle=False):
                mean, _ = self.model_.encode(batch.to(self.device_))
                outputs.append(self.model_.prior.responsibilities(mean).cpu())
        return torch.cat(outputs).numpy()

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        X = self._validate_X(X, reset=False)
        return self._posterior_means(to_tensor(X))

    def soft_assign(self, X) -> np.ndarray:
        """Return Gaussian-mixture posterior responsibilities."""
        check_is_fitted(self, "model_")
        X = self._validate_X(X, reset=False)
        return self._responsibilities(to_tensor(X))

    def predict_proba(self, X) -> np.ndarray:
        """Alias for mixture posterior responsibilities."""
        return self.soft_assign(X)

    def predict(self, X) -> np.ndarray:
        return self.soft_assign(X).argmax(axis=1).astype(np.int64)

    def sample(self, n_samples: int = 1, cluster: int | None = None) -> np.ndarray:
        """Draw and decode samples from the fitted mixture prior."""
        check_is_fitted(self, "model_")
        if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        if cluster is not None and (
            not isinstance(cluster, (int, np.integer))
            or not 0 <= cluster < self.n_clusters
        ):
            raise ValueError("cluster must be None or a valid cluster index.")
        if cluster is None:
            weights = torch.softmax(self.model_.prior.logits.detach().cpu(), dim=0)
            clusters = torch.multinomial(
                weights,
                num_samples=n_samples,
                replacement=True,
                generator=self.seed_manager_.torch,
            )
        else:
            clusters = torch.full((n_samples,), int(cluster), dtype=torch.long)
        means = self.model_.prior.means.detach().cpu()[clusters]
        scales = (0.5 * self.model_.prior.log_variances.detach().cpu()[clusters]).exp()
        noise = torch.randn(
            means.shape, generator=self.seed_manager_.torch, dtype=means.dtype
        )
        z = (means + scales * noise).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            generated = self.model_.decode(z)
            if self.reconstruction == "bernoulli":
                generated = torch.sigmoid(generated)
        return generated.cpu().numpy()
