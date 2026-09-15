"""Probabilistic components for generative deep clustering."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class GaussianMixturePrior(nn.Module):
    """Trainable categorical mixture of diagonal Gaussian latent priors."""

    def __init__(self, n_clusters: int, latent_dim: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_clusters))
        self.means = nn.Parameter(torch.empty(n_clusters, latent_dim))
        self.log_variances = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        nn.init.xavier_uniform_(self.means)

    def initialize(self, weights, means, variances) -> None:
        """Initialize mixture parameters from a fitted diagonal GMM."""
        with torch.no_grad():
            weights = torch.as_tensor(weights, dtype=self.logits.dtype)
            self.logits.copy_(weights.clamp_min(1e-12).log())
            self.means.copy_(torch.as_tensor(means, dtype=self.means.dtype))
            self.log_variances.copy_(
                torch.as_tensor(variances, dtype=self.log_variances.dtype)
                .clamp_min(1e-6)
                .log()
            )

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``log p(c, z)`` for every sample and cluster."""
        difference = z.unsqueeze(1) - self.means.unsqueeze(0)
        log_density = -0.5 * (
            math.log(2 * math.pi)
            + self.log_variances.unsqueeze(0)
            + difference.square() / self.log_variances.exp().unsqueeze(0)
        ).sum(dim=2)
        return torch.log_softmax(self.logits, dim=0).unsqueeze(0) + log_density

    def responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        """Return normalized mixture posterior ``p(c | z)``."""
        return torch.softmax(self.log_joint(z), dim=1)

    def expected_kl(
        self, posterior_mean: torch.Tensor, posterior_log_variance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return analytic VaDE KL per sample and variational responsibilities."""
        mean = posterior_mean.unsqueeze(1)
        variance = posterior_log_variance.exp().unsqueeze(1)
        prior_mean = self.means.unsqueeze(0)
        prior_log_variance = self.log_variances.unsqueeze(0)
        prior_variance = prior_log_variance.exp()
        gaussian_cross_entropy = -0.5 * (
            math.log(2 * math.pi)
            + prior_log_variance
            + (variance + (mean - prior_mean).square()) / prior_variance
        ).sum(dim=2)
        log_pi = torch.log_softmax(self.logits, dim=0).unsqueeze(0)
        log_responsibility = torch.log_softmax(log_pi + gaussian_cross_entropy, dim=1)
        responsibility = log_responsibility.exp()
        gaussian_kl = 0.5 * (
            prior_log_variance
            - posterior_log_variance.unsqueeze(1)
            + (variance + (mean - prior_mean).square()) / prior_variance
            - 1.0
        ).sum(dim=2)
        categorical_kl = log_responsibility - log_pi
        kl = (responsibility * (gaussian_kl + categorical_kl)).sum(dim=1)
        return kl, responsibility

    @property
    def weights(self) -> np.ndarray:
        return torch.softmax(self.logits.detach(), dim=0).cpu().numpy()
