"""Improved Deep Embedded Clustering estimator."""

from __future__ import annotations

import torch

from opendeepclustering.estimators._torch_dec import _BaseDEC


class IDEC(_BaseDEC):
    """IDEC estimator with reconstruction preservation during clustering."""

    algorithm_name = "IDEC"
    taxonomy = "Simultaneous"

    def __init__(
        self,
        n_clusters: int = 10,
        dims: tuple[int, ...] = (500, 500, 2000, 10),
        alpha: float = 1.0,
        gamma: float = 0.1,
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
        super().__init__(
            n_clusters=n_clusters,
            dims=dims,
            alpha=alpha,
            pretrain_epochs=pretrain_epochs,
            layerwise_pretrain_epochs=layerwise_pretrain_epochs,
            pretrain_method=pretrain_method,
            corruption=corruption,
            max_epochs=max_epochs,
            update_interval=update_interval,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            pretrain_lr=pretrain_lr,
            optimizer=optimizer,
            pretrain_optimizer=pretrain_optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            n_init=n_init,
            kmeans_max_iter=kmeans_max_iter,
            kmeans_tol=kmeans_tol,
            tol=tol,
            device=device,
            random_state=random_state,
            deterministic=deterministic,
            resume_from=resume_from,
            callbacks=callbacks,
            verbose=verbose,
        )
        self.gamma = gamma

    def _validate_parameters(self, X):
        super()._validate_parameters(X)
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative.")

    def _finetune_parameters(self):
        return self.model_.parameters()

    def _combine_losses(
        self,
        clustering_loss: torch.Tensor,
        reconstruction_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Use the paper objective: reconstruction + gamma * clustering."""
        return reconstruction_loss + self.gamma * clustering_loss
