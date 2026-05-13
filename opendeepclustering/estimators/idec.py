"""Improved Deep Embedded Clustering estimator."""

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
        super().__init__(
            n_clusters=n_clusters,
            dims=dims,
            alpha=alpha,
            pretrain_epochs=pretrain_epochs,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            n_init=n_init,
            kmeans_max_iter=kmeans_max_iter,
            tol=tol,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.gamma = gamma
        self.reconstruction_weight = gamma
