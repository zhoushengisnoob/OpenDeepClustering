"""Deep Embedded Clustering estimator."""

from opendeepclustering.estimators._torch_dec import _BaseDEC


class DEC(_BaseDEC):
    """Deep Embedded Clustering with a scikit-learn style API.

    Examples
    --------
    >>> from opendeepclustering import DEC
    >>> model = DEC(n_clusters=10, pretrain_epochs=10, max_epochs=20)
    >>> labels = model.fit_predict(X)
    >>> embeddings = model.transform(X)
    """

    algorithm_name = "DEC"
    taxonomy = "Simultaneous"
    reconstruction_weight = 0.0
