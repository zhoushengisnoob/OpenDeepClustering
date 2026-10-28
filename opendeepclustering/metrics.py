"""Validated clustering metrics exposed by the package API."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score


@dataclass(frozen=True)
class ClusteringScores:
    """Named result object for a clustering evaluation."""

    acc: float
    f1: float
    nmi: float
    ari: float

    def as_dict(self):
        return asdict(self)


def _validate_labels(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    return y_true, y_pred


def evaluate_clustering(y_true, y_pred) -> ClusteringScores:
    """Return named ACC, macro-F1, NMI, and ARI scores."""
    y_true, y_pred = _validate_labels(y_true, y_pred)
    acc, f1 = cluster_acc_f1(y_true, y_pred)
    return ClusteringScores(acc, f1, cluster_nmi(y_true, y_pred), cluster_ari(y_true, y_pred))


def eva(y_true, y_pred):
    """Return ACC, macro-F1, NMI, and ARI for clustering predictions."""
    result = evaluate_clustering(y_true, y_pred)
    return result.acc, result.f1, result.nmi, result.ari


def cluster_acc_f1(y_true, y_pred):
    """Return clustering accuracy and macro-F1 after Hungarian label matching."""
    y_true, y_pred = _validate_labels(y_true, y_pred)
    true_values, true_inverse = np.unique(y_true, return_inverse=True)
    pred_values, pred_inverse = np.unique(y_pred, return_inverse=True)
    size = max(len(true_values), len(pred_values))
    contingency = np.zeros((size, size), dtype=np.int64)
    np.add.at(contingency, (pred_inverse, true_inverse), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    mapping = {row: column for row, column in zip(rows, columns)}
    mapped = np.asarray([mapping[index] for index in pred_inverse])
    labels = np.arange(len(true_values))
    return accuracy_score(true_inverse, mapped), f1_score(
        true_inverse, mapped, labels=labels, average="macro", zero_division=0
    )


def cluster_nmi(y_true, y_pred):
    """Return normalized mutual information."""
    return nmi_score(y_true, y_pred, average_method="arithmetic")


def cluster_ari(y_true, y_pred):
    """Return adjusted Rand index."""
    return ari_score(y_true, y_pred)


def embedding_effective_rank(embedding) -> float:
    """Return the effective rank of an embedding as a collapse indicator.

    The value is the exponential of the Shannon entropy of the normalized
    singular-value-squared spectrum of the centered embedding. It ranges from
    1.0, when all variance lies along a single direction, to the embedding
    dimension when variance is spread evenly.

    Unlike counting distinct cluster assignments, this detects representation
    collapse: KMeans returns ``n_clusters`` non-empty clusters regardless of
    whether the embedding carries usable structure.
    """
    array = np.asarray(embedding, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("embedding must be a 2D array of shape (n_samples, n_features).")
    if array.size == 0:
        raise ValueError("embedding must not be empty.")
    centered = array - array.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = float(np.sum(np.square(singular)))
    if energy <= 0:
        return 1.0
    share = np.square(singular) / energy
    positive = share[share > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return float(np.exp(entropy))


__all__ = [
    "ClusteringScores",
    "cluster_acc_f1",
    "cluster_ari",
    "cluster_nmi",
    "embedding_effective_rank",
    "eva",
    "evaluate_clustering",
]
