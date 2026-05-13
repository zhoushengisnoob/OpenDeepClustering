"""Clustering metrics exposed by the package API."""

import numpy as np
from munkres import Munkres, make_cost_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score


def eva(y_true, y_pred):
    """Return ACC, macro-F1, NMI, and ARI for clustering predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc, f1 = cluster_acc_f1(y_true, y_pred)
    return acc, f1, cluster_nmi(y_true, y_pred), cluster_ari(y_true, y_pred)


def cluster_acc_f1(y_true, y_pred):
    """Return clustering accuracy and macro-F1 after Hungarian label matching."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    y_true = y_true - np.min(y_true)
    y_pred = y_pred - np.min(y_pred)
    n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    weights = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(y_pred, y_true):
        weights[int(pred), int(true)] += 1

    indexes = Munkres().compute(make_cost_matrix(weights))
    label_mapping = {old: new for old, new in indexes}
    mapped_pred = np.asarray([label_mapping[label] for label in y_pred])
    return accuracy_score(y_true, mapped_pred), f1_score(
        y_true, mapped_pred, average="macro"
    )


def cluster_nmi(y_true, y_pred):
    """Return normalized mutual information."""
    return nmi_score(y_true, y_pred, average_method="arithmetic")


def cluster_ari(y_true, y_pred):
    """Return adjusted Rand index."""
    return ari_score(y_true, y_pred)

__all__ = ["cluster_acc_f1", "cluster_ari", "cluster_nmi", "eva"]
