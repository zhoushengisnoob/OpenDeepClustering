# -*- coding: utf-8 -*-
"""
This is a program for evaluation of clustering.
Author: Guanbao Liang
License: BSD 2 clause
"""

import numpy as np

from sklearn.metrics import accuracy_score as acc_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score


from munkres import Munkres, make_cost_matrix


def eva(y_true, y_pred):
    """
    Function calculates the acc(accuracy), f1(F1 score), nmi(Normalized Mutual Information), ari(Adjusted Rand Index).

    Parameters
    ----------
    y_true : List[int]
        The ground-truth labels.
    y_pred : List[int]
        The predictions of model.

    Returns
    -------
    acc : float
        The accuracy.
    f1 : float
        The F1 score.
    nmi : float
        The Normalized Mutual Information.
    ari : float
        The Adjusted Rand Index.
    """
    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    acc, f1 = cluster_acc_f1(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    return acc, f1, nmi, ari


def cluster_acc_f1(y_true, y_pred):
    """
    Function calculates the acc(accuracy), f1(F1 score), nmi(Normalized Mutual Information), ari(Adjusted Rand Index).

    Parameters
    ----------
    y_true : List[int]
        The ground-truth labels.
    y_pred : List[int]
        The predictions of model.

    Returns
    -------
    acc : float
        The accuracy.
    f1 : float
        The F1 score.
    """
    assert y_true.shape == y_pred.shape
    y_true = y_true - np.min(y_true)
    y_pred = y_pred - np.min(y_pred)

    sample_size = len(y_pred)
    classes = max(y_true.max(), y_pred.max()) + 1
    w = np.zeros((classes, classes))

    for i in range(sample_size):
        w[y_pred[i], y_true[i]] += 1

    w = w.astype(int)
    cost = make_cost_matrix(w)
    m = Munkres()
    indexes = m.compute(cost)
    old_pred_label = [item[0] for item in indexes]
    new_pred_label = [item[1] for item in indexes]
    mapping = dict(zip(old_pred_label, new_pred_label))
    y_pred_new = [mapping[i] for i in y_pred]
    acc = acc_score(y_true, y_pred_new)
    f1 = f1_score(y_true, y_pred_new, average="macro")
    return acc, f1


def cluster_nmi(y_true, y_pred):
    """
    Function calculates the acc(accuracy), f1(F1 score), nmi(Normalized Mutual Information), ari(Adjusted Rand Index).

    Parameters
    ----------
    y_true : List[int]
        The ground-truth labels.
    y_pred : List[int]
        The predictions of model.

    Returns
    -------
    nmi : float
        The Normalized Mutual Information.
    """
    return nmi_score(y_true, y_pred, average_method="arithmetic")


def cluster_ari(y_true, y_pred):
    """
    Function calculates the acc(accuracy), f1(F1 score), nmi(Normalized Mutual Information), ari(Adjusted Rand Index).

    Parameters
    ----------
    y_true : List[int]
        The ground-truth labels.
    y_pred : List[int]
        The predictions of model.

    Returns
    -------
    ari : float
        The Adjusted Rand Index.
    """
    return ari_score(y_true, y_pred)
