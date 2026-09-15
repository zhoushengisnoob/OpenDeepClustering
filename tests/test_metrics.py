import numpy as np
import pytest

from opendeepclustering.metrics import cluster_acc_f1, eva, evaluate_clustering


def test_metrics_support_string_and_non_contiguous_labels():
    truth = np.array(["cat", "cat", "dog", "dog"])
    predicted = np.array([20, 20, 10, 10])
    result = evaluate_clustering(truth, predicted)
    assert result.acc == result.f1 == result.nmi == result.ari == 1.0
    assert eva(truth, predicted) == (1.0, 1.0, 1.0, 1.0)


def test_metrics_validate_shape_and_empty_input():
    with pytest.raises(ValueError, match="same shape"):
        cluster_acc_f1([0], [0, 1])
    with pytest.raises(ValueError, match="must not be empty"):
        cluster_acc_f1([], [])
