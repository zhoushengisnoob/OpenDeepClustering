import numpy as np
import pytest

from opendeepclustering.metrics import (
    cluster_acc_f1,
    embedding_effective_rank,
    eva,
    evaluate_clustering,
)


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


def test_effective_rank_separates_collapsed_from_spread_embeddings():
    rng = np.random.default_rng(0)
    # One direction carries all variance: the degenerate case.
    direction = rng.normal(size=(1, 8))
    collapsed = rng.normal(size=(200, 1)) @ direction
    assert embedding_effective_rank(collapsed) == pytest.approx(1.0, abs=1e-6)

    # Isotropic noise spreads variance across all eight dimensions.
    spread = rng.normal(size=(200, 8))
    assert embedding_effective_rank(spread) > 6.0

    # A constant embedding has no variance at all and reports the floor.
    assert embedding_effective_rank(np.ones((10, 4))) == 1.0


def test_effective_rank_validates_input():
    with pytest.raises(ValueError, match="2D array"):
        embedding_effective_rank(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="must not be empty"):
        embedding_effective_rank(np.zeros((0, 3)))
