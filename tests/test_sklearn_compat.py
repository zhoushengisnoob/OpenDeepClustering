import pytest
from sklearn.utils.estimator_checks import check_estimator

from opendeepclustering import DEC, IDEC


@pytest.mark.parametrize("estimator_class", [DEC, IDEC])
def test_sklearn_estimator_contract(estimator_class):
    estimator = estimator_class(
        n_clusters=2,
        dims=(4, 2),
        pretrain_epochs=0,
        max_epochs=0,
        batch_size=16,
        n_init=1,
        random_state=0,
        device="cpu",
    )
    check_estimator(estimator)
