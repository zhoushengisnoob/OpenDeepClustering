import pytest
from sklearn.utils.estimator_checks import check_estimator

from opendeepclustering import AutoencoderKMeans, DEC, DeepCluster, IDEC, VaDE


@pytest.mark.parametrize(
    ("estimator_class", "extra"),
    [
        (AutoencoderKMeans, {"dims": (4, 2), "max_epochs": 0}),
        (DEC, {"dims": (4, 2), "pretrain_epochs": 0, "max_epochs": 0}),
        (DeepCluster, {"hidden_dim": 4, "embedding_dim": 2, "rounds": 0}),
        (IDEC, {"dims": (4, 2), "pretrain_epochs": 0, "max_epochs": 0}),
        (
            VaDE,
            {
                "hidden_dims": (4,),
                "latent_dim": 2,
                "pretrain_epochs": 0,
                "max_epochs": 0,
            },
        ),
    ],
)
def test_sklearn_estimator_contract(estimator_class, extra):
    estimator = estimator_class(
        n_clusters=2,
        batch_size=16,
        n_init=1,
        random_state=0,
        device="cpu",
        **extra,
    )
    check_estimator(estimator)
