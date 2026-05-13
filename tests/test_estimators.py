import numpy as np

from opendeepclustering import DEC, IDEC


def make_data():
    rng = np.random.default_rng(0)
    left = rng.normal(loc=-2.0, scale=0.2, size=(12, 4))
    right = rng.normal(loc=2.0, scale=0.2, size=(12, 4))
    return np.vstack([left, right]).astype("float32")


def test_dec_fit_predict_transform_shapes():
    X = make_data()
    model = DEC(
        n_clusters=2,
        dims=(8, 2),
        pretrain_epochs=1,
        max_epochs=1,
        batch_size=8,
        random_state=0,
        device="cpu",
    )

    labels = model.fit_predict(X)

    assert labels.shape == (24,)
    assert model.transform(X).shape == (24, 2)
    assert model.predict_proba(X).shape == (24, 2)


def test_idec_fit_predict_transform_shapes():
    X = make_data()
    model = IDEC(
        n_clusters=2,
        dims=(8, 2),
        gamma=0.1,
        pretrain_epochs=1,
        max_epochs=1,
        batch_size=8,
        random_state=0,
        device="cpu",
    )

    labels = model.fit_predict(X)

    assert labels.shape == (24,)
    assert model.transform(X).shape == (24, 2)
