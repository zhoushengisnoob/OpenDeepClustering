import numpy as np
import torch
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

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


def test_reproducible_fit_and_sklearn_clone():
    X = make_data()
    prototype = DEC(
        n_clusters=2,
        dims=(6, 2),
        pretrain_epochs=1,
        max_epochs=2,
        update_interval=1,
        batch_size=8,
        n_init=2,
        random_state=11,
        deterministic=True,
        device="cpu",
    )
    first = clone(prototype).fit(X)
    second = clone(prototype).fit(X)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_allclose(first.embedding_, second.embedding_)
    assert first.get_params()["update_interval"] == 1


def test_callbacks_and_fitted_state_are_exposed():
    events = []
    model = DEC(
        n_clusters=2,
        dims=(4, 2),
        pretrain_epochs=0,
        max_epochs=0,
        n_init=1,
        callbacks=[lambda event, estimator, state: events.append(event)],
        random_state=0,
        device="cpu",
    ).fit(make_data())
    assert events[0] == "fit_start"
    assert events[-1] == "fit_end"
    assert model.stop_reason_ == "max_epochs"
    np.testing.assert_allclose(model.get_embeddings(), model.embedding_)


def test_idec_paper_loss_weights_clustering_term():
    model = IDEC(gamma=0.25)
    clustering = torch.tensor(2.0)
    reconstruction = torch.tensor(3.0)
    assert model._combine_losses(clustering, reconstruction).item() == 3.5


def test_prefit_error_and_checkpoint_resume(tmp_path):
    X = make_data()
    model = DEC(
        n_clusters=2,
        dims=(4, 2),
        pretrain_epochs=0,
        max_epochs=1,
        update_interval=1,
        n_init=1,
        random_state=5,
        device="cpu",
    )
    with pytest.raises(NotFittedError):
        model.transform(X)
    model.fit(X)
    checkpoint = tmp_path / "state.pt"
    model.save_checkpoint(checkpoint)
    resumed = clone(model).set_params(resume_from=str(checkpoint), max_epochs=1).fit(X)
    assert resumed.n_iter_ > model.n_iter_


def test_target_update_schedule_uses_global_passes():
    class CountingDEC(DEC):
        def _soft_assign_tensor(self, data):
            self.target_updates_ = getattr(self, "target_updates_", 0) + 1
            first = 0.9 if self.target_updates_ % 2 else 0.1
            q = torch.empty((len(data), 2))
            q[:, 0] = first
            q[:, 1] = 1 - first
            return q

    model = CountingDEC(
        n_clusters=2,
        dims=(4, 2),
        pretrain_epochs=0,
        max_epochs=2,
        update_interval=2,
        batch_size=8,
        n_init=1,
        tol=1e-12,
        random_state=0,
        device="cpu",
    ).fit(make_data())
    assert model.n_iter_ == 6
    assert model.target_updates_ == 3


def test_image_inputs_require_explicit_adapter():
    with pytest.raises(ValueError, match="flatten_samples"):
        DEC(n_clusters=2, dims=(2,), pretrain_epochs=0, max_epochs=0).fit(
            np.zeros((4, 2, 2), dtype=np.float32)
        )
