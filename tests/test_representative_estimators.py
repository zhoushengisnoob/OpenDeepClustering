import json

import numpy as np
import pytest
import yaml
from sklearn.cluster import MiniBatchKMeans

from opendeepclustering import AutoencoderKMeans, DeepCluster, VaDE
from opendeepclustering.cli import run_benchmark


def make_tabular_data():
    rng = np.random.default_rng(4)
    left = rng.normal(-1.5, 0.15, size=(12, 6))
    right = rng.normal(1.5, 0.15, size=(12, 6))
    return np.vstack((left, right)).astype(np.float32)


def make_image_data():
    rng = np.random.default_rng(5)
    images = np.zeros((16, 1, 8, 8), dtype=np.float32)
    images[:8, :, :, :3] = 1.0
    images[8:, :, :, 5:] = 1.0
    images += rng.normal(0, 0.02, size=images.shape).astype(np.float32)
    return images


def test_autoencoder_kmeans_composes_arbitrary_shallow_estimator():
    X = make_tabular_data()
    shallow = MiniBatchKMeans(n_clusters=2, n_init=1, random_state=9)
    model = AutoencoderKMeans(
        n_clusters=2,
        dims=(8, 2),
        clusterer=shallow,
        max_epochs=1,
        batch_size=8,
        random_state=3,
        deterministic=True,
        device="cpu",
    ).fit(X)
    assert model.clusterer_ is not shallow
    assert model.labels_.shape == (24,)
    assert model.transform(X).shape == (24, 2)
    assert model.predict(X).shape == (24,)
    assert model.get_capabilities()["taxonomy"] == "Multi-stage"


def test_deepcluster_runs_image_augmentation_and_alternating_round():
    X = make_image_data()
    model = DeepCluster(
        n_clusters=2,
        hidden_dim=8,
        embedding_dim=3,
        rounds=1,
        epochs_per_round=1,
        batch_size=4,
        n_init=1,
        augment=True,
        random_state=2,
        deterministic=True,
        device="cpu",
    ).fit(X)
    assert model.labels_.shape == (16,)
    assert model.transform(X).shape == (16, 3)
    assert model.predict(X).shape == (16,)
    assert model.augmentation_calls_ > 0
    assert len(model.history_["active_clusters"]) == 1
    assert model.get_capabilities()["input_modalities"] == (
        "tabular",
        "image_nchw",
    )


def test_vade_exposes_posterior_and_generation_interfaces():
    X = make_tabular_data()
    model = VaDE(
        n_clusters=2,
        hidden_dims=(8,),
        latent_dim=2,
        pretrain_epochs=1,
        max_epochs=1,
        batch_size=8,
        n_init=1,
        random_state=7,
        deterministic=True,
        device="cpu",
    ).fit(X)
    probabilities = model.predict_proba(X)
    assert model.transform(X).shape == (24, 2)
    assert probabilities.shape == (24, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert model.sample(3).shape == (3, 6)
    assert model.sample(2, cluster=1).shape == (2, 6)
    assert model.get_capabilities()["sample"] is True


@pytest.mark.parametrize(
    ("estimator", "parameters", "dataset"),
    [
        (
            "AutoencoderKMeans",
            {"n_clusters": 2, "dims": [6, 2], "max_epochs": 0, "n_init": 1},
            {},
        ),
        (
            "DeepCluster",
            {
                "n_clusters": 2,
                "hidden_dim": 8,
                "embedding_dim": 2,
                "rounds": 1,
                "epochs_per_round": 1,
                "batch_size": 10,
                "n_init": 1,
            },
            {"reshape": [1, 2, 3], "flatten": False},
        ),
        (
            "VaDE",
            {
                "n_clusters": 2,
                "hidden_dims": [6],
                "latent_dim": 2,
                "pretrain_epochs": 0,
                "max_epochs": 0,
                "n_init": 1,
            },
            {},
        ),
    ],
)
def test_cli_runs_each_representative_core(tmp_path, estimator, parameters, dataset):
    config = {
        "estimator": estimator,
        "seeds": [1],
        "dataset": {
            "kind": "blobs",
            "params": {
                "n_samples": 20,
                "n_features": 6,
                "centers": 2,
                "random_state": 1,
            },
            "normalization": "unit",
            **dataset,
        },
        "parameters": {**parameters, "device": "cpu", "deterministic": True},
        "output": str(tmp_path / f"{estimator}.json"),
    }
    path = tmp_path / f"{estimator}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_benchmark(path)
    assert result["config"]["estimator"] == estimator
    assert result["runs"][0]["metrics"]["acc"] >= 0
    output = json.loads((tmp_path / f"{estimator}.json").read_text())
    assert output["schema_version"] == 1
