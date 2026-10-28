import json

import numpy as np
import pytest
import yaml
from sklearn.cluster import KMeans, MiniBatchKMeans

from opendeepclustering import AutoencoderKMeans, DeepCluster, VaDE
from opendeepclustering.cli import run_benchmark
from opendeepclustering.metrics import cluster_acc_f1


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
    # One entry per alternating round plus one for the final clustering pass.
    assert len(model.history_["active_clusters"]) == 2
    assert model.get_capabilities()["input_modalities"] == (
        "tabular",
        "image_nchw",
    )


def make_separable_clusters(n_clusters=6, per_cluster=40, n_features=24, seed=11):
    """Clusters that are separable but not along any single direction.

    Centers sit on a circle inside the first two dimensions, all at the same
    radius, so no one-dimensional projection separates them. Well-separated
    blobs would not do: a collapsed embedding can still rank them along its
    single surviving axis and score perfectly, which is exactly how the
    previous fixture hid the image backbone's collapse.
    """
    rng = np.random.default_rng(seed)
    angles = np.arange(n_clusters) * 2.0 * np.pi / n_clusters
    centers = np.zeros((n_clusters, n_features))
    centers[:, 0] = 6.0 * np.cos(angles)
    centers[:, 1] = 6.0 * np.sin(angles)
    X = np.repeat(centers, per_cluster, axis=0)
    X = X + rng.normal(0.0, 0.25, size=X.shape)
    y = np.repeat(np.arange(n_clusters), per_cluster)
    return X.astype(np.float32), y


@pytest.mark.parametrize("rounds", [1, 5])
def test_deepcluster_does_not_collapse_the_representation(rounds):
    """Guard the defect recorded in docs/benchmarks/deepcluster-collapse-2026-10-28.md.

    On trivially separable data the learned embedding must stay usable across
    multiple alternating rounds. The shipped image backbone instead collapsed
    to an effectively one-dimensional representation, which no prior assertion
    could detect: cluster counts stay at ``n_clusters`` regardless.
    """
    X, y = make_separable_clusters()
    model = DeepCluster(
        n_clusters=6,
        hidden_dim=32,
        embedding_dim=8,
        rounds=rounds,
        epochs_per_round=1,
        batch_size=32,
        n_init=5,
        augment=False,
        random_state=4,
        deterministic=True,
        device="cpu",
    ).fit(X)

    assert model.effective_rank_ > 1.5, (
        f"embedding collapsed after {rounds} round(s): "
        f"effective rank {model.effective_rank_:.3f}"
    )
    assert len(model.history_["effective_rank"]) == rounds + 1
    assert all(value > 0.0 for value in model.history_["mean_dim_std"])

    accuracy = cluster_acc_f1(y, model.labels_)[0]
    assert accuracy >= 0.70, (
        f"clustering accuracy {accuracy:.4f} after {rounds} round(s) is below "
        "the floor for linearly separable input"
    )


@pytest.mark.xfail(
    reason=(
        "Known defect: the image backbone's global average pooling to 1x1 "
        "collapses the embedding. See "
        "docs/benchmarks/deepcluster-collapse-2026-10-28.md"
    ),
    strict=True,
)
def test_deepcluster_image_path_beats_clustering_the_raw_input():
    """Pin the defect on the path that actually carries it.

    The tabular path uses an MLP and passes the quality floors. The image path
    uses the convolutional backbone whose pooling destroys the representation,
    so this is marked strict-xfail: it documents the open defect and will fail
    loudly once the backbone is repaired, prompting removal of the marker.
    """
    X, y = make_separable_clusters(n_features=64)
    images = X.reshape(len(X), 1, 8, 8)
    baseline = cluster_acc_f1(
        y, KMeans(n_clusters=6, n_init=5, random_state=0).fit_predict(X)
    )[0]
    model = DeepCluster(
        n_clusters=6,
        hidden_dim=32,
        embedding_dim=8,
        rounds=5,
        epochs_per_round=1,
        batch_size=32,
        n_init=5,
        augment=False,
        random_state=4,
        deterministic=True,
        device="cpu",
    ).fit(images)
    learned = cluster_acc_f1(y, model.labels_)[0]
    assert learned >= baseline - 0.05, (
        f"image-path accuracy {learned:.4f} is below raw-input KMeans "
        f"{baseline:.4f} by more than the allowed margin"
    )


def test_deepcluster_beats_clustering_the_raw_input():
    """A representation learner must not do worse than clustering its own input.

    This is the property the MNIST image path violates: 0.2040 accuracy against
    0.5342 for KMeans on raw pixels.
    """
    X, y = make_separable_clusters()
    baseline = cluster_acc_f1(
        y, KMeans(n_clusters=6, n_init=5, random_state=0).fit_predict(X)
    )[0]
    model = DeepCluster(
        n_clusters=6,
        hidden_dim=32,
        embedding_dim=8,
        rounds=3,
        epochs_per_round=1,
        batch_size=32,
        n_init=5,
        augment=False,
        random_state=4,
        deterministic=True,
        device="cpu",
    ).fit(X)
    learned = cluster_acc_f1(y, model.labels_)[0]
    assert learned >= baseline - 0.05, (
        f"DeepCluster accuracy {learned:.4f} is below raw-input KMeans "
        f"{baseline:.4f} by more than the allowed margin"
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
    assert output["schema_version"] == 2
