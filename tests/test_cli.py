import json
import hashlib

import numpy as np
import pytest
import yaml
from sklearn.datasets import make_blobs

from opendeepclustering import DEC
from opendeepclustering.cli import _load_dataset, main, run_benchmark


def test_cli_and_python_runner_share_output_schema(tmp_path, capsys):
    output = tmp_path / "result.json"
    config = {
        "estimator": "DEC",
        "seeds": [3],
        "dataset": {
            "kind": "blobs",
            "params": {
                "n_samples": 30,
                "n_features": 4,
                "centers": 2,
                "random_state": 3,
            },
        },
        "parameters": {
            "n_clusters": 2,
            "dims": [4, 2],
            "pretrain_epochs": 0,
            "max_epochs": 0,
            "n_init": 1,
            "device": "cpu",
        },
        "output": str(output),
    }
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    direct = run_benchmark(path)
    assert direct["runs"][0]["seed"] == 3
    saved = json.loads(output.read_text())
    assert saved["schema_version"] == 2
    assert saved["dataset"]["shape_signature"] == {
        "sample_shape": [4], "dtype": np.dtype(np.float32).str,
    }
    X, _ = make_blobs(n_samples=30, n_features=4, centers=2, random_state=3)
    labels = DEC(
        n_clusters=2,
        dims=[4, 2],
        pretrain_epochs=0,
        max_epochs=0,
        n_init=1,
        device="cpu",
        random_state=3,
    ).fit_predict(X.astype(np.float32))
    digest = hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()
    assert direct["runs"][0]["labels_sha256"] == digest
    assert main(["benchmark", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["estimator"] == "DEC"


def _blob_config(tmp_path, normalization, **dataset):
    return {
        "estimator": "AutoencoderKMeans",
        "seeds": [0],
        "dataset": {
            "kind": "blobs",
            "params": {
                "n_samples": 40,
                "n_features": 6,
                "centers": 2,
                "random_state": 0,
            },
            "normalization": normalization,
            **dataset,
        },
        "parameters": {
            "n_clusters": 2, "dims": [8, 2], "max_epochs": 0, "n_init": 1,
            "device": "cpu", "deterministic": True,
        },
        "output": str(tmp_path / f"{normalization}.json"),
    }


def _write(tmp_path, config, name):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_unknown_normalization_raises_instead_of_passing_data_through(tmp_path):
    """A misspelled recipe must fail, not silently train on unnormalized data."""
    config = _blob_config(tmp_path, "globalrms")  # missing underscore
    with pytest.raises(ValueError, match="Unknown normalization"):
        run_benchmark(_write(tmp_path, config, "typo"))


def test_global_rms_normalization_sets_dataset_mean_square_to_one(tmp_path):
    result = run_benchmark(
        _write(tmp_path, _blob_config(tmp_path, "global_rms"), "global_rms")
    )
    assert result["dataset"]["samples"] == 40
    # The whole-dataset constant is chosen so mean(x^2) is 1 over the dataset.
    X, _ = _load_dataset(
        _blob_config(tmp_path, "global_rms")["dataset"], tmp_path
    )
    assert float(np.mean(np.square(X))) == pytest.approx(1.0, abs=1e-5)
    # Per-sample magnitudes must still differ, which is what distinguishes this
    # from the per-sample "dec" recipe.
    per_sample = np.mean(np.square(X), axis=1)
    assert per_sample.max() - per_sample.min() > 1e-3


def test_dec_normalization_collapses_every_sample_to_unit_mean_square(tmp_path):
    """Pin the behaviour that cost 0.083 MNIST accuracy, so it stays visible."""
    X, _ = _load_dataset(_blob_config(tmp_path, "dec")["dataset"], tmp_path)
    per_sample = np.mean(np.square(X), axis=1)
    np.testing.assert_allclose(per_sample, 1.0, atol=1e-5)


def test_global_rms_divisor_is_taken_before_subsampling(tmp_path):
    """A subset and the full dataset must land on one scale.

    Computing the divisor after subsampling would give each subset its own
    constant, so a 20-sample run and a 40-sample run would not be comparable.
    """
    full, _ = _load_dataset(
        _blob_config(tmp_path, "global_rms")["dataset"], tmp_path
    )
    subset, _ = _load_dataset(
        _blob_config(
            tmp_path, "global_rms", subset_size=20, subset_seed=0
        )["dataset"],
        tmp_path,
    )
    rng = np.random.default_rng(0)
    indices = rng.choice(len(full), size=20, replace=False)
    np.testing.assert_allclose(subset, full[indices], rtol=1e-6)


@pytest.mark.parametrize("dtype", ["<f4", ">f4"])
def test_shape_signature_distinguishes_same_bytes_in_cli_records(tmp_path, capsys, monkeypatch, dtype):
    """Record layout and dtype without changing the existing byte-view digest."""
    from opendeepclustering import cli

    class MetadataOnlyEstimator:
        n_iter_ = 0
        converged_ = False
        stop_reason_ = "synthetic-provenance-fixture"

        def __init__(self, **parameters):
            pass

        def fit_predict(self, X):
            return np.arange(len(X), dtype=np.int64) % 2

    monkeypatch.setattr(cli, "ESTIMATORS", {"MetadataOnly": MetadataOnlyEstimator})
    monkeypatch.setattr(cli, "_environment", lambda: {"synthetic_only": True})
    monkeypatch.setattr(cli, "_git_sha", lambda: "synthetic-provenance-fixture")
    # Interpret the same little-endian backing bytes from both byte orders.
    matrix = np.arange(8, dtype="<f4").reshape(2, 4).view(dtype)
    arrays = [
        matrix, matrix.reshape(2, 1, 2, 2), matrix.view(np.uint32),
        matrix.view(matrix.dtype.newbyteorder("S")),
    ]
    signatures = []
    feature_digests = []
    output = tmp_path / "metadata.json"
    config = {
        "estimator": "MetadataOnly", "dataset": {"kind": "blobs"}, "output": str(output),
    }
    path = _write(tmp_path, config, "metadata")
    for X in arrays:
        monkeypatch.setattr(
            cli, "_load_dataset", lambda config, config_dir: (X, np.array([0, 1])),
        )
        assert cli.main(["benchmark", "--config", str(path)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert json.loads(output.read_text()) == result
        assert result["schema_version"] == 2
        dataset = result["dataset"]
        assert dataset["samples"] == 2
        assert dataset["features"] == 4
        assert dataset["sample_shape"] == list(X.shape[1:])
        assert dataset["shape_signature"] == {
            "sample_shape": list(X.shape[1:]), "dtype": X.dtype.str,
        }
        signatures.append(dataset["shape_signature"])
        feature_digests.append(dataset["features_sha256"])
    assert len(set(feature_digests)) == 1
    assert len({json.dumps(signature, sort_keys=True) for signature in signatures}) == len(arrays)
