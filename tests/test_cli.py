import json
import hashlib

import numpy as np
import yaml
from sklearn.datasets import make_blobs

from opendeepclustering import DEC
from opendeepclustering.cli import main, run_benchmark


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
    assert json.loads(output.read_text())["schema_version"] == 1
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
