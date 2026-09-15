# Quickstart

## Install

OpenDeepClustering supports Python 3.10 and 3.11 in continuous integration.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For MNIST and other torchvision-backed image datasets, add the optional dependency:

```bash
python -m pip install -e ".[image]"
```

`pyproject.toml` is the authoritative dependency specification. FAISS is not required by the current DEC/IDEC implementation.

## Python API

Estimators accept a finite, dense, two-dimensional array shaped `(n_samples, n_features)`. Flatten image tensors before fitting.

```python
from sklearn.datasets import make_blobs
from opendeepclustering import DEC

X, _ = make_blobs(
    n_samples=300,
    n_features=12,
    centers=3,
    random_state=7,
)

model = DEC(
    n_clusters=3,
    dims=(32, 10),
    pretrain_epochs=5,
    max_epochs=10,
    random_state=7,
    deterministic=True,
    device="cpu",
)
labels = model.fit_predict(X)
embedding = model.transform(X)
assignments = model.soft_assign(X)
```

Fitted estimators expose `labels_`, `embedding_`, `cluster_centers_`, `n_iter_`, `converged_`, `stop_reason_` and `history_`.

## Reproducible CLI

The CLI calls the same estimator classes as the Python API:

```bash
odc benchmark --config configs/benchmarks/dec_smoke.yaml
```

Run repository benchmark configurations from a source checkout. The result records the resolved configuration, Git commit, software/hardware environment, dataset checksums, per-seed metrics and aggregate statistics.
