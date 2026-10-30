"""Public dataset loading shared by the Python API and the benchmark CLI.

The CLI used to carry this logic privately, so Python users had no access to
the same data path their benchmark results came from. One implementation now
serves both, and returns the provenance needed to identify exactly which array
an estimator was given.

See docs/architecture/dataset-contract.md for the specification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NORMALIZATIONS = ("global_rms", "dec", "unit", "none")


def array_sha256(array) -> str:
    """Digest an array's bytes.

    Invariant to reshaping, so pair it with :func:`shape_signature` when the
    sample layout matters.
    """
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def shape_signature(array) -> str:
    """Describe sample layout and dtype, which a byte digest cannot."""
    array = np.asarray(array)
    return f"{tuple(array.shape[1:])}:{array.dtype}"


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded arrays with the provenance identifying them."""

    features: np.ndarray
    labels: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Allow ``X, y = load_dataset(...)``."""
        return iter((self.features, self.labels))


def normalize(X: np.ndarray, normalization: str | None, *, divisor: float | None = None):
    """Apply one of the closed set of preprocessing recipes.

    ``divisor`` supplies the ``global_rms`` constant. Pass the value computed
    over the full dataset so a subset and the whole dataset stay on one scale;
    recomputing it per subset would give each its own.
    """
    if normalization == "global_rms":
        scale = divisor if divisor is not None else float(
            np.sqrt(np.mean(np.square(X)))
        )
        return X / scale if scale else X
    if normalization == "dec":
        if X.ndim != 2:
            raise ValueError("DEC normalization requires flattened samples.")
        # Forces every sample to mean(x^2) == 1, which discards per-sample
        # magnitude. On MNIST that costs about 0.083 accuracy; see
        # docs/benchmarks/dec-idec-normalization-2026-10-30.md.
        scale = np.sqrt(np.mean(np.square(X), axis=1, keepdims=True)).clip(1e-12)
        return X / scale
    if normalization == "unit":
        maximum = float(np.max(np.abs(X)))
        return X / maximum if maximum else X
    if normalization in (None, "none"):
        return X
    raise ValueError(
        f"Unknown normalization {normalization!r}. Supported values are "
        f"{', '.join(repr(name) for name in NORMALIZATIONS)}."
    )


def _load_source(spec: dict, config_dir: Path):
    kind = spec["kind"]
    if kind == "blobs":
        from sklearn.datasets import make_blobs

        return make_blobs(**dict(spec.get("params", {})))
    if kind == "npz":
        path = Path(spec["path"])
        if not path.is_absolute():
            path = config_dir / path
        with np.load(path) as archive:
            return (
                archive[spec.get("features_key", "X")],
                archive[spec.get("labels_key", "y")],
            )
    if kind == "mnist":
        try:
            from torchvision.datasets import MNIST
        except ImportError as exc:
            raise RuntimeError(
                "MNIST loading requires the optional torchvision package."
            ) from exc
        root = Path(spec.get("root", "datasets"))
        if not root.is_absolute():
            root = config_dir / root
        download = spec.get("download", False)
        train = MNIST(root, train=True, download=download)
        split = spec.get("split", "all")
        if split == "all":
            test = MNIST(root, train=False, download=download)
            return (
                np.concatenate([train.data.numpy(), test.data.numpy()]),
                np.concatenate([train.targets.numpy(), test.targets.numpy()]),
            )
        if split == "train":
            return train.data.numpy(), train.targets.numpy()
        raise ValueError("MNIST split must be 'all' or 'train'.")
    raise ValueError(f"Unsupported dataset kind: {kind!r}.")


def load_dataset(spec: dict, config_dir: str | Path = ".") -> DatasetBundle:
    """Load, subsample, reshape and normalize one dataset specification.

    Labels are returned separately and never enter fitting; estimators accept
    ``y=None`` and discard it. They reach evaluation only, after ``fit``.
    """
    config_dir = Path(config_dir)
    X, y = _load_source(spec, config_dir)
    X = np.asarray(X, dtype=np.float32)

    # Captured before subsampling so a subset and the full dataset are divided
    # by the same constant.
    full_dataset_rms = float(np.sqrt(np.mean(np.square(X))))

    subset_indices = None
    subset_size = spec.get("subset_size")
    if subset_size is not None:
        if not isinstance(subset_size, int) or not 0 < subset_size <= len(X):
            raise ValueError(
                "subset_size must be a positive integer within the dataset."
            )
        rng = np.random.default_rng(spec.get("subset_seed", 0))
        subset_indices = rng.choice(len(X), size=subset_size, replace=False)
        X, y = X[subset_indices], np.asarray(y)[subset_indices]

    reshape = spec.get("reshape")
    if reshape is not None:
        if (
            not isinstance(reshape, (list, tuple))
            or not reshape
            or any(not isinstance(size, int) or size <= 0 for size in reshape)
            or int(np.prod(reshape)) != int(np.prod(X.shape[1:]))
        ):
            raise ValueError(
                "reshape must be a positive shape preserving sample size."
            )
        X = X.reshape(len(X), *reshape)

    if spec.get("flatten", True):
        X = X.reshape(len(X), -1)
    elif X.ndim == 3:
        X = X[:, None, :, :]

    normalization = spec.get("normalization")
    X = normalize(X, normalization, divisor=full_dataset_rms)
    y = np.asarray(y)

    provenance = {
        "samples": int(len(X)),
        "features": int(np.prod(X.shape[1:])),
        "sample_shape": list(X.shape[1:]),
        "features_sha256": array_sha256(X),
        "labels_sha256": array_sha256(y),
        "shape_signature": shape_signature(X),
        "normalization": normalization,
        "split": spec.get("split"),
    }
    if subset_indices is not None:
        provenance["subset_size"] = int(len(subset_indices))
        provenance["subset_seed"] = spec.get("subset_seed", 0)
        provenance["subset_indices_sha256"] = array_sha256(
            np.asarray(subset_indices, dtype=np.int64)
        )
    return DatasetBundle(features=X, labels=y, provenance=provenance)
