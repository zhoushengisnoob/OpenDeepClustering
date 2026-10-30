"""Command-line experiments backed by the same estimators as the Python API."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import yaml

from opendeepclustering.estimators import (
    AutoencoderKMeans,
    DEC,
    DeepCluster,
    IDEC,
    VaDE,
)
from opendeepclustering.data import load_dataset
from opendeepclustering.metrics import embedding_effective_rank, evaluate_clustering


ESTIMATORS = {
    "AutoencoderKMeans": AutoencoderKMeans,
    "DEC": DEC,
    "DeepCluster": DeepCluster,
    "IDEC": IDEC,
    "VaDE": VaDE,
}


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Benchmark configuration must be a mapping.")
    allowed = {"estimator", "dataset", "parameters", "seed", "seeds", "output", "checkpoint"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown benchmark configuration keys: {sorted(unknown)}")
    for required in ("estimator", "dataset"):
        if required not in config:
            raise ValueError(f"Missing required configuration key: {required}.")
    if not isinstance(config["dataset"], dict) or "kind" not in config["dataset"]:
        raise ValueError("dataset must be a mapping containing kind.")
    if "parameters" in config and not isinstance(config["parameters"], dict):
        raise ValueError("parameters must be a mapping.")
    seeds = config.get("seeds", [config.get("seed", 0)])
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a non-empty list of integers.")
    checkpoint = config.get("checkpoint", {})
    if not isinstance(checkpoint, dict) or set(checkpoint) - {"resume_from", "save_to"}:
        raise ValueError("checkpoint supports only resume_from and save_to.")


def _load_dataset(config, config_dir: Path):
    """Delegate to the shared public loader.

    Retained as a thin wrapper so existing call sites and tests keep working
    while one implementation serves both the CLI and the Python API.
    """
    bundle = load_dataset(config, config_dir)
    return bundle.features, bundle.labels


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _shape_signature(array):
    """Describe sample layout and dtype, which a byte digest cannot.

    SHA-256 over a byte view is invariant to reshaping, so flattened and NCHW
    views of one dataset produce the identical digest. Recording the sample
    shape and dtype alongside it is what makes the two distinguishable, and
    what lets a result name the exact array the estimator received.
    """
    return f"{tuple(array.shape[1:])}:{array.dtype}"


def run_benchmark(config_path: str | Path):
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_config(config)
    estimator_name = config["estimator"]
    if estimator_name not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator_name!r}.")
    X, y = _load_dataset(config["dataset"], path.parent)
    seeds = config.get("seeds", [config.get("seed", 0)])
    runs = []
    for seed in seeds:
        params = dict(config.get("parameters", {}))
        params["random_state"] = seed
        checkpoint = config.get("checkpoint", {})
        if checkpoint.get("resume_from"):
            resume_path = Path(checkpoint["resume_from"])
            if not resume_path.is_absolute():
                resume_path = path.parent / resume_path
            params["resume_from"] = str(resume_path)
        estimator = ESTIMATORS[estimator_name](**params)
        started = time.perf_counter()
        labels = estimator.fit_predict(X)
        duration = time.perf_counter() - started
        scores = evaluate_clustering(y, labels)
        if checkpoint.get("save_to"):
            checkpoint_path = Path(checkpoint["save_to"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = path.parent / checkpoint_path
            if len(seeds) > 1:
                checkpoint_path = checkpoint_path.with_stem(
                    f"{checkpoint_path.stem}-seed-{seed}"
                )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            estimator.save_checkpoint(checkpoint_path)
        run = {
            "seed": seed,
            "metrics": scores.as_dict(),
            "labels_sha256": hashlib.sha256(
                np.asarray(labels, dtype=np.int64).tobytes()
            ).hexdigest(),
            "duration_seconds": duration,
            "n_iter": estimator.n_iter_,
            "converged": estimator.converged_,
            "stop_reason": estimator.stop_reason_,
        }
        embedding = getattr(estimator, "embedding_", None)
        if embedding is not None:
            run["effective_rank"] = embedding_effective_rank(embedding)
        runs.append(run)
    metric_names = ("acc", "f1", "nmi", "ari")
    aggregate = {
        metric: {
            "mean": float(np.mean([run["metrics"][metric] for run in runs])),
            "std": float(np.std([run["metrics"][metric] for run in runs])),
        }
        for metric in metric_names
    }
    result = {
        # 2 added the optional per-run effective_rank collapse indicator.
        # 3 added dataset.shape_signature.
        "schema_version": 3,
        "config": config,
        "config_path": str(path),
        "git_sha": _git_sha(),
        "environment": _environment(),
        "dataset": {
            "samples": len(X),
            "features": int(np.prod(X.shape[1:])),
            "sample_shape": list(X.shape[1:]),
            "features_sha256": _array_sha256(X),
            "labels_sha256": _array_sha256(y),
            # Pairs the digest with the layout it was taken over, since the
            # digest alone cannot distinguish a flattened matrix from an NCHW
            # tensor holding the same bytes.
            "shape_signature": _shape_signature(X),
        },
        "runs": runs,
        "aggregate": aggregate,
    }
    output = config.get("output")
    if output:
        destination = Path(output)
        if not destination.is_absolute():
            destination = path.parent / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser():
    parser = argparse.ArgumentParser(prog="odc")
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="Run a reproducible benchmark.")
    benchmark.add_argument("--config", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        print(json.dumps(run_benchmark(args.config), indent=2))
    return 0


def legacy_main(method: str):
    """Compatibility entry point for retired script locations."""
    print(
        f"This {method} script is deprecated; use `odc benchmark --config ...`.",
        file=sys.stderr,
    )
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
