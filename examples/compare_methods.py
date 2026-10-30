"""Compare every estimator on CPU, and show when not to use one.

The scores here are not the point — the datasets are small and the training
budgets are tiny. The procedure is: one input matrix per dataset, one metric
implementation, one seed set, and a floor that says whether a learned
representation contributed anything.

Two datasets, because the answer differs between them:

* Well-separated blobs, where KMeans on the raw input is already perfect and
  no deep method can improve on it. Recognising this case saves real time.
* Concentric rings embedded in a higher-dimensional space, where classes are
  separable but not by distance to a single centroid, so a learned
  representation has something to contribute.

    python examples/compare_methods.py

Needs no GPU, no download and no optional extras. Runtime is a couple of
minutes. For real multi-seed evidence on MNIST see
docs/benchmarks/comparison.md.
"""

from __future__ import annotations

import argparse
import statistics as st
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from opendeepclustering import DEC, IDEC, AutoencoderKMeans, DeepCluster, VaDE
from opendeepclustering.metrics import cluster_acc_f1, embedding_effective_rank

SEEDS = (0, 1, 2)
N_CLUSTERS = 5
N_FEATURES = 20


def _to_unit_range(X):
    """Scale to [0, 1] so VaDE's Bernoulli option would also be valid."""
    X = X - X.min()
    return (X / X.max()).astype(np.float32)


def make_blob_dataset():
    """Compact, well-separated clusters: the case KMeans already solves."""
    X, y = make_blobs(
        n_samples=600,
        n_features=N_FEATURES,
        centers=N_CLUSTERS,
        cluster_std=1.8,
        random_state=0,
    )
    return _to_unit_range(X), y


def make_ring_dataset():
    """Concentric rings rotated into N_FEATURES dimensions.

    Each class is a ring at its own radius, all sharing one centre, so the
    classes are separable but not by distance to a single centroid. KMeans on
    the raw input therefore cannot recover them.
    """
    rng = np.random.default_rng(0)
    per_class = 120
    points, labels = [], []
    for index in range(N_CLUSTERS):
        radius = 1.0 + 1.6 * index
        angle = rng.uniform(0, 2 * np.pi, size=per_class)
        jitter = rng.normal(0, 0.18, size=per_class)
        points.append(
            np.stack(
                [
                    (radius + jitter) * np.cos(angle),
                    (radius + jitter) * np.sin(angle),
                ],
                axis=1,
            )
        )
        labels.append(np.full(per_class, index))
    plane = np.concatenate(points)
    y = np.concatenate(labels)

    # Embed the 2D structure with a random rotation, plus low-variance noise
    # in the remaining directions.
    basis = rng.normal(size=(2, N_FEATURES))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    X = plane @ basis + rng.normal(0, 0.05, size=(len(plane), N_FEATURES))

    order = rng.permutation(len(X))
    return _to_unit_range(X[order]), y[order]


DATASETS = (
    ("Well-separated blobs", make_blob_dataset),
    ("Concentric rings in 20D", make_ring_dataset),
)


def build(name, seed):
    """Comparable small budgets; not tuned, and not paper protocols."""
    common = dict(random_state=seed, deterministic=True, device="cpu")
    if name == "AutoencoderKMeans":
        return AutoencoderKMeans(
            n_clusters=N_CLUSTERS, dims=(32, 8), max_epochs=8, batch_size=64,
            n_init=5, **common,
        )
    if name == "DEC":
        return DEC(
            n_clusters=N_CLUSTERS, dims=(32, 8), pretrain_method="joint",
            pretrain_epochs=8, max_epochs=10, batch_size=64, update_interval=5,
            **common,
        )
    if name == "IDEC":
        return IDEC(
            n_clusters=N_CLUSTERS, dims=(32, 8), pretrain_method="joint",
            pretrain_epochs=8, max_epochs=10, batch_size=64, update_interval=5,
            **common,
        )
    if name == "VaDE":
        return VaDE(
            n_clusters=N_CLUSTERS, hidden_dims=(32,), latent_dim=8,
            pretrain_epochs=8, max_epochs=10, batch_size=64, n_init=5, **common,
        )
    if name == "DeepCluster":
        return DeepCluster(
            n_clusters=N_CLUSTERS, hidden_dim=32, embedding_dim=8, rounds=5,
            epochs_per_round=2, batch_size=64, n_init=5, augment=False, **common,
        )
    raise ValueError(name)


def evaluate(X, y):
    """Return the raw-input floor and one row per estimator."""
    # The honest floor: cluster the input directly, with no learned
    # representation. A method below this line is not contributing.
    floor = st.mean(
        cluster_acc_f1(y, KMeans(n_clusters=N_CLUSTERS, n_init=5,
                                 random_state=seed).fit_predict(X))[0]
        for seed in SEEDS
    )

    rows = []
    for name in ("AutoencoderKMeans", "DEC", "IDEC", "VaDE", "DeepCluster"):
        accuracies, ranks = [], []
        started = time.perf_counter()
        for seed in SEEDS:
            model = build(name, seed).fit(X)
            accuracies.append(cluster_acc_f1(y, model.labels_)[0])
            ranks.append(embedding_effective_rank(model.embedding_))
        rows.append(
            {
                "name": name,
                "mean": st.mean(accuracies),
                "sd": st.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "rank": st.mean(ranks),
                "seconds": time.perf_counter() - started,
            }
        )

    return floor, rows


def report(title, X, floor, rows):
    print(f"=== {title} ===")
    print(f"{len(X)} samples x {N_FEATURES} features, {N_CLUSTERS} clusters, "
          f"seeds {list(SEEDS)}, identical input for every method.\n")
    print("%-20s %-18s %-9s %-9s %s" % (
        "estimator", "ACC mean ± sd", "vs floor", "eff.rank", "seconds"))
    for row in sorted(rows, key=lambda item: -item["mean"]):
        print("%-20s %.4f ± %.4f    %+.4f   %6.2f   %6.1f" % (
            row["name"], row["mean"], row["sd"],
            row["mean"] - floor, row["rank"], row["seconds"]))
    print("%-20s %.4f" % ("KMeans on raw input", floor))

    best = max(rows, key=lambda item: item["mean"])
    if best["mean"] <= floor:
        print(
            "\n-> No method beats KMeans on the raw input here. The clusters are\n"
            "   already compact and well separated, so there is nothing for a\n"
            "   learned representation to add. On data like this, use KMeans."
        )
    else:
        print(
            "\n-> %s exceeds the floor by %+.4f. Distance to a single centroid\n"
            "   does not separate these classes, so a learned representation has\n"
            "   something to contribute — though every method stays weak at this\n"
            "   deliberately tiny budget." % (best["name"], best["mean"] - floor)
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the tables; still exits non-zero on failure. Used by CI.",
    )
    args = parser.parse_args()

    for title, builder in DATASETS:
        X, y = builder()
        floor, rows = evaluate(X, y)
        if not args.quiet:
            report(title, X, floor, rows)

        # The example doubles as a smoke test: every estimator must produce
        # finite results on every dataset.
        assert len(rows) == 5
        for row in rows:
            assert np.isfinite(row["mean"]), f"{title}/{row['name']}"
            assert np.isfinite(row["rank"]), f"{title}/{row['name']}"

    if not args.quiet:
        print(
            "The floor is KMeans on the unmodified input. A positive 'vs floor'\n"
            "means the learned representation helped; negative means it did not.\n"
            "Effective rank near 1.0 indicates a collapsed embedding. Treat the\n"
            "orderings as illustrative — the budgets are far below the reference\n"
            "protocols. See docs/benchmarks/comparison.md for real evidence."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
