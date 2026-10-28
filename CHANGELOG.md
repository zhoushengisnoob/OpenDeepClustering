# Changelog

All notable changes to OpenDeepClustering are recorded here. The project follows semantic versioning from v0.2.0 onward.

## [Unreleased]

### Added

- `metrics.embedding_effective_rank` for detecting representation collapse, with `DeepCluster` recording `effective_rank` and `mean_dim_std` per clustering pass and exposing `effective_rank_` after fitting.
- Benchmark results carry a per-run `effective_rank` field for estimators that expose an `embedding_` attribute.
- Quality-floor tests for `DeepCluster`: an effective-rank floor across multiple alternating rounds and an accuracy floor relative to KMeans on the raw input. A strict-`xfail` test pins the defective image path and will fail once that backbone is repaired.
- `docs/benchmarks/deepcluster-collapse-2026-10-28.md` isolates the cause of DeepCluster's near-chance MNIST accuracy and bounds what the identified repairs achieve.
- `docs/benchmarks/three-pattern-mnist-2026-09-18.md` records the previously unsummarised 20-seed MNIST baselines for `AutoencoderKMeans`, `VaDE` and `DeepCluster`.

### Changed

- Benchmark `schema_version` raised from 1 to 2 for the added `effective_rank` field.
- `DeepCluster`'s documented status changed from architecture reference to known defect: its 20-seed MNIST accuracy of 0.2040 ± 0.0137 is below the 0.5342 reached by KMeans on raw pixels, so it must not be used as a baseline or correctness oracle.

## [0.2.0] - 2026-09-15

### Added

- A scikit-learn-style estimator contract with shared data validation, fit state, callbacks, checkpoints and deterministic seed control.
- Paper-traceable DEC and IDEC implementations with full-dataset target updates, assignment-change stopping and reference MNIST configurations.
- Architecture-reference implementations for Autoencoder + KMeans, DeepCluster and VaDE, covering the Survey's multi-stage, iterative and generative patterns alongside the simultaneous DEC/IDEC family.
- A shared YAML-driven benchmark CLI with configuration, environment, Git revision, data checksum, metric, duration and stop-reason capture.
- Mathematical component tests, estimator compatibility checks, end-to-end benchmark tests and Python 3.10/3.11 CI.
- Versioned method specifications, architecture reviews, provenance records, benchmark reports and a MkDocs documentation site.
- BSD-2-Clause licensing, citation metadata and reproducible source/wheel package validation.

### Changed

- Package dependencies and archive contents are now explicitly bounded and validated.
- Legacy scripts remain available as historical reproduction material while the supported Python API and CLI share one estimator core.

### Evidence

- Issues #2–#8 and their linked pull requests contain the implementation history and review boundary.
- `docs/benchmarks/mnist-reference-2026-09-15.md` records the five-seed DEC/IDEC GPU baseline.
- `docs/benchmarks/four-pattern-smoke-2026-09-15.md` records the four-pattern server smoke validation.
