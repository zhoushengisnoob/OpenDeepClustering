# DeepCluster representation collapse (2026-10-28)

This record documents why the `DeepCluster` estimator produces near-chance clustering on MNIST, isolates the cause to an architectural choice rather than to hyperparameters, and quantifies how far a set of candidate repairs gets. It is a negative-result and defect record, not a reproduction report.

The headline finding is that the shipped implementation scores **below raw-pixel KMeans on the same data**, and that repairing every defect identified here does not close that gap.

## Provenance

- Host: `lab-gpu-32`, six NVIDIA GeForce RTX 3090, driver 555.42.02
- Git commit: `58b981acf753bcacef627680f3d2b2dad4c583c4`
- Python: 3.10.21
- NumPy: 1.26.4
- SciPy: 1.15.3
- scikit-learn: 1.4.2
- PyTorch: 2.1.2+cu121, CUDA runtime 12.1
- Test suite before the runs: 28 passed
- Dataset: MNIST train+test concatenated, scaled by the global maximum absolute value
- Ablation matrix: 24 runs (8 variants × 3 seeds), zero failures, 77 minutes wall clock across six GPUs

Environment note: this is the same interpreter that produced the existing 20-seed baselines, and it differs from the PyTorch 2.5.1 / Python 3.12 profile documented in the README installation section. Per-run JSON remains on the GPU server under `benchmark_outputs/dc_ablation_20261028/`, which is an ignored build artifact; this page is the versioned summary.

## Observed defect

A 20-seed MNIST baseline recorded on 2026-09-18 gives `DeepCluster` an accuracy of 0.2040 ± 0.0137. For a ten-class problem, chance is 0.10. The standard deviation shows this is stable, not high-variance behaviour.

Per-round instrumentation on a 10,000-sample subset shows the embedding is already degenerate before any training occurs:

| Stage | ACC | NMI | Effective rank | Top-1 variance share | Mean per-dimension std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untrained backbone (round 0) | 0.1937 | 0.1311 | 1.31 | 0.954 | 0.0045 |
| After 20 alternating rounds | 0.1723 | 0.0920 | 1.02 | 0.998 | 1.2958 |

Effective rank is the exponential of the entropy of the normalised singular-value spectrum of the centred embedding. A value of 1.31 on a 32-dimensional embedding means the representation is effectively one-dimensional: 95% of the variance lies along a single direction.

The alternating loop is therefore not the origin of the collapse. It inherits a degenerate representation and degrades it further, from 0.1937 to 0.1723 over 20 rounds.

## Cause isolation

Four embeddings were scored by KMeans with identical settings and seed. All backbones are randomly initialised and untrained, so differences are attributable to architecture alone.

| Representation | ACC | NMI | Effective rank | Top-1 variance share | Mean per-dimension std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw 784-dimensional pixels | 0.5342 | 0.4807 | 69.06 | 0.094 | 0.1924 |
| CNN with `AdaptiveAvgPool2d((1, 1))`, as shipped | 0.2193 | 0.1431 | 1.61 | 0.900 | 0.0040 |
| Same CNN pooled to a 4×4 grid | 0.3508 | 0.3050 | 10.30 | 0.348 | 0.0096 |
| The tabular path's MLP on flattened input | 0.4130 | 0.2601 | 17.42 | 0.213 | 0.0522 |

Global average pooling to a single spatial cell is the cause. Averaging a randomly initialised two-layer convolutional feature map over all spatial positions yields a nearly constant output per channel. Changing only the pooling target from 1×1 to 4×4 raises untrained accuracy from 0.2193 to 0.3508 and effective rank from 1.61 to 10.30.

## Ablation of candidate repairs

Each variant runs the library's genuine `DeepCluster.fit` loop at the reference configuration (`rounds: 20`, `hidden_dim: 64`, `embedding_dim: 32`, `batch_size: 256`, `lr: 0.05`, `n_init: 20`) on the full 70,000-sample dataset. Only the backbone class and the flip-augmentation policy are substituted. Three seeds per variant; the deviations reported are population standard deviations over those three runs.

`pool` is the spatial pooling target, `flip` the horizontal-flip augmentation, `epr` the `epochs_per_round` value, and `AE` the number of autoencoder pretraining epochs applied to the backbone before the alternating loop.

| Variant | pool | flip | epr | AE | ACC | NMI | Effective rank | Minutes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 shipped defaults | 1×1 | yes | 1 | 0 | 0.1946 ± 0.0108 | 0.1339 ± 0.0135 | 1.02 | 14 |
| A1 pooling only | 4×4 | yes | 1 | 0 | 0.3023 ± 0.0421 | 0.2479 ± 0.0383 | 1.63 | 14 |
| A2 flip removal only | 1×1 | no | 1 | 0 | 0.2000 ± 0.0006 | 0.1369 ± 0.0015 | 1.03 | 14 |
| A3 pooling + flip removal | 4×4 | no | 1 | 0 | 0.2874 ± 0.0382 | 0.2178 ± 0.0541 | 1.78 | 14 |
| A4 A3 + longer rounds | 4×4 | no | 5 | 0 | 0.3248 ± 0.0199 | 0.2384 ± 0.0178 | 2.38 | 24 |
| A5 A3 + pretraining | 4×4 | no | 1 | 10 | 0.2699 ± 0.0541 | 0.1900 ± 0.0501 | 1.95 | 14 |
| A6 A3 + both | 4×4 | no | 5 | 10 | 0.3297 ± 0.0063 | 0.2729 ± 0.0129 | 2.59 | 24 |
| A7 no pooling + both | none | no | 5 | 10 | 0.4140 ± 0.0250 | 0.3116 ± 0.0269 | 4.17 | 23 |

A0 reproduces the independent 20-seed baseline of 0.2040 ± 0.0137, which establishes that the substitution harness measures the shipped behaviour.

For reference on the same data and preprocessing: raw-pixel KMeans reaches 0.5342 accuracy and `AutoencoderKMeans` reaches 0.7916 ± 0.0188 over 20 seeds.

## Interpretation

Spatial pooling is the dominant single factor, worth +0.108 accuracy from A0 to A1. The effect is an order of magnitude larger than the three-seed standard error and is corroborated by the untrained-backbone comparison.

Removing flip augmentation in isolation is worth +0.005 (A0 to A2), which is within noise. Horizontally flipping a handwritten digit changes its class, so the augmentation is semantically wrong on this dataset and should be removed regardless, but it is not a performance lever while the representation is degenerate. Once pooling is fixed, A1 at 0.3023 slightly exceeds A3 at 0.2874; that 0.015 difference lies inside both standard deviations and does not establish a benefit either way.

Autoencoder pretraining alone is not beneficial: A3 to A5 moves accuracy from 0.2874 to 0.2699. It only becomes neutral-to-positive in combination with more epochs per round (A4 0.3248 to A6 0.3297, again within noise).

Effective rank tracks accuracy monotonically across the matrix, from 1.02 at 0.1946 to 4.17 at 0.4140, which supports treating the degree of collapse as the proximate determinant of clustering quality.

Two conclusions follow that bound what parameter work can achieve.

First, **the best repaired variant remains well below the no-representation-learning reference.** A7 at 0.4140 against raw-pixel KMeans at 0.5342 is a gap of 0.12. On MNIST this estimator is a net negative contribution even after every defect identified here is addressed.

Second, **the alternating loop still degrades a corrected representation.** A two-round probe of the A6 configuration reached 0.4435, while the same configuration at 20 rounds settles at 0.3297. The published method relies on ImageNet-scale data, Sobel preprocessing, strong augmentation and empty-cluster reassignment to hold pseudo-label training stable; none of those are present here. Repairing the backbone raises the starting point without supplying a mechanism that prevents the loop from drifting.

## Why the test suite did not detect this

Three gaps combine, and they are worth recording because the same pattern would hide a regression in any other estimator.

The two DeepCluster tests assert output shapes, `augmentation_calls_ > 0`, and `len(history_["active_clusters"]) == 1`. The CLI test asserts `metrics["acc"] >= 0`, which is true for any non-negative number and therefore always passes. No test asserts a quality lower bound.

The test fixture is 16 synthetic 8×8 images split into a left-bright and a right-bright group. A two-cluster problem with that much separation is solvable even from a largely degenerate embedding, so the fixture cannot surface a ten-class collapse.

No test exercises more than one alternating round; `rounds` appears only as 0 and 1 across the suite, while collapse-driven drift accumulates over rounds.

Separately, the collapse indicator the implementation recorded was the wrong statistic. `active_clusters` counts distinct values in the KMeans assignment, and KMeans returns `n_clusters` non-empty clusters regardless of embedding quality, so the value was a constant 10 in every affected run. Effective rank or per-dimension standard deviation would have exposed the problem. The value was also written only to `history_` and never reached the benchmark JSON, so none of the 20 baseline runs preserved it. Both gaps are now closed; see the status section below.

## Status and next steps

The findings establish that:

1. Spatial pooling must be revised; global average pooling to 1×1 is not viable for this backbone and dataset.
2. Flip augmentation should be removed for digit data on correctness grounds, independent of its measured effect.
3. The alternating loop needs an explicit anti-collapse mechanism before the method can be expected to exceed its own input representation.
4. The quality floor for `DeepCluster` must be expressed as a test assertion, and a collapse metric that can actually detect collapse should be recorded in the benchmark schema.

Item 4 is addressed. `embedding_effective_rank` is now part of the public metrics API, `DeepCluster` records `effective_rank` and `mean_dim_std` alongside `active_clusters` for every clustering pass and exposes `effective_rank_` as a fitted attribute, and the benchmark CLI writes a per-run `effective_rank` field for any estimator with an `embedding_` attribute. That schema addition raises `schema_version` from 1 to 2.

Three test assertions now guard the behaviour. Two run on the tabular path and enforce an effective-rank floor across multiple rounds and an accuracy floor relative to KMeans on the raw input. The third, `test_deepcluster_image_path_beats_clustering_the_raw_input`, exercises the defective convolutional path and is marked strict `xfail`: it records the open defect and will fail as an unexpected pass once the backbone is repaired, which forces the marker to be removed deliberately.

Designing those assertions surfaced a trap worth recording. Well-separated Gaussian blobs do not work as a fixture: a collapsed embedding can still order them along its single surviving axis and score perfect accuracy. Verified against the defective backbone, an accuracy floor on blob data passes and catches nothing, which is the same weakness as the original 16-sample two-class fixture. The committed fixture places cluster centres on a circle at equal radius so that no one-dimensional projection separates them. Conversely the effective-rank floor catches the defect on blob data but not on the ring data, where the collapsed embedding retains rank 2.68. Neither indicator is sufficient alone, so both are asserted.

Items 1 through 3 change the method's defining behaviour and its published numbers, so they remain tracked as implementation work rather than applied as part of this diagnostic record.
