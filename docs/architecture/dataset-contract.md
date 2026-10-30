# Dataset loading contract

One implementation serves both the Python API and the benchmark CLI. This page defines what identifies a dataset, what preprocessing is permitted, and what must be recorded for a result to be traceable to the exact array the estimator saw.

## Using it

`opendeepclustering.data.load_dataset` is the shared implementation; the benchmark CLI calls it rather than carrying its own copy.

```python
from opendeepclustering.data import load_dataset

X, y = load_dataset({"kind": "mnist", "root": "datasets", "normalization": "global_rms"})
```

It also returns provenance, which is what the benchmark output records:

```python
bundle = load_dataset({"kind": "mnist", "root": "datasets", "normalization": "global_rms"})
bundle.provenance["features_sha256"]
bundle.provenance["shape_signature"]
```

## Identity

A `DatasetSpec` names everything that changes the returned arrays:

| Field | Meaning |
| --- | --- |
| `kind` | `blobs`, `npz`, or a named dataset (`mnist`, `fashion_mnist`, `usps`, `cifar10`, `cifar100`, `stl10`, `reuters10k`) |
| `root` | Directory holding the source files. Never written to unless `download` is true. |
| `download` | Explicit opt-in. Defaults to false; no loader downloads implicitly. |
| `split` | `train`, `test`, or `all` for the transductive concatenation of both labeled splits |
| `labels_key` / `features_key` | Array names for `npz`, so a file carrying several label sets is unambiguous |
| `subset_size` / `subset_seed` | Deterministic subsampling, applied before preprocessing |
| `reshape` | Explicit target sample shape, validated to preserve the element count |
| `flatten` | Whether samples are flattened to a 2D matrix or kept in their multi-dimensional layout |
| `normalization` | One of the recipes below. Unknown values raise rather than passing through. |

Two loads agreeing on every field return byte-identical arrays in identical order. Any field changing means the results are not comparable, which is what the [method comparison](../benchmarks/comparison.md) keys on.

## Preprocessing recipes

`normalization` is a closed set. An unrecognised value is an error, never a silent no-op.

| Value | Operation | Intended inputs |
| --- | --- | --- |
| `none` | Cast to float32 only | Data already on its intended scale, such as Reuters TF-IDF features |
| `unit` | Divide by the global maximum absolute value | uint8 images, giving [0, 1] |
| `uint8_to_unit` | Divide by 255.0 exactly | uint8 images where the scale must not depend on the observed maximum |
| `affine_to_unit` | Map the declared source range to [0, 1] | USPS, whose source features lie in [-1, 1] |
| `dec` | Divide each sample by the root mean square of its own features | DEC and IDEC, per their papers |

`unit` and `uint8_to_unit` differ when no sample attains 255: `unit` then stretches the data while `uint8_to_unit` does not. `unit` is retained because the committed reference results used it; prefer `uint8_to_unit` for new image configurations.

`affine_to_unit` requires an explicit `source_range`. Inferring it from the observed minimum and maximum would make the transform depend on the subset, so a subset and the full dataset would be scaled differently.

Likelihood compatibility is checked, not assumed: a Bernoulli reconstruction with values outside [0, 1] raises before training starts.

## Fitted preprocessing and the split boundary

No recipe above is fitted — each is a fixed function of a single sample or a declared constant. That is deliberate. A step fitted across data, such as PCA or a whitening transform, must record the split it was fitted on, because fitting on `all` and then reporting test metrics would describe information the model should not have had.

`split: all` is the transductive protocol used by the DEC and IDEC papers and by every committed reference configuration here: both labeled splits are concatenated, clustered together, and evaluated against all labels. It is not a train/test evaluation and must not be described as one. A future `train`-fit with `test`-only evaluation is a different protocol whose numbers are not comparable with these.

## Labels

Labels are returned alongside features but never enter fitting. Estimators take `y=None` and discard it. Labels reach only `metrics`, after `fit`, and no configuration is selected by consulting them. The loader returns them as a separate array for exactly this reason: there is no code path by which a label can influence an optimisation step.

STL-10's unlabeled split has no ground truth, so ACC, NMI and ARI are undefined for it. Loading it for evaluation raises.

## Provenance

`DatasetBundle` carries the arrays and the facts needed to identify them:

- `features_sha256` and `labels_sha256`, over the final arrays
- `shape_signature`: the sample shape and dtype, so a 2D matrix and an NCHW tensor over the same bytes are distinguishable. Digests alone are not sufficient: SHA-256 over a byte view is invariant to reshaping, and the committed records show the identical digest `d27fb913...` for flattened MNIST and for DeepCluster's NCHW view of it. Benchmark results carry this field from `schema_version` 3.
- `spec_digest`: a digest of the normalised specification, so a result can be matched back to the request that produced it
- `n_samples`, `n_features`, `split`, `normalization`, and the resolved subset indices when subsampling applied

Benchmark output records all of these. A result that cannot name the exact post-preprocessing array it measured is not usable as evidence.
