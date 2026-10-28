# Multi-stage, iterative and generative MNIST baseline (2026-09-18)

This record reports 20-seed MNIST results for the three representatives that had no multi-seed quality evidence: `AutoencoderKMeans`, `DeepCluster` and `VaDE`. The runs completed on 2026-09-18 but were not summarised at the time, because per-run JSON lands in an ignored output directory and does not reach the repository on its own.

## Provenance

- Host: `lab-gpu-32`, six NVIDIA GeForce RTX 3090, driver 555.42.02
- Python: 3.10.21
- NumPy: 1.26.4
- scikit-learn: 1.4.2
- PyTorch: 2.1.2+cu121, CUDA runtime 12.1
- Dataset: MNIST train+test concatenated, 70,000 samples, per the committed reference configurations
- Scheduling: six concurrent single-GPU workers, one seed per process, isolated by `CUDA_VISIBLE_DEVICES`
- Outcome: 60 planned runs, 60 completed, zero failures and zero timeouts, 2 hours 1 minute wall clock

The runs predate the current commit. Verify that `opendeepclustering/` is unchanged between the recorded commit and the revision you compare against before treating these as current numbers. This interpreter also differs from the PyTorch 2.5.1 / Python 3.12 profile in the README installation section; it is the same environment that produced the earlier DEC/IDEC reference evidence.

## Results

Seeds 0 through 19. Deviations are population standard deviations over the 20 runs. Every run reported `completed` as its stop reason, which for these estimators means the configured round or epoch budget was exhausted rather than that a convergence criterion fired.

| Estimator | Pattern | ACC | NMI | Minutes per run |
| --- | --- | ---: | ---: | ---: |
| `AutoencoderKMeans` | Multi-stage | 0.7916 ± 0.0188 | 0.7634 ± 0.0118 | 3.6 |
| `VaDE` | Generative | 0.7158 ± 0.0443 | 0.7483 ± 0.0305 | 25.7 |
| `DeepCluster` | Iterative | 0.2040 ± 0.0137 | 0.1439 ± 0.0197 | 2.4 |

For context on the same data and preprocessing, KMeans applied directly to the raw 784-dimensional pixels reaches 0.5342 accuracy and 0.4807 NMI.

## Interpretation

`AutoencoderKMeans` at 0.7916 is consistent with the range commonly reported for autoencoder features followed by KMeans on MNIST, and its 0.0188 deviation is the tightest of the three.

`VaDE` at 0.7158 is the same order as published VaDE results but below them, and its 0.0443 deviation is more than twice that of `AutoencoderKMeans`. Any claim about VaDE's level on this setup needs the seed count kept high: at this variance, a three-seed mean carries roughly ±0.026 standard error.

`DeepCluster` at 0.2040 is below the raw-pixel reference, so the method is a net negative contribution on this dataset. The deviation of 0.0137 shows this is stable behaviour rather than seed sensitivity. This is a defect, not a tuning gap; the cause and the bound on candidate repairs are recorded in the [collapse record](deepcluster-collapse-2026-10-28.md).

## Scope

These are architecture-reference numbers under the committed configurations, not reproductions of the original papers. Preprocessing, backbone sizes, optimisation schedules and training budgets differ from the published recipes, and the configurations were not tuned against labels. Results are not comparable with other sources unless preprocessing, sample split, hyperparameters, seeds and metric definitions all match.
