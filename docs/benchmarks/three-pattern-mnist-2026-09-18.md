# Multi-stage, iterative and generative MNIST baseline (2026-09-18)

This is the audited 20-seed MNIST baseline for the three representatives that previously had no multi-seed quality evidence: `AutoencoderKMeans`, `DeepCluster` and `VaDE`. The runs completed on 2026-09-18 but were not summarised at the time, because per-run JSON lands in a Git-ignored output directory and does not reach the repository on its own.

Every number below is regenerated from a committed record rather than transcribed. No new GPU time was spent on this report.

## Reproducing this report

The per-seed metrics are committed in [`records/mnist-20seed-2026-09-18.json`](records/mnist-20seed-2026-09-18.json) (37 KB, SHA-256 `d064e66b196d29e5147736106342a23478a2d53e0a87f7f1814e75fcbd5e87f5`). To rebuild and verify every table:

```bash
python tools/report_benchmark_record.py docs/benchmarks/records/mnist-20seed-2026-09-18.json --check
```

The command recomputes each aggregate from the per-seed values and exits non-zero if any stored figure drifts. The record itself was produced by `tools/audit_benchmark_batch.py` from the server directory `benchmark_outputs/v03_reference_20260918_2240/results`, which holds the bulky raw JSON and logs and stays off Git.

## Provenance

- Host: `lab-gpu-32`, six NVIDIA GeForce RTX 3090, driver 555.42.02
- Git commit of the runs: `e01d2c1bcebde8d254f09dbd74cb418a16f6138c`
- Python 3.10.21, NumPy 1.26.4, scikit-learn 1.4.2, PyTorch 2.1.2+cu121, CUDA runtime 12.1
- Dataset: MNIST train+test concatenated, 70,000 × 784 after preprocessing
- Feature digest `d27fb91380ff036e...`, label digest `818800...`, identical across all 60 runs
- Scheduling: six concurrent single-GPU workers, one seed per process, isolated by `CUDA_VISIBLE_DEVICES`
- Outcome: 60 planned runs, 60 completed, zero failures and zero timeouts, 2 hours 1 minute wall clock

`opendeepclustering/` and `configs/` are byte-identical between the run commit `e01d2c1` and the current `58b981a`; only `.gitignore`, `README.md` and `datasets/README.md` differ. These results therefore still describe the current estimator code. Verify this again before citing them against a later revision.

This interpreter differs from the PyTorch 2.5.1 / Python 3.12 profile in the README installation section. It is the same environment that produced the earlier DEC/IDEC reference evidence.

## Integrity audit

The audit checked all 60 records and reported **zero problems**:

- All 20 seeds (0–19) present for each estimator, with no duplicates and no omissions.
- `git_sha`, feature digest, label digest, sample count, feature count and schema version identical within each estimator group.
- Estimator parameters, dataset configuration and environment identical within each group, excluding `random_state`, which varies with the seed by design.
- 20 distinct predicted-label digests per estimator, confirming the seeds produced genuinely different solutions rather than duplicated records.
- Every run reported `completed`, which for these estimators means the configured round or epoch budget was exhausted. None of them implements a convergence criterion, so `completed` carries no information about convergence.

No run is excluded from any figure below.

## Results

Seeds 0–19, n = 20 per estimator. Deviations are sample standard deviations; intervals are two-sided 95% Student-t confidence intervals for the mean.

| Estimator | Pattern | ACC | NMI | ARI | Median minutes |
| --- | --- | ---: | ---: | ---: | ---: |
| `AutoencoderKMeans` | Multi-stage | 0.7916 ± 0.0193 | 0.7634 ± 0.0121 | 0.7027 ± 0.0184 | 3.65 |
| `VaDE` | Generative | 0.7158 ± 0.0454 | 0.7483 ± 0.0313 | 0.6210 ± 0.0582 | 25.49 |
| `DeepCluster` | Iterative | 0.2040 ± 0.0140 | 0.1439 ± 0.0202 | 0.0644 ± 0.0115 | 2.44 |

Confidence intervals and medians for the primary metrics:

| Estimator | ACC mean | ACC 95% CI | ACC median | NMI mean | NMI 95% CI |
| --- | ---: | :---: | ---: | ---: | :---: |
| `AutoencoderKMeans` | 0.7916 | [0.7825, 0.8006] | 0.7869 | 0.7634 | [0.7577, 0.7691] |
| `VaDE` | 0.7158 | [0.6945, 0.7370] | 0.7090 | 0.7483 | [0.7337, 0.7630] |
| `DeepCluster` | 0.2040 | [0.1974, 0.2106] | 0.2022 | 0.1439 | [0.1344, 0.1534] |

For context on the same data and preprocessing, KMeans applied directly to the raw 784-dimensional pixels reaches 0.5342 accuracy and 0.4807 NMI.

## Interpretation

`AutoencoderKMeans` at 0.7916 is consistent with the range commonly reported for autoencoder features followed by KMeans on MNIST, and its interval is the tightest of the three.

`VaDE` at 0.7158 is the same order as published VaDE results but below them, and its 0.0454 deviation is more than twice that of `AutoencoderKMeans`. Its ARI interval spans 0.055, so any claim about VaDE's level on this setup needs the seed count kept high; a three-seed mean at this variance carries roughly ±0.026 standard error on accuracy.

`DeepCluster` at 0.2040 is below the raw-pixel reference, so the method is a net negative contribution on this dataset. The interval [0.1974, 0.2106] excludes any plausible reading as high variance around a usable mean. This is a defect, not a tuning gap. The cause and the bound on candidate repairs are recorded in the [collapse record](deepcluster-collapse-2026-10-28.md), and this baseline is the diagnostic input for that work rather than a result to present as representative evidence.

## Scope

These are architecture-reference numbers under the committed configurations, not reproductions of the original papers. Backbone sizes, optimisation schedules, preprocessing and training budgets differ from the published recipes.

The configurations were fixed before the runs and were not adjusted against these metrics; test labels measured the outcome and did not select the setup. Results are not comparable with other sources unless preprocessing, sample split, hyperparameters, seeds and metric definitions all match.
