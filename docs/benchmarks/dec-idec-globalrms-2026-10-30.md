# DEC/IDEC reference under global_rms normalization (2026-10-30)

Five-seed MNIST reference for DEC and IDEC after the normalization fix recorded in the [normalization defect record](dec-idec-normalization-2026-10-30.md). These supersede the five-seed figures in the [2026-09-15 record](mnist-reference-2026-09-15.md), which used the defective per-sample `dec` normalization.

## Reproducing this report

Per-seed metrics are committed in [`records/mnist-5seed-dec-idec-globalrms-2026-10-30.json`](records/mnist-5seed-dec-idec-globalrms-2026-10-30.json) (SHA-256 `83013598d3a9a046f709b20b315522f67095ffd1d3f9ed8b88be254c8fab5b4f`):

```bash
python tools/report_benchmark_record.py docs/benchmarks/records/mnist-5seed-dec-idec-globalrms-2026-10-30.json --check
```

## Provenance

- Host: `lab-gpu-32`, six NVIDIA GeForce RTX 3090, driver 555.42.02
- Git commit: `ec55123c8ad93d31f1b4f49826661009da467f3b`
- Python 3.10.21, NumPy 1.26.4, scikit-learn 1.4.2, PyTorch 2.1.2+cu121, CUDA runtime 12.1
- Dataset: MNIST train+test concatenated, 70,000 × 784, `global_rms` normalization
- Feature digest: `1aff75b934807dbd5fbf9b0a788f8f636be5fee49096486a988a3c05b4d7f58b`
- Configurations: [`dec_mnist_reference.yaml`](https://github.com/zhoushengisnoob/OpenDeepClustering/blob/master/configs/benchmarks/dec_mnist_reference.yaml), [`idec_mnist_reference.yaml`](https://github.com/zhoushengisnoob/OpenDeepClustering/blob/master/configs/benchmarks/idec_mnist_reference.yaml), unchanged apart from `normalization`
- Scheduling: six concurrent single-GPU workers, one seed per process
- Outcome: 10 planned, 10 completed, zero failures, zero integrity problems

Pretraining is this repository's own greedy layerwise schedule. The original authors' released weights were used only as a diagnostic in the defect record and are not involved here.

## Results

Seeds 0–4. Deviations are sample standard deviations; intervals are two-sided 95% Student-t confidence intervals for the mean. Every run stopped on `label_change_below_tol`, the assignment-change criterion, rather than by exhausting the epoch budget.

| Estimator | ACC | ACC 95% CI | NMI | ARI | Median minutes |
| --- | ---: | :---: | ---: | ---: | ---: |
| `IDEC` | 0.8771 ± 0.0089 | [0.8661, 0.8882] | 0.8631 ± 0.0099 | 0.8262 ± 0.0154 | 13.31 |
| `DEC` | 0.8616 ± 0.0094 | [0.8500, 0.8733] | 0.8338 ± 0.0127 | 0.7966 ± 0.0175 | 12.72 |

## Comparison with the superseded record and with published figures

| Estimator | `dec` normalization (2026-09-15) | `global_rms` (this record) | Change | Published |
| --- | ---: | ---: | ---: | --- |
| DEC | 0.7788 ± 0.0313 | 0.8616 ± 0.0094 | **+0.0828** | 0.8430; 0.8655 as quoted in the IDEC paper |
| IDEC | 0.7938 ± 0.0327 | 0.8771 ± 0.0089 | **+0.0833** | 0.8806 |

Both methods now sit at published level. DEC exceeds its own paper's 0.8430 and approaches the 0.8655 the IDEC paper attributes to it. IDEC's 0.8771 is 0.0035 below its paper's 0.8806, well inside one standard deviation.

Seed variance fell by roughly two thirds, from 0.0313 to 0.0094 for DEC and from 0.0327 to 0.0089 for IDEC. The defective normalization was not only depressing the mean but destabilising the result: with per-sample magnitude removed, KMeans initialisation on the pretrained embedding settled into more varied local solutions.

These are configuration-matched against each other — one dataset, one preprocessing recipe, one metric implementation, one seed set — and are therefore directly comparable within this record. They are not comparable with the `unit`-normalized records for the other three representatives; see the [method comparison](comparison.md).

## Status

DEC and IDEC remain **reference implementations**. This record raises the confidence in that status rather than changing it: the clustering loop was already shown correct against the original authors' released weights, and it now reaches published level from this repository's own pretraining.

The remaining 0.0035 gap for IDEC and the surplus for DEC are not pursued. Chasing a published figure more closely would mean tuning against test labels, and the published figures themselves are a single run from a different framework.
