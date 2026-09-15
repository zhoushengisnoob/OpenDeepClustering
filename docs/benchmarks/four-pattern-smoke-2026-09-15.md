# Four-pattern architecture smoke record (2026-09-15)

This record validates that one representative of every interaction pattern can run through the shared estimator and benchmark infrastructure. It is wiring and architecture evidence, not a paper-reproduction or algorithm-quality result.

## Provenance

- Host: `lab-gpu-32`
- Git commit: `2a66475ea88e8cb940829da063aac5d8cf7e8d2c`
- Python: 3.10.21
- NumPy: 1.26.4
- scikit-learn: 1.4.2
- PyTorch: 2.1.2+cu121
- CUDA reported by PyTorch: 12.1
- GPU reported by PyTorch: NVIDIA GeForce RTX 3090
- Execution device: CPU, as fixed by every smoke configuration
- Test suite before the runs: 27 passed

The benchmark CLI captured the resolved configuration, environment, input and output SHA-256 digests, per-seed duration, fit status and metrics. Generated JSON files are deliberately ignored build artifacts; this page is the versioned summary.

## Results

All runs used seed 7 and 120 synthetic samples. The metrics only confirm that predictions and common evaluation flow are operational.

| Pattern | Estimator | Input shape | ACC | F1 | NMI | ARI | Duration (s) | Fit status | Configuration |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Multi-stage | AutoencoderKMeans | 120×8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.013 | completed | [`autoencoder_kmeans_smoke.yaml`](https://github.com/zhoushengisnoob/OpenDeepClustering/blob/master/configs/benchmarks/autoencoder_kmeans_smoke.yaml) |
| Iterative | DeepCluster | 120×1×8×8 | 0.8167 | 0.8138 | 0.6547 | 0.5988 | 1.049 | completed | [`deepcluster_smoke.yaml`](https://github.com/zhoushengisnoob/OpenDeepClustering/blob/master/configs/benchmarks/deepcluster_smoke.yaml) |
| Generative | VaDE | 120×8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.023 | completed | [`vade_smoke.yaml`](https://github.com/zhoushengisnoob/OpenDeepClustering/blob/master/configs/benchmarks/vade_smoke.yaml) |
| Simultaneous | DEC | 120×8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.004 | label change below tolerance | [`dec_smoke.yaml`](https://github.com/zhoushengisnoob/OpenDeepClustering/blob/master/configs/benchmarks/dec_smoke.yaml) |
| Simultaneous extension | IDEC | 120×8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.018 | label change below tolerance | [`idec_smoke.yaml`](https://github.com/zhoushengisnoob/OpenDeepClustering/blob/master/configs/benchmarks/idec_smoke.yaml) |

The common 120×8 dataset had feature digest `d2a5cf67481c9b4df778f0d2b21072091e0d863197db8aae23d2b47ef26b39a5` and label digest `5bfc85ab07a541b0ee35db8cd0ef0621daacbf205cad14101deaa553b1454fbd`. DeepCluster's explicit NCHW adaptation produced feature digest `2244f4b915f9422ccb197962c51b4aea0d13c9f21acfc7e9abe178eb9565bccc` and label digest `72474fb279f7666d779c5fef48d64812727b0b8181a3644d4dc01040190fe19f`.

## Acceptance interpretation

The run confirms that all four patterns share configuration parsing, seed control, environment capture, dataset provenance, metrics and result schema while retaining separate method-specific training loops. The committed MNIST reference configurations are the next layer of experimental evidence; they were not run as part of this architecture smoke check.
