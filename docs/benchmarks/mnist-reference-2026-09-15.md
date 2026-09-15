# MNIST reference benchmark — 2026-09-15

This is the first traceable v0.2 scientific baseline. It is a repository reference benchmark, not a claim of bit-for-bit reproduction of the original frameworks.

## Provenance

- Git commit: `7a0a4d2d5df93e4f624e29bb3ca6a5a6fcc5fd2c`
- Host: `lab-gpu-32`; NVIDIA GeForce RTX 3090
- Python 3.10.21; NumPy 1.26.4; scikit-learn 1.4.2; Torch 2.1.2+cu121; CUDA 12.1
- Dataset: all 70,000 MNIST train+test samples, 784 flattened features, per-sample DEC normalization
- Feature SHA-256 after preprocessing: `e28edaf37036180081334c48697607b262608b061888b80f7b8cf24846b2c3a9`
- Label SHA-256: `818800b46032126b329f9306cb69a6842cc53ea30318374769bb6f46cc861467`
- Seeds: 0, 1, 2, 3, 4; labels were used only after fitting for evaluation
- Executable specifications: `configs/benchmarks/dec_mnist_reference.yaml` and `configs/benchmarks/idec_mnist_reference.yaml`
- Full ignored JSON artifacts remain on the server under `benchmark_outputs/reference_seeds/`.

## Results

| Method | Seed | ACC | F1 | NMI | ARI | Updates | Stop |
|---|---:|---:|---:|---:|---:|---:|---|
| DEC | 0 | 79.23% | 77.60% | 81.28% | 74.22% | 9,940 | label delta |
| DEC | 1 | 72.29% | 70.04% | 76.69% | 66.04% | 8,120 | label delta |
| DEC | 2 | 79.62% | 78.10% | 82.38% | 75.40% | 9,100 | label delta |
| DEC | 3 | 79.16% | 77.55% | 81.46% | 74.21% | 9,940 | label delta |
| DEC | 4 | 79.09% | 77.24% | 80.55% | 72.74% | 8,400 | label delta |
| **DEC mean±std** | | **77.88%±2.80%** | **76.10%±3.05%** | **80.47%±1.98%** | **72.52%±3.35%** | | |
| IDEC | 0 | 80.79% | 79.20% | 84.92% | 77.52% | 6,440 | label delta |
| IDEC | 1 | 73.57% | 71.25% | 79.46% | 68.59% | 7,980 | label delta |
| IDEC | 2 | 81.24% | 79.74% | 85.85% | 78.49% | 6,860 | label delta |
| IDEC | 3 | 81.03% | 79.52% | 85.78% | 78.30% | 6,440 | label delta |
| IDEC | 4 | 80.29% | 78.47% | 83.67% | 75.38% | 7,280 | label delta |
| **IDEC mean±std** | | **79.38%±2.92%** | **77.64%±3.22%** | **83.93%±2.37%** | **75.65%±3.70%** | | |

All runs converged through the published assignment-change stopping rule. Mean wall time was about 22 minutes per seed when ten seeds were distributed over six GPUs.

## Published comparison and investigation

The original DEC paper reports 84.30% MNIST ACC. The IDEC paper's common implementation reports 86.55% for DEC and 88.06% ACC/86.72% NMI for IDEC. This baseline is lower by 6.42 ACC points for DEC versus its original paper and 8.68 points for IDEC, so the greater-than-two-point investigation gate was triggered.

The discrepancy is not treated as unexplained. The committed configuration exposes these concrete reproduction gaps:

1. The current epoch-based profile performs about 13,700 updates per greedy layer and 27,400 global autoencoder updates. The DEC paper specifies 50,000 per layer and 100,000 global updates.
2. The paper reduces the pretraining learning rate tenfold every 20,000 updates; this profile keeps it constant.
3. The original DEC experiments vary an annealing-speed parameter that is not part of the later fixed-target-interval formulation implemented here.
4. The framework changed from Caffe/Keras to PyTorch, including loss-reduction and random-stream details.

These are implementation/profile differences fixed before evaluation, not label-selected hyperparameters. The benchmark validates the corrected full-dataset target, stopping, IDEC objective and reproducibility infrastructure, while showing that exact training-budget reproduction remains worthwhile future scientific work.
