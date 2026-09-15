# Reproducible benchmarks

The command below uses the exact estimator core imported by Python users:

```bash
odc benchmark --config configs/benchmarks/dec_smoke.yaml
```

Reference runs use `dec_mnist_reference.yaml` and `idec_mnist_reference.yaml`; both combine the 60,000 training and 10,000 test samples used by the papers. MNIST is not downloaded implicitly; place it under the configured root or set `download: true` deliberately. Each output contains the resolved experiment configuration, Git commit, Python/package/CUDA/GPU environment, post-preprocessing feature and label checksums, per-seed metrics and aggregate mean/std.

The reference configurations encode the paper-facing profile: 784-500-500-2000-10 autoencoder, greedy denoising layer pretraining, full autoencoder fine-tuning, KMeans centroid initialization, full-dataset DEC targets, update interval and label-change stopping. IDEC uses `L_r + gamma L_c`; DEC updates only encoder and clustering parameters after initialization.

The smoke configuration is a CI wiring test, not an algorithm-quality claim. Formal acceptance requires five MNIST seeds on the GPU server and comparison against the historical and paper-reported range.

Checkpoint writes are opt-in. Add `checkpoint: {save_to: path.pt}` to save model, optimizer and fit state; use `checkpoint: {resume_from: path.pt}` to continue from it. With multiple seeds, the CLI appends the seed to each saved filename. Estimators never create a checkpoint unless `save_checkpoint` is called explicitly.
