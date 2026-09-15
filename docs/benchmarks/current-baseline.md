# Pre-v0.2 baseline record

The repository README reported the following single-run results before the v0.2 engineering work. They are retained as historical observations, not reproducible reference results, because the prior logs did not record a commit SHA, complete environment, dataset checksum, preprocessing recipe or seed aggregation.

| Method | MNIST ACC | STL10 ACC | CIFAR10 ACC |
|---|---:|---:|---:|
| DEC | 69.79% | 26.56% | 21.13% |
| IDEC | 69.31% | 26.60% | 21.19% |

The snapshot is Git commit `4e7d0edee09b0bab9404c203989dae1220a3744e` (2026-05-13). Its checked-in defaults were seed 2024, Adam, learning rate 0.4, weight decay 0.0001, batch size 256 and 1000 epochs, with method dimensions 500-500-2000-10 and IDEC gamma 0.1. The README did not identify whether all reported rows used those defaults, nor preserve dependency versions, raw logs or preprocessing checksums; those provenance fields are therefore recorded as unknown rather than reconstructed after the fact.

Known semantic defects in that baseline were: auxiliary targets computed independently inside each mini-batch; one tolerance parameter reused for KMeans and training; no DEC label-change stop rule; and IDEC computing `cluster_loss + gamma * reconstruction_loss` rather than the published `reconstruction_loss + gamma * cluster_loss`. The numbers therefore must not be used as the v0.2 correctness oracle.

The authoritative replacement is the JSON artifact produced by the reference YAML files in `configs/benchmarks`. It records all seeds, mean±standard deviation, environment, commit SHA and stopping state. Until those five-seed GPU runs are completed, the reference result is intentionally marked pending.
