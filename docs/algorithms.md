# Algorithm status

Status is evidence-based: an algorithm is listed as a reference implementation only after paper traceability, deterministic CPU tests and an end-to-end benchmark path are present.

| Algorithm | Survey category | API | Correctness evidence | Benchmark evidence | Status |
| --- | --- | --- | --- | --- | --- |
| Autoencoder + KMeans | Multi-stage | `AutoencoderKMeans` | Frozen stage boundary, arbitrary cloned shallow clusterer and estimator checks | CPU smoke config; [20-seed MNIST](benchmarks/three-pattern-mnist-2026-09-18.md) accuracy 0.7916 ± 0.0188 | Architecture reference |
| DeepCluster | Iterative | `DeepCluster` | Alternating KMeans/pseudo-label classification, balanced resampling, image augmentation and estimator checks | 20-seed MNIST accuracy 0.2040 ± 0.0137, below raw-pixel KMeans at 0.5342; cause and repair ablation in the [collapse record](benchmarks/deepcluster-collapse-2026-10-28.md) | **Known defect** |
| VaDE | Generative | `VaDE` | GMM posterior normalization, analytic identity-KL test, ELBO training, sampling and estimator checks | CPU smoke config; [20-seed MNIST](benchmarks/three-pattern-mnist-2026-09-18.md) accuracy 0.7158 ± 0.0443 | Architecture reference |
| DEC | Simultaneous | `DEC` | Objective, full-dataset target distribution, update schedule and estimator checks | Five-seed MNIST record plus CPU smoke config | Reference implementation |
| IDEC | Simultaneous | `IDEC` | Reconstruction-preserving objective, shared DEC core and estimator checks | Five-seed MNIST record plus CPU smoke config | Reference implementation |

“Architecture reference” means the defining interaction pattern and mathematical core are implemented and tested, while the compact default backbone or protocol is not a claim of paper-level reproduction. See the individual method specifications, [four-pattern architecture review](architecture/four-pattern-review.md), [DEC/IDEC traceability](architecture/dec-idec-traceability.md), [four-pattern smoke record](benchmarks/four-pattern-smoke-2026-09-15.md) and [DEC/IDEC benchmark records](benchmarks/current-baseline.md). Results are not directly comparable unless preprocessing, sample split, hyperparameters, seeds and metric definitions match.

“Known defect” means the interaction pattern runs end to end but the method's clustering quality is measurably worse than omitting representation learning entirely, so the implementation must not be used as a baseline or as a correctness oracle. `DeepCluster` carries that status because its embedding collapses to an effectively one-dimensional representation at initialisation. The [collapse record](benchmarks/deepcluster-collapse-2026-10-28.md) isolates the cause and bounds what the identified repairs achieve.
