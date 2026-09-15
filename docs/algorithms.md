# Algorithm status

Status is evidence-based: an algorithm is listed as a reference implementation only after paper traceability, deterministic CPU tests and an end-to-end benchmark path are present.

| Algorithm | Survey category | API | Correctness evidence | Benchmark evidence | Status |
| --- | --- | --- | --- | --- | --- |
| Autoencoder + KMeans | Multi-stage | `AutoencoderKMeans` | Frozen stage boundary, arbitrary cloned shallow clusterer and estimator checks | CPU smoke config; five-seed MNIST reference config committed | Architecture reference |
| DeepCluster | Iterative | `DeepCluster` | Alternating KMeans/pseudo-label classification, balanced resampling, image augmentation and estimator checks | CPU NCHW smoke config; five-seed MNIST practical reference config committed | Architecture reference |
| VaDE | Generative | `VaDE` | GMM posterior normalization, analytic identity-KL test, ELBO training, sampling and estimator checks | CPU smoke config; five-seed MNIST reference config committed | Architecture reference |
| DEC | Simultaneous | `DEC` | Objective, full-dataset target distribution, update schedule and estimator checks | Five-seed MNIST record plus CPU smoke config | Reference implementation |
| IDEC | Simultaneous | `IDEC` | Reconstruction-preserving objective, shared DEC core and estimator checks | Five-seed MNIST record plus CPU smoke config | Reference implementation |

“Architecture reference” means the defining interaction pattern and mathematical core are implemented and tested, while the compact default backbone or protocol is not a claim of paper-level reproduction. See the individual method specifications, [four-pattern architecture review](architecture/four-pattern-review.md), [DEC/IDEC traceability](architecture/dec-idec-traceability.md) and [benchmark records](benchmarks/current-baseline.md). Results are not directly comparable unless preprocessing, sample split, hyperparameters, seeds and metric definitions match.
