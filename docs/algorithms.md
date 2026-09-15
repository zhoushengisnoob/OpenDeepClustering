# Algorithm status

Status is evidence-based: an algorithm is listed as a reference implementation only after paper traceability, deterministic CPU tests and an end-to-end benchmark path are present.

| Algorithm | Survey category | API | Correctness evidence | Benchmark evidence | Status |
| --- | --- | --- | --- | --- | --- |
| DEC | Simultaneous | `DEC` | Objective, full-dataset target distribution, update schedule and estimator checks | Five-seed MNIST record plus CPU smoke config | Reference implementation |
| IDEC | Simultaneous | `IDEC` | Reconstruction-preserving objective, shared DEC core and estimator checks | Five-seed MNIST record plus CPU smoke config | Reference implementation |
| Representation-based representative | Representation-based | — | — | — | Planned |
| Pseudo-label-based representative | Pseudo-label-based | — | — | — | Planned |
| Generative representative | Generative | — | — | — | Planned |

See [DEC/IDEC traceability](architecture/dec-idec-traceability.md) for the implementation-to-paper map and [benchmark records](benchmarks/current-baseline.md) for the evidence boundary. Results are not directly comparable unless preprocessing, sample split, hyperparameters, seeds and metric definitions match.
