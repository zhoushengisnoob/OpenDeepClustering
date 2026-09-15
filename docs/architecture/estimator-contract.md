# Estimator contract and extension boundary

OpenDeepClustering exposes algorithms as scikit-learn estimators. Constructor arguments are configuration only; learned state is created by `fit` and carries a trailing underscore.

## Public contract

- Required: `fit`, `fit_predict`, `transform`, `get_params`, `set_params`.
- Required learned attributes: `labels_`, `embedding_`, `cluster_centers_`, `n_features_in_`, `n_iter_`, `converged_`, `stop_reason_`, `history_`.
- Optional capability: `soft_assign`/`predict_proba`, declared by `supports_soft_assignment`.
- Optional capability: `sample`, declared by `supports_sample`.
- Discovery: `get_capabilities()` reports taxonomy, supported input modalities and optional APIs without fitting.
- Input: dense 2D array-like values or CPU/GPU Torch tensors. Images require the explicit `flatten_samples` adapter; sparse arrays, NaN and infinity are rejected explicitly.
- Reproducibility: `random_state` controls model initialization, batch order and KMeans; `deterministic=True` requests deterministic Torch kernels without permanently changing process-global RNG state. CUDA reproducibility still depends on the GPU, driver, CUDA/cuDNN and Torch versions, which are captured by benchmark results; cross-platform bitwise identity is not promised.

## Shared layers

`DeepClusterMixin` owns validation and public fitted-state conventions. `data` owns array/tensor and stable-index adaptation. `training` owns fit state, callbacks and scoped random generators. `components` contains reusable autoencoder, Student-t assignment and full-dataset target distribution primitives. An estimator owns only its stage orchestration and objective.

This boundary supports the Survey taxonomy without forcing every method into a DEC loop:

| Family | Representation | Cluster state | Training stages | Expected optional API |
|---|---|---|---|---|
| Multi-stage | frozen after representation learning | external/second stage | representation→clustering | `transform` |
| Iterative | updated between stages | pseudo-label/assignment state | alternating rounds | `history_` |
| Generative | generator/discriminator/latent model | method-specific | adversarial or variational | sample/generate extension |
| Simultaneous | jointly updated | differentiable assignment | pretrain→initialize→joint fit | `soft_assign` |

`AutoencoderKMeans`, `DeepCluster`, `VaDE`, and DEC/IDEC now exercise the multi-stage, iterative, generative and simultaneous boundaries. Each reuses validation, adapters, components and fit state while retaining a method-specific orchestration class.

The fit state is stage-neutral: a sequential estimator records representation and clustering stages; an iterative estimator records alternating rounds; a generative estimator records generator/discriminator stages; and a simultaneous estimator records a joint stage. None is forced into DEC's training loop.

## Compatibility policy

The old files under `models/Simultaneous/*/{main,pretrain}.py` are deprecated command shims and are scheduled for removal in v0.3. They contain no training logic. CLI and Python calls both instantiate `opendeepclustering.DEC` or `opendeepclustering.IDEC`, so bug fixes have one implementation point.
