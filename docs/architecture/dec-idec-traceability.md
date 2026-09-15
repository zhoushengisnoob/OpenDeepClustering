# DEC and IDEC traceability

| Paper concept | Public parameter/state | Implementation |
|---|---|---|
| Encoder mapping z=f(x) | `dims`, `embedding_`, `transform` | `StackedAutoEncoder.encode` |
| Student-t assignment q | `alpha`, `soft_assign` | `StudentTClustering.forward` |
| Soft frequency f_j=sum_i q_ij | internal full dataset state | `target_distribution` |
| Auxiliary target p_ij proportional to q_ij²/f_j | `update_interval` | `_BaseDEC._finetune` and `target_distribution` |
| DEC KL(P||Q) | `cluster_loss` history | `_BaseDEC._combine_losses` |
| IDEC L=L_r+gamma L_c | `gamma`, loss histories | `IDEC._combine_losses` |
| Centroid initialization | `n_init`, `kmeans_max_iter`, `kmeans_tol` | `_initialize_clusters` |
| Assignment-change stopping | `tol`, `converged_`, `stop_reason_`, `n_iter_` | `_finetune` |
| Greedy denoising SAE | `pretrain_method`, `corruption`, `layerwise_pretrain_epochs` | `_greedy_layerwise_pretrain` |
| Global SAE fine-tuning | `pretrain_epochs`, `pretrain_optimizer`, `pretrain_lr` | `_joint_autoencoder_pretrain` |

The DEC reference is Xie et al., “Unsupervised Deep Embedding for Clustering Analysis,” ICML 2016. The IDEC reference is Guo et al., “Improved Deep Embedded Clustering with Local Structure Preservation,” IJCAI 2017. The committed reference YAML files are the executable specification; the smoke YAML is a fast practical wiring profile and is not presented as paper reproduction.

## Recorded reproduction differences

- This implementation is PyTorch rather than the original Caffe DEC and Keras IDEC code.
- Pretraining duration is currently expressed as full-dataset epochs. The DEC paper specified 50,000 updates per greedy layer and 100,000 global autoencoder updates with a tenfold learning-rate drop every 20,000 updates.
- The target is refreshed at a fixed `update_interval`, following the published IDEC algorithm and common later DEC implementations. The original DEC experiments additionally varied an annealing-speed parameter.
- Exact random initialization streams and low-level floating-point kernels differ across frameworks. Benchmark output captures the concrete software and GPU environment.

These differences are fixed before a run and are not selected using evaluation labels. A benchmark is called a reference run because it is the repository's traceable scientific reference, not because it is claimed to reproduce the authors' framework bit for bit.
