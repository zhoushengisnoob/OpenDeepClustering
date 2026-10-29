# VaDE seed variance (2026-10-29)

This record characterises why VaDE's MNIST result varies materially across seeds, using training-state evidence rather than final scores alone. The conclusion is that the spread originates in the pretrained embedding the Gaussian mixture is initialised on, and that ELBO training amplifies rather than closes the resulting gap.

No implementation change is applied here. One candidate correction is measured and reported as insufficiently supported for adoption at this sample size.

## Provenance

- Host: `lab-gpu-32`, six NVIDIA GeForce RTX 3090, driver 555.42.02
- Git commit: `58b981acf753bcacef627680f3d2b2dad4c583c4`
- Python 3.10.21, NumPy 1.26.4, scikit-learn 1.4.2, PyTorch 2.1.2+cu121, CUDA runtime 12.1
- Dataset: MNIST train+test concatenated, 70,000 × 784, scaled by the global maximum absolute value
- Reference protocol as committed in `vade_mnist_reference.yaml`: `hidden_dims` 500-500-2000, `latent_dim` 10, `pretrain_epochs` 10, `max_epochs` 100, `batch_size` 100, `lr` 0.002, `n_init` 20, Bernoulli reconstruction, `deterministic: true`
- Paired trajectory diagnostic: 2 runs. Stability screen: 12 runs (4 variants × 3 seeds), zero failures
- Test suite after the work: 39 passed, 1 xfailed

Both experiments reproduce the [audited 20-seed baseline](three-pattern-mnist-2026-09-18.md) exactly on shared seeds, which establishes that the instrumented and substituted runs measure shipped behaviour. The trajectory probe reached 0.6513 on seed 12 and 0.8005 on seed 18 against recorded values of 0.6513 and 0.8005; the screen's control variant reached 0.7734, 0.7465 and 0.6731 on seeds 0–2 against recorded 0.7734, 0.7465 and 0.6731.

Instrumentation attaches through the estimator's existing callback hook, so no part of the optimisation path is reimplemented. Per-run JSON is on the server under `benchmark_outputs/vade_diag_20261028/` and `benchmark_outputs/vade_gateA_20261028/`, both Git-ignored; this page is the versioned summary.

## Observed spread

The audited baseline gives VaDE accuracy 0.7158 ± 0.0454 with a 95% confidence interval of [0.6945, 0.7370], and ARI 0.6210 ± 0.0582. The extremes are seed 12 at 0.6513 and seed 18 at 0.8005, a range of 0.149. All 20 runs completed; none failed and none exhausted a different stopping path, because the estimator has no convergence criterion and always runs the full epoch budget.

## Where the variance originates

Two seeds at the extremes were instrumented at every fifth ELBO epoch, plus once immediately after mixture initialisation and before any ELBO step.

| Stage | Seed 12 ACC | Seed 18 ACC |
| --- | ---: | ---: |
| After autoencoder pretraining and GMM fit | 0.6028 | 0.6773 |
| ELBO epoch 10 | 0.6033 | 0.7604 |
| ELBO epoch 40 | 0.6353 | 0.7899 |
| ELBO epoch 100 (final) | 0.6513 | 0.8005 |

The gap is 0.075 before ELBO training begins and 0.149 at the end. Seed 12 gained 0.049 over 100 epochs; seed 18 gained 0.123. Training amplifies the initial difference instead of converging the two runs toward a shared solution.

Nothing in the loss or mixture state distinguishes the two runs:

| Final quantity | Seed 12 | Seed 18 |
| --- | ---: | ---: |
| Reconstruction loss | 78.03 | 78.07 |
| KL | 18.44 | 18.39 |
| Pretraining loss at its last epoch | 77.88 | 77.86 |
| Components holding over 1% of samples | 10 | 9 |
| Mean posterior confidence | 0.940 | 0.963 |
| Prior log-variance range | [-4.18, -0.87] | [-3.99, -0.46] |

The worse run has the marginally *lower* reconstruction loss and the marginally *higher* component count. Pretraining loss is effectively identical, so the autoencoder reconstructs equally well in both cases while producing embeddings of visibly different cluster separability. Neither the training objective nor component occupancy is a usable proxy for clustering quality here.

This also rules out the mechanism the investigation initially targeted. There is no component collapse: occupancy stays at 9 or 10 of 10 across every run in both experiments, and the run that dropped to 9 is the better one.

## Stability screen

Four configurations, three seeds each, fixed before any metric was consulted. `V0` is the committed reference protocol and acts as the control. `V1` and `V2` vary parameters the estimator already exposes. `V3` tests one implementation observation: `_pretrain` omits `posterior_log_variance` from its optimiser parameter list, so the variance head enters ELBO training at its random initialisation; `V3` includes it and trains through the reparameterised sample.

| Variant | Change | Init ACC | Final ACC | Final range | NMI |
| --- | --- | ---: | ---: | ---: | ---: |
| V0 reference | — | 0.6180 ± 0.0140 | 0.7310 ± 0.0424 | 0.100 | 0.7455 ± 0.0218 |
| V1 pretrain50 | `pretrain_epochs` 10 → 50 | 0.6553 ± 0.0239 | 0.7288 ± 0.0272 | 0.067 | 0.7268 ± 0.0113 |
| V2 ninit100 | `n_init` 20 → 100 | 0.6180 ± 0.0140 | 0.7302 ± 0.0411 | 0.098 | 0.7456 ± 0.0195 |
| V3 pretrain variance head | train `posterior_log_variance` during pretraining | 0.6076 ± 0.0198 | **0.7693 ± 0.0287** | 0.067 | **0.7578 ± 0.0223** |

Deviations are population standard deviations over three seeds; at n = 3 they are a weak estimate of spread and are shown to compare effect sizes within the screen, not as uncertainty claims.

**Raising GMM restarts changes nothing.** `V2` matches `V0` almost exactly: identical initialisation accuracy to four decimals, and on seed 0 an identical final accuracy of 0.7734. Eighty additional restarts find no better mixture, so the spread does not come from the mixture fit being under-optimised. It comes from the embedding the mixture is fitted on.

**Longer pretraining improves the start and worsens the end.** `V1` raises initialisation accuracy by 0.037, the largest initialisation gain in the screen, yet final accuracy falls slightly and NMI falls by 0.019. Its reconstruction loss is the lowest of all variants at about 77.1 against 78.0 for the control. Reconstructing better produced worse clustering, which is the same objective misalignment visible in the paired trajectory.

**Training the variance head during pretraining is the only improvement.** `V3` raises final accuracy by 0.038 and NMI by 0.012, and narrows the three-seed range from 0.100 to 0.067, despite having the *lowest* initialisation accuracy of any variant. The benefit therefore appears in ELBO trainability rather than in the starting point, which is consistent with the variance head no longer entering ELBO optimisation at a random initialisation.

## Numerical robustness

The prior's `log_variances` is unconstrained, while the posterior's is clamped to [-12, 12] in `_VaDEModel.encode`. The prior value appears in the denominator of `expected_kl` as `prior_variance`, so an unbounded drift toward large negative values would be a divide-by-near-zero risk.

Across all 14 instrumented and screened runs the prior log-variance stayed within [-5.01, -0.27], corresponding to variances in [0.0067, 0.76]. The risk is therefore real as a robustness gap but did not materialise in any run here, and it is not a contributor to the observed variance. It is recorded as a hardening item, not as a cause.

`GaussianMixturePrior.initialize` does clamp incoming GMM covariances at 1e-6 and incoming weights at 1e-12 before taking logarithms, so a degenerate component returned by scikit-learn cannot produce a non-finite prior. That behaviour is now covered by tests.

## Tests added

Six deterministic tests, none requiring a GPU:

- Mixture parameter shapes, and that `initialize` round-trips weights through log-space so `weights` recovers the input and sums to 1.
- Responsibility normalisation per sample for both `responsibilities` and the variational responsibilities returned by `expected_kl`, including a point far from every component mean.
- Covariance clamping: a zero and a 1e-12 covariance stay finite and respect the 1e-6 floor, and `expected_kl` remains finite on them.
- Zero mixture weight stays representable with finite logits and finite KL.
- Fitted mixture state is well formed: finite parameters, normalised weights, correct `cluster_centers_` and `embedding_` shapes, responsibilities in [0, 1] summing to 1.
- Restart reproducibility: two deterministic fits at one seed agree exactly on labels, centers, responsibilities and mixture weights.

VaDE has no checkpoint or resume path — only DEC and IDEC implement one through `_torch_dec.py` — so restart reproducibility is the property available to test. It also underpins this report: per-seed spread is only interpretable if a single seed is itself repeatable.

## Declared status: validated with stated instability

VaDE is a **validated practical reference with stated instability**.

The mean of 0.7158 ± 0.0454 over 20 audited seeds is a usable reference figure, and it is the same order as published VaDE results on MNIST while sitting below them. The method clears the bar that `DeepCluster` fails: it substantially exceeds KMeans on raw pixels at 0.5342, so the representation learning contributes. No run failed, no component collapsed, and the mixture parameters are well formed throughout.

The instability must be stated with the number. The seed range on the audited batch is 0.149 in accuracy, and the 95% confidence interval for the mean spans 0.0425. Any claim quoting a single VaDE figure needs the seed count and the interval attached. At this variance a three-seed mean carries roughly ±0.026 standard error, which is large relative to the differences usually of interest between configurations.

The instability is intrinsic to the present two-stage reference boundary rather than a defect. Pretraining optimises reconstruction, the mixture is then fitted to whatever embedding that produced, and nothing in either stage selects for cluster separability. The screen shows this directly: more mixture restarts cannot help, and more pretraining helps the initialisation while hurting the final result.

## Gate B not run

No 20-seed confirmation was performed. `V3` is the only variant that improved on the control, but at three seeds its +0.038 accuracy gain is not separable from a spread whose within-variant range is still 0.067, and adopting it would change a published reference number on that basis. The issue's gate directs confirmation only after a screen "improves training stability without violating the method"; the stability improvement here is suggestive and under-powered.

The defensible next step is a 20-seed comparison of `V0` against `V3` alone, which would cost roughly 9 GPU-card-hours at the observed 26 minutes per run and would resolve whether the gain and the narrowing survive. That is a separate, pre-registered decision rather than an extension of this screen, and it should be taken on the algorithmic rationale — that the variance head should not enter ELBO training untrained — rather than on these three seeds.

## Open items

1. Decide `V3` on 20 seeds, or document the `_pretrain` omission as intentional.
2. Bound the prior `log_variances`, matching the posterior's existing clamp.
3. VaDE reports `converged_ = True` and `stop_reason_ = "completed"` unconditionally after exhausting `max_epochs`, with no convergence criterion. As with `DeepCluster`, those fields carry no information for this estimator.
4. Consider recording mixture occupancy and posterior confidence in the benchmark schema, alongside the `effective_rank` field added for collapse detection, so future batches preserve the training-state evidence this report had to regenerate.
