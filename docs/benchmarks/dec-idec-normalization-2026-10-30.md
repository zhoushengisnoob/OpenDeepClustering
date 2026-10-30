# DEC/IDEC normalization defect and gold-standard validation (2026-10-30)

Our DEC and IDEC reached 0.7788 and 0.7938 MNIST accuracy against the 0.8430 and 0.8806 quoted in the papers. This record establishes that **the clustering implementation is correct** and locates the whole shortfall in one preprocessing choice.

The validation method matters as much as the result. Published figures in this field are widely transcribed between papers rather than re-measured, and the IDEC author's own quick-start repository states that its results are worse than the paper's. So the reference used here is not a quoted number: it is the original authors' released pretrained weights, run through our clustering loop.

## Provenance

- Host: `lab-gpu-32`, NVIDIA GeForce RTX 3090, driver 555.42.02
- Git commit: `ec55123c8ad93d31f1b4f49826661009da467f3b`
- Python 3.10.21, PyTorch 2.1.2+cu121, NumPy 1.26.4, scikit-learn 1.4.2
- Dataset: MNIST train+test concatenated, 70,000 × 784
- Reference source: [`XifengGuo/IDEC`](https://github.com/XifengGuo/IDEC) at `4a661c2`, the Keras implementation covering both DEC and IDEC
- Reference weights: `mnist_ae_weights.h5` from [`XifengGuo/data-and-models`](https://github.com/XifengGuo/data-and-models), the file that repository's instructions use to reproduce its paper numbers
- Probes: 5 single-seed runs, seed 0, each the clustering stage only

## Method: released weights through our loop

The shortfall had two possible homes — the autoencoder pretraining, which the original work performed in Caffe with a greedy layerwise schedule, or our clustering loop. Loading the authors' released weights and running only our clustering stage separates them.

The architectures match exactly. The released file holds a symmetric 784-500-500-2000-10 autoencoder with ReLU on every layer except the input, the embedding and the output, which is what `StackedAutoEncoder` builds. Keras stores kernels as `(in, out)` and PyTorch as `(out, in)`, so each of the eight weight matrices is transposed on load; the mapping is by layer name, not by position.

## Result

| Configuration | ACC | NMI | Paper |
| --- | ---: | ---: | --- |
| IDEC, released weights, `/50` preprocessing | **0.8858** | 0.8765 | 0.8806 / 0.8672 |
| IDEC, released weights, `global_rms` | 0.8706 | 0.8517 | — |
| IDEC, released weights, our `dec` normalization | 0.7909 | 0.8429 | — |
| IDEC, our pretraining, our `dec` normalization (5-seed mean) | 0.7938 | 0.8393 | — |
| DEC, released weights, `/50` preprocessing | **0.8624** | 0.8341 | 0.8430 (0.8655 as quoted in the IDEC paper) |
| DEC, released weights, `global_rms` | 0.8548 | 0.8254 | — |
| DEC, our pretraining, our `dec` normalization (5-seed mean) | 0.7788 | 0.8047 | — |

**Our clustering loop reaches or exceeds both papers.** With the authors' weights and the authors' preprocessing, IDEC scores 0.8858 against a published 0.8806, and DEC scores 0.8624 against a published 0.8430.

**The shortfall is preprocessing, not pretraining.** Rows three and four are the decisive comparison: under our `dec` normalization, the authors' weights (0.7909) and our own pretrained weights (0.7938) perform the same. Pretraining is therefore not the bottleneck. Holding the weights fixed and changing only the normalization moves IDEC from 0.7909 to 0.8858, a difference of 0.095.

## Cause: per-sample scaling discards class information

The DEC paper specifies normalizing "all datasets so that `(1/d)·||x_i||₂²` is approximately 1". We implemented that literally, as a per-sample division by each sample's own root mean square. The original code instead divides the entire dataset by the constant 50, with the comment "normalize as it does in DEC paper".

Per-sample scaling forces every sample to `mean(x²) = 1` exactly. On MNIST that destroys a discriminative signal, because overall ink mass differs systematically by class:

| Digit | Mean per-sample RMS |
| ---: | ---: |
| 0 | 98.07 |
| 1 | 64.46 |
| 2 | 90.48 |
| 8 | 90.51 |

The between-class standard deviation of per-sample RMS is 8.52 against an overall standard deviation of 15.52, so roughly a third of that variation is class information. A thin `1` and a heavy `1` should remain close while a `1` and a `0` should not, and per-sample normalization collapses exactly that distinction. Measured across the dataset:

| Normalization | Dataset `mean(x²)` | Per-sample `mean(x²)` range |
| --- | ---: | --- |
| `/50` (original code) | 2.920 | [0.423, 9.791] |
| `global_rms` (now ours) | 1.000 | [0.145, 3.353] |
| `dec` (previously ours) | 1.000 | [1.000, 1.000] |

The collapsed range in the last row is the defect.

## Fix

A `global_rms` normalization divides the whole dataset by one constant, its global root mean square, so `mean(x²)` is 1 over the dataset as the paper requires while per-sample magnitude differences survive. It carries no magic number, unlike the original `/50`, and it is within 0.015 accuracy of that constant on both methods.

The divisor is computed **before** subsampling. Computing it after would scale a subset and the full dataset differently, making their results incomparable.

`normalization` is now a closed set: `global_rms`, `dec`, `unit`, `none`. An unrecognised value raises. Previously an unknown value was silently ignored and the data passed through unnormalized, so a typo produced a plausible-looking but unnormalized run.

`dec` is retained rather than removed. It is the literal reading of the paper's text and the configuration that produced the earlier reference records, so deleting it would make those records unreproducible. Its docstring and the dataset contract now state that it discards per-sample magnitude.

## What this says about published figures

The paper's text and the authors' own code disagree. The text states a per-sample condition; the code applies a global constant and calls it the same thing. Implementing the text faithfully costs 0.095 accuracy on MNIST.

This is a concrete case for preferring executable references over quoted ones. A number transcribed from a paper carries neither the preprocessing that produced it nor the discrepancy between a paper and its own implementation. The [IDEC author's quick-start repository](https://github.com/XifengGuo/IDEC-toy) states plainly that its results are worse than the paper's, and the DCEC paper tabulates a separate "DEC†" row for figures obtained from public code rather than from the DEC paper. Both are the same caution.

Validation in this repository therefore prefers, in order: the original authors' released weights or code run locally; a configuration-matched local reproduction; and only then a published figure, labelled as transcribed.

## Scope

Seed 0 only, for the probes. These runs establish which component holds the defect, not a multi-seed reference level; that is the [`global_rms` reference record](dec-idec-globalrms-2026-10-30.md). The released weights are not redistributed here and are not required to use the package — they were a diagnostic instrument.
