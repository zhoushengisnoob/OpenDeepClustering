# Provenance and licenses

This page records the source and redistribution boundary reviewed for the 0.1 packaging baseline. It is an engineering provenance record, not legal advice.

## Repository material

| Material | Provenance evidence | Treatment |
| --- | --- | --- |
| `opendeepclustering/`, tests, CI and documentation | Developed in this repository; Git history identifies the contributing authors | Distributed under the repository BSD-2-Clause license |
| Historical code under `models/`, `data/` and `utils/` | File headers identify Guanbao Liang as author and state “BSD 2 clause”; Git history preserves the commits | Headers are retained; code is source-checkout material and excluded from the wheel and source distribution |
| `pic/deepclustering-logo.png` | Added to this repository by Guanbao Liang in commit `36e02b1`; no third-party source is recorded | Included only in the source distribution as project documentation artwork; do not reuse independently without confirming rights |
| Benchmark JSON and YAML | Generated or authored in this repository | JSON outputs are excluded from distributions; maintained benchmark YAML is included in the source distribution |

The installable wheel contains only the `opendeepclustering` Python package and generated distribution metadata. It does not bundle datasets, model checkpoints, paper PDFs, historical scripts or dependency source code.

## Runtime and optional dependencies

Dependencies are installed separately by the Python package manager and are not redistributed in the OpenDeepClustering wheel. The audit used the upstream projects' license files as the authoritative references.

| Dependency | Role | Upstream license reference |
| --- | --- | --- |
| NumPy | Arrays and numerical operations | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| SciPy | Hungarian assignment | [SciPy license](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| scikit-learn | Estimator protocol, KMeans and metrics | [scikit-learn COPYING](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| PyTorch | Neural-network training | [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| PyYAML | Benchmark configuration parsing | [PyYAML LICENSE](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| torchvision | Optional MNIST/image loading | [torchvision LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE) |

FAISS, EasyDict, Matplotlib, Joblib and tqdm were present in the historical requirements file but are not direct dependencies of the installable package. They were removed from the authoritative package metadata. Transitive dependencies remain governed by their own licenses.

The Autoencoder + KMeans, DeepCluster and VaDE estimators were newly written in this repository from the published method descriptions and survey taxonomy. No third-party implementation code was copied. Their method specification pages link the primary papers and distinguish reproduced semantics from modern practical alternatives.

## Contribution rule

New algorithms must record whether their implementation is original, adapted, or vendored. Adapted or vendored material requires an exact source URL or revision, a compatible license, preservation of required notices, and a note describing the changes. See [CONTRIBUTING.md](https://github.com/zhoushengisnoob/OpenDeepClustering/blob/master/CONTRIBUTING.md) for the contributor workflow.
