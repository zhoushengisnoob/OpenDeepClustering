# Contributing to OpenDeepClustering

Thank you for helping make deep-clustering implementations easier to inspect and reproduce.

## Development setup

OpenDeepClustering supports Python 3.10 and 3.11 in continuous integration. Create an isolated environment, then install the development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

Install the `image` extra only when working with image datasets. The `benchmark` extra provides the same optional image-loading dependency used by the reference MNIST configurations.

## Before opening a pull request

Run the same checks used by continuous integration:

```bash
python -m pytest -q
odc benchmark --config configs/benchmarks/dec_smoke.yaml
mkdocs build --strict
python -m build
python -m twine check dist/*
check-wheel-contents dist/*.whl
python tools/check_dist.py dist
cffconvert --validate
```

Algorithm changes should include a focused correctness test, a deterministic smoke configuration, and an update to the algorithm status page. Do not commit downloaded datasets, checkpoints, logs, benchmark outputs, or credentials. GPU reference runs stay outside public CI; record their configuration, environment and aggregate results under `docs/benchmarks/`.

## Provenance and licensing

Contributions are accepted under the repository's BSD-2-Clause license. Submit only work you have the right to license, retain required upstream notices, and identify adapted code or assets in `docs/provenance.md`. A contribution should not copy an implementation merely because it accompanies a paper; translate the paper into independently reviewable code or document the compatible upstream license and the exact source.
