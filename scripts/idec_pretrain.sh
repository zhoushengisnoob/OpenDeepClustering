#!/usr/bin/env bash
set -euo pipefail
echo "Pretraining is part of the IDEC estimator pipeline; edit pretrain_epochs in the benchmark YAML." >&2
python -m opendeepclustering benchmark --config configs/benchmarks/idec_mnist_reference.yaml
