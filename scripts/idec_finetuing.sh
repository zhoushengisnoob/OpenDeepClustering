#!/usr/bin/env bash
set -euo pipefail
python -m opendeepclustering benchmark --config configs/benchmarks/idec_mnist_reference.yaml
