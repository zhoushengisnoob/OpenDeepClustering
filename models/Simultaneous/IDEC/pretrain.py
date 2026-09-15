"""Deprecated IDEC pretraining shim; use the estimator configuration instead."""

from opendeepclustering.cli import legacy_main

if __name__ == "__main__":
    raise SystemExit(legacy_main("IDEC"))
