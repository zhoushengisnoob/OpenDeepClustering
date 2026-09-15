"""Deprecated IDEC CLI shim; training lives in opendeepclustering.IDEC."""

from opendeepclustering.cli import legacy_main

if __name__ == "__main__":
    raise SystemExit(legacy_main("IDEC"))
