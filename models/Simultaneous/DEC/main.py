"""Deprecated DEC CLI shim; training lives in opendeepclustering.DEC."""

from opendeepclustering.cli import legacy_main

if __name__ == "__main__":
    raise SystemExit(legacy_main("DEC"))
