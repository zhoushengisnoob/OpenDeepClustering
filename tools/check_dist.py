"""Validate that built archives contain only intentional project material."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path


WHEEL_PACKAGE_PREFIX = "opendeepclustering/"
WHEEL_METADATA_MARKER = ".dist-info/"
SDIST_ALLOWED_ROOTS = {
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "PKG-INFO",
    "README.md",
    "configs",
    "docs",
    "mkdocs.yml",
    "opendeepclustering",
    "opendeepclustering.egg-info",
    "pic",
    "pyproject.toml",
    "setup.cfg",
    "tests",
    "tools",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    "benchmark_outputs",
    "checkpoints",
    "data",
    "datasets",
    "exps",
    "logs",
    "models",
    "scripts",
    "utils",
}
FORBIDDEN_WHEEL_PARTS = FORBIDDEN_PARTS - {"data"}


def _check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
    unexpected = [
        name
        for name in names
        if not name.startswith(WHEEL_PACKAGE_PREFIX)
        and WHEEL_METADATA_MARKER not in name
    ]
    forbidden = [
        name for name in names if FORBIDDEN_WHEEL_PARTS.intersection(Path(name).parts)
    ]
    if unexpected or forbidden:
        raise RuntimeError(
            f"Unexpected wheel content in {path}: {sorted(set(unexpected + forbidden))}"
        )
    required = {
        "opendeepclustering/__init__.py",
        "opendeepclustering/estimators/autoencoder_kmeans.py",
        "opendeepclustering/estimators/dec.py",
        "opendeepclustering/estimators/deepcluster.py",
        "opendeepclustering/estimators/idec.py",
        "opendeepclustering/estimators/vade.py",
    }
    missing = required.difference(names)
    if missing:
        raise RuntimeError(f"Missing wheel content in {path}: {sorted(missing)}")


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = [name for name in archive.getnames() if archive.getmember(name).isfile()]
    stripped = [Path(*Path(name).parts[1:]) for name in names]
    roots = {item.parts[0] for item in stripped if item.parts}
    unexpected = roots.difference(SDIST_ALLOWED_ROOTS)
    forbidden = [
        str(item) for item in stripped if item.parts and item.parts[0] in FORBIDDEN_PARTS
    ]
    if unexpected or forbidden:
        raise RuntimeError(
            f"Unexpected sdist content in {path}: {sorted(unexpected)} {sorted(forbidden)}"
        )
    required = {
        Path("CITATION.cff"),
        Path("CONTRIBUTING.md"),
        Path("LICENSE"),
        Path("README.md"),
        Path("mkdocs.yml"),
        Path("pyproject.toml"),
    }
    missing = required.difference(stripped)
    if missing:
        raise RuntimeError(f"Missing sdist content in {path}: {sorted(map(str, missing))}")


def main(directory: str = "dist") -> int:
    dist = Path(directory)
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("Expected exactly one wheel and one source distribution.")
    _check_wheel(wheels[0])
    _check_sdist(sdists[0])
    print(f"Validated {wheels[0].name} and {sdists[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "dist"))
