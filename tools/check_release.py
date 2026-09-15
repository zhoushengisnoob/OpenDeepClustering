"""Check that repository release metadata agrees before building artifacts."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def main() -> int:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        package_version = tomllib.load(stream)["project"]["version"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(
        r'^version:\s*["\']?([^"\'\s]+)["\']?\s*$', citation, re.MULTILINE
    )
    if citation_match is None:
        raise RuntimeError("CITATION.cff has no top-level version field")
    citation_version = citation_match.group(1)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    if not SEMVER.fullmatch(package_version):
        raise RuntimeError(f"Package version is not stable SemVer: {package_version}")
    if citation_version != package_version:
        raise RuntimeError(
            "Version mismatch: "
            f"pyproject.toml={package_version}, CITATION.cff={citation_version}"
        )
    if f"## [{package_version}]" not in changelog:
        raise RuntimeError(f"CHANGELOG.md has no section for {package_version}")

    print(f"Release metadata agrees on version {package_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
