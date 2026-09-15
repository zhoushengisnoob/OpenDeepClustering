# Release procedure

This checklist separates repository preparation from publication. Completing a release-preparation pull request does not create a tag, a GitHub Release or a PyPI upload.

## Prepare the release pull request

- [ ] Confirm every roadmap dependency is closed or explicitly deferred.
- [ ] Update the version in `pyproject.toml` and `CITATION.cff`.
- [ ] Move user-facing changes from `Unreleased` into a dated version section in `CHANGELOG.md`.
- [ ] Run `python tools/check_release.py` and `cffconvert --validate`.
- [ ] Run `python -m pytest -q` and `mkdocs build --strict`.
- [ ] Build into an empty output directory with `python -m build --outdir <directory>`.
- [ ] Run `python -m twine check <directory>/*`, `check-wheel-contents <directory>/*.whl` and `python tools/check_dist.py <directory>`.
- [ ] Install the wheel in a fresh environment and verify `opendeepclustering.__version__`, all public estimators and `odc --help`.
- [ ] Review benchmark records and confirm that smoke results are not presented as paper-reproduction evidence.
- [ ] Merge only after every required CI check passes.

## Publish after maintainer approval

- [ ] Pull the reviewed merge commit on `master` and repeat the metadata and package checks.
- [ ] Create an annotated `v<version>` tag on that exact commit and push the tag.
- [ ] Create a GitHub Release from the tag using the matching changelog section.
- [ ] If PyPI publication is intended, upload the already-validated artifacts through the configured trusted publisher and verify the installed public release in a fresh environment.
- [ ] Record the release URL and any publication exception in the roadmap issue.

Tags and external publication are irreversible public actions relative to ordinary repository changes, so they remain explicit maintainer decisions.
