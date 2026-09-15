from importlib.metadata import version

import opendeepclustering


def test_public_version_matches_installed_distribution():
    assert opendeepclustering.__version__ == version("opendeepclustering")
