"""User-facing API for OpenDeepClustering."""

from importlib.metadata import version as _distribution_version

from opendeepclustering.estimators import (
    AutoencoderKMeans,
    DEC,
    DeepCluster,
    IDEC,
    VaDE,
)
from opendeepclustering.metrics import ClusteringScores, evaluate_clustering

__version__ = _distribution_version("opendeepclustering")

__all__ = [
    "__version__",
    "AutoencoderKMeans",
    "ClusteringScores",
    "DEC",
    "DeepCluster",
    "IDEC",
    "VaDE",
    "evaluate_clustering",
]
