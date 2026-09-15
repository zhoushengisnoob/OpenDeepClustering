# OpenDeepClustering

OpenDeepClustering turns the taxonomy from the companion survey into inspectable, scikit-learn-style reference implementations. The current foundation exposes DEC and IDEC through one Python API and one reproducible command-line benchmark path.

The project is intentionally conservative about algorithm status. “Reference implementation” means that the objective, update schedule and initialization have been traced to the paper and exercised by correctness tests; it does not imply that every published result has been reproduced on every hardware stack.

## Start here

- Follow the [quickstart](quickstart.md) to install the package and fit an estimator.
- Check the [algorithm status](algorithms.md) before using a method in a comparison.
- Use the [benchmark protocol](benchmarking.md) when recording results.
- Consult the [Python API](api.md) for public classes and metrics.

## Project scope

The installable wheel contains the `opendeepclustering` package and its license metadata. Repository-only benchmark configurations, documentation, tests and historical experiment code remain available from the source tree but are not imported as part of the public package.
