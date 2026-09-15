# Four-pattern architecture review

## Decision

The estimator foundation covers all four survey interaction patterns without changing the existing DEC or IDEC constructor, fitted attributes or training semantics. No taxonomy-specific training loop was added to `DeepClusterMixin`; the mixin remains limited to validation, discovery and public fitted-state conventions.

| Pattern | Representative | Method-owned orchestration | Shared pieces exercised |
| --- | --- | --- | --- |
| Multi-stage | `AutoencoderKMeans` | autoencoder training→frozen embedding→cloned shallow estimator | MLP autoencoder, device/random state, callbacks, validation |
| Iterative | `DeepCluster` | feature extraction↔KMeans pseudo-labels→balanced classifier updates | KMeans factory, device/random state, fit history; NCHW validation stays method-specific |
| Generative | `VaDE` | autoencoder warm-up→GMM initialization→ELBO optimization | Gaussian-mixture component, device/random state, validation |
| Simultaneous | `DEC`/`IDEC` | pretraining→centroid initialization→joint clustering objective | autoencoder, Student-t assignment, target distribution, fit state |

## Compatibility result

- All representatives implement `fit`, `fit_predict`, `transform`, `predict`, parameter cloning and the required fitted attributes.
- Optional behavior is explicit: VaDE and DEC-family estimators advertise soft assignment; VaDE additionally advertises sampling; DeepCluster advertises NCHW images.
- CLI configuration resolves every representative through the same estimator registry and emits the same provenance/metric schema.
- Existing DEC/IDEC tests and reference configurations remain unchanged.
- Adding the three estimators required one additive capability-discovery method and a shape-preserving CLI dataset option, not incompatible public API changes.

## Deliberate non-abstractions

The DeepCluster alternating loop and VaDE ELBO remain in their estimator modules. Unifying them behind a generic “trainer” would hide the exact update order and probabilistic terms that make the methods useful as references. Only stable mechanics—validation, random-state scoping, reusable network/probability components, KMeans construction, callbacks and fit state—are shared.

## Conclusion

The architecture review is complete for the initial representative set. The foundation can now accept further algorithms one at a time, provided each addition supplies a method specification, capability declaration, mathematical tests, CLI smoke configuration and provenance-aware benchmark record.
