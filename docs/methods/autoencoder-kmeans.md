# Autoencoder + KMeans specification

## Role in the taxonomy

This estimator is the minimal multi-stage representative. The survey defines multi-stage deep clustering as separately optimizing a representation learner and then passing its frozen embeddings to a shallow clusterer. It explicitly identifies autoencoder followed by KMeans as the straightforward early pattern.

Primary taxonomy source: [Zhou et al., *A Comprehensive Survey on Deep Clustering*, Section 5.1](https://arxiv.org/abs/2206.07579).

## Method contract

1. Train a symmetric multilayer autoencoder using mean-squared reconstruction loss.
2. Freeze the learned representation for the clustering stage.
3. Clone and fit the supplied scikit-learn clusterer on the embeddings. When no clusterer is supplied, use KMeans with the estimator's `n_clusters`, `n_init`, `kmeans_max_iter` and `kmeans_tol` settings.
4. Expose the encoder output through `transform`. Out-of-sample `predict` is available when the fitted shallow clusterer implements `predict`.

The required `cluster_centers_` attribute is always computed as the empirical mean embedding for each fitted label, so the public contract remains stable even when the supplied clusterer has no centroid attribute.

## Evidence boundary

This is a taxonomy reference composition, not a reproduction of one named paper. The stage separation and arbitrary shallow-estimator composition are the behavior under test. The compact MLP, Adam optimizer and default dimensions are modern practical defaults chosen for a small, dependency-light API.
