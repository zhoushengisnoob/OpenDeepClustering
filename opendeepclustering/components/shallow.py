"""Shallow clustering factories shared by deep estimators."""

from sklearn.cluster import KMeans


def make_kmeans(*, n_clusters, n_init, max_iter, tol, random_state):
    """Construct the consistently configured centroid initializer."""
    return KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
