"""Configurable launch size; otherwise the unchanged V2.0 kernels."""
from .common import run_kmeans
from .common import load_baseline

assign = load_baseline().assign


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4, *,
           initial_centroids=None, early_stopping=True, threads_per_block=256):
    return run_kmeans(
        X, num_clusters, num_iterations, threshold,
        initial_centroids=initial_centroids, early_stopping=early_stopping,
        threads=threads_per_block, assign=assign,
    )
