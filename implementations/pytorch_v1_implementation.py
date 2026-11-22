"""V1 K-Means with matrix multiplication for nearest-centroid assignment."""

import torch

if __package__:
    from .pytorch_v0_implementation import _fit_kmeans, _run_cuda_benchmark
else:
    from pytorch_v0_implementation import _fit_kmeans, _run_cuda_benchmark


def _assign_labels(X, centroids):
    centroid_norm_sq = centroids.square().sum(dim=1)
    # The sample's squared norm is the same for every centroid.
    scores = torch.addmm(centroid_norm_sq, X, centroids.T, alpha=-2)
    return scores.argmin(dim=1)


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4, *,
           initial_centroids=None, early_stopping=True):
    """Fit on X's device and return labels, centroids, iterations, convergence, and inertia."""
    return _fit_kmeans(X, num_clusters, num_iterations, threshold, _assign_labels,
                       initial_centroids=initial_centroids,
                       early_stopping=early_stopping)


def main():
    _run_cuda_benchmark(kmeans, "V1")


if __name__ == "__main__":
    main()
