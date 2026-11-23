"""Scikit-learn CPU baseline with the NumPy baseline's initialization."""
if __package__:
    from .profiling import stage
else:
    from profiling import stage

from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans

if __package__:
    from .vectorized_implementation import load_fashion_mnist
else:
    from vectorized_implementation import load_fashion_mnist


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4, *,
           initial_centroids=None, early_stopping=True, n_threads=1):
    """Return a fitted KMeans model using one run of Lloyd's algorithm.

    Scikit-learn's tolerance is relative and differs from the NumPy baseline's
    absolute centroid-shift threshold, even when their values are identical.
    """

    with stage("initial_work", device=False):
        if initial_centroids is None:
            indices = np.random.RandomState(42).choice(X.shape[0], num_clusters, replace=False)
            initial_centroids = X[indices]
        initial_centroids = np.array(initial_centroids, dtype=np.result_type(X.dtype, np.float32), copy=True)
        if initial_centroids.shape != (num_clusters, X.shape[1]) or not np.isfinite(initial_centroids).all():
            raise ValueError("initial_centroids must be finite with shape (K, D).")

        model = KMeans(
            n_clusters=num_clusters,
            init=initial_centroids,
            n_init=1,
            max_iter=num_iterations,
            tol=threshold,
            random_state=42,
            algorithm="lloyd",
        )
    if early_stopping:
        return model.fit(X)
    return _fit_fixed_lloyd(model, X, initial_centroids, num_iterations, n_threads)


def _fit_fixed_lloyd(model, X, initial_centroids, num_iterations, n_threads):
    """Use sklearn's compiled Lloyd steps without its unconditional label-stop.

    This private API is deliberately isolated; incompatible releases fail loudly.
    Empty clusters retain sklearn's relocation behavior.
    """
    from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense

    with stage("initial_work", device=False):
        X = np.array(X, dtype=initial_centroids.dtype, order="C", copy=True)
        if X.ndim != 2 or not X.size or not np.isfinite(X).all():
            raise ValueError("X must be a finite nonempty matrix.")
        if not 1 <= model.n_clusters <= len(X) or num_iterations < 1 or n_threads < 1:
            raise ValueError("Invalid clusters, iterations, or thread count.")
        # Match sklearn's centering to improve numerical stability.
        mean = X.mean(axis=0)
        X -= mean
        centers = np.ascontiguousarray(initial_centroids - mean)
        new_centers = np.empty_like(centers)
        weights = np.ones(len(X), dtype=X.dtype)
        cluster_weights = np.zeros(model.n_clusters, dtype=X.dtype)
        shifts = np.zeros(model.n_clusters, dtype=X.dtype)
        labels = np.full(len(X), -1, dtype=np.int32)
    for _ in range(num_iterations):
        with stage("lloyd_fused", device=False, step=_ + 1):
            lloyd_iter_chunked_dense(X, weights, centers, new_centers,
                                     cluster_weights, labels, shifts, n_threads)
        centers, new_centers = new_centers, centers
    with stage("final_assignment_inertia", device=False):
        lloyd_iter_chunked_dense(X, weights, centers, new_centers,
                                 cluster_weights, labels, shifts, n_threads,
                                 update_centers=False)
        residuals = X - centers[labels]
        model.inertia_ = float(np.sum(residuals * residuals, dtype=np.float64))
        model.cluster_centers_ = centers + mean
        model.labels_ = labels
        model.n_iter_ = num_iterations
        model.n_features_in_ = X.shape[1]
        model._n_threads = n_threads
        model._n_features_out = model.n_clusters
    return model


if __name__ == "__main__":
    X, y = load_fashion_mnist()
    start = perf_counter()
    model = kmeans(X, num_clusters=10, num_iterations=1000, threshold=1e-4)
    elapsed = perf_counter() - start

    print("Scikit-learn K-means completed.", model.labels_.shape, model.cluster_centers_.shape)
    print(f"Clustering time: {elapsed:.3f} seconds")
    print(f"Iterations: {model.n_iter_}; iteration limit reached: {model.n_iter_ == model.max_iter}")
    print(f"Final inertia: {model.inertia_:.6f}")
