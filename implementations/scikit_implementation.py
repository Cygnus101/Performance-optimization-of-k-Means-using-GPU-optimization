"""Scikit-learn CPU baseline with the NumPy baseline's initialization."""

from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans

if __package__:
    from .vectorized_implementation import load_fashion_mnist
else:
    from vectorized_implementation import load_fashion_mnist


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4):
    """Return a fitted KMeans model using one run of Lloyd's algorithm.

    Scikit-learn's tolerance is relative and differs from the NumPy baseline's
    absolute centroid-shift threshold, even when their values are identical.
    """

    rng = np.random.RandomState(42)
    random_indices = rng.choice(X.shape[0], num_clusters, replace=False)
    initial_centroids = X[random_indices].astype(np.result_type(X.dtype, np.float32))

    model = KMeans(
        n_clusters=num_clusters,
        init=initial_centroids,
        n_init=1,
        max_iter=num_iterations,
        tol=threshold,
        random_state=42,
        algorithm="lloyd",
    )
    return model.fit(X)


if __name__ == "__main__":
    X, y = load_fashion_mnist()
    start = perf_counter()
    model = kmeans(X, num_clusters=10, num_iterations=1000, threshold=1e-4)
    elapsed = perf_counter() - start

    print("Scikit-learn K-means completed.", model.labels_.shape, model.cluster_centers_.shape)
    print(f"Clustering time: {elapsed:.3f} seconds")
    print(f"Iterations: {model.n_iter_}; iteration limit reached: {model.n_iter_ == model.max_iter}")
    print(f"Final inertia: {model.inertia_:.6f}")
