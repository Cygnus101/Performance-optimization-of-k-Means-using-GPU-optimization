from sklearn.datasets import fetch_openml
from pathlib import Path
from time import perf_counter
import numpy as np

def load_fashion_mnist():                   #data loading function

    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "datasets"

    X, y = fetch_openml(
        "Fashion-MNIST",
        version=1,
        return_X_y=True,
        as_frame=False,
        data_home=cache_dir,
        cache=True
    )

    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    return X, y 

def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4):
    """Return labels, centroids, iteration count, convergence status, and inertia."""
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional array.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")
    if not 1 <= num_clusters <= X.shape[0]:
        raise ValueError("num_clusters must be between 1 and the number of samples.")
    if num_iterations < 1:
        raise ValueError("num_iterations must be positive.")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative.")

    np.random.seed(42)
    random_indices = np.random.choice(X.shape[0], num_clusters, replace=False)
    centroids = X[random_indices].astype(np.result_type(X.dtype, np.float32))
    converged = False
    for iteration in range(num_iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for k in range(num_clusters):
            cluster_points = X[labels == k]
            if cluster_points.shape[0] > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
        
        if np.linalg.norm(new_centroids - centroids) < threshold:
            centroids = new_centroids
            converged = True
            break
        centroids = new_centroids

    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    residuals = X - centroids[labels]
    inertia = float(np.sum(residuals * residuals, dtype=np.float64))

    return labels, centroids, iteration + 1, converged, inertia


if __name__ == "__main__":
    X, y = load_fashion_mnist()
    start = perf_counter()
    kmeans_labels, kmeans_centroids, n_iter, converged, inertia = kmeans(
        X, num_clusters=10, num_iterations=1000, threshold=1e-4
    )
    elapsed = perf_counter() - start

    print("K-means clustering completed.", kmeans_labels.shape, kmeans_centroids.shape)
    print(f"Clustering time: {elapsed:.3f} seconds")
    print(f"Iterations: {n_iter}; converged: {converged}")
    print(f"Final inertia: {inertia:.6f}")
