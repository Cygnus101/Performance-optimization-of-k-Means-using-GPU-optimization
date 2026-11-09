"""V0 K-Means with PyTorch's direct Euclidean distance computation."""

from math import isfinite
from time import perf_counter

import numpy as np
import torch

if __package__:
    from .vectorized_implementation import load_fashion_mnist
else:
    from vectorized_implementation import load_fashion_mnist


@torch.no_grad()
def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4):
    """Fit on X's device and return labels, centroids, iterations, convergence, and inertia."""
    if X.ndim != 2 or X.numel() == 0:
        raise ValueError("X must be a nonempty two-dimensional tensor.")
    if X.dtype not in (torch.float32, torch.float64):
        raise ValueError("X must use float32 or float64.")
    if not torch.isfinite(X).all().item():
        raise ValueError("X must contain only finite values.")
    if not 1 <= num_clusters <= X.shape[0]:
        raise ValueError("num_clusters must be between 1 and the number of samples.")
    if num_iterations < 1:
        raise ValueError("num_iterations must be positive.")
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative.")

    indices = np.random.RandomState(42).choice(X.shape[0], num_clusters, replace=False)
    centroids = X[torch.as_tensor(indices, device=X.device)]
    sample_counts = X.new_ones(X.shape[0])
    converged = False

    for iteration in range(1, num_iterations + 1):
        labels = torch.cdist(
            X, centroids, p=2, compute_mode="donot_use_mm_for_euclid_dist"
        ).argmin(dim=1)

        sums = torch.zeros_like(centroids).index_add_(0, labels, X)
        counts = X.new_zeros(num_clusters).index_add_(0, labels, sample_counts)
        means = sums / counts.clamp_min(1).unsqueeze(1)
        new_centroids = torch.where(counts.unsqueeze(1) > 0, means, centroids)

        # Reading the scalar on the host makes the stopping decision each iteration.
        shift = torch.linalg.vector_norm(new_centroids - centroids).item()
        centroids = new_centroids
        if shift < threshold:
            converged = True
            break

    labels = torch.cdist(
        X, centroids, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    ).argmin(dim=1)
    inertia = (X - centroids[labels]).square().sum(dtype=torch.float64).item()
    return labels, centroids, iteration, converged, inertia


def main():
    if not torch.cuda.is_available():
        raise SystemExit("V0 requires a CUDA-enabled PyTorch install and an NVIDIA GPU.")

    device = torch.device("cuda", torch.cuda.current_device())
    X, _ = load_fashion_mnist()
    samples = torch.as_tensor(X, device=device)

    # Warm up the same operations on a small slice before the full run.
    kmeans(samples[:1024], num_iterations=2)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = perf_counter()
    labels, centroids, iterations, converged, inertia = kmeans(
        samples, num_clusters=10, num_iterations=1000, threshold=1e-4
    )
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("PyTorch V0 K-means completed.", labels.shape, centroids.shape)
    print(f"Clustering time: {elapsed:.3f} seconds")
    print(f"Peak PyTorch CUDA tensor memory: {peak_mib:.2f} MiB")
    print(f"Iterations: {iterations}; converged: {converged}")
    print(f"Final inertia: {inertia:.6f}")


if __name__ == "__main__":
    main()
