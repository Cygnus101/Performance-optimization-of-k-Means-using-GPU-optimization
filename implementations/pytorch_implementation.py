"""PyTorch library baseline: GEMM assignment and tensor centroid updates."""
if __package__:
    from .profiling import stage
else:
    from profiling import stage

from math import isfinite
from time import perf_counter

import numpy as np
import torch

if __package__:
    from .vectorized_implementation import load_fashion_mnist
else:
    from vectorized_implementation import load_fashion_mnist


def _assign_labels(X, centroids):
    # The sample norm does not affect the nearest-centroid decision.
    scores = torch.addmm(centroids.square().sum(dim=1), X, centroids.T, alpha=-2)
    return scores.argmin(dim=1)


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4, *,
           initial_centroids=None, early_stopping=True):
    """Fit on X's device and return labels, centroids, iterations, convergence, and inertia."""
    return _fit_kmeans(X, num_clusters, num_iterations, threshold, _assign_labels,
                       initial_centroids=initial_centroids,
                       early_stopping=early_stopping)


@torch.no_grad()
def _fit_kmeans(X, num_clusters, num_iterations, threshold, assign_labels, *,
                initial_centroids=None, early_stopping=True):
    with stage("initial_work", device=True):
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

        if initial_centroids is None:
            indices = np.random.RandomState(42).choice(X.shape[0], num_clusters, replace=False)
            initial_centroids = X[torch.as_tensor(indices, device=X.device)]
        centroids = torch.as_tensor(initial_centroids, dtype=X.dtype, device=X.device).clone()
        if centroids.shape != (num_clusters, X.shape[1]) or not torch.isfinite(centroids).all().item():
            raise ValueError("initial_centroids must be finite with shape (K, D).")
        sample_counts = X.new_ones(X.shape[0])
    converged = False

    for iteration in range(1, num_iterations + 1):
        with stage("assignment", device=True, step=iteration):
            labels = assign_labels(X, centroids)

        with stage("centroid_update", device=True, step=iteration):
            sums = torch.zeros_like(centroids).index_add_(0, labels, X)
            counts = X.new_zeros(num_clusters).index_add_(0, labels, sample_counts)
            means = sums / counts.clamp_min(1).unsqueeze(1)
            new_centroids = torch.where(counts.unsqueeze(1) > 0, means, centroids)

        # Avoid a host synchronization per iteration in fixed-work benchmarks.
        shift = (torch.linalg.vector_norm(new_centroids - centroids).item()
                 if early_stopping else None)
        centroids = new_centroids
        if early_stopping and shift < threshold:
            converged = True
            break

    with stage("final_assignment_inertia", device=True):
        labels = assign_labels(X, centroids)
        inertia = (X - centroids[labels]).square().sum(dtype=torch.float64).item()
    return labels, centroids, iteration, converged, inertia


def main():
    _run_cuda_benchmark(kmeans, "baseline")


def _run_cuda_benchmark(fit, version):
    if not torch.cuda.is_available():
        raise SystemExit(f"{version} requires a CUDA-enabled PyTorch install and an NVIDIA GPU.")

    device = torch.device("cuda", torch.cuda.current_device())
    X, _ = load_fashion_mnist()
    samples = torch.as_tensor(X, device=device)

    # Warm up the same operations on a small slice before the full run.
    fit(samples[:1024], num_iterations=2)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = perf_counter()
    labels, centroids, iterations, converged, inertia = fit(
        samples, num_clusters=10, num_iterations=1000, threshold=1e-4
    )
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch {version} K-means completed.", labels.shape, centroids.shape)
    print(f"Clustering time: {elapsed:.3f} seconds")
    print(f"Peak PyTorch CUDA tensor memory: {peak_mib:.2f} MiB")
    print(f"Iterations: {iterations}; converged: {converged}")
    print(f"Final inertia: {inertia:.6f}")


if __name__ == "__main__":
    main()
