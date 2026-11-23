from implementations.profiling import stage

import numpy as np

from implementations.vectorized_implementation import load_fashion_mnist

THREADS = 256


def blocks(n):
    return (n + THREADS - 1) // THREADS


def initialize_centroids(X, k):
    indices = np.random.RandomState(42).choice(
        len(X), k, replace=False
    )
    return np.ascontiguousarray(X[indices], dtype=np.float32)


def load_baseline():
    """Load the original V2.0 kernels, shared by the independent variants."""
    from . import v2_0_baseline
    return v2_0_baseline


def require_cuda():
    """Benchmarks must never silently use the simulator or an Apple GPU."""
    from numba import cuda, config

    if config.ENABLE_CUDASIM:
        raise RuntimeError("CUDA simulator enabled. Unset NUMBA_ENABLE_CUDASIM for benchmarks.")
    if not cuda.is_available():
        raise RuntimeError("NVIDIA CUDA required. In Colab select Runtime > Change runtime type > GPU.")
    device = cuda.get_current_device()
    if device.compute_capability < (6, 0):
        raise RuntimeError("Compute capability >= 6.0 required for baseline float64 atomic inertia.")
    return device


def run_kmeans(X, num_clusters, num_iterations, threshold, *, initial_centroids,
               early_stopping, threads, assign, layout="row", warp=False,
               reduction=None, shards=32):
    """Common host loop; variants replace one stage, never call one another.

    This public fit path deliberately includes device allocation, transfers,
    layout preparation and result copies. See the dedicated sweep for timing.
    """
    from numba import cuda, config

    with stage("initial_work", device=False):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a finite nonempty matrix.")
        n, d = X.shape
        if not isinstance(num_clusters, (int, np.integer)) or not 1 <= num_clusters <= n:
            raise ValueError("num_clusters must be an integer between 1 and N.")
        if not isinstance(num_iterations, (int, np.integer)) or num_iterations < 1:
            raise ValueError("num_iterations must be a positive integer.")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative.")
        if not isinstance(threads, (int, np.integer)) or not 32 <= threads <= 1024 or threads % 32:
            raise ValueError("threads_per_block must be a multiple of 32 in [32, 1024].")
        if warp and config.ENABLE_CUDASIM:
            raise RuntimeError("Warp shuffle requires a real NVIDIA GPU; CUDA simulator is unsupported.")
        if not isinstance(shards, (int, np.integer)) or shards < 1:
            raise ValueError("shards must be a positive integer.")
        if initial_centroids is None:
            initial_centroids = initialize_centroids(X, num_clusters)
        C = np.ascontiguousarray(initial_centroids, dtype=np.float32)
        if C.shape != (num_clusters, d) or not np.isfinite(C).all():
            raise ValueError("initial_centroids must be finite with shape (K, D).")
        if not config.ENABLE_CUDASIM:
            device = require_cuda()
            if threads > device.MAX_THREADS_PER_BLOCK:
                raise ValueError("threads_per_block exceeds device limit.")
        base = load_baseline()
        grid = lambda size: (size + threads - 1) // threads
    with stage("h2d", device=True):
        d_X = cuda.to_device(X)
    # Assignment alone consumes the transposed copy; accumulation uses row-major X.
    d_assign_X = d_X
    if layout == "transposed":
        with stage("layout_and_h2d", device=True):
            d_assign_X = cuda.to_device(np.ascontiguousarray(X.T))
    with stage("h2d", device=True):
        d_C = cuda.to_device(C)
    with stage("allocation", device=False):
        labels = cuda.device_array(n, dtype=np.int32)
        sums = cuda.device_array((num_clusters, d), dtype=np.float32)
        counts = cuda.device_array(num_clusters, dtype=np.int32)
        shift = cuda.device_array(1, dtype=np.float32)
    with stage("h2d", device=True):
        inertia = cuda.to_device(np.zeros(1, dtype=np.float64))
    assignment_blocks = grid(n * 32 if warp else n)
    with stage("allocation", device=False):
        if reduction is not None:
            partials = cuda.device_array((shards, num_clusters, d), dtype=np.float32)
            partial_counts = cuda.device_array((shards, num_clusters), dtype=np.int32)
            clear, accumulate, merge = reduction
    converged = False
    for iteration in range(1, num_iterations + 1):
        with stage("centroid_reset", device=True, step=iteration):
            base.reset[grid(num_clusters * d), threads](sums, counts, shift)
        with stage("assignment", device=True, step=iteration):
            assign[assignment_blocks, threads](d_assign_X, d_C, labels)
        with stage("centroid_accumulation", device=True, step=iteration):
            if reduction is None:
                base.accumulate[grid(n * d), threads](d_X, labels, sums, counts)
            else:
                clear[grid(partials.size), threads](partials, partial_counts)
                accumulate[grid(n * d), threads](d_X, labels, partials, partial_counts)
                merge[grid(num_clusters * d), threads](partials, partial_counts, sums, counts)
        with stage("centroid_update", device=True, step=iteration):
            base.update[grid(num_clusters * d), threads](sums, counts, d_C, shift, early_stopping)
        if early_stopping and np.sqrt(float(shift.copy_to_host()[0])) < threshold:
            converged = True
            break
    with stage("final_assignment_inertia", device=True):
        assign[assignment_blocks, threads](d_assign_X, d_C, labels)
        base.calculate_inertia[grid(n), threads](d_X, d_C, labels, inertia)
    with stage("synchronization", device=False):
        cuda.synchronize()
    with stage("d2h", device=True):
        return (labels.copy_to_host(), d_C.copy_to_host(), iteration,
                converged, float(inertia.copy_to_host()[0]))
