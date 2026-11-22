"""One warp per image; lanes sum feature-strided squared distances."""
from .common import run_kmeans
from numba import cuda, float32


@cuda.jit
def assign(X, C, labels):
    lane = cuda.threadIdx.x % 32
    image = cuda.grid(1) // 32
    # threads/block is a multiple of 32: either all or no lanes enter.
    if image < X.shape[0]:
        best = float32(3.4028235e38)
        cluster = 0
        for j in range(C.shape[0]):
            dist = float32(0.0)
            for f in range(lane, X.shape[1], 32):
                diff = X[image, f] - C[j, f]
                dist += diff * diff
            for offset in (16, 8, 4, 2, 1):
                dist += cuda.shfl_down_sync(0xffffffff, dist, offset)
            if lane == 0 and dist < best:
                best = dist
                cluster = j
        if lane == 0:
            labels[image] = cluster


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4, *,
           initial_centroids=None, early_stopping=True, threads_per_block=256):
    return run_kmeans(
        X, num_clusters, num_iterations, threshold,
        initial_centroids=initial_centroids, early_stopping=early_stopping,
        threads=threads_per_block, assign=assign, warp=True,
    )
