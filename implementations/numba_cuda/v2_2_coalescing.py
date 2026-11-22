"""One image per thread with feature-major input for adjacent image reads."""
from .common import run_kmeans
from numba import cuda, float32


@cuda.jit
def assign(XT, C, labels):
    i = cuda.grid(1)
    if i < XT.shape[1]:
        best = float32(3.4028235e38)
        cluster = 0
        for j in range(C.shape[0]):
            dist = float32(0.0)
            for f in range(XT.shape[0]):
                diff = XT[f, i] - C[j, f]
                dist += diff * diff
            if dist < best:
                best = dist
                cluster = j
        labels[i] = cluster


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4, *,
           initial_centroids=None, early_stopping=True, threads_per_block=256):
    return run_kmeans(
        X, num_clusters, num_iterations, threshold,
        initial_centroids=initial_centroids, early_stopping=early_stopping,
        threads=threads_per_block, assign=assign, layout="transposed",
    )
