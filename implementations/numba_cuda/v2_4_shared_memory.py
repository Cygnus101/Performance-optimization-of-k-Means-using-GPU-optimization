"""One image per thread; stage 4 x 128 centroid tiles in shared memory."""
from .common import run_kmeans
from numba import cuda, float32

CENTROID_TILE = 4
FEATURE_TILE = 128


@cuda.jit
def assign(X, C, labels):
    tile = cuda.shared.array((CENTROID_TILE, FEATURE_TILE), float32)
    distances = cuda.local.array(CENTROID_TILE, float32)
    i = cuda.grid(1)
    tid = cuda.threadIdx.x
    best = float32(3.4028235e38)
    cluster = 0
    for first in range(0, C.shape[0], CENTROID_TILE):
        for j in range(CENTROID_TILE):
            distances[j] = float32(0.0)
        for start in range(0, X.shape[1], FEATURE_TILE):
            for t in range(tid, CENTROID_TILE * FEATURE_TILE, cuda.blockDim.x):
                j = t // FEATURE_TILE
                f = t % FEATURE_TILE
                value = float32(0.0)
                if first + j < C.shape[0] and start + f < X.shape[1]:
                    value = C[first + j, start + f]
                tile[j, f] = value
            cuda.syncthreads()
            if i < X.shape[0]:
                for j in range(CENTROID_TILE):
                    for f in range(min(FEATURE_TILE, X.shape[1] - start)):
                        diff = X[i, start + f] - tile[j, f]
                        distances[j] += diff * diff
            # Inactive image threads also reach both barriers.
            cuda.syncthreads()
        if i < X.shape[0]:
            for j in range(CENTROID_TILE):
                if first + j < C.shape[0] and distances[j] < best:
                    best = distances[j]
                    cluster = first + j
    if i < X.shape[0]:
        labels[i] = cluster


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4, *,
           initial_centroids=None, early_stopping=True, threads_per_block=256):
    return run_kmeans(
        X, num_clusters, num_iterations, threshold,
        initial_centroids=initial_centroids, early_stopping=early_stopping,
        threads=threads_per_block, assign=assign,
    )
