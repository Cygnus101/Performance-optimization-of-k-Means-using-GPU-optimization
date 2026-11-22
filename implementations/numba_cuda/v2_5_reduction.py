"""Shard centroid atomics by block, then merge partial sums without atomics."""
from .common import run_kmeans
from numba import cuda, float32
from .common import load_baseline

assign = load_baseline().assign


@cuda.jit
def clear(partials, counts):
    i = cuda.grid(1)
    d = partials.shape[2]
    k = partials.shape[1]
    if i < partials.size:
        shard = i // (k * d)
        cluster = (i // d) % k
        f = i % d
        partials[shard, cluster, f] = 0.0
        if f == 0:
            counts[shard, cluster] = 0


@cuda.jit
def accumulate(X, labels, partials, counts):
    i = cuda.grid(1)
    if i < X.size:
        sample = i // X.shape[1]
        f = i % X.shape[1]
        cluster = labels[sample]
        shard = cuda.blockIdx.x % partials.shape[0]
        cuda.atomic.add(partials, (shard, cluster, f), X[sample, f])
        if f == 0:
            cuda.atomic.add(counts, (shard, cluster), 1)


@cuda.jit
def merge(partials, partial_counts, sums, counts):
    i = cuda.grid(1)
    if i < sums.size:
        cluster = i // sums.shape[1]
        f = i % sums.shape[1]
        total = float32(0.0)
        count = 0
        for shard in range(partials.shape[0]):
            total += partials[shard, cluster, f]
            if f == 0:
                count += partial_counts[shard, cluster]
        sums[cluster, f] = total
        if f == 0:
            counts[cluster] = count


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4, *,
           initial_centroids=None, early_stopping=True, threads_per_block=256, shards=32):
    return run_kmeans(
        X, num_clusters, num_iterations, threshold,
        initial_centroids=initial_centroids, early_stopping=early_stopping,
        threads=threads_per_block, assign=assign, reduction=(clear, accumulate, merge), shards=shards,
    )
