"""V2.6: V2.3 warp assignment with V2.5 sharded centroid accumulation.

Keep row-major images so neighbouring warp lanes read consecutive features.
The existing runner includes shard allocation, clearing and merging in fit time.
"""
from .common import run_kmeans
from .v2_3_warp import assign
from .v2_5_reduction import clear, accumulate, merge


def kmeans(X, num_clusters=10, num_iterations=100, threshold=1e-4, *,
           initial_centroids=None, early_stopping=True, threads_per_block=256, shards=32):
    return run_kmeans(
        X, num_clusters, num_iterations, threshold,
        initial_centroids=initial_centroids, early_stopping=early_stopping,
        threads=threads_per_block, assign=assign, warp=True,
        reduction=(clear, accumulate, merge), shards=shards,
    )
