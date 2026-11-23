from implementations.profiling import stage

import numpy as np
from numba import cuda, float32, float64

from .common import (
    THREADS,
    blocks,
    initialize_centroids,
    load_fashion_mnist,
)


@cuda.jit
def assign(X, C, labels):
    i = cuda.grid(1)    #index of the image

    if i < X.shape[0]:      #blocks excecss threads
        best = float32(3.4028235e38)
        cluster = 0
        for j in range(C.shape[0]): #index of cluster
            dist = float32(0.0)
            for f in range(X.shape[1]):
                diff = X[i, f] - C[j, f]
                dist += diff * diff
            if dist < best:
                best = dist
                cluster = j
        labels[i] = cluster


@cuda.jit
def reset(sums, counts, shift):
    i = cuda.grid(1)

    if i < sums.size:
        k = i // sums.shape[1]
        f = i % sums.shape[1]

        sums[k, f] = 0.0

        if f == 0:
            counts[k] = 0

    if i == 0:
        shift[0] = 0.0


@cuda.jit
def accumulate(X, labels, sums, counts):
    i = cuda.grid(1)

    if i < X.size:
        sample = i // X.shape[1]
        feature = i % X.shape[1]
        cluster = labels[sample]

        cuda.atomic.add(
            sums, (cluster, feature), X[sample, feature]
        )

        if feature == 0:
            cuda.atomic.add(counts, cluster, 1)


@cuda.jit
def update(sums, counts, C, shift, track_shift):
    i = cuda.grid(1)

    if i < C.size:
        k = i // C.shape[1]
        f = i % C.shape[1]

        if counts[k] > 0:
            value = sums[k, f] / counts[k]

            if track_shift:
                diff = value - C[k, f]
                cuda.atomic.add(shift, 0, diff * diff)

            C[k, f] = value


@cuda.jit
def calculate_inertia(X, C, labels, inertia):
    i = cuda.grid(1)

    if i < X.shape[0]:
        k = labels[i]
        dist = float32(0.0)

        for f in range(X.shape[1]):
            diff = X[i, f] - C[k, f]
            dist += diff * diff

        cuda.atomic.add(inertia, 0, float64(dist))


def kmeans(
    X,
    num_clusters=10,
    num_iterations=100,
    threshold=1e-4,
    *,
    initial_centroids=None,
    early_stopping=True,
):
    with stage("initial_work", device=False):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        k = num_clusters

        if initial_centroids is None:
            initial_centroids = initialize_centroids(X, k)

        initial_centroids = np.ascontiguousarray(
            initial_centroids, dtype=np.float32
        )

    with stage("h2d", device=True):
        d_X = cuda.to_device(X)
        d_C = cuda.to_device(initial_centroids)

    with stage("allocation", device=False):
        d_labels = cuda.device_array(n, dtype=np.int32)
        d_sums = cuda.device_array((k, d), dtype=np.float32)
        d_counts = cuda.device_array(k, dtype=np.int32)
        d_shift = cuda.device_array(1, dtype=np.float32)

    with stage("h2d", device=True):
        d_inertia = cuda.to_device(
            np.zeros(1, dtype=np.float64)
        )

    converged = False

    for iteration in range(1, num_iterations + 1):
        with stage("centroid_reset", device=True, step=iteration):
            reset[blocks(k * d), THREADS](
                d_sums, d_counts, d_shift
            )

        with stage("assignment", device=True, step=iteration):
            assign[blocks(n), THREADS](
                d_X, d_C, d_labels
            )

        with stage("centroid_accumulation", device=True, step=iteration):
            accumulate[blocks(n * d), THREADS](
                d_X, d_labels, d_sums, d_counts
            )

        with stage("centroid_update", device=True, step=iteration):
            update[blocks(k * d), THREADS](
                d_sums, d_counts, d_C,
                d_shift, early_stopping
            )

        if early_stopping:
            shift = float(d_shift.copy_to_host()[0])

            if np.sqrt(shift) < threshold:
                converged = True
                break

    with stage("final_assignment_inertia", device=True):
        assign[blocks(n), THREADS](
            d_X, d_C, d_labels
        )

        calculate_inertia[blocks(n), THREADS](
            d_X, d_C, d_labels, d_inertia
        )

    with stage("synchronization", device=False):
        cuda.synchronize()

    with stage("d2h", device=True):
        return (
            d_labels.copy_to_host(),
            d_C.copy_to_host(),
            iteration,
            converged,
            float(d_inertia.copy_to_host()[0]),
        )
