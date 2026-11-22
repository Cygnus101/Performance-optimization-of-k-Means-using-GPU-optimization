"""Numba-only fixed-work Fashion-MNIST sweep: python -m benchmarks.sweep_numba."""
import argparse
import csv
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from implementations.numba_cuda.common import (
    initialize_centroids, load_baseline, load_fashion_mnist, require_cuda,
)

K_VALUES = (2, 4, 8, 16, 32, 64)
ITERATIONS, RUNS = 20, 5
TIMING = (
    "Synchronized perf_counter wall time for complete kmeans call: host float32 "
    "preparation/validation, device allocation, H2D (including transposed copy in "
    "V2.2), 20 Lloyd updates, final assignment/inertia, synchronization and D2H "
    "results. Excludes dataset loading, centroid selection, JIT warmup, CPU "
    "reference, correctness checks and plotting. No early stopping."
)


def squared_distances(batch, centers):
    """Float64 reference distances, bounded temporary size and no N x K x D array."""
    distances = (np.sum(batch * batch, axis=1)[:, None]
                 + np.sum(centers * centers, axis=1)[None, :]
                 - 2.0 * (batch @ centers.T))
    return np.maximum(distances, 0.0)


def cpu_reference(X, initial, iterations=ITERATIONS):
    """Independent float64 Lloyd loop; lowest-index ties, retain empty centers."""
    X64 = X.astype(np.float64)
    centers = initial.astype(np.float64).copy()
    labels = np.empty(len(X), dtype=np.int32)
    for _ in range(iterations):
        for start in range(0, len(X), 1024):
            stop = start + 1024
            labels[start:stop] = squared_distances(X64[start:stop], centers).argmin(axis=1)
        for j in range(len(centers)):
            members = X64[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    objective = 0.0
    for start in range(0, len(X), 1024):
        batch = X64[start:start + 1024]
        distances = squared_distances(batch, centers)
        nearest = distances.argmin(axis=1)
        labels[start:start + len(batch)] = nearest
        # Direct residuals avoid norm/dot cancellation in the reported objective.
        objective += np.sum((batch - centers[nearest]) ** 2)
    return labels, centers, iterations, False, float(objective)


def validate_result(X, result, k, iterations=ITERATIONS, reference_inertia=None):
    """Validate final assignments and inertia outside timing; fail on divergence."""
    labels, centers, actual_iterations, converged, inertia = result
    if actual_iterations != iterations or converged:
        raise AssertionError("Expected fixed iterations without early stopping")
    if (labels.shape != (len(X),) or centers.shape != (k, X.shape[1])
            or centers.dtype != np.float32 or not np.issubdtype(labels.dtype, np.integer)):
        raise AssertionError("Incorrect output shape/dtype")
    if not np.isfinite(centers).all() or not np.isfinite(inertia):
        raise AssertionError("Nonfinite output")
    if np.any(labels < 0) or np.any(labels >= k):
        raise AssertionError("Labels outside [0, K)")
    centers64 = centers.astype(np.float64)
    objective = 0.0
    for start in range(0, len(X), 1024):
        batch = X[start:start + 1024].astype(np.float64)
        assigned = labels[start:start + len(batch)]
        selected = np.sum((batch - centers64[assigned]) ** 2, axis=1)
        minimum = squared_distances(batch, centers64).min(axis=1)
        np.testing.assert_allclose(selected, minimum, rtol=2e-5, atol=2e-5)
        objective += selected.sum()
    np.testing.assert_allclose(inertia, objective, rtol=2e-5, atol=2e-5)
    if reference_inertia is not None:
        # Different FP32 reduction orders can change near ties and later trajectories.
        np.testing.assert_allclose(objective, reference_inertia, rtol=5e-3, atol=1e-4)
    return float(objective)


def configurations(thread_sizes):
    configs = [("V2.0 baseline", load_baseline().kmeans, {})]
    tuning = importlib.import_module("implementations.numba_cuda.v2_1_block_tuning")
    for threads in thread_sizes:
        configs.append((f"V2.1 threads={threads}", tuning.kmeans,
                        {"threads_per_block": threads}))
    for module in ("v2_2_coalescing", "v2_3_warp", "v2_4_shared_memory", "v2_5_reduction"):
        fit = importlib.import_module(f"implementations.numba_cuda.{module}").kmeans
        configs.append((module, fit, {}))
    return configs


def plot_results(rows, output, speedup=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(row["implementation"] for row in rows))
    ks = sorted({row["k"] for row in rows})
    medians = {name: [float(np.median([r["seconds"] for r in rows
                                     if r["implementation"] == name and r["k"] == k]))
                      for k in ks] for name in names}
    for is_speedup in ([False, True] if speedup else [False]):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name in names:
            values = medians[name]
            if is_speedup:
                values = np.array(medians["V2.0 baseline"]) / values
            ax.plot(ks, values, "o-", label=name)
        ax.set(xlabel="K (clusters)", ylabel="Speedup vs V2.0" if is_speedup else "Median complete fit time (s)",
               title="Fashion-MNIST: Numba CUDA, 20 iterations, 5 runs")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / ("speedup_vs_k.png" if is_speedup else "runtime_vs_k.png"), dpi=160)
        plt.close(fig)
    return medians


def run_sweep(output, thread_sizes=(128, 256, 512), speedup=False):
    from numba import cuda

    device = require_cuda()  # Before loading data or writing any timing artifacts.
    if (not thread_sizes or len(set(thread_sizes)) != len(thread_sizes)
            or any(t < 32 or t > device.MAX_THREADS_PER_BLOCK or t % 32 for t in thread_sizes)):
        raise ValueError("Thread sizes must be distinct multiples of 32 within the device limit")
    print(f"NVIDIA CUDA device: {device.name}; loading full Fashion-MNIST", flush=True)
    X, _ = load_fashion_mnist()
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.shape != (70_000, 784) or not np.isfinite(X).all():
        raise ValueError(f"Expected full finite Fashion-MNIST (70000, 784), got {X.shape}")
    fits = configurations(thread_sizes)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("runtime.csv", "medians.csv", "environment.json", "runtime_vs_k.png", "speedup_vs_k.png"):
        (output / filename).unlink(missing_ok=True)
    metadata = dict(dataset="OpenML Fashion-MNIST v1", shape=list(X.shape), dtype=str(X.dtype),
                    preprocessing="existing shared loader; pixels divided by 255; labels unused",
                    k=list(K_VALUES), iterations=ITERATIONS, runs=RUNS, warmups=1, seed=42,
                    configurations=[dict(name=name, options=opts) for name, _, opts in fits],
                    default_threads_per_block=256, reduction_shards=32,
                    timing=TIMING, gpu=str(device.name), compute_capability=device.compute_capability,
                    platform=platform.platform(), python=platform.python_version(),
                    cuda_runtime=cuda.runtime.get_version(),
                    versions={name: importlib.metadata.version(name) for name in
                              ("numpy", "numba", "scikit-learn", "matplotlib")},
                    validation="float64 CPU Lloyd reference; nearest/inertia rtol=2e-5 atol=2e-5; reference objective rtol=0.005 atol=0.0001",
                    status="running")
    try:
        metadata["versions"]["numba-cuda"] = importlib.metadata.version("numba-cuda")
    except importlib.metadata.PackageNotFoundError:
        pass
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for k in K_VALUES:
        initial = initialize_centroids(X, k)
        print(f"K={k}: computing independent float64 CPU reference", flush=True)
        reference = cpu_reference(X, initial)[-1]
        kwargs = dict(num_clusters=k, num_iterations=ITERATIONS, initial_centroids=initial,
                      threshold=0.0, early_stopping=False)
        # Rotate the order across K to reduce systematic device-temperature bias.
        offset = K_VALUES.index(k) % len(fits)
        for name, fit, options in fits[offset:] + fits[:offset]:
            print(f"K={k}: {name}, full warmup", flush=True)
            warmup = fit(X, **kwargs, **options)
            cuda.synchronize()
            validate_result(X, warmup, k, reference_inertia=reference)
            for run in range(1, RUNS + 1):
                cuda.synchronize()
                start = perf_counter()
                result = fit(X, **kwargs, **options)
                cuda.synchronize()
                seconds = perf_counter() - start
                objective = validate_result(X, result, k, reference_inertia=reference)
                rows.append(dict(implementation=name, k=k, run=run, seconds=seconds,
                                 inertia=objective, reference_inertia=reference))
                with (output / "runtime.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  run {run}/{RUNS}: {seconds:.6f}s; validated", flush=True)
    medians = plot_results(rows, output, speedup)
    with (output / "medians.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["implementation", "k", "median_seconds"])
        for name, values in medians.items():
            writer.writerows((name, k, seconds) for k, seconds in zip(K_VALUES, values))
    metadata["status"] = "complete"
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/numba_sweep"))
    parser.add_argument("--threads-per-block", type=int, nargs="+", default=[128, 256, 512],
                        help="V2.1 configurations; other variants use 256")
    parser.add_argument("--speedup", action="store_true")
    args = parser.parse_args()
    run_sweep(args.output, tuple(args.threads_per_block), args.speedup)


if __name__ == "__main__":
    main()
