"""Unified complete-fit CPU, PyTorch and Numba CUDA benchmark."""
import argparse
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
from implementations import vectorized_implementation as numpy_impl
from implementations import scikit_implementation as sklearn_impl
from .validation import cpu_reference, validate_result

K_VALUES = (2, 4, 8, 16, 32, 64)
N, D, ITERATIONS, RUNS = 100_000, 32, 20, 5
SUITES = ("all", "original", "numba", "gpu", "cpu")
TIMING = (
    "Complete fit, CPU inputs to CPU results: host fit preparation/checks, device "
    "allocation, input/initial-centroid H2D, layout preparation (including V2.2 "
    "transpose/extra transfer), 20 Lloyd updates, final assignment/inertia, and "
    "result D2H. perf_counter with GPU synchronization before and after fit. "
    "Excludes data loading/normalization, seed selection, context initialization, "
    "JIT warmup, independent CPU reference, result validation, CSV and plotting. "
    "Warm allocator caches may persist; no cold-start or kernel-only claim."
)
ARTIFACTS = ("runtime.csv", "medians.csv", "environment.json", "runtime_vs_k.png", "speedup_vs_k.png")


def as_result(model):
    return model.labels_, model.cluster_centers_, model.n_iter_, False, model.inertia_


def load_benchmark_data(dataset="fashion-mnist", *, smoke=False):
    """Load/preprocess once, outside timing. Never substitute another dataset."""
    if dataset == "fashion-mnist":
        X, _ = numpy_impl.load_fashion_mnist()
        if not smoke and X.shape != (70_000, 784):
            raise ValueError(f"Expected full Fashion-MNIST (70000, 784), got {X.shape}")
        if smoke:
            X = X[:256]  # Keep all image features.
        preprocessing = "OpenML Fashion-MNIST v1; float32 pixels / 255; labels unused"
    elif dataset == "synthetic":
        n, d = (256, 4) if smoke else (N, D)
        X = np.random.default_rng(42).standard_normal((n, d), dtype=np.float32)
        preprocessing = "standard normal; np.random.default_rng(42)"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("Dataset must be a finite nonempty matrix")
    return X, preprocessing


def torch_complete_fit(X, initial, *, torch, fit, device, **kwargs):
    """Called INSIDE timing: fresh input transfers and CPU result copies every fit."""
    samples = torch.as_tensor(X, device=device)
    centers = torch.as_tensor(initial, device=device)
    result = fit(samples, initial_centroids=centers, **kwargs)
    return (result[0].cpu().numpy(), result[1].cpu().numpy(), *result[2:])


def prepare_backends(suite, thread_sizes):
    """Initialize selected contexts outside timing; never upload benchmark inputs here."""
    backends = {}
    info = {"gpu_status": "omitted (CPU suite)"}
    if suite in ("all", "original", "gpu"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA required; use --suite cpu for explicitly partial CPU results.")
        from implementations import pytorch_implementation
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.synchronize(device)
        backends["torch"] = (torch, pytorch_implementation.kmeans, device)
        info.update(gpu_status="CUDA", gpu=torch.cuda.get_device_name(device),
                    torch=torch.__version__, torch_cuda=torch.version.cuda, tf32=False)
    if suite in ("all", "numba", "gpu"):
        from numba import cuda
        from implementations.numba_cuda.common import require_cuda
        device = require_cuda()  # Explicitly rejects CUDA simulator.
        cuda.select_device(0)  # Match PyTorch's device for cross-library comparisons.
        device = require_cuda()
        if (not thread_sizes or len(set(thread_sizes)) != len(thread_sizes)
                or any(t < 32 or t > min(1024, device.MAX_THREADS_PER_BLOCK) or t % 32
                       for t in thread_sizes)):
            raise ValueError("Thread sizes must be distinct multiples of 32 within the device limit")
        from .numba_configs import configurations
        backends["numba"] = (cuda, configurations(thread_sizes))
        info.update(gpu_status="CUDA", numba_gpu=str(device.name),
                    compute_capability=device.compute_capability,
                    numba_cuda_runtime=cuda.runtime.get_version())
    return backends, info


def build_fits(X, initial, kwargs, suite, threads, backends):
    """Each zero-argument fit starts with the same host arrays and returns host results."""
    fits = []
    if suite in ("all", "original", "cpu"):
        fits.extend([
            ("NumPy CPU", lambda: numpy_impl.kmeans(X, initial_centroids=initial, **kwargs), None),
            ("scikit-learn CPU", lambda: as_result(sklearn_impl.kmeans(
                X, initial_centroids=initial, n_threads=threads, **kwargs)), None),
        ])
    if "torch" in backends:
        torch, fit, device = backends["torch"]
        fits.append(("PyTorch GPU", lambda fit=fit: torch_complete_fit(
            X, initial, torch=torch, fit=fit, device=device, **kwargs),
                     lambda: torch.cuda.synchronize(device)))
    if "numba" in backends:
        cuda, configs = backends["numba"]
        for name, fit, options in configs:
            # Bind loop values now; every variant must keep its own callable/options.
            fits.append((name, lambda fit=fit, options=options: fit(
                X, initial_centroids=initial, **kwargs, **options), cuda.synchronize))
    return fits


def measure_fit(fit, synchronize=None):
    if synchronize is not None:
        synchronize()
    start = perf_counter()
    result = fit()
    if synchronize is not None:
        synchronize()
    return result, perf_counter() - start


def run_sweep(output, *, dataset="fashion-mnist", suite="all", cpu_only=False, smoke=False,
              threads=1, thread_sizes=(128, 256, 512), k_values=None,
              speedup=False, no_progress=False):
    if suite not in SUITES:
        raise ValueError(f"Unknown suite: {suite}")
    if cpu_only:
        if suite not in ("all", "cpu"):
            raise ValueError("--cpu-only conflicts with the selected suite")
        suite = "cpu"
    if threads < 1:
        raise ValueError("threads must be positive")
    ks = tuple(k_values) if k_values is not None else ((2, 4) if smoke else K_VALUES)
    if (not ks or any(not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1
                      for k in ks) or len(set(ks)) != len(ks)):
        raise ValueError("K values must be a nonempty list of distinct positive integers")
    ks = tuple(int(k) for k in ks)
    backends, device_info = prepare_backends(suite, thread_sizes)
    tqdm.write(f"Loading dataset: {dataset}...")
    X, preprocessing = load_benchmark_data(dataset, smoke=smoke)
    n, d = X.shape
    if n < max(ks):
        raise ValueError("Dataset has fewer rows than the requested K")
    kwargs = dict(num_clusters=ks[0], num_iterations=ITERATIONS, threshold=0,
                  early_stopping=False)
    names = [name for name, _, _ in build_fits(X, X[:ks[0]], kwargs, suite, threads, backends)]
    baseline = "NumPy CPU" if "NumPy CPU" in names else "V2.0 baseline"
    metadata = dict(n=n, d=d, k=list(ks), iterations=ITERATIONS, runs=RUNS,
                    warmups=1, seed=42, threads=threads, smoke=smoke, suite=suite,
                    dataset=dataset, preprocessing=preprocessing, implementations=names,
                    platform=platform.platform(), cpu=platform.processor(),
                    python=platform.python_version(), dtype="float32", timing=TIMING,
                    timing_protocol="complete_fit_v1", speedup_baseline=baseline,
                    reference="independent float64 CPU Lloyd, 20 iterations",
                    validation="nearest/inertia rtol=2e-5 atol=2e-5; reference objective rtol=0.005 atol=0.0001",
                    versions={m: importlib.metadata.version(m) for m in
                              ("numpy", "scikit-learn", "matplotlib", "threadpoolctl", "tqdm")},
                    status="running", **device_info)
    if "numba" in backends:
        metadata.update(numba_thread_sizes=list(thread_sizes), numba_default_threads=256, shards=32)
        metadata["versions"]["numba"] = importlib.metadata.version("numba")
        try:
            metadata["versions"]["numba-cuda"] = importlib.metadata.version("numba-cuda")
        except importlib.metadata.PackageNotFoundError:
            pass
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Clear only our previous artifacts once dataset/device setup succeeds.
    for filename in ARTIFACTS:
        (output / filename).unlink(missing_ok=True)
    environment = output / "environment.json"
    environment.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    try:
        with threadpool_limits(limits=threads), tqdm(
                total=len(ks) * len(names), desc=f"{dataset}: configurations", unit="config",
                dynamic_ncols=True, disable=no_progress) as configurations:
            for index, k in enumerate(ks):
                indices = np.random.RandomState(42).choice(n, k, replace=False)
                initial = X[indices].copy()
                tqdm.write(f"K={k}: computing independent float64 CPU reference")
                reference = cpu_reference(X, initial, ITERATIONS)[-1]
                kwargs["num_clusters"] = k
                fits = build_fits(X, initial, kwargs, suite, threads, backends)
                offset = index % len(fits)
                for name, fit, synchronize in fits[offset:] + fits[:offset]:
                    configurations.set_postfix_str(f"K={k} | {name}")
                    with tqdm(total=RUNS + 1, desc=f"K={k} | {name}", unit="fit", position=1,
                              leave=False, dynamic_ncols=True, disable=no_progress) as runs:
                        runs.set_postfix_str("warmup")
                        warmup = fit()  # Full work and JIT compilation, excluded from measurements.
                        if synchronize is not None:
                            synchronize()
                        validate_result(X, warmup, k, ITERATIONS, reference)
                        runs.update(1)
                        for run in range(1, RUNS + 1):
                            runs.set_postfix_str(f"run {run}/{RUNS}")
                            result, seconds = measure_fit(fit, synchronize)
                            runs.set_postfix_str("validation")
                            objective = validate_result(X, result, k, ITERATIONS, reference)
                            rows.append(dict(implementation=name, k=k, run=run, seconds=seconds,
                                             inertia=objective, reference_inertia=reference))
                            with (output / "runtime.csv").open("w", newline="") as stream:
                                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                                writer.writeheader()
                                writer.writerows(rows)
                            runs.update(1)
                    median = np.median([r["seconds"] for r in rows[-RUNS:]])
                    tqdm.write(f"K={k:3} {name}: median {median:.6f}s")
                    configurations.update(1)
        from .plot_k import plot_results
        medians = plot_results(output / "runtime.csv", speedup=speedup, baseline=baseline)
        with (output / "medians.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["implementation", "k", "median_seconds"])
            for name, series in medians.items():
                writer.writerows((name, k, seconds) for k, seconds in sorted(series.items()))
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        environment.write_text(json.dumps(metadata, indent=2) + "\n")
        for filename in ("medians.csv", "runtime_vs_k.png", "speedup_vs_k.png"):
            (output / filename).unlink(missing_ok=True)
        raise
    metadata["status"] = "complete"
    environment.write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/k_sweep"))
    parser.add_argument("--dataset", choices=("fashion-mnist", "synthetic"), default="fashion-mnist")
    parser.add_argument("--suite", choices=SUITES, default="all",
                        help="all=CPU+PyTorch+Numba; original=CPU+PyTorch; gpu=PyTorch+Numba")
    parser.add_argument("--cpu-only", action="store_true", help="Alias for --suite cpu")
    parser.add_argument("--smoke", action="store_true", help="K=2,4; 256 rows; still 20 iterations/5 runs")
    parser.add_argument("--k", type=int, nargs="+", dest="k_values",
                        help="Distinct positive cluster counts; overrides smoke's K=2,4 default")
    parser.add_argument("--threads", type=int, default=1, help="CPU thread count")
    parser.add_argument("--threads-per-block", type=int, nargs="+", default=[128, 256, 512],
                        help="V2.1 configurations; other Numba variants use 256")
    parser.add_argument("--speedup", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.cpu_only and args.suite not in ("all", "cpu"):
        parser.error("--cpu-only conflicts with --suite")
    run_sweep(args.output, dataset=args.dataset, suite=args.suite, cpu_only=args.cpu_only,
              smoke=args.smoke, threads=args.threads, thread_sizes=tuple(args.threads_per_block),
              k_values=None if args.k_values is None else tuple(args.k_values),
              speedup=args.speedup, no_progress=args.no_progress)


if __name__ == "__main__":
    main()
