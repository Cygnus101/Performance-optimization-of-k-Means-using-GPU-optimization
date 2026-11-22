"""Fixed-work benchmark: python -m benchmarks.sweep_k."""
import argparse
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from implementations import vectorized_implementation as numpy_impl
from implementations import scikit_implementation as sklearn_impl

K_VALUES = (2, 4, 8, 16, 32, 64, 128)
N, D, ITERATIONS, RUNS = 100_000, 32, 20, 5


def validate_result(X, result, k, iterations, reference_inertia=None):
    """Independent float64 checks in batches; outside the timing region."""
    labels, centers, actual_iterations, _, inertia = result
    assert actual_iterations == iterations, (actual_iterations, iterations)
    assert centers.shape == (k, X.shape[1]) and centers.dtype == np.float32
    assert labels.shape == (len(X),) and np.issubdtype(labels.dtype, np.integer)
    assert np.isfinite(centers).all() and np.isfinite(inertia)
    assert np.all((labels >= 0) & (labels < k))
    objective = 0.0
    batch_size = max(1, min(1024, (16 * 1024**2) // (k * X.shape[1] * 8)))
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size].astype(np.float64)
        distances = ((batch[:, None] - centers.astype(np.float64)) ** 2).sum(axis=2)
        selected = distances[np.arange(len(batch)), labels[start:start + len(batch)]]
        np.testing.assert_allclose(selected, distances.min(axis=1), rtol=2e-5, atol=2e-5)
        objective += selected.sum()
    np.testing.assert_allclose(inertia, objective, rtol=2e-5, atol=2e-5)
    if reference_inertia is not None:
        # Near ties can change subsequent Lloyd trajectories.
        np.testing.assert_allclose(objective, reference_inertia, rtol=5e-3, atol=1e-4)
    return float(objective)


def as_result(model):
    return model.labels_, model.cluster_centers_, model.n_iter_, False, model.inertia_


def load_benchmark_data(dataset="fashion-mnist", *, smoke=False):
    """Load/preprocess once, outside timing. Never substitute another dataset."""
    if dataset == "fashion-mnist":
        X, _ = numpy_impl.load_fashion_mnist()
        if smoke:
            X = X[:256]  # Keep all image features.
        preprocessing = "OpenML Fashion-MNIST v1; float32 pixels / 255; labels unused"
    elif dataset == "synthetic":
        n, d = (256, 4) if smoke else (N, D)
        X = np.random.default_rng(42).standard_normal((n, d), dtype=np.float32)
        preprocessing = "standard normal; np.random.default_rng(42)"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return np.ascontiguousarray(X, dtype=np.float32), preprocessing


def run_sweep(output, *, dataset="fashion-mnist", cpu_only=False, smoke=False,
              threads=1, speedup=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Use an empty output directory: {output}")
    X, preprocessing = load_benchmark_data(dataset, smoke=smoke)
    n, d = X.shape
    ks = (2, 4) if smoke else K_VALUES
    print(f"Dataset: {dataset}; N={n}, D={d}; smoke={smoke}", flush=True)
    metadata = dict(n=n, d=d, k=list(ks), iterations=ITERATIONS, runs=RUNS,
                    warmups=1, seed=42, threads=threads, smoke=smoke,
                    dataset=dataset, preprocessing=preprocessing,
                    implementations=["NumPy CPU", "scikit-learn CPU"] + ([] if cpu_only else
                        ["PyTorch GPU"]),
                    platform=platform.platform(), cpu=platform.processor(),
                    python=platform.python_version(), dtype="float32",
                    timing="fit including final assignment/inertia; excludes input transfer and validation",
                    versions={m: importlib.metadata.version(m) for m in
                              ("numpy", "scikit-learn", "matplotlib", "threadpoolctl")})
    torch = None
    if not cpu_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use --cpu-only for explicitly partial CPU results.")
        from implementations import pytorch_implementation as pytorch_impl
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda", torch.cuda.current_device())
        samples = torch.as_tensor(X, device=device)
        metadata.update(gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                        cuda=torch.version.cuda, tf32=False)
    metadata["gpu_status"] = "omitted (--cpu-only)" if cpu_only else "CUDA"
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with threadpool_limits(limits=threads):
        for k in ks:
            indices = np.random.RandomState(42).choice(n, k, replace=False)
            initial = X[indices].copy()
            kwargs = dict(num_clusters=k, num_iterations=ITERATIONS, threshold=0,
                          initial_centroids=initial, early_stopping=False)
            fits = [("NumPy CPU", lambda: numpy_impl.kmeans(X, **kwargs), False),
                    ("scikit-learn CPU", lambda: as_result(sklearn_impl.kmeans(
                        X, **kwargs, n_threads=threads)), False)]
            if torch is not None:
                gpu_kwargs = dict(kwargs, initial_centroids=torch.as_tensor(initial, device=device))
                fits.append(("PyTorch GPU", lambda: pytorch_impl.kmeans(samples, **gpu_kwargs), True))
            reference = None
            for name, fit, gpu in fits:
                fit()  # One full configuration warmup, excluded from the five runs.
                if gpu:
                    torch.cuda.synchronize(device)
                for run in range(1, RUNS + 1):
                    if gpu:
                        torch.cuda.synchronize(device)
                    start = perf_counter()
                    result = fit()
                    if gpu:
                        torch.cuda.synchronize(device)
                    seconds = perf_counter() - start
                    if gpu:
                        result = (result[0].cpu().numpy(), result[1].cpu().numpy(), *result[2:])
                    objective = validate_result(X, result, k, ITERATIONS, reference)
                    if name == "NumPy CPU" and reference is None:
                        reference = objective
                    rows.append(dict(implementation=name, k=k, run=run, seconds=seconds))
                median = np.median([r['seconds'] for r in rows[-RUNS:]])
                print(f"K={k:3} {name}: median {median:.6f}s", flush=True)
                with (output / "runtime.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
    from .plot_k import plot_results
    plot_results(output / "runtime.csv", speedup=speedup)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/k_sweep"))
    parser.add_argument("--dataset", choices=("fashion-mnist", "synthetic"),
                        default="fashion-mnist", help="Default: full Fashion-MNIST; synthetic uses N=100000, D=32")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="K=2,4 and 256 rows; Fashion-MNIST keeps all features, synthetic uses D=4")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--speedup", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run_sweep(args.output, dataset=args.dataset, cpu_only=args.cpu_only, smoke=args.smoke,
              threads=args.threads, speedup=args.speedup)


if __name__ == "__main__":
    main()
