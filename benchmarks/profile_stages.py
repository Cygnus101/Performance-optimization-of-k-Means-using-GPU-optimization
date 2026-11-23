"""Comparative profiling on actual implementations (host wall and CUDA-event views)."""
import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from implementations.profiling import Collector
from .sweep_k import (K_VALUES, ITERATIONS, build_fits, load_benchmark_data,
                      measure_fit, prepare_backends)
from .validation import cpu_reference, validate_result

SPECS = {
    "numpy": "NumPy CPU", "sklearn": "scikit-learn CPU", "pytorch": "PyTorch GPU",
    "v2.0": "V2.0 baseline",
    **{f"v2.1:{t}": f"V2.1 threads={t}" for t in (128, 256, 512)},
    **{f"v2.{i}": name for i, name in enumerate((
        "v2_2_coalescing", "v2_3_warp", "v2_4_shared_memory", "v2_5_reduction",
        "v2_6_warp_reduction"), 2)},
}
STAGES = ("cpu_overhead", "initial_work", "h2d", "layout_and_h2d", "allocation", "assignment",
          "lloyd_fused", "centroid_reset", "centroid_accumulation", "centroid_update",
          "final_assignment_inertia", "synchronization", "d2h", "unscoped_host")
ARTIFACTS = ("runtime.csv", "steps.csv", "stages.csv", "summary.csv", "environment.json",
             "report.md", "elapsed.png", "host_breakdown.png", "cuda_intervals.png")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(runtime, stages):
    groups = defaultdict(list)
    for row in runtime:
        groups[(row["implementation"], row["k"], row["mode"], "end_to_end", "wall")].append(row["seconds"])
    for row in stages:
        for metric in ("wall", "event"):
            value = row[metric+"_seconds"]
            key = (row["implementation"], row["k"], "profiled", row["stage"], metric)
            groups[key]  # Keep explicit N/A groups.
            if value is not None:
                groups[key].append(value)
    rows = []
    for (name, k, mode, stage, metric), values in groups.items():
        rows.append(dict(implementation=name, k=k, mode=mode, stage=stage, metric=metric,
                         count=len(values), median_seconds=float(np.median(values)) if values else None,
                         std_seconds=float(np.std(values, ddof=1)) if len(values)>1 else None,
                         q25_seconds=float(np.percentile(values, 25)) if values else None,
                         q75_seconds=float(np.percentile(values, 75)) if values else None,
                         min_seconds=min(values) if values else None, max_seconds=max(values) if values else None))
    return rows


def make_outputs(output, runtime, stages, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r["implementation"] for r in runtime))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab20")
    for name_index, name in enumerate(names):
        for mode, style in (("unprofiled", "-"), ("profiled", "--")):
            rows = sorted((r for r in summary if r["implementation"] == name and
                           r["mode"] == mode and r["stage"] == "end_to_end"), key=lambda r:r["k"])
            ax.errorbar([r["k"] for r in rows], [r["median_seconds"] for r in rows],
                        yerr=[r["std_seconds"] or 0 for r in rows], linestyle=style,
                        marker="o", color=colors(name_index % 20), label=f"{name} ({mode})")
    ax.set(xlabel="K", ylabel="Complete fit seconds (median ± sample SD)", yscale="log")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(output/"elapsed.png"); plt.close(fig)
    labels = list(dict.fromkeys((r["implementation"], r["k"]) for r in stages))
    for metric, filename in (("wall", "host_breakdown.png"), ("event", "cuda_intervals.png")):
        fig, ax = plt.subplots(figsize=(max(10, len(labels)*.6), 7))
        bottom = np.zeros(len(labels))
        any_values = False
        for stage_index, stage in enumerate(STAGES):
            values = []
            stage_has_values = False
            for name, k in labels:
                data = [r[metric+"_seconds"] for r in stages if r["implementation"] == name and
                        r["k"] == k and r["stage"] == stage and r[metric+"_seconds"] is not None]
                values.append(float(np.mean(data)) if data else 0)
                stage_has_values |= bool(data)
            any_values |= stage_has_values
            if not stage_has_values:
                continue
            ax.bar(range(len(labels)), values, bottom=bottom, label=stage, color=colors(stage_index % 20))
            bottom += values
        if not any_values:
            ax.text(.5, .5, "N/A: no CUDA event measurements", transform=ax.transAxes, ha="center")
        ax.set_xticks(range(len(labels)), [f"{name}\nK={k}" for name,k in labels], rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("Mean seconds (exclusive host wall scopes)" if metric == "wall" else
                      "Mean CUDA stream intervals, not active kernel time")
        if any_values:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1,1))
        fig.tight_layout(); fig.savefig(output/filename); plt.close(fig)
    lines = ["# Comparative stage profile", "", "Measured complete-fit medians; profiled/unprofiled is a latency ratio, not CPU overhead.", "",
             "| Implementation | K | Unprofiled (s) | Profiled (s) | Ratio |", "|---|---:|---:|---:|---:|"]
    for name, k in labels:
        times = {mode:float(np.median([r["seconds"] for r in runtime if r["implementation"]==name and
                 r["k"]==k and r["mode"]==mode])) for mode in ("unprofiled", "profiled")}
        lines.append(f'| {name} | {k} | {times["unprofiled"]:.6f} | {times["profiled"]:.6f} | {times["profiled"]/times["unprofiled"]:.3f} |')
    lines += ["", "Empty CSV numeric fields mean N/A, never zero. See docs/profiling.md for stage boundaries, fused operations, event limitations and accounting.",
              "Host breakdown uses means so stacked scopes account for mean elapsed time. Do not sum per-stage medians or add wall and event views.",
              "One unprofiled and one instrumented full warmup per configuration/K are excluded from repeated samples; warmup includes execution and any JIT, not isolated compiler time."]
    (output/"report.md").write_text("\n".join(lines)+"\n")


def run_profile(output, *, dataset="fashion-mnist", smoke=False, versions=None,
                backend="all", device=0, k_values=None, runs=5, iterations=ITERATIONS, threads=1):
    if min(runs, iterations, threads) < 1 or device < 0:
        raise ValueError("runs, iterations, threads must be positive; device must be nonnegative")
    selected = list(SPECS) if versions is None else list(versions)
    if not selected or len(set(selected)) != len(selected) or any(v not in SPECS for v in selected):
        raise ValueError("Use distinct current version IDs from --list-versions; V0/V1 are consolidated into pytorch")
    allowed = {"all":list(SPECS), "cpu":["numpy","sklearn"], "torch":["pytorch"],
               "numba":[v for v in SPECS if v.startswith("v2")],
               "gpu":[v for v in SPECS if v not in ("numpy","sklearn")]}
    if backend not in allowed:
        raise ValueError("Unknown backend")
    if versions is not None and any(v not in allowed[backend] for v in selected):
        raise ValueError("--versions conflicts with --backend")
    selected = [v for v in selected if v in allowed[backend]]
    ks = list(k_values) if k_values is not None else ([2,4] if smoke else list(K_VALUES))
    if not ks or len(set(ks))!=len(ks) or any(not isinstance(k, (int,np.integer)) or isinstance(k,bool) or k<1 for k in ks):
        raise ValueError("K must be distinct positive integers")
    ks = [int(k) for k in ks]
    has_torch = "pytorch" in selected
    has_numba = any(v.startswith("v2") for v in selected)
    suite = "gpu" if has_torch and has_numba else "original" if has_torch else "numba" if has_numba else "cpu"
    backends, device_info = prepare_backends(suite, (128,256,512), device_index=device)
    X, preprocessing = load_benchmark_data(dataset, smoke=smoke)
    if max(ks)>len(X):
        raise ValueError("K exceeds sample count")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for filename in ARTIFACTS:
        (output/filename).unlink(missing_ok=True)
    metadata = dict(started_at=datetime.now(timezone.utc).isoformat(), protocol="stage_profile_v1", status="running", dataset=dataset,
                    shape=list(X.shape), preprocessing=preprocessing, smoke=smoke,
                    seed=42, k=ks, iterations=iterations, runs=runs, warmups=2, threads=threads,
                    selected=selected, device=device, platform=platform.platform(), python=platform.python_version(),
                    versions={}, warmup=[], **device_info)
    try:
        metadata["source_commit"] = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
        metadata["source_dirty"] = bool(subprocess.check_output(["git","status","--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        metadata["source_commit"] = None
    metadata["accounting"] = "sum exclusive wall stages including unscoped_host = profiled end-to-end per run; CUDA intervals are a separate non-additive view"
    metadata["validation"] = "same independent float64 reference and tolerances as sweep_k; validate every warmup and measured fit outside timing"
    metadata["limitations"] = "docs/profiling.md; cpu_overhead is N/A, not inferred by subtracting device time; transfer scopes include allocation/staging; implicit waits remain in calling scope"
    for package in ("numpy","scikit-learn","torch","numba","numba-cuda","matplotlib","threadpoolctl"):
        try: metadata["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    runtime, steps, stages = [], [], []
    def checkpoint():
        (output/"environment.json").write_text(json.dumps(metadata, indent=2)+"\n")
        for filename, rows in (("runtime.csv", runtime), ("steps.csv", steps), ("stages.csv", stages)):
            write_csv(output/filename, rows)
    checkpoint()
    try:
        with threadpool_limits(limits=threads):
            for index, k in enumerate(ks):
                initial = X[np.random.RandomState(42).choice(len(X), k, replace=False)].copy()
                reference = cpu_reference(X, initial, iterations)[-1]
                kwargs = dict(num_clusters=k, num_iterations=iterations, threshold=0, early_stopping=False)
                # Build the actual unified adapters, then select requested configurations.
                fits = build_fits(X, initial, kwargs, "all", threads, backends)
                wanted = {SPECS[v] for v in selected}
                fits = [entry for entry in fits if entry[0] in wanted]
                if {f[0] for f in fits} != wanted:
                    raise RuntimeError("Selected implementation missing from actual registry")
                offset = index % len(fits)
                for name, fit, sync in fits[offset:]+fits[:offset]:
                    result, warmup_seconds = measure_fit(fit, sync)
                    validate_result(X, result, k, iterations, reference)
                    metadata["warmup"].append(dict(implementation=name, k=k, seconds=warmup_seconds,
                        definition="First full fit after context setup; JIT/cache setup plus execution, not isolated compilation"))
                    gpu = backends["torch"][0].cuda if name == "PyTorch GPU" else backends["numba"][0] if sync else None
                    result, instrumented_warmup, _ = Collector(gpu).measure(fit, sync)
                    validate_result(X, result, k, iterations, reference)
                    metadata["warmup"][-1]["instrumented_seconds"] = instrumented_warmup
                    for run in range(1, runs+1):
                        modes = ("unprofiled", "profiled") if run%2 else ("profiled", "unprofiled")
                        for mode in modes:
                            if mode == "profiled":
                                result, seconds, records = Collector(gpu).measure(fit, sync)
                            else:
                                result, seconds = measure_fit(fit, sync)
                            objective = validate_result(X, result, k, iterations, reference)
                            base = dict(implementation=name, k=k, run=run)
                            runtime.append(dict(**base, mode=mode, seconds=seconds,
                                                inertia=objective, reference_inertia=reference))
                            if mode == "profiled":
                                if not np.isclose(sum(r["wall_seconds"] for r in records), seconds, rtol=1e-12, atol=1e-12):
                                    raise RuntimeError("Host scope accounting failed")
                                steps.extend(dict(**base, **r) for r in records)
                                for stage in STAGES:
                                    data = [r for r in records if r["stage"]==stage]
                                    events = [r["event_seconds"] for r in data if r["event_seconds"] is not None]
                                    stages.append(dict(**base, stage=stage, calls=len(data),
                                        wall_seconds=sum(r["wall_seconds"] for r in data) if data else None,
                                        event_seconds=sum(events) if events else None))
                            checkpoint()
                    print(f"K={k} {name}: validated {runs} paired fits")
        summary = summarize(runtime, stages)
        write_csv(output/"summary.csv", summary)
        make_outputs(output, runtime, stages, summary)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        for filename in ("summary.csv","report.md","elapsed.png","host_breakdown.png","cuda_intervals.png"):
            (output/filename).unlink(missing_ok=True)
        raise
    finally:
        checkpoint()
    return runtime, steps, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-versions", action="store_true")
    parser.add_argument("--versions", nargs="+", choices=list(SPECS))
    parser.add_argument("--backend", choices=("all","cpu","torch","numba","gpu"), default="all")
    parser.add_argument("--device", type=int, default=0, help="NVIDIA CUDA device index")
    parser.add_argument("--dataset", choices=("fashion-mnist","synthetic"), default="fashion-mnist")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", dest="k_values")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/stage_profile"))
    args = vars(parser.parse_args())
    if args.pop("list_versions"):
        for version, name in SPECS.items(): print(f"{version:12} {name}")
        return
    output = args.pop("output")
    run_profile(output, **args)


if __name__ == "__main__":
    main()
