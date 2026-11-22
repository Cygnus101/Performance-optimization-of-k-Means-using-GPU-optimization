"""Plot complete-fit runtime and optional speedup from unified sweep results."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(csv_path, *, speedup=False, baseline=None):
    csv_path = Path(csv_path)
    groups = {}
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(row["implementation"], {}).setdefault(int(row["k"]), []).append(float(row["seconds"]))
    if not groups:
        raise ValueError("No measurements to plot")
    if baseline is None:
        baseline = "NumPy CPU" if "NumPy CPU" in groups else "V2.0 baseline"
    if speedup and baseline not in groups:
        raise ValueError(f"Missing speedup baseline: {baseline}")
    medians_by_name = {name: {k: float(np.median(values)) for k, values in series.items()}
                       for name, series in groups.items()}
    for metric in (["runtime", "speedup"] if speedup else ["runtime"]):
        fig, ax = plt.subplots(figsize=(12, 7))
        for name, series in groups.items():
            ks = sorted(series)
            medians = [np.median(series[k]) for k in ks]
            if metric == "runtime":
                ax.errorbar(ks, medians, yerr=[np.std(series[k], ddof=1) for k in ks],
                            marker="o", capsize=3, label=name)
            else:
                ax.plot(ks, [np.median(groups[baseline][k]) / t for k, t in zip(ks, medians)],
                        marker="o", label=name)
        ax.set(xlabel="Number of clusters (K)",
               ylabel="Complete fit runtime (seconds)" if metric == "runtime" else f"Speedup vs {baseline}",
               title="Complete fit vs K (median ± run standard deviation)" if metric == "runtime"
                     else "Speedup vs K (ratio of median runtimes)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, labels=ks)
        ax.grid(alpha=.25)
        if metric == "runtime":
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(csv_path.parent / f"{metric}_vs_k.png", dpi=160)
        plt.close(fig)
    if not speedup:
        (csv_path.parent / "speedup_vs_k.png").unlink(missing_ok=True)
    return medians_by_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--speedup", action="store_true")
    parser.add_argument("--baseline", help="Speedup denominator; default NumPy CPU or V2.0 baseline")
    args = parser.parse_args()
    plot_results(args.csv, speedup=args.speedup, baseline=args.baseline)
