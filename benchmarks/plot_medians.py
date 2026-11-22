"""Plot imported console medians without inventing individual runs or error bars."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_medians(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    datasets = {row["dataset"] for row in rows}
    if len(datasets) != 1:
        raise ValueError("Expected exactly one dataset")
    ks = sorted({int(row["k"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in dict.fromkeys(row["implementation"] for row in rows):
        series = {int(row["k"]): row for row in rows if row["implementation"] == name}
        values = [float(series[k]["median_runtime_seconds"])
                  if k in series and series[k]["status"] == "completed"
                  else float("nan") for k in ks]
        ax.plot(ks, values, marker="o", label=name)
    ax.set(xlabel="Number of clusters (K)", ylabel="Median runtime (seconds)",
           title=f"{next(iter(datasets))}: runtime vs K (completed console medians)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks, labels=ks)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    output = csv_path.with_name(csv_path.stem + "_runtime_vs_k.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    print(plot_medians(args.csv))
