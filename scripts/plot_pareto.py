"""
Plot accuracy vs. median latency (ms/clip) Pareto curves from bench-latency summaries.

Usage:
    python scripts/plot_pareto.py
    python scripts/plot_pareto.py --latency-dir artifacts/latency --output artifacts/latency/pareto.pdf
    python scripts/plot_pareto.py --dataset kabr      # filter to one dataset
"""
import argparse
import json
import statistics
from pathlib import Path


def load_summaries(latency_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(latency_dir.glob("*.latency.json")):
        with p.open(encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the Pareto-optimal (higher acc, lower latency) subset, sorted by latency."""
    sorted_pts = sorted(points, key=lambda x: x[1])  # sort by latency ascending
    front = []
    best_acc = -1.0
    for lat, acc in sorted_pts:
        if acc > best_acc:
            best_acc = acc
            front.append((lat, acc))
    return front


def plot(rows: list[dict], output: Path, dataset_filter: str | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    COLORS = {
        "zero-shot": "#00A8E8",
        "lora":      "#F9A825",
    }
    MARKERS = {
        "kabr":       "o",
        "baboonland": "s",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    datasets = ["kabr", "baboonland"] if dataset_filter is None else [dataset_filter]

    for ax, ds in zip(axes if dataset_filter is None else [axes], datasets):
        ds_rows = [r for r in rows if r.get("dataset") == ds and r.get("latency_median_ms")]
        if not ds_rows:
            ax.set_title(f"{ds} (no data)")
            continue

        for mode, color in COLORS.items():
            mode_rows = [r for r in ds_rows if r.get("mode") == mode]
            if not mode_rows:
                continue
            xs = [r["latency_median_ms"] for r in mode_rows]
            ys = [r["accuracy"] for r in mode_rows]
            ax.scatter(xs, ys, c=color, marker=MARKERS.get(ds, "o"),
                       s=90, zorder=3, label=mode)
            for r in mode_rows:
                slug = Path(r.get("predictions_path", "")).stem
                slug = slug.replace(f"_{ds}_{mode}", "").replace("_zero-shot", "").replace("_lora", "")
                ax.annotate(
                    slug,
                    (r["latency_median_ms"], r["accuracy"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color=color,
                )

            # draw Pareto front for this mode
            pts = list(zip(xs, ys))
            if len(pts) >= 2:
                front = pareto_front(pts)
                if front:
                    fx, fy = zip(*[(lat, acc) for lat, acc in front])
                    ax.step(fx, fy, where="post", color=color, lw=1.5,
                            linestyle="--", alpha=0.7)

        ax.set_xlabel("Median latency (ms/clip)", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(f"{ds.upper()} — Accuracy vs. Latency", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("Pareto Frontier: Wildlife VLM Accuracy vs. Inference Latency",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")

    # Also write a CSV summary table
    csv_path = output.with_suffix(".csv")
    import csv
    fieldnames = [
        "model_id", "mode", "dataset",
        "accuracy", "latency_median_ms", "latency_p90_ms",
        "clips_per_sec", "peak_gpu_mb", "n_clips",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (x.get("dataset",""), x.get("mode",""), x.get("accuracy", 0))):
            writer.writerow(r)
    print(f"CSV  → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-dir", default="artifacts/latency",
                        help="Directory containing *.latency.json files")
    parser.add_argument("--output", default="artifacts/latency/pareto.pdf",
                        help="Output PDF path")
    parser.add_argument("--dataset", default=None, choices=["kabr", "baboonland"],
                        help="Filter to one dataset (default: both side-by-side)")
    args = parser.parse_args()

    latency_dir = Path(args.latency_dir)
    rows = load_summaries(latency_dir)
    if not rows:
        print(f"No *.latency.json files found in {latency_dir}")
        return
    print(f"Loaded {len(rows)} benchmark summaries")
    plot(rows, output=Path(args.output), dataset_filter=args.dataset)


if __name__ == "__main__":
    main()
