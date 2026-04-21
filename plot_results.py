from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot total-cost comparison for predictive maintenance policies.")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("artifacts") / "cost_comparison.csv",
        help="CSV file produced by train_compare.py",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("artifacts") / "total_cost_comparison.png",
        help="Path to save the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = pd.read_csv(args.metrics_file)
    summary = dataframe.groupby("policy", as_index=False)["total_cost"].mean().sort_values("total_cost")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(summary["policy"], summary["total_cost"], color=["#4c78a8", "#f58518", "#54a24b"][: len(summary)])
    axis.set_title("Average Total Cost: RL vs Fixed-Interval Baseline")
    axis.set_xlabel("Policy")
    axis.set_ylabel("Average Total Cost")
    for bar in axis.patches:
        axis.annotate(
            f"{bar.get_height():.2f}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(args.output_file, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
