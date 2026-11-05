"""Plot the author's released QC curves beside this project's evaluation logs.

The checked-in reference CSV is an extraction of the author's
``plot_data/ogbench-individual.pkl``; see results/reproduction/README.md.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduction"
REFERENCE = OUT / "paper_qc_reference.csv"
CUBE_LOG = ROOT / "results" / "modal_cube_triple" / "learning_curve_evaluation.csv"
PUZZLE_LOG = ROOT / "results" / "puzzle_3x3_task4" / "metrics.csv"


def read_reference(task: str) -> list[dict[str, float]]:
    with REFERENCE.open(newline="") as file:
        return [
            {
                "steps": float(row["total_steps"]),
                "mean": float(row["success_mean"]),
                "low": float(row["ci_low"]),
                "high": float(row["ci_high"]),
            }
            for row in csv.DictReader(file)
            if row["task"] == task
        ]


def read_cube() -> list[tuple[float, float]]:
    with CUBE_LOG.open(newline="") as file:
        return [
            (1_000_000 + float(row["online_step"]), float(row["success_rate"]))
            for row in csv.DictReader(file)
        ]


def read_puzzle() -> list[tuple[float, float]]:
    latest: dict[tuple[str, int], float] = {}
    with PUZZLE_LOG.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["metric"] == "success_rate" and row["phase"] in {
                "evaluation",
                "online_evaluation",
            }:
                latest[(row["phase"], int(row["step"]))] = float(row["value"])
    return sorted(
        (
            (float(step) if phase == "evaluation" else 1_000_000.0 + step),
            rate,
        )
        for (phase, step), rate in latest.items()
    )


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cases = (
        ("cube-triple-play-task2", "Cube-triple task2", read_cube()),
        ("puzzle-3x3-play-task4", "Puzzle-3x3 task4", read_puzzle()),
    )
    for ax, (task, title, local) in zip(axes, cases, strict=True):
        reference = read_reference(task)
        x = [r["steps"] / 1_000_000 for r in reference]
        ax.fill_between(
            x,
            [r["low"] * 100 for r in reference],
            [r["high"] * 100 for r in reference],
            color="#4878a8",
            alpha=0.18,
            linewidth=0,
            label="KTC 95% của tác giả (4 seed)",
        )
        ax.plot(x, [r["mean"] * 100 for r in reference], color="#4878a8", label="Tác giả: trung bình 4 seed")
        ax.plot(
            [step / 1_000_000 for step, _ in local],
            [rate * 100 for _, rate in local],
            "o-",
            color="#c06431",
            markersize=3.5,
            linewidth=1.7,
            label="Cài đặt lại: 1 seed, 50 episode",
        )
        ax.axvline(1, color="#777777", linestyle="--", linewidth=1)
        ax.set(title=title, xlim=(0, 2.02), ylim=(-4, 105), xlabel="Tổng bước (triệu)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Tỷ lệ thành công (%)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("QC: đối chiếu đường công bố và log cài đặt lại", fontsize=12)
    fig.tight_layout()
    path = OUT / "paper_vs_reimplementation.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    for task, _, local in cases:
        ref = {int(r["steps"]): r for r in read_reference(task)}
        for step, rate in local:
            if int(step) in (1_000_000, 2_000_000):
                print(task, int(step), f"ours={rate:.3f}", f"paper={ref[int(step)]['mean']:.3f}")
    print(path)


if __name__ == "__main__":
    main()
