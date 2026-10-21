"""Plot offline and online learning curves from qchunk metrics.csv."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path


def load_metrics(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Load the long-form metrics CSV into phase/step metric dictionaries."""
    phases: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            phases[row["phase"]][int(row["step"])][row["metric"]] = float(
                row["value"]
            )
    return phases


def rolling_mean(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    items: deque[float] = deque()
    total = 0.0
    for value in values:
        items.append(value)
        total += value
        if len(items) > window:
            total -= items.popleft()
        result.append(total / len(items))
    return result


def rolling_wilson(
    values: list[float], window: int, z: float = 1.96
) -> tuple[list[float], list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    items: deque[float] = deque()
    successes = 0.0
    for value in values:
        items.append(value)
        successes += value
        if len(items) > window:
            successes -= items.popleft()
        n = len(items)
        proportion = successes / n
        denominator = 1.0 + z * z / n
        centre = (proportion + z * z / (2.0 * n)) / denominator
        margin = (
            z
            * math.sqrt(
                proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
            )
            / denominator
        )
        lower.append(max(0.0, centre - margin))
        upper.append(min(1.0, centre + margin))
    return lower, upper


def ordered_metric(
    records: dict[int, dict[str, float]], metric: str
) -> tuple[list[int], list[float]]:
    pairs = sorted(
        (step, values[metric])
        for step, values in records.items()
        if metric in values
    )
    return [step for step, _ in pairs], [value for _, value in pairs]


def evaluation_records(
    phases: dict[str, dict[int, dict[str, float]]],
) -> list[tuple[int, str, dict[str, float]]]:
    records: list[tuple[int, str, dict[str, float]]] = []
    offline = phases.get("evaluation", {})
    if offline:
        _, values = max(offline.items())
        records.append((0, "offline", values))
    records.extend(
        (step, "online", values)
        for step, values in sorted(phases.get("online_evaluation", {}).items())
    )
    return records


def write_evaluation_summary(
    path: Path, records: list[tuple[int, str, dict[str, float]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "online_step",
                "checkpoint_phase",
                "success_rate",
                "mean_return",
                "mean_length",
            ],
        )
        writer.writeheader()
        for step, phase, values in records:
            writer.writerow(
                {
                    "online_step": step,
                    "checkpoint_phase": phase,
                    "success_rate": values.get("success_rate"),
                    "mean_return": values.get("mean_return"),
                    "mean_length": values.get("mean_length"),
                }
            )


def plot(metrics_path: Path, output_stem: Path, window: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    phases = load_metrics(metrics_path)
    episodes = phases.get("online_episode", {})
    episode_steps, successes = ordered_metric(episodes, "success")
    _, episode_returns = ordered_metric(episodes, "episode_return")
    if not episode_steps or len(episode_steps) != len(episode_returns):
        raise ValueError("online_episode metrics are missing or misaligned")

    eval_records = evaluation_records(phases)
    if not eval_records:
        raise ValueError("no evaluation metrics found")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    write_evaluation_summary(
        output_stem.with_name(f"{output_stem.name}_evaluation.csv"), eval_records
    )

    x_episode = [step / 1_000_000 for step in episode_steps]
    smooth_success = rolling_mean(successes, window)
    success_low, success_high = rolling_wilson(successes, window)
    smooth_return = rolling_mean(episode_returns, window)

    x_eval = [step / 1_000_000 for step, _, _ in eval_records]
    eval_success = [values["success_rate"] for _, _, values in eval_records]
    eval_return = [values["mean_return"] for _, _, values in eval_records]
    eval_length = [values["mean_length"] for _, _, values in eval_records]

    offline_steps, critic_loss = ordered_metric(
        phases.get("offline", {}), "critic_loss"
    )
    _, flow_loss = ordered_metric(phases.get("offline", {}), "flow_loss")
    x_offline = [step / 1_000_000 for step in offline_steps]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    blue = "#2F6BFF"
    orange = "#E67E22"
    green = "#1B9E77"
    purple = "#7B61FF"
    gray = "#6B7280"

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    figure.suptitle("Q-chunking on Cube Triple — Modal run", fontweight="bold")

    ax = axes[0, 0]
    ax.fill_between(
        x_episode,
        success_low,
        success_high,
        color=blue,
        alpha=0.14,
        linewidth=0,
        label="95% Wilson interval",
    )
    ax.plot(
        x_episode,
        smooth_success,
        color=blue,
        linewidth=1.8,
        label=f"Online episodes ({window}-episode rolling)",
    )
    ax.plot(
        x_eval,
        eval_success,
        color=orange,
        marker="o",
        markersize=5,
        linewidth=1.8,
        label="Evaluation (50 episodes)",
        zorder=3,
    )
    ax.annotate(
        "96%",
        (x_eval[-1], eval_success[-1]),
        xytext=(-8, -18),
        textcoords="offset points",
        ha="right",
        color=orange,
        fontweight="bold",
    )
    ax.set(title="Task success", xlabel="Online environment steps (millions)")
    ax.set_ylabel("Success rate")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.03, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right", frameon=False)

    ax = axes[0, 1]
    ax.scatter(
        x_episode,
        episode_returns,
        s=6,
        color=gray,
        alpha=0.10,
        linewidths=0,
        label="Completed episodes",
        rasterized=True,
    )
    ax.plot(
        x_episode,
        smooth_return,
        color=green,
        linewidth=1.8,
        label=f"{window}-episode rolling mean",
    )
    ax.plot(
        x_eval,
        eval_return,
        color=orange,
        marker="o",
        markersize=5,
        linewidth=1.8,
        label="Evaluation mean",
        zorder=3,
    )
    ax.annotate(
        f"{eval_return[-1]:.0f}",
        (x_eval[-1], eval_return[-1]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color=orange,
        fontweight="bold",
    )
    ax.set(title="Episode return", xlabel="Online environment steps (millions)")
    ax.set_ylabel("Return (higher is better)")
    ax.set_xlim(0, 1.02)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1, 0]
    ax.plot(x_offline, critic_loss, color=purple, linewidth=1.6, label="Critic loss")
    ax.plot(x_offline, flow_loss, color=green, linewidth=1.6, label="Flow loss")
    ax.set_yscale("log")
    ax.set(
        title="Offline optimization",
        xlabel="Offline gradient updates (millions)",
        ylabel="Training loss (log scale)",
    )
    ax.set_xlim(0, 1.02)
    ax.grid(axis="y", alpha=0.2, which="both")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(
        x_eval,
        eval_length,
        color=orange,
        marker="o",
        markersize=5,
        linewidth=1.8,
    )
    ax.axhline(1000, color=gray, linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"{eval_length[-1]:.0f} steps",
        (x_eval[-1], eval_length[-1]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color=orange,
        fontweight="bold",
    )
    ax.set(
        title="Evaluation episode length",
        xlabel="Online environment steps (millions)",
        ylabel="Mean episode length (steps)",
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1050)
    ax.grid(axis="y", alpha=0.2)

    figure.text(
        0.5,
        -0.01,
        "At x=0: offline checkpoint. Evaluation markers use 50 episodes; "
        f"online rolling curves use the latest {window} completed episodes.",
        ha="center",
        color=gray,
        fontsize=9,
    )
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="Path to long-form metrics.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/learning_curve"),
        help="Output path without extension",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Trailing episode window for online curves",
    )
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    plot(args.metrics, args.output, args.window)


if __name__ == "__main__":
    main()
