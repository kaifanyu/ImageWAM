#!/usr/bin/env python
"""Compare controlled SVGD trials that optimize different latent distances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


METRICS = ("rms", "cosine", "token_cosine")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size != a.size:
        return None
    rank_a = _average_ranks(a)
    rank_b = _average_ranks(b)
    if np.std(rank_a) == 0.0 or np.std(rank_b) == 0.0:
        return None
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _load_trial(history_path: Path) -> dict[str, Any]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    records = payload["history"]
    objective = payload.get("config", {}).get("latent_distance", "rms")
    iterations = np.asarray([record["iteration"] for record in records], dtype=int)
    positions = np.asarray(
        [record["terminal_eefs"] for record in records], dtype=np.float64
    )
    start = np.asarray(payload["actual_start_eef"], dtype=np.float64)
    goal = np.asarray(payload["diagnostic_goal_eef"], dtype=np.float64)
    goal_delta = goal - start
    goal_distance = float(np.linalg.norm(goal_delta))
    goal_unit = goal_delta / max(goal_distance, 1e-12)
    centroids = positions.mean(axis=1)
    progress_fraction = ((centroids - start[None, :]) @ goal_unit) / max(
        goal_distance, 1e-12
    )
    centroid_goal_error = np.linalg.norm(centroids - goal[None, :], axis=1)
    goal_errors = np.asarray(
        [record["goal_errors_m"] for record in records], dtype=np.float64
    )
    latent_metrics = {
        metric: np.asarray(
            [record["latent_metrics"][metric] for record in records],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    objective_values = latent_metrics[objective]
    tracking_rows = []
    for record, terminal_row in zip(records, positions):
        if "target_tracking_errors_m" in record:
            tracking_rows.append(record["target_tracking_errors_m"])
            continue
        targets = np.asarray(
            record.get("particles_before_update", terminal_row), dtype=np.float64
        )
        tracking_rows.append(np.linalg.norm(terminal_row - targets, axis=1))
    tracking = np.asarray(tracking_rows, dtype=np.float64)
    update_norm = np.asarray(
        [record.get("applied_update_norm_mean_m", 0.0) for record in records],
        dtype=np.float64,
    )
    correlations = {
        metric: _spearman(values, goal_errors)
        for metric, values in latent_metrics.items()
    }
    repeatability_abs = []
    for record in records:
        deltas = [
            abs(float(item["latent_metric_deltas"][objective]))
            for item in record.get("repeatability", [])
            if item is not None
        ]
        repeatability_abs.append(float(np.mean(deltas)) if deltas else np.nan)
    return {
        "name": history_path.parent.name,
        "history_path": str(history_path),
        "objective": objective,
        "iterations": iterations,
        "progress_fraction": progress_fraction,
        "centroid_goal_error_m": centroid_goal_error,
        "goal_errors_m": goal_errors,
        "latent_metrics": latent_metrics,
        "objective_values": objective_values,
        "tracking_errors_m": tracking,
        "update_norm_m": update_norm,
        "correlations": correlations,
        "repeatability_objective_abs_delta": np.asarray(
            repeatability_abs, dtype=np.float64
        ),
    }


def _rows(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        for index, iteration in enumerate(trial["iterations"]):
            row = {
                "trial": trial["name"],
                "objective": trial["objective"],
                "population": int(iteration),
                "objective_mean": float(trial["objective_values"][index].mean()),
                "objective_min": float(trial["objective_values"][index].min()),
                "centroid_goal_axis_fraction": float(
                    trial["progress_fraction"][index]
                ),
                "centroid_goal_error_m": float(
                    trial["centroid_goal_error_m"][index]
                ),
                "tracking_error_mean_m": float(
                    trial["tracking_errors_m"][index].mean()
                ),
                "tracking_error_max_m": float(
                    trial["tracking_errors_m"][index].max()
                ),
                "applied_update_norm_mean_m": float(trial["update_norm_m"][index]),
                "objective_repeat_abs_delta_mean": (
                    None
                    if np.isnan(trial["repeatability_objective_abs_delta"][index])
                    else float(
                        trial["repeatability_objective_abs_delta"][index]
                    )
                ),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(
                    trial["latent_metrics"][metric][index].mean()
                )
            rows.append(row)
    return rows


def _plot(trials: list[dict[str, Any]], output: Path, dpi: int) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    figure.suptitle(
        "Controlled 20-particle SVGD latent-distance comparison", fontsize=16
    )

    for trial in trials:
        label = trial["objective"]
        iterations = trial["iterations"]
        axes[0, 0].plot(
            iterations, 100.0 * trial["progress_fraction"], marker="o", label=label
        )
        axes[0, 1].plot(
            iterations,
            100.0 * trial["centroid_goal_error_m"],
            marker="o",
            label=label,
        )
        objective_mean = trial["objective_values"].mean(axis=1)
        axes[0, 2].plot(
            iterations,
            objective_mean / max(float(objective_mean[0]), 1e-12),
            marker="o",
            label=label,
        )
        axes[1, 0].plot(
            iterations,
            1000.0 * trial["update_norm_m"],
            marker="o",
            label=label,
        )
        axes[1, 1].plot(
            iterations,
            1000.0 * trial["tracking_errors_m"].mean(axis=1),
            marker="o",
            label=label,
        )

    axes[0, 0].axhline(100.0, color="tab:green", linewidth=1, alpha=0.5)
    axes[0, 0].set(
        title="Centroid progress toward physical goal",
        xlabel="evaluated population",
        ylabel="goal-axis progress (%)",
    )
    axes[0, 1].set(
        title="Centroid physical goal error",
        xlabel="evaluated population",
        ylabel="error (cm)",
    )
    axes[0, 2].axhline(1.0, color="black", linewidth=1, alpha=0.4)
    axes[0, 2].set(
        title="Own objective normalized to population 0",
        xlabel="evaluated population",
        ylabel="relative objective",
    )
    axes[1, 0].set(
        title="Mean applied particle update",
        xlabel="optimizer update",
        ylabel="update norm (mm)",
    )
    axes[1, 1].set(
        title="Mean rollout target tracking error",
        xlabel="evaluated population",
        ylabel="tracking error (mm)",
    )
    for axis in axes.ravel()[:5]:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    width = 0.24
    x = np.arange(len(METRICS), dtype=float)
    for trial_index, trial in enumerate(trials):
        values = [
            np.nan
            if trial["correlations"][metric] is None
            else trial["correlations"][metric]
            for metric in METRICS
        ]
        offset = (trial_index - 0.5 * (len(trials) - 1)) * width
        axes[1, 2].bar(x + offset, values, width=width, label=trial["objective"])
    axes[1, 2].axhline(0.0, color="black", linewidth=1)
    axes[1, 2].set_xticks(x, METRICS)
    axes[1, 2].set(
        title="Distance vs physical-error Spearman correlation",
        ylabel="correlation (positive is desired)",
        ylim=(-1.0, 1.0),
    )
    axes[1, 2].grid(axis="y", alpha=0.3)
    axes[1, 2].legend(fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).expanduser().absolute()
    histories = sorted(suite_dir.glob("*/history.json"))
    if not histories:
        parser.error(f"No trial history files found under {suite_dir}")
    trials = [_load_trial(path) for path in histories]
    rows = _rows(trials)

    csv_path = suite_dir / "trial_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "trials": [
            {
                "name": trial["name"],
                "objective": trial["objective"],
                "initial_goal_axis_fraction": float(trial["progress_fraction"][0]),
                "final_goal_axis_fraction": float(trial["progress_fraction"][-1]),
                "initial_centroid_goal_error_m": float(
                    trial["centroid_goal_error_m"][0]
                ),
                "final_centroid_goal_error_m": float(
                    trial["centroid_goal_error_m"][-1]
                ),
                "initial_objective_mean": float(
                    trial["objective_values"][0].mean()
                ),
                "final_objective_mean": float(
                    trial["objective_values"][-1].mean()
                ),
                "tracking_error_mean_m": float(
                    trial["tracking_errors_m"].mean()
                ),
                "tracking_error_max_m": float(trial["tracking_errors_m"].max()),
                "metric_physical_error_spearman": trial["correlations"],
                "objective_repeatability_abs_delta_mean": (
                    None
                    if np.isnan(
                        trial["repeatability_objective_abs_delta"]
                    ).all()
                    else float(
                        np.nanmean(
                            trial["repeatability_objective_abs_delta"]
                        )
                    )
                ),
            }
            for trial in trials
        ]
    }
    summary_path = suite_dir / "trial_comparison.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_path = suite_dir / "trial_comparison.png"
    _plot(trials, plot_path, args.dpi)

    print(f"trials:     {len(trials)}")
    print(f"comparison: {plot_path}")
    print(f"summary:    {summary_path}")
    print(f"table:      {csv_path}")


if __name__ == "__main__":
    main()
