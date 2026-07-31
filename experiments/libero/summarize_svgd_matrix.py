#!/usr/bin/env python
"""Summarize a heterogeneous SVGD ablation matrix into CSV, JSON, and plots."""

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
    if a.size < 2 or a.size != b.size:
        return None
    rank_a, rank_b = _average_ranks(a), _average_ranks(b)
    if np.std(rank_a) == 0.0 or np.std(rank_b) == 0.0:
        return None
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _summarize(history_path: Path) -> dict[str, Any]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    records = payload["history"]
    config = payload["config"]
    objective = config.get("latent_distance", "rms")
    energies = np.asarray(
        [record["latent_metrics"][objective] for record in records],
        dtype=np.float64,
    )
    goal_errors = np.asarray(
        [record["goal_errors_m"] for record in records], dtype=np.float64
    )
    tracking = np.asarray(
        [record["target_tracking_errors_m"] for record in records],
        dtype=np.float64,
    )
    update_records = [
        record for record in records if record.get("update_applied", False)
    ]
    expected_populations = int(config["iterations"]) + 1
    completed = (
        len(records) == expected_populations
        and records[-1].get("phase") == "final_evaluation"
        and (history_path.parent / "best_metadata.json").is_file()
    )
    flat_best = int(np.argmin(energies))
    best_population, best_particle = np.unravel_index(
        flat_best, energies.shape
    )
    initial_centroid_error = float(
        records[0]["terminal_diagnostics"]["centroid_goal_error_m"]
    )
    final_centroid_error = float(
        records[-1]["terminal_diagnostics"]["centroid_goal_error_m"]
    )
    trust_scales = np.asarray(
        [
            scale
            for record in update_records
            for scale in record.get("trust_region_scales", [])
        ],
        dtype=np.float64,
    )
    clipped = sum(
        np.count_nonzero(np.asarray(record.get("bounds_clipped", []), dtype=bool))
        for record in update_records
    )
    clip_denominator = (
        len(update_records) * int(config["particles"]) * 3
    )
    fd_eps = np.asarray(config["fd_eps"], dtype=np.float64).reshape(-1)
    if fd_eps.size == 1:
        fd_eps = np.repeat(fd_eps, 3)
    if fd_eps.size != 3:
        raise ValueError(
            f"{history_path}: fd_eps needs one or three values, got {fd_eps.tolist()}"
        )
    isotropic_fd_eps = (
        float(fd_eps[0]) if np.allclose(fd_eps, fd_eps[0]) else None
    )
    return {
        "trial": history_path.parent.name,
        "completed": completed,
        "measured_populations": len(records),
        "expected_populations": expected_populations,
        "objective": objective,
        "feature_encoder": config.get("feature_encoder", "flux_ae"),
        "encoder_model": config.get("dino_model"),
        "transport": config.get("transport", "svgd"),
        "init_mode": config.get("init_mode"),
        "seed": int(config.get("seed", 0)),
        "particles": int(config["particles"]),
        "iterations_requested": int(config["iterations"]),
        "fd_eps_m": isotropic_fd_eps,
        "fd_eps_x_m": float(fd_eps[0]),
        "fd_eps_y_m": float(fd_eps[1]),
        "fd_eps_z_m": float(fd_eps[2]),
        "step_size": float(config["step_size"]),
        "temperature": float(config["temperature"]),
        "repulsion_weight": float(config.get("repulsion_weight", 0.0)),
        "latent_views": config.get("latent_views", "both"),
        "move_steps": int(
            payload.get("effective_rollout", {}).get(
                "move_steps", config.get("move_steps") or 0
            )
        ),
        "settle_steps": int(
            payload.get("effective_rollout", {}).get(
                "settle_steps", config.get("settle_steps") or 0
            )
        ),
        "initial_objective_mean": float(energies[0].mean()),
        "final_objective_mean": float(energies[-1].mean()),
        "objective_mean_change_percent": float(
            100.0 * (energies[-1].mean() / max(energies[0].mean(), 1e-12) - 1.0)
        ),
        "particles_objective_improved_fraction": float(
            np.mean(energies[-1] < energies[0])
        ),
        "initial_centroid_goal_error_m": initial_centroid_error,
        "final_centroid_goal_error_m": final_centroid_error,
        "centroid_goal_error_reduction_m": (
            initial_centroid_error - final_centroid_error
        ),
        "initial_goal_axis_fraction": float(
            records[0]["terminal_diagnostics"]["centroid_goal_axis_fraction"]
        ),
        "final_goal_axis_fraction": float(
            records[-1]["terminal_diagnostics"]["centroid_goal_axis_fraction"]
        ),
        "particles_physically_improved_fraction": float(
            np.mean(goal_errors[-1] < goal_errors[0])
        ),
        "final_success_1cm_fraction": float(np.mean(goal_errors[-1] <= 0.01)),
        "final_success_2cm_fraction": float(np.mean(goal_errors[-1] <= 0.02)),
        "final_success_5cm_fraction": float(np.mean(goal_errors[-1] <= 0.05)),
        "best_objective_population": int(records[best_population]["iteration"]),
        "best_objective_particle": int(best_particle),
        "best_objective_value": float(energies[best_population, best_particle]),
        "best_objective_physical_error_m": float(
            goal_errors[best_population, best_particle]
        ),
        "best_physical_error_m": float(goal_errors.min()),
        "objective_physical_error_spearman": _spearman(energies, goal_errors),
        "tracking_error_median_m": float(np.median(tracking)),
        "tracking_error_p95_m": float(np.quantile(tracking, 0.95)),
        "trust_region_limited_fraction": (
            float(np.mean(trust_scales < 0.999999)) if trust_scales.size else 0.0
        ),
        "bounds_clipped_coordinate_fraction": (
            float(clipped / clip_denominator) if clip_denominator else 0.0
        ),
    }


def _plot(rows: list[dict[str, Any]], path: Path, dpi: int) -> None:
    rows = sorted(
        rows, key=lambda row: row["centroid_goal_error_reduction_m"]
    )
    labels = [
        row["trial"] + ("" if row["completed"] else " [incomplete]")
        for row in rows
    ]
    y = np.arange(len(rows))
    colors = {
        "rms": "tab:blue",
        "cosine": "tab:orange",
        "token_cosine": "tab:green",
    }
    bar_colors = [colors.get(row["objective"], "tab:gray") for row in rows]
    figure, axes = plt.subplots(
        1, 4, figsize=(20, max(7, 0.42 * len(rows))), constrained_layout=True
    )
    series = (
        (
            [100.0 * row["centroid_goal_error_reduction_m"] for row in rows],
            "Centroid error reduction",
            "cm, larger is better",
        ),
        (
            [-row["objective_mean_change_percent"] for row in rows],
            "Objective reduction",
            "%, larger is better",
        ),
        (
            [100.0 * row["final_success_2cm_fraction"] for row in rows],
            "Final particles within 2 cm",
            "%",
        ),
        (
            [1000.0 * row["tracking_error_p95_m"] for row in rows],
            "Tracking error p95",
            "mm, smaller is better",
        ),
    )
    for index, (values, title, xlabel) in enumerate(series):
        axes[index].barh(y, values, color=bar_colors, alpha=0.85)
        axes[index].set(title=title, xlabel=xlabel)
        axes[index].grid(axis="x", alpha=0.3)
        axes[index].set_yticks(y)
        axes[index].set_yticklabels(labels if index == 0 else [])
    figure.suptitle(
        "SVGD endpoint ablation matrix (bar color = optimized latent metric)",
        fontsize=15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).expanduser().absolute()
    out_dir = (
        Path(args.out_dir).expanduser().absolute()
        if args.out_dir
        else suite_dir
    )
    histories = sorted(suite_dir.glob("trials/*/history.json"))
    if not histories:
        parser.error(f"No trial histories found under {suite_dir / 'trials'}")
    rows = [_summarize(path) for path in histories]
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "matrix_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "matrix_summary.json"
    json_path.write_text(
        json.dumps({"trials": rows}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_path = out_dir / "matrix_summary.png"
    _plot(rows, plot_path, args.dpi)

    print(f"trials:  {len(rows)}")
    print(f"plot:    {plot_path}")
    print(f"summary: {json_path}")
    print(f"table:   {csv_path}")


if __name__ == "__main__":
    main()
