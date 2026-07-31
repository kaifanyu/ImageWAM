#!/usr/bin/env python
"""Summarize goal-image endpoint and mug-path hyperparameter trials."""

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


def _load_trial(trial_dir: Path) -> dict[str, Any]:
    endpoint_history = json.loads(
        (trial_dir / "endpoint_search" / "history.json").read_text(
            encoding="utf-8"
        )
    )
    endpoint_best = json.loads(
        (trial_dir / "endpoint_search" / "best_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    path_history = json.loads(
        (trial_dir / "path_search" / "history.json").read_text(
            encoding="utf-8"
        )
    )
    path_best = json.loads(
        (trial_dir / "path_search" / "best_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    endpoint_config = endpoint_history["config"]
    path_config = path_history["config"]
    endpoint_final = endpoint_history["history"][-1]
    path_initial = path_history["history"][0]
    path_final = path_history["history"][-1]
    endpoint_particles = np.asarray(
        endpoint_final["particles_before_update"], dtype=np.float64
    )
    path_particles = np.asarray(
        path_final["parameters"], dtype=np.float64
    )
    target = endpoint_best["selection"]["target_eef"]
    endpoint_replay = endpoint_best["replay"]
    path_replay = path_best["replay"]
    return {
        "trial": trial_dir.name,
        "metric": endpoint_config["latent_distance"],
        "seed": int(endpoint_config["seed"]),
        "endpoint_repulsion": float(endpoint_config["repulsion_weight"]),
        "path_repulsion": float(path_config["repulsion_weight"]),
        "endpoint_step_size": float(endpoint_config["step_size"]),
        "path_step_size": float(path_config["step_size"]),
        "endpoint_temperature": float(endpoint_config["temperature"]),
        "path_temperature": float(path_config["temperature"]),
        "endpoint_bandwidth": float(endpoint_config["bandwidth_scale"]),
        "path_bandwidth": float(path_config["bandwidth_scale"]),
        "inferred_x_m": float(target[0]),
        "inferred_y_m": float(target[1]),
        "inferred_z_m": float(target[2]),
        "endpoint_replay_energy": float(
            endpoint_replay["objective_energy"]
        ),
        "endpoint_physical_error_m": float(
            endpoint_replay["physical_goal_error_m"]
        ),
        "endpoint_final_spread_m": float(
            np.mean(np.std(endpoint_particles, axis=0))
        ),
        "path_initial_success_fraction": float(
            path_initial["success_fraction"]
        ),
        "path_final_success_fraction": float(
            path_final["success_fraction"]
        ),
        "path_final_spread_m": float(
            np.mean(np.std(path_particles, axis=0))
        ),
        "selected_midpoint_x_m": float(
            path_best["selected_parameters"]["midpoint_x_m"]
        ),
        "selected_arc_height_m": float(
            path_best["selected_parameters"]["arc_height_m"]
        ),
        "path_replay_energy": float(path_replay["energy"]),
        "path_goal_error_m": float(path_replay["goal_error_m"]),
        "path_target_tracking_error_m": float(
            path_replay.get(
                "target_tracking_error_m", path_replay["goal_error_m"]
            )
        ),
        "mug_displacement_m": float(path_replay["mug_displacement_m"]),
        "mug_rotation_deg": float(path_replay["mug_rotation_deg"]),
        "physical_success": bool(path_replay["physical_success"]),
    }


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["trial"] for row in rows]
    positions = np.arange(len(rows))
    colors = [
        "tab:blue" if row["physical_success"] else "tab:red" for row in rows
    ]
    figure, axes = plt.subplots(
        2, 3, figsize=(18, 10), constrained_layout=True
    )
    series = [
        (
            [100.0 * row["endpoint_physical_error_m"] for row in rows],
            "Inferred endpoint error",
            "cm",
        ),
        (
            [100.0 * row["endpoint_final_spread_m"] for row in rows],
            "Final endpoint-particle spread",
            "cm",
        ),
        (
            [row["endpoint_replay_energy"] for row in rows],
            "Endpoint replay latent energy",
            "energy",
        ),
        (
            [100.0 * row["path_final_success_fraction"] for row in rows],
            "Final safe path particles",
            "%",
        ),
        (
            [100.0 * row["path_final_spread_m"] for row in rows],
            "Final path-particle spread",
            "cm",
        ),
        (
            [100.0 * row["mug_displacement_m"] for row in rows],
            "Selected replay mug displacement",
            "cm",
        ),
    ]
    for axis, (values, title, ylabel) in zip(axes.ravel(), series):
        axis.bar(positions, values, color=colors, alpha=0.82)
        axis.set(
            title=title,
            ylabel=ylabel,
            xticks=positions,
            xticklabels=labels,
        )
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle(
        "Goal-image endpoint/path sweep (blue = selected replay succeeded)",
        fontsize=15,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True)
    args = parser.parse_args()
    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    trial_dirs = sorted(
        path
        for path in (sweep_dir / "trials").iterdir()
        if (path / "endpoint_search" / "best_metadata.json").is_file()
        and (path / "path_search" / "best_metadata.json").is_file()
    )
    if not trial_dirs:
        parser.error(f"No completed trials under {sweep_dir / 'trials'}")
    rows = [_load_trial(path) for path in trial_dirs]

    csv_path = sweep_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = sweep_dir / "summary.json"
    json_path.write_text(
        json.dumps({"trials": rows}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_path = sweep_dir / "summary.png"
    _plot(rows, plot_path)
    print(f"trials:  {len(rows)}")
    print(f"table:   {csv_path}")
    print(f"summary: {json_path}")
    print(f"plot:    {plot_path}")


if __name__ == "__main__":
    main()
