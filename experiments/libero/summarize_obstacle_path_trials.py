#!/usr/bin/env python
"""Summarize trajectory-shape optimization trials for mug avoidance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_trial(history_path: Path) -> dict[str, Any]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    records = payload["history"]
    config = payload["config"]
    metadata_path = history_path.parent / "best_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else None
    )
    initial, final = records[0], records[-1]
    replay = metadata["replay"] if metadata else {}
    selected = metadata["selected_parameters"] if metadata else {}
    return {
        "trial": history_path.parent.name,
        "completed": bool(
            metadata is not None and final["phase"] == "final_evaluation"
        ),
        "objective": config["latent_distance"],
        "transport": config["transport"],
        "particles": int(config["particles"]),
        "iterations": int(config["iterations"]),
        "init_mode": config["init_mode"],
        "initial_energy_mean": float(initial["energy_mean"]),
        "final_energy_mean": float(final["energy_mean"]),
        "energy_reduction_percent": float(
            100.0
            * (
                1.0
                - final["energy_mean"]
                / max(initial["energy_mean"], 1e-12)
            )
        ),
        "initial_success_fraction": float(initial["success_fraction"]),
        "final_success_fraction": float(final["success_fraction"]),
        "initial_mug_displacement_mean_m": float(
            initial["mug_displacement_mean_m"]
        ),
        "final_mug_displacement_mean_m": float(
            final["mug_displacement_mean_m"]
        ),
        "selected_midpoint_x_m": selected.get("midpoint_x_m"),
        "selected_arc_height_m": selected.get("arc_height_m"),
        "selected_replay_energy": replay.get("energy"),
        "selected_replay_goal_error_m": replay.get("goal_error_m"),
        "selected_replay_mug_displacement_m": replay.get(
            "mug_displacement_m"
        ),
        "selected_replay_mug_rotation_deg": replay.get("mug_rotation_deg"),
        "selected_replay_physical_success": replay.get("physical_success"),
    }


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["trial"] for row in rows]
    y = list(range(len(rows)))
    figure, axes = plt.subplots(
        1, 4, figsize=(20, max(6, 0.7 * len(rows))), constrained_layout=True
    )
    series = [
        (
            [row["energy_reduction_percent"] for row in rows],
            "Latent objective reduction",
            "%",
        ),
        (
            [100.0 * row["final_success_fraction"] for row in rows],
            "Final safe-path particles",
            "%",
        ),
        (
            [
                100.0 * row["final_mug_displacement_mean_m"]
                for row in rows
            ],
            "Final mean mug displacement",
            "cm",
        ),
        (
            [
                100.0
                * (
                    row["selected_replay_mug_displacement_m"]
                    if row["selected_replay_mug_displacement_m"] is not None
                    else 0.0
                )
                for row in rows
            ],
            "Objective-selected mug displacement",
            "cm",
        ),
    ]
    colors = [
        "tab:blue" if row["selected_replay_physical_success"] else "tab:red"
        for row in rows
    ]
    for index, (values, title, xlabel) in enumerate(series):
        axes[index].barh(y, values, color=colors, alpha=0.82)
        axes[index].set(
            title=title,
            xlabel=xlabel,
            yticks=y,
            yticklabels=labels if index == 0 else [],
        )
        axes[index].grid(axis="x", alpha=0.3)
    figure.suptitle(
        "Mug avoidance path optimization (blue = selected replay succeeded)",
        fontsize=15,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    histories = sorted(suite_dir.glob("trials/*/history.json"))
    if not histories:
        parser.error(f"No histories found under {suite_dir / 'trials'}")
    rows = [_load_trial(path) for path in histories]

    csv_path = suite_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = suite_dir / "summary.json"
    json_path.write_text(
        json.dumps({"trials": rows}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_path = suite_dir / "summary.png"
    _plot(rows, plot_path)
    print(f"trials:  {len(rows)}")
    print(f"table:   {csv_path}")
    print(f"summary: {json_path}")
    print(f"plot:    {plot_path}")


if __name__ == "__main__":
    main()
