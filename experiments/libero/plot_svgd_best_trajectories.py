#!/usr/bin/env python
"""Plot the lowest-objective measured rollout from each trial in a suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


OBJECTIVE_ORDER = {"rms": 0, "cosine": 1, "token_cosine": 2}


def _load_best(history_path: Path) -> dict[str, Any]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    records = payload["history"]
    objective = payload["config"]["latent_distance"]
    objective_values = np.asarray(
        [record["latent_metrics"][objective] for record in records],
        dtype=np.float64,
    )
    flat_index = int(np.argmin(objective_values))
    record_index, particle_index = np.unravel_index(
        flat_index, objective_values.shape
    )
    record = records[record_index]
    trace_relative = record["rollout_trace_files"][particle_index]
    if trace_relative is None:
        raise ValueError(
            f"Best particle has no saved rollout trace in {history_path}"
        )
    trace_path = history_path.parent / trace_relative
    with np.load(trace_path) as trace:
        target = np.asarray(trace["target_eef"], dtype=np.float64)
        terminal = np.asarray(trace["terminal_eef"], dtype=np.float64)
        actual_path = np.asarray(trace["eef_path"], dtype=np.float64)
        desired_path = np.asarray(trace["desired_eefs"], dtype=np.float64)

    start = np.asarray(payload["actual_start_eef"], dtype=np.float64)
    goal = np.asarray(payload["diagnostic_goal_eef"], dtype=np.float64)
    return {
        "trial": history_path.parent.name,
        "objective": objective,
        "completed_populations": len(records),
        "last_completed_iteration": int(records[-1]["iteration"]),
        "selection_iteration": int(record["iteration"]),
        "selection_particle": int(particle_index),
        "objective_value": float(objective_values[record_index, particle_index]),
        "physical_goal_error_m": float(
            record["goal_errors_m"][particle_index]
        ),
        "target_tracking_error_m": float(
            record["target_tracking_errors_m"][particle_index]
        ),
        "start_eef": start,
        "goal_eef": goal,
        "target_eef": target,
        "terminal_eef": terminal,
        "actual_path": actual_path,
        "desired_path": desired_path,
        "trace_path": str(trace_path),
    }


def _jsonable(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
        if key not in {"actual_path", "desired_path"}
    }


def _plot(results: list[dict[str, Any]], output: Path, dpi: int) -> None:
    columns = min(3, len(results))
    rows = math.ceil(len(results) / columns)
    figure = plt.figure(
        figsize=(6.2 * columns, 5.4 * rows), constrained_layout=True
    )
    figure.suptitle(
        "Lowest-objective measured EEF rollout from each trial",
        fontsize=15,
    )

    for index, result in enumerate(results):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        actual = result["actual_path"]
        desired = result["desired_path"]
        start = result["start_eef"]
        goal = result["goal_eef"]
        target = result["target_eef"]
        terminal = result["terminal_eef"]

        axis.plot(
            desired[:, 0],
            desired[:, 1],
            desired[:, 2],
            linestyle="--",
            color="0.55",
            linewidth=1.2,
            label="minimum-jerk desired",
        )
        axis.plot(
            actual[:, 0],
            actual[:, 1],
            actual[:, 2],
            color="tab:blue",
            linewidth=2.2,
            label="actual EEF",
        )
        axis.scatter(
            actual[:, 0],
            actual[:, 1],
            actual[:, 2],
            c=np.arange(actual.shape[0]),
            cmap="viridis",
            s=16,
            alpha=0.85,
        )
        axis.scatter(*start, marker="*", s=180, color="tab:blue", label="start")
        axis.scatter(*goal, marker="X", s=130, color="tab:green", label="goal")
        axis.scatter(
            *target, marker="D", s=70, color="tab:orange", label="particle target"
        )
        axis.scatter(
            *terminal, marker="o", s=65, color="tab:red", label="actual terminal"
        )
        axis.set(
            xlabel="EEF X (m)",
            ylabel="EEF Y (m)",
            zlabel="EEF Z (m)",
            title=(
                f"{result['objective']}: population "
                f"{result['selection_iteration']}, particle "
                f"{result['selection_particle']:02d}\n"
                f"E={result['objective_value']:.6f}, "
                f"goal error={1000 * result['physical_goal_error_m']:.1f} mm"
            ),
        )
        axis.view_init(elev=24, azim=-58)
        axis.set_box_aspect((1.0, 2.2, 1.0))
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend(loc="upper left", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).expanduser().absolute()
    output = Path(args.output).expanduser().absolute()
    histories = list(suite_dir.glob("trials/*/history.json"))
    if not histories:
        histories = list(suite_dir.glob("*/history.json"))
    histories = sorted(
        histories,
        key=lambda path: (
            OBJECTIVE_ORDER.get(path.parent.name, 99),
            path.parent.name,
        ),
    )
    if not histories:
        parser.error(f"No trial histories found under {suite_dir}")
    results = [_load_best(path) for path in histories]
    _plot(results, output, args.dpi)

    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {"selection_rule": "lowest measured objective", "trials": [
                _jsonable(result) for result in results
            ]},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"plot:    {output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
