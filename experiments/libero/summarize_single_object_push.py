#!/usr/bin/env python
"""Summarize image-only push trials using held-out object-pose diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_events(path: Path, object_name: str) -> dict[tuple[int, int], dict[str, Any]]:
    events: dict[tuple[int, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("evaluation") != "base":
            continue
        motion = (event.get("object_motion") or {}).get(object_name)
        if motion and motion.get("terminal_position") is not None:
            events[(int(event["iteration"]), int(event["particle"]))] = event
    return events


def _diagnostics(
    event: dict[str, Any], object_name: str, start: np.ndarray, goal: np.ndarray
) -> dict[str, float]:
    terminal = np.asarray(
        event["object_motion"][object_name]["terminal_position"], dtype=np.float64
    )
    axis = goal - start
    distance = float(np.linalg.norm(axis))
    unit = axis / max(distance, 1e-12)
    offset = terminal - start
    projection = float(offset @ unit)
    lateral = float(np.linalg.norm(offset - projection * unit))
    return {
        "object_goal_error_m": float(np.linalg.norm(terminal - goal)),
        "object_goal_axis_fraction": projection / max(distance, 1e-12),
        "object_lateral_error_m": lateral,
        "object_displacement_m": float(np.linalg.norm(offset)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--success-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).expanduser().resolve()
    trial_root = suite_dir / "trials"
    trial_dirs = sorted(path.parent for path in trial_root.glob("*/history.json"))
    if not trial_dirs:
        parser.error(f"No completed trial histories under {trial_root}")

    first_history = json.loads(
        (trial_dirs[0] / "history.json").read_text(encoding="utf-8")
    )
    run_dir = Path(first_history["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    tracked = manifest["tracked_objects"]
    if len(tracked) != 1:
        raise RuntimeError(f"Expected one tracked object, found {list(tracked)}")
    object_name, object_spec = next(iter(tracked.items()))
    start = np.asarray(object_spec["start_position"], dtype=np.float64)
    goal = np.asarray(object_spec["goal_position"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for trial_dir in trial_dirs:
        history = json.loads((trial_dir / "history.json").read_text(encoding="utf-8"))
        metadata = json.loads(
            (trial_dir / "best_metadata.json").read_text(encoding="utf-8")
        )
        events = _load_events(trial_dir / "evaluations.jsonl", object_name)
        selection = metadata["selection"]
        selected_key = (int(selection["iteration"]), int(selection["particle"]))
        selected = _diagnostics(events[selected_key], object_name, start, goal)
        final_iteration = int(history["history"][-1]["iteration"])
        final = [
            _diagnostics(event, object_name, start, goal)
            for (iteration, _), event in events.items()
            if iteration == final_iteration
        ]
        final_errors = np.asarray(
            [record["object_goal_error_m"] for record in final], dtype=np.float64
        )
        row = {
            "trial": trial_dir.name,
            "feature_encoder": history["config"]["feature_encoder"],
            "gradient_source": history.get("gradient_source", "finite_difference"),
            "latent_views": history["config"]["latent_views"],
            "objective": history["config"]["latent_distance"],
            "selected_iteration": selected_key[0],
            "selected_particle": selected_key[1],
            "selected_image_energy": float(selection["energy"]),
            **{f"selected_{key}": value for key, value in selected.items()},
            "final_object_goal_error_min_m": float(final_errors.min()),
            "final_object_goal_error_mean_m": float(final_errors.mean()),
            "final_object_success_fraction": float(
                np.mean(final_errors <= args.success_tolerance)
            ),
            "success_tolerance_m": args.success_tolerance,
        }
        rows.append(row)

    (suite_dir / "object_push_summary.json").write_text(
        json.dumps(
            {
                "object_name": object_name,
                "object_start_position": start.tolist(),
                "object_goal_position": goal.tolist(),
                "goal_is_optimizer_input": False,
                "trials": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (suite_dir / "object_push_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = (
        "trial",
        "latent_views",
        "selected_image_energy",
        "selected_object_goal_error_m",
        "selected_object_goal_axis_fraction",
        "final_object_success_fraction",
    )
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row[column]) for column in columns))
    print(f"[done] {suite_dir / 'object_push_summary.json'}")


if __name__ == "__main__":
    main()
