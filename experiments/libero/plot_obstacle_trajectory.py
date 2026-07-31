#!/usr/bin/env python
"""Plot EEF motion and tracked-object motion for an endpoint candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _select_candidate(manifest: dict[str, Any], selector: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in manifest["candidates"]
        if candidate["id"] == selector or candidate["kind"] == selector
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Candidate selector {selector!r} matched {len(matches)} records"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--candidate",
        default="oracle",
        help="Candidate id or unique kind; defaults to oracle.",
    )
    parser.add_argument("--output")
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().absolute()
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    candidate = _select_candidate(manifest, args.candidate)
    candidate_dir = run_dir / candidate["id"]
    object_trace_path = candidate.get("object_trace")
    if not object_trace_path:
        parser.error(f"{candidate['id']} does not contain a tracked object")

    eef = np.asarray(
        np.load(candidate_dir / "eef_path.npy"), dtype=np.float64
    )
    with np.load(run_dir / object_trace_path) as trace:
        names = [str(name) for name in trace["object_names"]]
        positions = np.asarray(trace["positions"], dtype=np.float64)
    if positions.shape[0] != eef.shape[0]:
        raise ValueError("EEF and object traces have different lengths")

    output = (
        Path(args.output).expanduser().absolute()
        if args.output
        else run_dir / f"{candidate['id']}_obstacle_analysis.png"
    )
    figure = plt.figure(figsize=(17, 10), constrained_layout=True)
    spatial = figure.add_subplot(2, 2, 1, projection="3d")
    top_down = figure.add_subplot(2, 2, 2)
    distance_axis = figure.add_subplot(2, 2, 3)
    motion_axis = figure.add_subplot(2, 2, 4)
    colors = plt.get_cmap("tab10")
    steps = np.arange(eef.shape[0])

    spatial.plot(
        eef[:, 0],
        eef[:, 1],
        eef[:, 2],
        color="tab:blue",
        linewidth=2.2,
        label="actual EEF",
    )
    top_down.plot(
        eef[:, 0],
        eef[:, 1],
        color="tab:blue",
        linewidth=2.2,
        label="actual EEF",
    )
    spatial.scatter(
        *eef[0], color="tab:blue", marker="*", s=150, label="EEF start"
    )
    spatial.scatter(
        *eef[-1], color="tab:green", marker="X", s=110, label="EEF terminal"
    )
    top_down.scatter(
        eef[0, 0], eef[0, 1], color="tab:blue", marker="*", s=150
    )
    top_down.scatter(
        eef[-1, 0], eef[-1, 1], color="tab:green", marker="X", s=110
    )

    result: dict[str, Any] = {
        "candidate": candidate["id"],
        "kind": candidate["kind"],
        "eef_start": eef[0],
        "eef_terminal": eef[-1],
        "objects": {},
    }
    for index, name in enumerate(names):
        color = colors(index + 3)
        path = positions[:, index]
        center_distance = np.linalg.norm(eef - path, axis=1)
        xy_distance = np.linalg.norm(eef[:, :2] - path[:, :2], axis=1)
        displacement = np.linalg.norm(path - path[0], axis=1)
        spatial.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            linewidth=2.0,
            label=f"{name} center",
        )
        spatial.scatter(*path[0], color=color, marker="D", s=80)
        top_down.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=2.0,
            label=f"{name} center",
        )
        top_down.scatter(path[0, 0], path[0, 1], color=color, marker="D", s=80)
        distance_axis.plot(
            steps,
            100.0 * center_distance,
            color=color,
            label="3D center distance",
        )
        distance_axis.plot(
            steps,
            100.0 * xy_distance,
            color=color,
            linestyle="--",
            label="XY center distance",
        )
        displacement_mm = 1000.0 * displacement
        if float(displacement_mm.max()) < 1e-6:
            displacement_mm = np.zeros_like(displacement_mm)
        motion_axis.plot(steps, displacement_mm, color=color, label=name)
        closest = int(np.argmin(center_distance))
        result["objects"][name] = {
            "initial_position": path[0],
            "terminal_position": path[-1],
            "terminal_displacement_m": float(displacement[-1]),
            "maximum_displacement_m": float(displacement.max()),
            "minimum_eef_center_distance_m": float(center_distance.min()),
            "minimum_eef_xy_distance_m": float(xy_distance.min()),
            "closest_eef_state_index": closest,
            "eef_at_closest_state": eef[closest],
        }

    spatial.set(
        title="EEF and object paths in world coordinates",
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
    )
    spatial.view_init(elev=23, azim=-58)
    spatial.legend(fontsize=8)
    top_down.set(
        title="Top-down path: obstacle placement",
        xlabel="X (m)",
        ylabel="Y (m)",
    )
    all_x = np.concatenate([eef[:, 0], positions[:, :, 0].reshape(-1)])
    all_y = np.concatenate([eef[:, 1], positions[:, :, 1].reshape(-1)])
    x_padding = max(0.03, 0.08 * float(np.ptp(all_x)))
    y_padding = max(0.03, 0.08 * float(np.ptp(all_y)))
    top_down.set_xlim(float(all_x.min() - x_padding), float(all_x.max() + x_padding))
    top_down.set_ylim(float(all_y.min() - y_padding), float(all_y.max() + y_padding))
    top_down.set_aspect("auto")
    top_down.legend(fontsize=8)
    distance_axis.set(
        title="EEF distance to tracked object center",
        xlabel="controller state index",
        ylabel="distance (cm)",
    )
    distance_axis.legend(fontsize=8)
    motion_axis.set(
        title="Object displacement from its initial pose",
        xlabel="controller state index",
        ylabel="displacement (mm)",
    )
    if all(
        item["maximum_displacement_m"] < 1e-9
        for item in result["objects"].values()
    ):
        motion_axis.text(
            0.5,
            0.5,
            "No measurable object movement",
            ha="center",
            va="center",
            transform=motion_axis.transAxes,
        )
        motion_axis.set_ylim(-0.05, 1.0)
    motion_axis.legend(fontsize=8)
    for axis in (top_down, distance_axis, motion_axis):
        axis.grid(alpha=0.3)
    figure.suptitle(
        f"{candidate['id']} ({candidate['kind']}): EEF/object interaction analysis",
        fontsize=15,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"plot:    {output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
