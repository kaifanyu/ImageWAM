#!/usr/bin/env python
"""Visualize an SVGD latent-pull experiment from its ``history.json``.

This script is intentionally simulator- and model-free: it only reads the
diagnostics already saved by ``svgd_endpoint.py`` and writes two figures.

Example:

    python experiments/libero/plot_svgd_latent_pull.py \
        --history runs/empty_arm_preview/latent_pull_oracle_8x10/history.json

Outputs (beside history.json unless --out-dir is supplied):

    latent_pull_dashboard.png
    latent_pull_tracks.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_history(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_root = ("history", "actual_start_eef", "diagnostic_goal_eef")
    missing_root = [key for key in required_root if key not in payload]
    if missing_root:
        raise ValueError(
            f"{path} lacks detailed latent-pull fields {missing_root}. "
            "Plot a run produced by the current svgd_endpoint.py."
        )
    if not payload["history"]:
        raise ValueError(f"No population records in {path}")

    required_record = (
        "iteration",
        "energies",
        "goal_errors_m",
        "terminal_eefs",
        "terminal_diagnostics",
    )
    for index, record in enumerate(payload["history"]):
        missing = [key for key in required_record if key not in record]
        if missing:
            raise ValueError(f"history[{index}] lacks fields {missing}")
    return payload


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    """Make a path absolute while preserving writable workspace symlinks."""
    path = path.expanduser()
    logical_cwd = Path(os.environ.get("PWD", str(Path.cwd())))
    return path if path.is_absolute() else logical_cwd / path


def _arrays(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload["history"]
    result = {
        "iterations": np.asarray([record["iteration"] for record in records]),
        "energies": np.asarray([record["energies"] for record in records], dtype=float),
        "goal_errors": np.asarray(
            [record["goal_errors_m"] for record in records], dtype=float
        ),
        "positions": np.asarray(
            [record["terminal_eefs"] for record in records], dtype=float
        ),
        "start": np.asarray(payload["actual_start_eef"], dtype=float),
        "goal": np.asarray(payload["diagnostic_goal_eef"], dtype=float),
    }
    if result["start"].shape != (3,) or result["goal"].shape != (3,):
        raise ValueError(
            "actual_start_eef and diagnostic_goal_eef must each have shape [3], got "
            f"{result['start'].shape} and {result['goal'].shape}"
        )
    if result["positions"].ndim != 3 or result["positions"].shape[2] != 3:
        raise ValueError(
            "terminal_eefs must have shape [population, particle, xyz], got "
            f"{result['positions'].shape}"
        )
    expected = result["positions"].shape[:2]
    if result["energies"].shape != expected or result["goal_errors"].shape != expected:
        raise ValueError(
            "energies, goal_errors_m, and terminal_eefs disagree on population/particle "
            f"shape: {result['energies'].shape}, {result['goal_errors'].shape}, {expected}"
        )
    for name, values in result.items():
        if not isinstance(values, np.ndarray):
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values")
    metric_names = set.intersection(
        *[
            set(record.get("latent_metrics", {}).keys())
            for record in records
        ]
    )
    result["latent_metrics"] = {
        name: np.asarray(
            [record["latent_metrics"][name] for record in records], dtype=float
        )
        for name in sorted(metric_names)
    }
    return result


def _geometry(arrays: dict[str, Any]) -> dict[str, np.ndarray | float]:
    positions = arrays["positions"]
    start = arrays["start"]
    goal = arrays["goal"]
    centroids = positions.mean(axis=1)
    goal_vector = goal - start
    goal_distance = float(np.linalg.norm(goal_vector))
    if goal_distance <= 0.0:
        raise ValueError("Start and diagnostic goal EEF must be different")
    goal_unit = goal_vector / goal_distance
    displacement = centroids - start[None, :]
    progress_m = displacement @ goal_unit
    off_axis = displacement - progress_m[:, None] * goal_unit[None, :]
    return {
        "centroids": centroids,
        "goal_distance_m": goal_distance,
        "goal_unit": goal_unit,
        "progress_m": progress_m,
        "progress_fraction": progress_m / goal_distance,
        "off_axis_m": np.linalg.norm(off_axis, axis=1),
        "centroid_goal_error_m": np.linalg.norm(centroids - goal[None, :], axis=1),
    }


def _update_series(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    records = [record for record in payload["history"] if record.get("update_applied")]
    keys = (
        "latent_update_goal_projection_m",
        "repulsion_update_goal_projection_m",
        "applied_update_goal_projection_m",
    )
    return {
        "iterations": np.asarray([record["iteration"] for record in records]),
        **{
            key: np.asarray([record.get(key, 0.0) for record in records], dtype=float)
            for key in keys
        },
    }


def _plot_dashboard(
    payload: dict[str, Any],
    arrays: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray | float],
    output: Path,
    title: str,
    dpi: int,
) -> None:
    iterations = arrays["iterations"]
    energies = arrays["energies"]
    goal_errors_cm = 100.0 * arrays["goal_errors"]
    positions = arrays["positions"]
    start = arrays["start"]
    goal = arrays["goal"]
    centroids = np.asarray(geometry["centroids"])
    centroid_goal_error_cm = 100.0 * np.asarray(geometry["centroid_goal_error_m"])
    progress_cm = 100.0 * np.asarray(geometry["progress_m"])
    off_axis_cm = 100.0 * np.asarray(geometry["off_axis_m"])

    best_physical_index = int(np.argmin(centroid_goal_error_cm))
    best_latent_index = int(np.argmin(energies.mean(axis=1)))
    objective = payload.get("config", {}).get("latent_distance", "rms")
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    figure.suptitle(title, fontsize=16)

    axis = axes[0, 0]
    axis.fill_between(
        iterations, energies.min(axis=1), energies.max(axis=1), alpha=0.15, label="range"
    )
    axis.plot(iterations, energies.mean(axis=1), marker="o", label="particle mean")
    axis.plot(iterations, energies.min(axis=1), marker=".", label="best particle")
    axis.axvline(iterations[best_latent_index], color="tab:purple", linestyle=":",
                 label="lowest mean latent")
    axis.set(
        title="Feature objective",
        xlabel="evaluated population",
        ylabel=objective,
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    axis.plot(iterations, centroid_goal_error_cm, marker="s", label="centroid")
    axis.plot(iterations, goal_errors_cm.min(axis=1), marker=".", label="best particle")
    axis.axhline(100.0 * float(geometry["goal_distance_m"]), color="black", linestyle="--",
                 linewidth=1, label="start error")
    axis.axvline(iterations[best_physical_index], color="tab:green", linestyle=":",
                 label="closest centroid")
    axis.set(
        title="Physical endpoint error (diagnostic only)",
        xlabel="evaluated population",
        ylabel="distance to goal (cm)",
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[0, 2]
    axis.plot(iterations, progress_cm, marker="o", label="toward-goal progress")
    axis.plot(iterations, off_axis_cm, marker="s", label="off-axis drift")
    axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axis.set(
        title="Useful motion versus shortcut motion",
        xlabel="evaluated population",
        ylabel="centroid displacement (cm)",
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    displacement_cm = 100.0 * (centroids - start[None, :])
    for dimension, label in enumerate(("Δx", "Δy", "Δz")):
        axis.plot(iterations, displacement_cm[:, dimension], marker=".", label=label)
    desired_cm = 100.0 * (goal - start)
    desired_text = ", ".join(
        f"{label}={value:+.1f}" for label, value in zip(("x", "y", "z"), desired_cm)
    )
    axis.text(0.02, 0.03, f"desired Δ (cm): {desired_text}", transform=axis.transAxes,
              fontsize=8, va="bottom")
    axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axis.set(
        title="Centroid coordinate drift",
        xlabel="evaluated population",
        ylabel="change from start (cm)",
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    repeated_iterations = np.repeat(iterations, energies.shape[1])
    scatter = axis.scatter(
        goal_errors_cm.ravel(),
        energies.ravel(),
        c=repeated_iterations,
        cmap="viridis",
        s=28,
        alpha=0.8,
    )
    figure.colorbar(scatter, ax=axis, label="population")
    axis.set(
        title="Does lower feature distance mean physically closer?",
        xlabel="particle physical goal error (cm)",
        ylabel=objective,
    )
    axis.grid(alpha=0.3)

    axis = axes[1, 2]
    updates = _update_series(payload)
    for key, label in (
        ("latent_update_goal_projection_m", "latent attraction"),
        ("repulsion_update_goal_projection_m", "repulsion"),
        ("applied_update_goal_projection_m", "total applied"),
    ):
        axis.plot(updates["iterations"], 100.0 * updates[key], marker="o", label=label)
    axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axis.set(
        title="Why the swarm moved",
        xlabel="optimizer update",
        ylabel="mean projection toward goal (cm)",
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def _plot_tracks(
    arrays: dict[str, Any],
    geometry: dict[str, np.ndarray | float],
    output: Path,
    title: str,
    dpi: int,
) -> None:
    iterations = arrays["iterations"]
    positions = arrays["positions"]
    start = arrays["start"]
    goal = arrays["goal"]
    goal_unit = np.asarray(geometry["goal_unit"])
    goal_distance = float(geometry["goal_distance_m"])
    particle_progress = np.einsum(
        "tnd,d->tn", positions - start[None, None, :], goal_unit
    ) / goal_distance
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, positions.shape[1]))

    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    figure.suptitle(title, fontsize=16)
    for particle, color in enumerate(colors):
        axes[0].plot(positions[:, particle, 0], positions[:, particle, 1], marker=".",
                     color=color, alpha=0.8)
        axes[1].plot(positions[:, particle, 1], positions[:, particle, 2], marker=".",
                     color=color, alpha=0.8)
        axes[2].plot(iterations, 100.0 * particle_progress[:, particle], marker=".",
                     color=color, alpha=0.55)

    for axis, horizontal, vertical, xlabel, ylabel in (
        (axes[0], 0, 1, "EEF x (m)", "EEF y (m)"),
        (axes[1], 1, 2, "EEF y (m)", "EEF z (m)"),
    ):
        axis.scatter(start[horizontal], start[vertical], marker="*", s=180,
                     color="tab:blue", label="start", zorder=5)
        axis.scatter(goal[horizontal], goal[vertical], marker="X", s=130,
                     color="tab:orange", label="goal", zorder=5)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_title("Particle tracks: X/Y")
    axes[1].set_title("Particle tracks: Y/Z")

    axes[2].plot(
        iterations,
        100.0 * particle_progress.mean(axis=1),
        color="black",
        marker="o",
        linewidth=2.5,
        label="particle mean",
    )
    axes[2].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[2].axhline(100.0, color="tab:green", linewidth=1, alpha=0.5)
    axes[2].set(
        title="Progress along start→goal axis",
        xlabel="evaluated population",
        ylabel="goal-axis progress (%)",
    )
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def _plot_latent_metrics(
    payload: dict[str, Any],
    arrays: dict[str, Any],
    output: Path,
    title: str,
    dpi: int,
) -> bool:
    metric_arrays = arrays.get("latent_metrics", {})
    if not metric_arrays:
        return False
    iterations = arrays["iterations"]
    objective = payload.get("config", {}).get("latent_distance", "rms")
    figure, axes = plt.subplots(
        1,
        len(metric_arrays),
        figsize=(5.5 * len(metric_arrays), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(f"{title}\noptimized objective: {objective}", fontsize=14)
    for axis, (name, values) in zip(axes[0], metric_arrays.items()):
        axis.fill_between(
            iterations,
            values.min(axis=1),
            values.max(axis=1),
            alpha=0.15,
            label="particle range",
        )
        axis.plot(iterations, values.mean(axis=1), marker="o", label="mean")
        axis.plot(iterations, values.min(axis=1), marker=".", label="minimum")
        axis.set(
            title=name + (" (objective)" if name == objective else ""),
            xlabel="evaluated population",
            ylabel="latent distance",
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return True


def _plot_particle_rollouts(
    payload: dict[str, Any],
    history_dir: Path,
    output_dir: Path,
    title: str,
    dpi: int,
) -> int:
    records = payload["history"]
    trace_rows = [record.get("rollout_trace_files") for record in records]
    if not trace_rows or not trace_rows[0]:
        return 0
    particle_count = len(trace_rows[0])
    iterations = np.asarray([record["iteration"] for record in records])
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.95, max(len(records), 2))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for particle in range(particle_count):
        figure = plt.figure(figsize=(14, 9), constrained_layout=True)
        axes = [
            figure.add_subplot(2, 2, 1),
            figure.add_subplot(2, 2, 2),
            figure.add_subplot(2, 2, 3),
        ]
        axis_3d = figure.add_subplot(2, 2, 4, projection="3d")
        loaded = 0
        for record_index, record in enumerate(records):
            files = record.get("rollout_trace_files") or []
            if particle >= len(files) or files[particle] is None:
                continue
            trace_path = history_dir / files[particle]
            if not trace_path.is_file():
                raise FileNotFoundError(
                    f"Missing rollout trace referenced by history: {trace_path}"
                )
            with np.load(trace_path) as trace:
                actual = np.asarray(trace["eef_path"], dtype=float)
                desired = np.asarray(trace["desired_eefs"], dtype=float)
                target = np.asarray(trace["target_eef"], dtype=float)
            action_steps = np.arange(1, desired.shape[0] + 1)
            state_steps = np.arange(actual.shape[0])
            color = colors[record_index]
            label = f"population {record['iteration']}"
            for dimension, axis in enumerate(axes):
                axis.plot(
                    state_steps,
                    actual[:, dimension],
                    color=color,
                    linewidth=1.4,
                    label=label,
                )
                axis.plot(
                    action_steps,
                    desired[:, dimension],
                    color=color,
                    linewidth=0.8,
                    linestyle=":",
                    alpha=0.65,
                )
                axis.scatter(
                    [actual.shape[0] - 1],
                    [target[dimension]],
                    color=color,
                    marker="x",
                    s=18,
                )
            axis_3d.plot(
                actual[:, 0],
                actual[:, 1],
                actual[:, 2],
                color=color,
                linewidth=1.4,
                label=label,
            )
            axis_3d.scatter(
                target[0], target[1], target[2], color=color, marker="x", s=22
            )
            loaded += 1

        if loaded == 0:
            plt.close(figure)
            continue
        for axis, coordinate in zip(axes, ("X", "Y", "Z")):
            axis.set(
                title=f"{coordinate}: solid actual, dotted desired, x target",
                xlabel="controller action",
                ylabel=f"EEF {coordinate.lower()} (m)",
            )
            axis.grid(alpha=0.3)
        axis_3d.set(
            title="Actual 3D EEF paths",
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
        )
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(6, len(iterations)),
            fontsize=8,
        )
        figure.suptitle(f"{title}\nparticle {particle:02d}", fontsize=14)
        figure.savefig(output_dir / f"particle_{particle:02d}.png", dpi=dpi)
        plt.close(figure)
        written += 1
    return written


def _print_summary(
    arrays: dict[str, Any], geometry: dict[str, np.ndarray | float]
) -> None:
    iterations = arrays["iterations"]
    energies = arrays["energies"]
    centroid_error = np.asarray(geometry["centroid_goal_error_m"])
    progress_fraction = np.asarray(geometry["progress_fraction"])
    off_axis = np.asarray(geometry["off_axis_m"])
    best_physical = int(np.argmin(centroid_error))
    best_latent = int(np.argmin(energies.mean(axis=1)))
    print(f"populations: {len(iterations)}, particles: {energies.shape[1]}")
    print(f"latent mean: {energies[0].mean():.6f} -> {energies[-1].mean():.6f}")
    print(
        f"centroid goal error: {100 * centroid_error[0]:.2f} -> "
        f"{100 * centroid_error[-1]:.2f} cm"
    )
    print(
        f"best physical centroid: population {iterations[best_physical]}, "
        f"{100 * centroid_error[best_physical]:.2f} cm"
    )
    print(
        f"lowest mean latent: population {iterations[best_latent]}, "
        f"physical centroid error {100 * centroid_error[best_latent]:.2f} cm"
    )
    print(
        f"final goal-axis progress: {100 * progress_fraction[-1]:.2f}%  |  "
        f"off-axis drift: {100 * off_axis[-1]:.2f} cm"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        required=True,
        help="Path to an SVGD history.json or the run directory containing it.",
    )
    parser.add_argument("--out-dir", help="Defaults to the directory containing history.json.")
    parser.add_argument("--prefix", default="latent_pull", help="Output filename prefix.")
    parser.add_argument("--title", help="Optional figure title; defaults to the run directory name.")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    history_path = _absolute_without_resolving_symlinks(Path(args.history))
    if history_path.is_dir():
        history_path = history_path / "history.json"
    if not history_path.is_file():
        parser.error(f"History file not found: {history_path}")
    out_dir = (
        _absolute_without_resolving_symlinks(Path(args.out_dir))
        if args.out_dir
        else history_path.parent
    )
    title = args.title or f"SVGD latent-pull analysis: {history_path.parent.name}"
    payload = _load_history(history_path)
    arrays = _arrays(payload)
    geometry = _geometry(arrays)

    dashboard_path = out_dir / f"{args.prefix}_dashboard.png"
    tracks_path = out_dir / f"{args.prefix}_tracks.png"
    metrics_path = out_dir / f"{args.prefix}_latent_metrics.png"
    rollouts_dir = out_dir / f"{args.prefix}_particle_rollouts"
    _plot_dashboard(payload, arrays, geometry, dashboard_path, title, args.dpi)
    _plot_tracks(arrays, geometry, tracks_path, title, args.dpi)
    metrics_written = _plot_latent_metrics(
        payload, arrays, metrics_path, title, args.dpi
    )
    rollout_plots = _plot_particle_rollouts(
        payload, history_path.parent, rollouts_dir, title, args.dpi
    )
    _print_summary(arrays, geometry)
    print(f"dashboard: {dashboard_path}")
    print(f"tracks:    {tracks_path}")
    if metrics_written:
        print(f"metrics:   {metrics_path}")
    if rollout_plots:
        print(f"rollouts:  {rollouts_dir} ({rollout_plots} particle plots)")


if __name__ == "__main__":
    main()
