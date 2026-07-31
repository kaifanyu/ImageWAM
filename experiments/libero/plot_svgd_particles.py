#!/usr/bin/env python
"""Plot SVGD endpoint search in 3D.

Two different paths live in the same coordinate space and are easy to conflate:

1. the *optimizer* path -- how a particle's candidate terminal position ``theta``
   drifts across iterations.  One point per iteration per particle, read from
   ``history.json``.
2. the *physical* path -- the arm's actual swing from the start state to one
   ``theta``, produced by the controller.  ``move_steps + settle_steps`` actions
   and one more end-effector sample than that.  Only the global best rollout is
   saved by ``svgd_endpoint.py``, as ``best_eef_path.npy``.

The left panel shows (1) for every particle, the right panel shows (2) for the
winner with its own optimizer path overlaid for comparison.

Example:

    python experiments/libero/plot_svgd_particles.py \
        --svgd-dir runs/empty_arm_preview/svgd_half_agentview \
        --goal-eef -0.0488 0.0011 1.0320
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _particle_paths(history: list[dict]) -> np.ndarray:
    """Stack per-iteration particle positions into ``(steps, particles, 3)``."""
    steps = [np.asarray(record["particles_before_update"], dtype=np.float64)
             for record in history]
    final = history[-1].get("particles_after_update")
    if final is not None:
        final_array = np.asarray(final, dtype=np.float64)
        if not np.array_equal(final_array, steps[-1]):
            steps.append(final_array)
    stacked = np.stack(steps)
    if stacked.ndim != 3 or stacked.shape[2] != 3:
        raise ValueError(f"Unexpected particle path shape: {stacked.shape}")
    return stacked


def _energies(history: list[dict], steps: int, particles: int) -> np.ndarray:
    values = np.full((steps, particles), np.nan)
    for index, record in enumerate(history[:steps]):
        recorded = record.get("energies")
        if recorded is not None:
            values[index, : len(recorded)] = np.asarray(recorded, dtype=np.float64)
    return values


def _set_equal_aspect(axis, points: np.ndarray) -> None:
    """Equal-length axes so path geometry is not visually distorted."""
    centre = points.reshape(-1, 3).mean(axis=0)
    span = float(np.max(np.ptp(points.reshape(-1, 3), axis=0))) or 1.0
    half = 0.55 * span
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--svgd-dir", required=True,
                        help="An svgd_endpoint.py output directory containing history.json.")
    parser.add_argument("--out", help="Defaults to <svgd-dir>/particle_paths_3d.png.")
    parser.add_argument("--goal-eef", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Physical goal marker; defaults to history diagnostic_goal_eef.")
    parser.add_argument("--show-terminal", action="store_true",
                        help="Also mark where the arm actually stopped, not just the request.")
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    svgd_dir = Path(args.svgd_dir).resolve()
    history_path = svgd_dir / "history.json"
    if not history_path.is_file():
        parser.error(f"history.json not found: {history_path}")
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload["history"]
    if not history:
        parser.error("history.json contains no iterations")

    paths = _particle_paths(history)
    steps, particle_count = paths.shape[0], paths.shape[1]
    energies = _energies(history, steps, particle_count)
    start = np.asarray(payload["actual_start_eef"], dtype=np.float64)
    goal = np.asarray(
        args.goal_eef if args.goal_eef is not None else payload["diagnostic_goal_eef"],
        dtype=np.float64,
    )
    views = (payload.get("config") or {}).get("latent_views", "both")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15, 7))
    figure.suptitle(
        f"SVGD endpoint search  |  {particle_count} particles x {steps - 1} updates  "
        f"|  latent_views={views}",
        fontsize=14,
    )
    colours = plt.get_cmap("tab10")

    axis = figure.add_subplot(1, 2, 1, projection="3d")
    for index in range(particle_count):
        track = paths[:, index]
        colour = colours(index % 10)
        axis.plot(track[:, 0], track[:, 1], track[:, 2], "-", color=colour,
                  linewidth=1.4, alpha=0.85, label=f"particle {index}")
        axis.scatter(*track[0], color=colour, marker="o", s=26, alpha=0.9)
        axis.scatter(*track[-1], color=colour, marker="*", s=170,
                     edgecolors="black", linewidths=0.4)
    if args.show_terminal:
        reached = np.asarray(history[-1]["terminal_eefs"], dtype=np.float64)
        axis.scatter(reached[:, 0], reached[:, 1], reached[:, 2], color="black",
                     marker="x", s=34, label="arm actually stopped")
    axis.scatter(*start, color="tab:green", marker="^", s=150,
                 edgecolors="black", linewidths=0.5, label="start (left)")
    axis.scatter(*goal, color="tab:red", marker="X", s=170,
                 edgecolors="black", linewidths=0.5, label="physical goal")
    axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
             title="Optimizer path: candidate endpoint per iteration\n"
                   "circle = start of search, star = final")
    _set_equal_aspect(axis, np.concatenate([paths.reshape(-1, 3), start[None], goal[None]]))
    axis.view_init(elev=args.elev, azim=args.azim)
    axis.legend(loc="upper left", fontsize=7)

    axis = figure.add_subplot(1, 2, 2, projection="3d")
    rollout_path = svgd_dir / "best_eef_path.npy"
    extras = [start[None], goal[None]]
    if rollout_path.is_file():
        rollout = np.load(rollout_path)
        extras.append(rollout)
        segments = np.stack([rollout[:-1], rollout[1:]], axis=1)
        shades = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, len(segments)))
        for segment, shade in zip(segments, shades):
            axis.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                      color=shade, linewidth=2.4)
        axis.scatter(*rollout[0], color="tab:green", marker="^", s=150,
                     edgecolors="black", linewidths=0.5)
        axis.scatter(*rollout[-1], color="tab:blue", marker="*", s=190,
                     edgecolors="black", linewidths=0.5, label="arm stopped here")
        subtitle = f"{len(rollout) - 1} actions, {len(rollout)} end-effector samples"
    else:
        subtitle = "best_eef_path.npy not found"
    best_particle = (payload.get("best_replay") or {}).get("particle")
    if best_particle is not None and best_particle < particle_count:
        track = paths[:, best_particle]
        axis.plot(track[:, 0], track[:, 1], track[:, 2], "--", color="tab:orange",
                  linewidth=1.3, label=f"optimizer path (particle {best_particle})")
    axis.scatter(*goal, color="tab:red", marker="X", s=170,
                 edgecolors="black", linewidths=0.5, label="physical goal")
    axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
             title=f"Physical rollout of the winning endpoint\n{subtitle}")
    _set_equal_aspect(axis, np.concatenate(extras))
    axis.view_init(elev=args.elev, azim=args.azim)
    axis.legend(loc="upper left", fontsize=7)

    output = Path(args.out).resolve() if args.out else svgd_dir / "particle_paths_3d.png"
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
