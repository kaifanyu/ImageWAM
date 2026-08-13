#!/usr/bin/env python
"""Re-render an existing run's start and goal poses with the side-profile camera.

Scene prep is expensive and its outputs are what everything downstream compares
against, so this exists to answer "what would the two-view goal image look like?"
without rebuilding a run.  It restores the run's ``start_state.npy`` and reaches the
goal pose the way the run did -- from the oracle candidate's stored
``terminal_state.npy`` when it is still on disk, otherwise by replaying the
oracle rollout, which is deterministic from the same snapshot -- then renders
both through the current camera settings.

    python -u -B experiments/libero/preview_side_camera.py \
      --run-dir runs/empty_arm_preview --out-dir runs/empty_arm_preview/side_preview

Writes ``start.png`` and ``goal.png`` (both ``[agentview | side profile]``), the
individual panels, and ``preview.png`` stacking the two so the motion is legible
in one glance.  Nothing in the run directory is overwritten unless ``--out-dir``
points back into it.

``--elevation-deg`` / ``--margin`` / ``--height`` / ``--x-center`` sweep the
framing; whatever looks right here is what ``sample_endpoint_trajectories.py``
should be given as ``--side-camera-*``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(Path(__file__).resolve().parent), str(REPO_ROOT / "third_party" / "LIBERO")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import side_camera  # noqa: E402
from sample_endpoint_trajectories import (  # noqa: E402
    _rollout_to_target,
    _synchronize_controllers_to_sim_state,
    _views_from_obs,
    set_composed_right_view,
)


def _oracle(run_dir: Path, manifest: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    """How to reach the run's goal pose: a stored snapshot, or a rollout to replay.

    Candidate directories are large and routinely deleted once a run has been
    summarised, so the snapshot is preferred but never required.
    """
    for candidate in manifest.get("candidates", []):
        if candidate.get("kind") != "oracle":
            continue
        state = run_dir / str(candidate["id"]) / "terminal_state.npy"
        if state.exists():
            return state, None
        return None, {
            "target": np.asarray(candidate["target_eef"], dtype=np.float64),
            "arc_height": float(candidate.get("arc_height_m", 0.0)),
            "midpoint_x": float(candidate.get("midpoint_x_m", 0.0)),
        }
    # The push/obstacle pipelines write a single goal state instead of a pool.
    goal_state = run_dir / "goal_state.npy"
    if goal_state.exists():
        return goal_state, None
    if "physical_goal_eef" in manifest:
        return None, {
            "target": np.asarray(manifest["physical_goal_eef"], dtype=np.float64),
            "arc_height": 0.0,
            "midpoint_x": 0.0,
        }
    return None, None


def _labelled(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 22), "white")
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((6, 6), label, fill="black")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", help="Defaults to <run-dir>/side_preview.")
    parser.add_argument("--margin", type=float, default=side_camera.DEFAULT_MARGIN)
    parser.add_argument("--height", type=float, default=side_camera.DEFAULT_HEIGHT)
    parser.add_argument("--elevation-deg", type=float, default=side_camera.DEFAULT_ELEVATION_DEG)
    parser.add_argument("--x-center", type=float, default=None)
    parser.add_argument("--view-size", type=int, default=None, help="Defaults to the run's view_size.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "side_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    view_size = int(args.view_size or manifest["view_size"])
    start_state = np.load(run_dir / manifest.get("start_state", "start_state.npy"))
    gripper_actions = [
        np.asarray(action) for action in manifest.get("start_gripper_controller_actions", [])
    ] or None

    goal_state_path, oracle_rollout = _oracle(run_dir, manifest)
    if goal_state_path is None and oracle_rollout is None:
        parser.error(f"{run_dir} records neither a goal snapshot nor a goal pose to roll out to")

    set_composed_right_view("sideview")
    env = side_camera.open_env(
        bddl_file_name=str(manifest["bddl"]),
        render_size=int(manifest["render_size"]),
    )
    # Re-install so the sweep flags override whatever open_env derived.
    pose = side_camera.install_side_camera(
        env,
        margin=args.margin,
        height=args.height,
        elevation_deg=args.elevation_deg,
        x_center=args.x_center,
    )
    print(
        f"[side] position={np.round(pose['position'], 3).tolist()} "
        f"target={np.round(pose['target'], 3).tolist()} "
        f"table_top_z={pose['table_top_z']:.3f}"
    )

    panels: list[Image.Image] = []
    try:
        env.seed(int(manifest.get("sim_seed", 0)))
        for name in ("start", "goal"):
            env.reset()
            state = (
                np.load(goal_state_path)
                if name == "goal" and goal_state_path is not None
                else start_state
            )
            obs = env.set_init_state(np.asarray(state, dtype=np.float64))
            _synchronize_controllers_to_sim_state(env, gripper_actions)
            if name == "goal" and goal_state_path is None:
                obs, _, _, _ = _rollout_to_target(
                    env,
                    obs,
                    oracle_rollout["target"],
                    move_steps=int(manifest["move_steps"]),
                    settle_steps=int(manifest["settle_steps"]),
                    gain=float(manifest["controller_gain"]),
                    arc_height=oracle_rollout["arc_height"],
                    midpoint_x=oracle_rollout["midpoint_x"],
                    view_size=view_size,
                )
            main_view, side_view, composed = _views_from_obs(obs, view_size)
            main_view.save(out_dir / f"{name}_agentview.png")
            side_view.save(out_dir / f"{name}_sideview.png")
            composed.save(out_dir / f"{name}.png")
            eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            print(f"[{name}] eef={np.round(eef, 3).tolist()}")
            panels.append(_labelled(composed, f"{name}   agentview | sideview   eef={np.round(eef, 3).tolist()}"))
    finally:
        env.close()

    preview = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), "white")
    offset = 0
    for panel in panels:
        preview.paste(panel, (0, offset))
        offset += panel.height
    preview.save(out_dir / "preview.png")
    _write_pose(out_dir / "side_camera.json", pose)
    print(f"\n[done] {out_dir / 'preview.png'}")


def _write_pose(path: Path, pose: dict[str, Any]) -> None:
    path.write_text(json.dumps(pose, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
