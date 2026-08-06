#!/usr/bin/env python
"""Prepare a deterministic one-object push scene and its oracle goal image.

The optimizer receives only ``goal_oracle.png``.  Object and end-effector poses
are saved as held-out diagnostics; they are not used by the image objective.
The nominal action family is a straight minimum-jerk Cartesian push with the
gripper held open, matching the endpoint experiments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (Path(__file__).resolve().parent, REPO_ROOT / "third_party" / "LIBERO"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from sample_endpoint_trajectories import (  # noqa: E402
    _object_motion_summary,
    _rollout_to_target,
    _synchronize_controllers_to_sim_state,
    _tracked_object_poses,
    _views_from_obs,
    _write_json,
    _write_video,
)


def _save_trace(path: Path, trace: dict[str, list[Any]]) -> None:
    np.savez_compressed(
        path,
        object_names=np.asarray(trace["tracked_object_names"], dtype=str),
        object_positions=np.asarray(trace["object_positions"], dtype=np.float64),
        object_quaternions_wxyz=np.asarray(
            trace["object_quaternions_wxyz"], dtype=np.float64
        ),
        desired_eefs=np.asarray(trace["desired_eefs"], dtype=np.float64),
        eef_before_actions=np.asarray(
            trace["eef_before_actions"], dtype=np.float64
        ),
        position_errors=np.asarray(trace["position_errors"], dtype=np.float64),
        normalized_times=np.asarray(trace["normalized_times"], dtype=np.float64),
        minimum_jerk_progress=np.asarray(
            trace["minimum_jerk_progress"], dtype=np.float64
        ),
        phases=np.asarray(trace["phases"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", default=str(REPO_ROOT / "runs" / "single_object_push_preview")
    )
    parser.add_argument(
        "--bddl",
        default=str(REPO_ROOT / "experiments" / "libero" / "single_object_push.bddl"),
    )
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--stage-offset-y", type=float, default=0.070)
    parser.add_argument("--push-target-offset-y", type=float, default=-0.1065)
    parser.add_argument("--push-height", type=float, default=0.925)
    parser.add_argument("--stage-steps", type=int, default=40)
    parser.add_argument("--move-steps", type=int, default=48)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--controller-gain", type=float, default=12.0)
    parser.add_argument("--minimum-object-displacement", type=float, default=0.05)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    bddl = Path(args.bddl).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if not bddl.is_file():
        parser.error(f"Missing BDDL file: {bddl}")
    if run_dir.exists():
        if not args.force:
            parser.error(f"Output exists: {run_dir}. Pass --force to replace it.")
        if not manifest_path.is_file():
            parser.error(
                f"Refusing to replace {run_dir}: it has no owned pipeline manifest"
            )
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parser.error(f"Refusing to replace malformed manifest: {manifest_path}")
        if prior.get("pipeline") != "single_object_push":
            parser.error(f"Refusing to replace a different pipeline at {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.render_size,
        camera_widths=args.render_size,
    )
    try:
        env.seed(args.sim_seed)
        obs = env.reset()
        object_names, reset_positions, _ = _tracked_object_poses(env)
        if len(object_names) != 1:
            raise RuntimeError(f"Expected exactly one tracked object, found {object_names}")

        # Align the gripper with the realized seeded object x coordinate.  BDDL
        # regions are fixture-local, so the corresponding world x is not the
        # literal number written in the region.
        object_reset = reset_positions[0]
        stage_target = np.asarray(
            [
                object_reset[0],
                object_reset[1] + args.stage_offset_y,
                args.push_height,
            ],
            dtype=np.float64,
        )
        obs, _, _, _ = _rollout_to_target(
            env,
            obs,
            stage_target,
            move_steps=args.stage_steps,
            settle_steps=args.settle_steps,
            gain=args.controller_gain,
            view_size=args.view_size,
        )
        _synchronize_controllers_to_sim_state(env)
        start_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        start_gripper_actions = [
            np.asarray(robot.gripper.current_action).copy() for robot in env.robots
        ]
        start_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        object_names, start_object_positions, start_object_quaternions = (
            _tracked_object_poses(env)
        )
        start_main, start_wrist, start_image = _views_from_obs(obs, args.view_size)
        start_main.save(run_dir / "start_agentview.png")
        start_wrist.save(run_dir / "start_wrist.png")
        start_image.save(run_dir / "start.png")
        np.save(run_dir / "start_state.npy", start_state)

        push_target = np.asarray(
            [
                start_object_positions[0, 0],
                start_object_positions[0, 1] + args.push_target_offset_y,
                args.push_height,
            ],
            dtype=np.float64,
        )
        trace: dict[str, list[Any]] = {}
        obs, actions, eef_path, frames = _rollout_to_target(
            env,
            obs,
            push_target,
            move_steps=args.move_steps,
            settle_steps=args.settle_steps,
            gain=args.controller_gain,
            capture_video=True,
            video_stride=args.video_stride,
            view_size=args.view_size,
            trace=trace,
        )
        goal_main, goal_wrist, goal_image = _views_from_obs(obs, args.view_size)
        goal_main.save(run_dir / "goal_oracle_agentview.png")
        goal_wrist.save(run_dir / "goal_oracle_wrist.png")
        goal_image.save(run_dir / "goal_oracle.png")
        np.save(run_dir / "oracle_actions.npy", actions.astype(np.float32))
        np.save(run_dir / "oracle_eef_path.npy", eef_path.astype(np.float32))
        np.save(run_dir / "goal_state.npy", np.asarray(env.get_sim_state()))
        _save_trace(run_dir / "oracle_trace.npz", trace)
        _write_video(run_dir / "oracle_rollout.mp4", frames, args.video_fps)

        motion = _object_motion_summary(trace, eef_path)[object_names[0]]
        goal_object_positions = np.asarray(
            trace["object_positions"][-1], dtype=np.float64
        )
        goal_object_quaternions = np.asarray(
            trace["object_quaternions_wxyz"][-1], dtype=np.float64
        )
        if motion["terminal_displacement_m"] < args.minimum_object_displacement:
            raise RuntimeError(
                "Oracle push did not move the object far enough: "
                f"{motion['terminal_displacement_m']:.4f} m < "
                f"{args.minimum_object_displacement:.4f} m"
            )

        manifest = {
            "schema_version": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pipeline": "single_object_push",
            "bddl": str(bddl),
            "sim_seed": args.sim_seed,
            "render_size": args.render_size,
            "view_size": args.view_size,
            "start_state": "start_state.npy",
            "start_image": "start.png",
            "goal_oracle_image": "goal_oracle.png",
            "actual_start_eef": start_eef,
            "physical_goal_eef": push_target,
            "start_gripper_controller_actions": start_gripper_actions,
            "move_steps": args.move_steps,
            "settle_steps": args.settle_steps,
            "controller_gain": args.controller_gain,
            "video_stride": args.video_stride,
            "video_fps": args.video_fps,
            "action_parameterization": {
                "kind": "minimum_jerk_cartesian_endpoint",
                "fixed_arc_height_m": 0.0,
                "fixed_midpoint_x_m": 0.0,
                "gripper_action": -1.0,
            },
            "tracked_objects": {
                object_names[0]: {
                    "start_position": start_object_positions[0],
                    "start_quaternion_wxyz": start_object_quaternions[0],
                    "goal_position": goal_object_positions[0],
                    "goal_quaternion_wxyz": goal_object_quaternions[0],
                    "oracle_motion": motion,
                }
            },
            "diagnostic_object_goal_is_optimizer_input": False,
            "artifacts": {
                "start_agentview": "start_agentview.png",
                "start_wrist": "start_wrist.png",
                "goal_agentview": "goal_oracle_agentview.png",
                "goal_wrist": "goal_oracle_wrist.png",
                "oracle_actions": "oracle_actions.npy",
                "oracle_eef_path": "oracle_eef_path.npy",
                "oracle_trace": "oracle_trace.npz",
                "oracle_video": "oracle_rollout.mp4",
            },
        }
        _write_json(manifest_path, manifest)
        print(f"[done] scene: {run_dir}")
        print(f"[done] object: {object_names[0]}")
        print(
            "[done] oracle displacement: "
            f"{100.0 * motion['terminal_displacement_m']:.2f} cm"
        )
        print(f"[done] goal image: {run_dir / 'goal_oracle.png'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
