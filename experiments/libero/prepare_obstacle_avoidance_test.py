#!/usr/bin/env python
"""Create collision and safe-path references for the mug avoidance test."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "third_party" / "LIBERO"):
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


def _quaternion_delta_degrees(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    a = np.asarray(a_wxyz, dtype=np.float64)
    b = np.asarray(b_wxyz, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cosine_half_angle = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine_half_angle))


def _save_trace(path: Path, trace: dict[str, list[Any]]) -> None:
    np.savez(
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


def _contact_sheet(
    entries: list[tuple[str, Image.Image]], path: Path
) -> None:
    width, height = entries[0][1].size
    label_height = 34
    sheet = Image.new(
        "RGB", (len(entries) * width, height + label_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(entries):
        x = index * width
        sheet.paste(image, (x, label_height))
        draw.text((x + 8, 9), label, fill="black")
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", default=str(REPO_ROOT / "runs" / "living_room_mug_obstacle")
    )
    parser.add_argument("--out-dir")
    parser.add_argument("--safe-arc-height", type=float, default=0.08)
    parser.add_argument("--safe-midpoint-x", type=float, default=0.0)
    parser.add_argument("--mug-displacement-tolerance", type=float, default=0.002)
    parser.add_argument("--mug-orientation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.03)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else run_dir / "mug_avoidance_test"
    )
    if out_dir.exists():
        if not args.force:
            parser.error(f"Output already exists: {out_dir}. Pass --force to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    start_state = np.load(run_dir / manifest["start_state"])
    physical_goal = np.asarray(manifest["physical_goal_eef"], dtype=np.float64)
    gripper_actions = [
        np.asarray(action)
        for action in manifest["start_gripper_controller_actions"]
    ]
    view_size = int(manifest["view_size"])
    variants = [
        ("collision_straight", 0.0, 0.0),
        ("safe_goal", args.safe_arc_height, args.safe_midpoint_x),
        ("safe_left", 0.0, -0.10),
        ("safe_right", 0.0, 0.10),
    ]

    env = OffScreenRenderEnv(
        bddl_file_name=manifest["bddl"],
        camera_heights=int(manifest["render_size"]),
        camera_widths=int(manifest["render_size"]),
    )
    records: list[dict[str, Any]] = []
    sheet_entries: list[tuple[str, Image.Image]] = []
    try:
        env.seed(int(manifest["sim_seed"]))
        env.reset()
        start_obs = env.set_init_state(start_state)
        _synchronize_controllers_to_sim_state(env, gripper_actions)
        object_names, _, initial_object_quaternions = _tracked_object_poses(env)
        if len(object_names) != 1:
            raise RuntimeError(
                f"Expected exactly one tracked mug, found {object_names}"
            )
        start_image = _views_from_obs(start_obs, view_size)[2]
        start_image.save(out_dir / "start.png")
        sheet_entries.append(("start", start_image))

        for name, arc_height, midpoint_x in variants:
            variant_dir = out_dir / name
            variant_dir.mkdir()
            env.reset()
            obs = env.set_init_state(start_state)
            _synchronize_controllers_to_sim_state(env, gripper_actions)
            trace: dict[str, list[Any]] = {}
            obs, actions, eef_path, frames = _rollout_to_target(
                env,
                obs,
                physical_goal,
                move_steps=int(manifest["move_steps"]),
                settle_steps=int(manifest["settle_steps"]),
                gain=float(manifest["controller_gain"]),
                arc_height=arc_height,
                midpoint_x=midpoint_x,
                capture_video=True,
                video_stride=args.video_stride,
                view_size=view_size,
                trace=trace,
            )
            terminal_main, terminal_wrist, terminal_image = _views_from_obs(
                obs, view_size
            )
            terminal_main.save(variant_dir / "terminal_agentview.png")
            terminal_wrist.save(variant_dir / "terminal_wrist.png")
            terminal_image.save(variant_dir / "terminal.png")
            np.save(variant_dir / "actions.npy", actions.astype(np.float32))
            np.save(variant_dir / "eef_path.npy", eef_path.astype(np.float32))
            _save_trace(variant_dir / "trace.npz", trace)
            _write_video(
                variant_dir / "rollout.mp4", frames, args.video_fps
            )

            object_motion = _object_motion_summary(trace, eef_path)
            terminal_object_quaternions = np.asarray(
                trace["object_quaternions_wxyz"][-1], dtype=np.float64
            )
            object_orientation_delta = {
                object_name: _quaternion_delta_degrees(
                    initial_object_quaternions[index],
                    terminal_object_quaternions[index],
                )
                for index, object_name in enumerate(object_names)
            }
            mug_motion = object_motion[object_names[0]]
            goal_error = float(np.linalg.norm(eef_path[-1] - physical_goal))
            mug_preserved = bool(
                mug_motion["terminal_displacement_m"]
                <= args.mug_displacement_tolerance
                and object_orientation_delta[object_names[0]]
                <= args.mug_orientation_tolerance_deg
            )
            success = bool(goal_error <= args.goal_tolerance and mug_preserved)
            record = {
                "name": name,
                "arc_height_m": arc_height,
                "midpoint_x_m": midpoint_x,
                "goal_error_m": goal_error,
                "object_motion": object_motion,
                "object_orientation_delta_deg": object_orientation_delta,
                "mug_preserved": mug_preserved,
                "success": success,
                "terminal_image": f"{name}/terminal.png",
                "rollout_video": f"{name}/rollout.mp4",
                "trace": f"{name}/trace.npz",
            }
            records.append(record)
            _write_json(variant_dir / "metadata.json", record)
            sheet_entries.append((name, terminal_image))
            print(
                f"[{name}] goal_error={goal_error:.4f}m "
                f"mug_displacement={mug_motion['terminal_displacement_m']:.4f}m "
                f"mug_rotation={object_orientation_delta[object_names[0]]:.2f}deg "
                f"success={success}",
                flush=True,
            )
            if name == "safe_goal":
                if not success:
                    raise RuntimeError(
                        "Configured safe goal did not preserve the upright mug"
                    )
                terminal_image.save(out_dir / "goal_avoidance.png")
                np.save(
                    out_dir / "goal_avoidance_state.npy",
                    np.asarray(env.get_sim_state()),
                )

        _contact_sheet(sheet_entries, out_dir / "references.png")
        test_manifest = {
            "schema_version": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "purpose": (
                "Optimize minimum-jerk path shape while holding the terminal EEF "
                "goal fixed and preserving the mug pose."
            ),
            "source_run_dir": str(run_dir),
            "source_manifest": str(run_dir / "manifest.json"),
            "start_state": str(run_dir / manifest["start_state"]),
            "start_image": "start.png",
            "goal_image": "goal_avoidance.png",
            "physical_goal_eef": physical_goal,
            "move_steps": int(manifest["move_steps"]),
            "settle_steps": int(manifest["settle_steps"]),
            "controller_gain": float(manifest["controller_gain"]),
            "tracked_object": object_names[0],
            "mug_displacement_tolerance_m": args.mug_displacement_tolerance,
            "mug_orientation_tolerance_deg": args.mug_orientation_tolerance_deg,
            "goal_tolerance_m": args.goal_tolerance,
            "safe_goal_parameters": {
                "midpoint_x_m": args.safe_midpoint_x,
                "arc_height_m": args.safe_arc_height,
            },
            "variants": records,
        }
        _write_json(out_dir / "manifest.json", test_manifest)
    finally:
        env.close()

    print(f"[done] test: {out_dir}")
    print(f"[done] goal: {out_dir / 'goal_avoidance.png'}")
    print(f"[done] references: {out_dir / 'references.png'}")


if __name__ == "__main__":
    main()
