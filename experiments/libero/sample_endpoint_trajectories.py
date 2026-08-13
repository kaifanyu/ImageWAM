#!/usr/bin/env python
"""Sample terminal robot-arm poses on an empty LIBERO table.

This is the simulator half of the direct-endpoint baseline:

    exact simulator snapshot -> K full action trajectories -> K terminal renders

It deliberately does not predict, edit, or score intermediate images.  Every
candidate is restored from the same flattened MuJoCo state.  The default set
contains diagnostic controls (no-op, wrong direction, under/over-shoot, oracle)
plus seeded stochastic targets around the physical goal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "third_party" / "LIBERO", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import robosuite.utils.transform_utils as T  # noqa: E402

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
import side_camera  # noqa: E402
from scene_geometry import arm_link_names, arm_link_positions  # noqa: E402


DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: "
    "move the robot arm from the left side of the empty table to the right side. "
    "Keep the end-effector height, orientation, and gripper unchanged. Update the "
    "agent and wrist views consistently with the motion. Keep the fixed camera, table, "
    "lighting, and background unchanged. Do not add or remove anything."
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clear_previous_run(run_dir: Path, manifest_path: Path) -> None:
    """Remove only artifacts owned by a previous endpoint-pipeline run."""
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot safely replace malformed prior manifest: {manifest_path}"
        ) from error
    if previous.get("pipeline") != "direct_terminal_endpoint":
        raise RuntimeError(
            f"Refusing to replace a run from another pipeline: {manifest_path}"
        )

    # Include directories from an interrupted candidate that was created but
    # had not yet been appended to the manifest.
    for candidate_path in run_dir.iterdir():
        if not re.fullmatch(r"candidate_\d{3}", candidate_path.name):
            continue
        if candidate_path.is_symlink() or candidate_path.is_file():
            candidate_path.unlink()
        elif candidate_path.is_dir():
            shutil.rmtree(candidate_path)

    downstream_files = (
        "start.png",
        "start_agentview.png",
        "start_wrist.png",
        "start_sideview.png",
        "start_state.npy",
        "start_proprio_raw.npy",
        "start_proprio_normalized.npy",
        "goal_oracle.png",
        "goal_edit.png",
        "goal_edit_compare.png",
        "goal_edit_metadata.json",
        "goal_editor_latent.npy",
        "goal_dynamics_latent.npy",
        "goal_dynamics_metadata.json",
        "metrics.json",
        "metrics.csv",
        "summary.md",
        "edit_mask.png",
        "edit_mask_agentview.png",
        "edit_mask_wrist.png",
        "contact_sheet_pixel.png",
        "contact_sheet_flux_vae.png",
        "contact_sheet_editor_final.png",
        "contact_sheet_dynamics_vae.png",
    )
    for filename in downstream_files:
        path = run_dir / filename
        if path.is_file() or path.is_symlink():
            path.unlink()
    manifest_path.unlink()


def _center_crop_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    resized = image.resize((round(src_w * scale), round(src_h * scale)), Image.BILINEAR)
    left = max((resized.width - width) // 2, 0)
    top = max((resized.height - height) // 2, 0)
    return resized.crop((left, top, left + width, top + height))


# Every render in this pipeline is a two-panel ``[agentview | right]`` image, and
# every latent built from one splits down its middle column, so which camera
# fills the right half is a property of the whole run rather than of one call
# site.  It is set once per process -- from the run manifest when there is one --
# and read by the many nested callers of ``_views_from_obs``.
RIGHT_VIEW_IMAGE_KEYS = {
    "wrist": "robot0_eye_in_hand_image",
    "sideview": side_camera.IMAGE_KEY,
}
DEFAULT_RIGHT_VIEW = "sideview"
_RIGHT_VIEW = DEFAULT_RIGHT_VIEW


def set_composed_right_view(name: str) -> str:
    """Choose the camera that fills the right half of every composed render."""
    global _RIGHT_VIEW
    if name not in RIGHT_VIEW_IMAGE_KEYS:
        raise ValueError(
            f"Unknown right view {name!r}; expected one of {sorted(RIGHT_VIEW_IMAGE_KEYS)}"
        )
    _RIGHT_VIEW = name
    return _RIGHT_VIEW


def composed_right_view() -> str:
    """The camera currently filling the right half of composed renders."""
    return _RIGHT_VIEW


def _views_from_obs(
    obs: dict[str, np.ndarray],
    view_size: int,
    right_view: str | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    # LIBERO training data rotates both camera observations by 180 degrees, and
    # the side camera is built to the same convention.
    right_view = _RIGHT_VIEW if right_view is None else right_view
    right_key = RIGHT_VIEW_IMAGE_KEYS.get(right_view)
    if right_key is None:
        raise ValueError(
            f"Unknown right view {right_view!r}; expected one of {sorted(RIGHT_VIEW_IMAGE_KEYS)}"
        )
    if right_key not in obs:
        raise KeyError(
            f"Observation has no {right_key!r}; build the env with "
            f"side_camera.open_env(...) so the {right_view!r} camera is rendered"
        )
    main = Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).convert("RGB")
    right = Image.fromarray(np.ascontiguousarray(obs[right_key][::-1, ::-1])).convert("RGB")
    main = _center_crop_resize(main, view_size, view_size)
    right = _center_crop_resize(right, view_size, view_size)
    composed = Image.new("RGB", (2 * view_size, view_size))
    composed.paste(main, (0, 0))
    composed.paste(right, (view_size, 0))
    return main, right, composed


def env_from_manifest(manifest: dict[str, Any], **kwargs: Any) -> OffScreenRenderEnv:
    """Open the run's scene framed exactly as its scene prep framed it.

    Reads back the two facts that decide what a composed render means -- which
    camera fills the right half, and where that camera sits -- so an optimiser
    never scores a render against a goal image built from different cameras.
    Manifests written before the side camera existed carry neither key and are
    reopened as the stock ``[agentview | wrist]`` pair.
    """
    right_view = str(manifest.get("composed_right_view", "wrist"))
    set_composed_right_view(right_view)
    pose = manifest.get("side_camera")
    return side_camera.open_env(
        bddl_file_name=str(manifest["bddl"]),
        render_size=int(manifest.get("render_size", 256)),
        side_camera=pose if pose is not None else (right_view == "sideview"),
        **kwargs,
    )


def _tracked_object_poses(
    env: OffScreenRenderEnv,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return world-frame poses for BDDL objects marked as objects of interest."""
    names = [str(name) for name in env.obj_of_interest]
    positions = np.empty((len(names), 3), dtype=np.float64)
    quaternions_wxyz = np.empty((len(names), 4), dtype=np.float64)
    for index, name in enumerate(names):
        state = env.env.object_states_dict[name].get_geom_state()
        positions[index] = np.asarray(state["pos"], dtype=np.float64)
        quaternions_wxyz[index] = np.asarray(state["quat"], dtype=np.float64)
    return names, positions, quaternions_wxyz


def _object_motion_summary(
    trace: dict[str, list[Any]],
    eef_path: np.ndarray,
) -> dict[str, dict[str, Any]]:
    names = [str(name) for name in trace.get("tracked_object_names", [])]
    if not names:
        return {}
    positions = np.asarray(trace["object_positions"], dtype=np.float64)
    eef_positions = np.asarray(eef_path, dtype=np.float64)
    if positions.shape[0] != eef_positions.shape[0]:
        raise ValueError(
            "Object and EEF traces must contain the same number of states"
        )
    summary: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        path = positions[:, index]
        step_distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        displacement_from_start = np.linalg.norm(path - path[0], axis=1)
        center_distances = np.linalg.norm(eef_positions - path, axis=1)
        xy_distances = np.linalg.norm(eef_positions[:, :2] - path[:, :2], axis=1)
        summary[name] = {
            "initial_position": path[0],
            "terminal_position": path[-1],
            "terminal_displacement_vector_m": path[-1] - path[0],
            "terminal_displacement_m": float(np.linalg.norm(path[-1] - path[0])),
            "maximum_displacement_from_start_m": float(
                displacement_from_start.max()
            ),
            "path_length_m": float(step_distances.sum()),
            "minimum_eef_center_distance_m": float(center_distances.min()),
            "minimum_eef_xy_distance_m": float(xy_distances.min()),
            "closest_eef_state_index": int(np.argmin(center_distances)),
        }
    return summary


def _quat_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(1.0 - float(quat[3]) ** 2, 0.0))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator).astype(np.float32)


def _proprio_from_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )


def _quaternion_error_degrees(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> float:
    a = np.asarray(a_xyzw, dtype=np.float64)
    b = np.asarray(b_xyzw, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    # q and -q encode the same rotation.
    cosine_half_angle = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine_half_angle))


def _normalize_proprio(raw: np.ndarray, stats_path: Path) -> np.ndarray:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))["state"]["default"]
    low = np.asarray(stats["global_min"], dtype=np.float32)
    high = np.asarray(stats["global_max"], dtype=np.float32)
    if raw.shape != low.shape or raw.shape != high.shape:
        raise ValueError(
            f"Proprio/stats shape mismatch: raw={raw.shape}, min={low.shape}, max={high.shape}"
        )
    span = high - low
    ignored = span < 1e-4
    safe_span = span.copy()
    safe_span[ignored] = 2.0
    normalized = 2.0 * (raw - low) / safe_span - 1.0
    # SingleFieldLinearNormalizer uses scale=1 and offset=-min for dimensions
    # whose observed range is effectively constant.
    normalized[ignored] = raw[ignored] - low[ignored]
    return np.clip(normalized, -5.0, 5.0).astype(np.float32)


def _minimum_jerk(u: float) -> float:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


# OSC_POSE with control_delta=true maps action[3:6] from [-1, 1] onto a
# world-frame axis-angle delta of +/-OSC_ROTATION_OUTPUT_MAX radians per control
# step (robosuite/controllers/config/osc_pose.json).
OSC_ROTATION_OUTPUT_MAX = 0.5


def _orientation_delta_axis_angle(
    current_quat_xyzw: np.ndarray,
    desired_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """World-frame axis-angle taking the current orientation onto the desired one.

    robosuite composes the commanded delta as ``goal = R_delta @ R_current``, so
    the delta is expressed in the world frame rather than the tool frame.
    """
    current = T.quat2mat(np.asarray(current_quat_xyzw, dtype=np.float64))
    desired = T.quat2mat(np.asarray(desired_quat_xyzw, dtype=np.float64))
    return np.asarray(T.quat2axisangle(T.mat2quat(desired @ current.T)), dtype=np.float64)


def _rollout_to_target(
    env: OffScreenRenderEnv,
    obs: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    move_steps: int,
    settle_steps: int,
    gain: float,
    target_quat_xyzw: np.ndarray | None = None,
    rotation_gain: float = 1.0,
    arc_height: float = 0.0,
    midpoint_x: float = 0.0,
    capture_video: bool = False,
    video_stride: int = 2,
    view_size: int = 224,
    trace: dict[str, list[Any]] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[Image.Image]]:
    start = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64)
    # `None` leaves action[3:6] at zero, which is what every position-only
    # caller relies on: OSC then holds the orientation it started from.
    start_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    target_quat = (
        None
        if target_quat_xyzw is None
        else np.asarray(target_quat_xyzw, dtype=np.float64)
        / np.linalg.norm(np.asarray(target_quat_xyzw, dtype=np.float64))
    )
    actions: list[np.ndarray] = []
    eef_path: list[np.ndarray] = [start.copy()]
    frames: list[Image.Image] = (
        [_views_from_obs(obs, view_size)[2]] if capture_video else []
    )

    if trace is not None:
        object_names, object_positions, object_quaternions = _tracked_object_poses(
            env
        )
        # Whole-arm poses, so a trace can be replayed as a skeleton and not just
        # as the end-effector point it traces out.
        link_names = arm_link_names(env)
        trace.update(
            {
                "desired_eefs": [],
                "eef_before_actions": [],
                "position_errors": [],
                "normalized_times": [],
                "minimum_jerk_progress": [],
                "phases": [],
                "tracked_object_names": object_names,
                "object_positions": [object_positions],
                "object_quaternions_wxyz": [object_quaternions],
                "arm_link_names": link_names,
                "arm_link_positions": [arm_link_positions(env, link_names)],
            }
        )

    def desired_quat(progress: float) -> np.ndarray | None:
        if target_quat is None:
            return None
        return np.asarray(
            T.quat_slerp(start_quat, target_quat, float(progress)), dtype=np.float64
        )

    def step_toward(
        desired: np.ndarray,
        step_idx: int,
        *,
        normalized_time: float,
        progress: float,
        phase: str,
    ) -> None:
        nonlocal obs
        eef_before = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        error = desired - eef_before
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(gain * error, -1.0, 1.0)
        quat_desired = desired_quat(progress)
        if quat_desired is not None:
            rotation_error = _orientation_delta_axis_angle(
                obs["robot0_eef_quat"], quat_desired
            )
            action[3:6] = np.clip(
                rotation_gain * rotation_error / OSC_ROTATION_OUTPUT_MAX, -1.0, 1.0
            )
        action[-1] = -1.0  # keep the gripper open
        if trace is not None:
            trace["desired_eefs"].append(np.asarray(desired, dtype=np.float64).copy())
            trace["eef_before_actions"].append(eef_before)
            trace["position_errors"].append(error.copy())
            trace["normalized_times"].append(float(normalized_time))
            trace["minimum_jerk_progress"].append(float(progress))
            trace["phases"].append(phase)
        obs, _, _, _ = env.step(action)
        actions.append(action)
        eef_path.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy())
        if trace is not None:
            _, object_positions, object_quaternions = _tracked_object_poses(env)
            trace["object_positions"].append(object_positions)
            trace["object_quaternions_wxyz"].append(object_quaternions)
            trace["arm_link_positions"].append(
                arm_link_positions(env, trace["arm_link_names"])
            )
        if capture_video and step_idx % max(video_stride, 1) == 0:
            frames.append(_views_from_obs(obs, view_size)[2])

    for step_idx in range(move_steps):
        u = float(step_idx + 1) / float(move_steps)
        progress = _minimum_jerk(u)
        bell = 4.0 * u * (1.0 - u)
        desired = start + progress * (target - start)
        desired[0] += float(midpoint_x) * bell
        desired[2] += float(arc_height) * bell
        step_toward(
            desired,
            step_idx,
            normalized_time=u,
            progress=progress,
            phase="move",
        )

    for settle_idx in range(settle_steps):
        step_toward(
            target,
            move_steps + settle_idx,
            normalized_time=1.0,
            progress=1.0,
            phase="settle",
        )

    final_step_index = move_steps + settle_steps - 1
    if capture_video and final_step_index % max(video_stride, 1) != 0:
        # Always include the exact terminal observation even when the last
        # control step does not land on video_stride.
        frames.append(_views_from_obs(obs, view_size)[2])

    return obs, np.stack(actions), np.stack(eef_path), frames


def _synchronize_controllers_to_sim_state(
    env: OffScreenRenderEnv,
    gripper_actions: list[np.ndarray] | None = None,
) -> None:
    """Refresh non-MuJoCo controller state after restoring a flat snapshot.

    ``set_init_state`` restores qpos/qvel and observations, but robosuite's OSC
    controller also caches the end-effector pose, Jacobian, goals, and
    null-space reference.  Leaving those values from the preceding ``reset``
    causes every candidate to begin with the same unintended transient.
    """
    if gripper_actions is not None and len(gripper_actions) != len(env.robots):
        raise ValueError("Saved gripper action count does not match robot count")
    for index, robot in enumerate(env.robots):
        if gripper_actions is not None:
            # GripperModel.current_action is an incremental command cache and
            # is not part of MuJoCo's flattened state.
            robot.gripper.current_action = np.asarray(gripper_actions[index]).copy()
        controller = robot.controller
        controller.update(force=True)
        controller.update_initial_joints(np.asarray(controller.joint_pos).copy())


def _write_video(path: Path, frames: list[Image.Image], fps: int) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()


def _candidate_specs(
    start: np.ndarray,
    goal: np.ndarray,
    count: int,
    trajectory_seed: int,
    endpoint_sigma: np.ndarray,
) -> list[dict[str, Any]]:
    if count < 5:
        raise ValueError("--num-trajectories must be at least 5 to retain all diagnostic controls")

    delta = goal - start
    z_padding = max(0.05, 6.0 * float(endpoint_sigma[2]))
    z_limits = (
        min(float(start[2]), float(goal[2])) - z_padding,
        max(float(start[2]), float(goal[2])) + z_padding,
    )
    wrong = start - 0.30 * delta
    undershoot = start + 0.55 * delta
    overshoot = start + 1.20 * delta
    for pose in (wrong, undershoot, overshoot):
        pose[0] = np.clip(pose[0], -0.18, 0.18)
        pose[1] = np.clip(pose[1], -0.32, 0.32)
        pose[2] = np.clip(pose[2], *z_limits)

    specs: list[dict[str, Any]] = [
        {"kind": "no_op", "target": start.copy(), "arc_height": 0.0, "midpoint_x": 0.0, "seed": None},
        {"kind": "wrong_direction", "target": wrong, "arc_height": 0.01, "midpoint_x": 0.0, "seed": None},
        {"kind": "undershoot", "target": undershoot, "arc_height": 0.03, "midpoint_x": 0.0, "seed": None},
        {"kind": "oracle", "target": goal.copy(), "arc_height": 0.04, "midpoint_x": 0.0, "seed": None},
        {"kind": "overshoot", "target": overshoot, "arc_height": 0.05, "midpoint_x": 0.0, "seed": None},
    ]

    while len(specs) < count:
        candidate_seed = int(trajectory_seed) + len(specs)
        candidate_rng = np.random.default_rng(candidate_seed)
        target = goal + candidate_rng.normal(0.0, endpoint_sigma, size=3)
        target[0] = np.clip(target[0], -0.18, 0.18)
        target[1] = np.clip(target[1], -0.32, 0.32)
        target[2] = np.clip(target[2], *z_limits)
        specs.append(
            {
                "kind": "sampled",
                "target": target,
                "arc_height": float(candidate_rng.uniform(0.0, 0.08)),
                "midpoint_x": float(candidate_rng.uniform(-0.05, 0.05)),
                "seed": candidate_seed,
            }
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(REPO_ROOT / "runs" / "empty_arm_endpoint"))
    parser.add_argument(
        "--bddl",
        default=str(REPO_ROOT / "experiments" / "libero" / "empty_table_move_arm.bddl"),
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=13,
        help="Total pool size: five diagnostic controls plus sampled trajectories.",
    )
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument("--trajectory-seed", type=int, default=7)
    parser.add_argument("--start-eef", type=float, nargs=3, default=[-0.05, 0.22, 1.03])
    parser.add_argument("--goal-eef", type=float, nargs=3, default=[-0.05, -0.22, 1.03])
    parser.add_argument(
        "--sample-endpoint-sigma",
        type=float,
        nargs=3,
        default=[0.006, 0.015, 0.004],
        metavar=("X", "Y", "Z"),
        help="Standard deviation in meters for sampled terminal targets around --goal-eef.",
    )
    parser.add_argument("--stage-steps", type=int, default=48)
    parser.add_argument("--move-steps", type=int, default=40)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--controller-gain", type=float, default=15.0)
    parser.add_argument("--stage-tolerance", type=float, default=0.01, help="Meters.")
    parser.add_argument("--success-tolerance", type=float, default=0.03, help="Meters.")
    parser.add_argument("--height-tolerance", type=float, default=0.015, help="Meters.")
    parser.add_argument("--orientation-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--gripper-tolerance", type=float, default=0.01)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument(
        "--right-view",
        choices=sorted(RIGHT_VIEW_IMAGE_KEYS),
        default=DEFAULT_RIGHT_VIEW,
        help=(
            "Camera filling the right half of every composed render. 'sideview' "
            "is a pure side profile of the table, which resolves the height and "
            "reach that agentview alone leaves ambiguous."
        ),
    )
    parser.add_argument(
        "--side-camera-margin",
        type=float,
        default=side_camera.DEFAULT_MARGIN,
        help="Meters between the table edge and the side camera.",
    )
    parser.add_argument(
        "--side-camera-height",
        type=float,
        default=side_camera.DEFAULT_HEIGHT,
        help="Meters above the table top that the side camera looks along.",
    )
    parser.add_argument(
        "--side-camera-elevation-deg",
        type=float,
        default=side_camera.DEFAULT_ELEVATION_DEG,
        help="Tilt above the horizontal; 0 keeps the table exactly edge on.",
    )
    parser.add_argument(
        "--side-camera-x",
        type=float,
        default=None,
        help="World x the side camera aims at; defaults to the robot/table midpoint.",
    )
    parser.add_argument("--reset-render-mae-tolerance", type=float, default=0.25)
    parser.add_argument("--reset-render-outlier-fraction", type=float, default=0.01)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dataset-stats", help="Optional stats JSON used to save normalized proprio.")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace artifacts owned by a previous endpoint-pipeline run.",
    )
    args = parser.parse_args()

    if args.stage_steps <= 0 or args.move_steps < 0 or args.settle_steps < 0:
        parser.error("--stage-steps must be positive; move/settle steps must be non-negative")
    if args.move_steps + args.settle_steps <= 0:
        parser.error("At least one move or settle step is required")
    if args.render_size <= 0 or args.view_size <= 0:
        parser.error("--render-size and --view-size must be positive")
    if args.video_stride <= 0 or args.video_fps <= 0:
        parser.error("--video-stride and --video-fps must be positive")
    tolerance_values = (
        args.stage_tolerance,
        args.success_tolerance,
        args.height_tolerance,
        args.orientation_tolerance_deg,
        args.gripper_tolerance,
        args.reset_render_mae_tolerance,
        args.reset_render_outlier_fraction,
    )
    if any(value < 0.0 for value in tolerance_values):
        parser.error("All tolerances must be non-negative")

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        parser.error(f"Run already exists: {manifest_path}. Pass --force or choose another --run-dir.")
    if manifest_path.exists() and args.force:
        _clear_previous_run(run_dir, manifest_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    bddl_path = Path(args.bddl).resolve()
    if not bddl_path.exists():
        parser.error(f"BDDL file not found: {bddl_path}")

    start_target = np.asarray(args.start_eef, dtype=np.float64)
    physical_goal = np.asarray(args.goal_eef, dtype=np.float64)
    endpoint_sigma = np.asarray(args.sample_endpoint_sigma, dtype=np.float64)
    if np.any(endpoint_sigma < 0.0):
        parser.error("--sample-endpoint-sigma values must be non-negative")

    set_composed_right_view(args.right_view)
    env = side_camera.open_env(
        bddl_file_name=str(bddl_path),
        render_size=args.render_size,
    )
    side_camera_pose = side_camera.install_side_camera(
        env,
        margin=args.side_camera_margin,
        height=args.side_camera_height,
        elevation_deg=args.side_camera_elevation_deg,
        x_center=args.side_camera_x,
    )
    print(
        f"[views] composed render = [agentview | {args.right_view}]; side camera at "
        f"{np.round(side_camera_pose['position'], 3).tolist()} aimed at "
        f"{np.round(side_camera_pose['target'], 3).tolist()}"
    )
    try:
        env.seed(args.sim_seed)
        obs = env.reset()
        obs, _, _, _ = _rollout_to_target(
            env,
            obs,
            start_target,
            move_steps=args.stage_steps,
            settle_steps=args.settle_steps,
            gain=args.controller_gain,
            view_size=args.view_size,
        )

        # Treat the staged pose as the controller's neutral pose. This is also
        # repeated after every exact state restore below.
        _synchronize_controllers_to_sim_state(env)
        if env.check_success():
            raise RuntimeError(
                "The BDDL goal is unexpectedly true at the staged start state. "
                "Choose a scene whose built-in goal is initially false; physical "
                "endpoint success is computed independently by this harness."
            )

        start_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        start_gripper_actions = [
            np.asarray(robot.gripper.current_action).copy() for robot in env.robots
        ]
        start_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        stage_tracking_error = float(np.linalg.norm(start_eef - start_target))
        if stage_tracking_error > args.stage_tolerance:
            raise RuntimeError(
                f"Failed to stage requested start pose: error={stage_tracking_error:.4f}m "
                f"> tolerance={args.stage_tolerance:.4f}m"
            )
        start_eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
        start_gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).copy()
        (
            tracked_object_names,
            tracked_object_positions,
            tracked_object_quaternions,
        ) = _tracked_object_poses(env)
        start_main, start_right, start_image = _views_from_obs(obs, args.view_size)
        start_main.save(run_dir / "start_agentview.png")
        start_right.save(run_dir / f"start_{args.right_view}.png")
        # Keep every camera on disk, not just the composed pair, so a run can be
        # inspected against a view it was not optimised on.
        for view_name in RIGHT_VIEW_IMAGE_KEYS:
            if view_name != args.right_view:
                _views_from_obs(obs, args.view_size, view_name)[1].save(
                    run_dir / f"start_{view_name}.png"
                )
        start_image.save(run_dir / "start.png")
        np.save(run_dir / "start_state.npy", start_state)

        raw_proprio = _proprio_from_obs(obs)
        np.save(run_dir / "start_proprio_raw.npy", raw_proprio)
        normalized_proprio_path: str | None = None
        stats_path: Path | None = None
        if args.dataset_stats:
            stats_path = Path(args.dataset_stats).resolve()
            if not stats_path.exists():
                parser.error(f"Dataset stats not found: {stats_path}")
            normalized = _normalize_proprio(raw_proprio, stats_path)
            np.save(run_dir / "start_proprio_normalized.npy", normalized)
            normalized_proprio_path = "start_proprio_normalized.npy"

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pipeline": "direct_terminal_endpoint",
            "simulator": "LIBERO/robosuite OffScreenRenderEnv",
            "bddl": str(bddl_path),
            "runtime_provenance": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pillow": _package_version("Pillow"),
                "libero": _package_version("libero"),
                "robosuite": _package_version("robosuite"),
                "mujoco": _package_version("mujoco"),
                "controller": type(env.robots[0].controller).__name__,
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
                "bddl_sha256": _sha256(bddl_path),
            },
            "sim_seed": args.sim_seed,
            "trajectory_seed": args.trajectory_seed,
            "num_trajectories": args.num_trajectories,
            "prompt": args.prompt,
            "render_size": args.render_size,
            "view_size": args.view_size,
            # Downstream optimisers read these back so their renders compose the
            # same two cameras the goal image was built from.
            "composed_views": ["agentview", args.right_view],
            "composed_right_view": args.right_view,
            "side_camera": side_camera_pose,
            "start_image": "start.png",
            "start_state": "start_state.npy",
            "start_proprio_raw": "start_proprio_raw.npy",
            "start_proprio_normalized": normalized_proprio_path,
            "dataset_stats": stats_path,
            "requested_start_eef": start_target,
            "actual_start_eef": start_eef,
            "stage_tracking_error_m": stage_tracking_error,
            "stage_tolerance_m": args.stage_tolerance,
            "start_eef_quat_xyzw": start_eef_quat,
            "start_gripper_qpos": start_gripper,
            "start_gripper_controller_actions": start_gripper_actions,
            "tracked_objects": {
                name: {
                    "initial_position": tracked_object_positions[index],
                    "initial_quaternion_wxyz": tracked_object_quaternions[index],
                }
                for index, name in enumerate(tracked_object_names)
            },
            "physical_goal_eef": physical_goal,
            "sample_endpoint_sigma_m": endpoint_sigma,
            "libero_builtin_goal": (
                "BDDL goal is diagnostic only; endpoint pose checks are authoritative"
            ),
            "libero_builtin_success_at_start": False,
            "controller_restore_sync": (
                "restore gripper current_action + update(force=True) + "
                "update_initial_joints(current)"
            ),
            "success_tolerance_m": args.success_tolerance,
            "height_tolerance_m": args.height_tolerance,
            "orientation_tolerance_deg": args.orientation_tolerance_deg,
            "gripper_tolerance": args.gripper_tolerance,
            "move_steps": args.move_steps,
            "settle_steps": args.settle_steps,
            "stage_steps": args.stage_steps,
            "controller_gain": args.controller_gain,
            "save_videos": args.save_videos,
            "video_stride": args.video_stride,
            "video_fps": args.video_fps,
            "reset_render_mae_tolerance": args.reset_render_mae_tolerance,
            "reset_render_outlier_fraction": args.reset_render_outlier_fraction,
            "candidates": [],
        }
        _write_json(manifest_path, manifest)

        specs = _candidate_specs(
            start_eef,
            physical_goal,
            args.num_trajectories,
            args.trajectory_seed,
            endpoint_sigma,
        )
        for index, spec in enumerate(specs):
            candidate_id = f"candidate_{index:03d}"
            candidate_dir = run_dir / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)

            # Reset controller internals, then restore the exact staged MuJoCo state.
            env.reset()
            obs = env.set_init_state(start_state)
            _synchronize_controllers_to_sim_state(env, start_gripper_actions)
            restored_image = _views_from_obs(obs, args.view_size)[2]
            restored_state = np.asarray(env.get_sim_state(), dtype=np.float64)
            reset_state_max_abs = float(np.max(np.abs(restored_state - start_state)))
            reset_rgb_abs = np.abs(
                np.asarray(restored_image, dtype=np.int16)
                - np.asarray(start_image, dtype=np.int16)
            )
            reset_rgb_max_abs = int(reset_rgb_abs.max())
            reset_rgb_mae = float(reset_rgb_abs.mean())
            reset_rgb_fraction_gt8 = float(np.mean(reset_rgb_abs > 8))
            # MuJoCo state should restore bit-for-bit. OSMesa edge anti-aliasing
            # has a small render noise floor, so validate its mean/tail rather
            # than requiring every boundary pixel to be identical.
            if reset_state_max_abs != 0.0:
                raise RuntimeError(
                    f"Non-deterministic simulator state for {candidate_id}: "
                    f"max absolute state diff={reset_state_max_abs}"
                )
            if (
                reset_rgb_mae > args.reset_render_mae_tolerance
                or reset_rgb_fraction_gt8 > args.reset_render_outlier_fraction
            ):
                raise RuntimeError(
                    f"Restored render drift for {candidate_id}: RGB MAE={reset_rgb_mae:.4f}, "
                    f"fraction(|diff|>8)={reset_rgb_fraction_gt8:.4f}"
                )

            target = np.asarray(spec["target"], dtype=np.float64)
            rollout_trace: dict[str, list[Any]] = {}
            obs, actions, eef_path, frames = _rollout_to_target(
                env,
                obs,
                target,
                move_steps=args.move_steps,
                settle_steps=args.settle_steps,
                gain=args.controller_gain,
                arc_height=float(spec["arc_height"]),
                midpoint_x=float(spec["midpoint_x"]),
                capture_video=args.save_videos,
                video_stride=args.video_stride,
                view_size=args.view_size,
                trace=rollout_trace,
            )

            terminal_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
            terminal_eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
            terminal_gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).copy()
            terminal_main, terminal_right, terminal_image = _views_from_obs(obs, args.view_size)
            terminal_main.save(candidate_dir / "terminal_agentview.png")
            terminal_right.save(candidate_dir / f"terminal_{args.right_view}.png")
            terminal_image.save(candidate_dir / "terminal.png")
            np.save(candidate_dir / "actions.npy", actions.astype(np.float32))
            np.save(candidate_dir / "eef_path.npy", eef_path.astype(np.float32))
            np.save(candidate_dir / "terminal_state.npy", np.asarray(env.get_sim_state()))
            object_trace_path: str | None = None
            if rollout_trace["tracked_object_names"]:
                np.savez(
                    candidate_dir / "object_trace.npz",
                    object_names=np.asarray(
                        rollout_trace["tracked_object_names"], dtype=str
                    ),
                    positions=np.asarray(
                        rollout_trace["object_positions"], dtype=np.float64
                    ),
                    quaternions_wxyz=np.asarray(
                        rollout_trace["object_quaternions_wxyz"], dtype=np.float64
                    ),
                )
                object_trace_path = f"{candidate_id}/object_trace.npz"
            if args.save_videos:
                _write_video(candidate_dir / "rollout.mp4", frames, args.video_fps)

            goal_error = float(np.linalg.norm(terminal_eef - physical_goal))
            height_error = float(abs(terminal_eef[2] - physical_goal[2]))
            orientation_error = _quaternion_error_degrees(start_eef_quat, terminal_eef_quat)
            gripper_error = float(np.max(np.abs(terminal_gripper - start_gripper)))
            tracking_error = float(np.linalg.norm(terminal_eef - target))
            displacement = physical_goal - start_eef
            denom = float(np.dot(displacement, displacement))
            progress = (
                float(np.dot(terminal_eef - start_eef, displacement) / denom)
                if denom > 1e-12
                else 0.0
            )
            record = {
                "id": candidate_id,
                "kind": spec["kind"],
                "trajectory_seed": spec["seed"],
                "target_eef": target,
                "terminal_eef": terminal_eef,
                "terminal_eef_quat_xyzw": terminal_eef_quat,
                "terminal_gripper_qpos": terminal_gripper,
                "arc_height_m": float(spec["arc_height"]),
                "midpoint_x_m": float(spec["midpoint_x"]),
                "goal_error_m": goal_error,
                "height_error_m": height_error,
                "orientation_error_deg": orientation_error,
                "gripper_error": gripper_error,
                "target_tracking_error_m": tracking_error,
                "object_motion": _object_motion_summary(
                    rollout_trace, eef_path
                ),
                "goal_progress": progress,
                "physical_success": (
                    goal_error <= args.success_tolerance
                    and height_error <= args.height_tolerance
                    and orientation_error <= args.orientation_tolerance_deg
                    and gripper_error <= args.gripper_tolerance
                ),
                "libero_builtin_success": bool(env.check_success()),
                "reset_state_max_abs": reset_state_max_abs,
                "reset_rgb_max_abs": reset_rgb_max_abs,
                "reset_rgb_mae": reset_rgb_mae,
                "reset_rgb_fraction_gt8": reset_rgb_fraction_gt8,
                "terminal_image": f"{candidate_id}/terminal.png",
                "actions": f"{candidate_id}/actions.npy",
                "eef_path": f"{candidate_id}/eef_path.npy",
                "object_trace": object_trace_path,
                "rollout_video": f"{candidate_id}/rollout.mp4" if args.save_videos else None,
            }
            manifest["candidates"].append(record)
            if spec["kind"] == "oracle":
                terminal_image.save(run_dir / "goal_oracle.png")
                manifest["goal_oracle_image"] = "goal_oracle.png"
            _write_json(candidate_dir / "metadata.json", record)
            _write_json(manifest_path, manifest)
            print(
                f"[{candidate_id}] {spec['kind']:<15s} "
                f"goal_error={goal_error:.4f}m reset_rgb_mae={reset_rgb_mae:.4f}"
            )

        print(f"\n[done] simulator artifacts: {run_dir}")
        print(f"[next] edit {run_dir / 'start.png'} into {run_dir / 'goal_edit.png'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
