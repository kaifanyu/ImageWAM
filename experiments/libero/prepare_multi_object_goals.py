#!/usr/bin/env python
"""Render a catalogue of candidate goal images for a multi-object LIBERO table.

The endpoint pipeline (``sample_endpoint_trajectories.py``) produces exactly one
goal image: the oracle terminal frame of a straight left-to-right reach.  A
scene with objects on the table admits many more interesting goals -- hover over
the mug, hover over the plate with the wrist spun 90 degrees, approach the bowl
at a tilt -- and picking between them is a judgement call best made by looking
at the pictures.

So this script does not optimize anything.  It restores the staged start state
written by the endpoint pipeline, servos the arm to each declared goal pose in
turn (position *and* orientation), and renders what it sees::

    runs/<scene>/goals/<goal_id>/goal.png            agentview | side profile, 2*view_size
    runs/<scene>/goals/<goal_id>/goal_agentview.png
    runs/<scene>/goals/<goal_id>/goal_sideview.png
    runs/<scene>/goals/<goal_id>/terminal_state.npy
    runs/<scene>/goals/<goal_id>/metadata.json
    runs/<scene>/goals/contact_sheet.png             all of them, labelled
    runs/<scene>/goals/index.json

``goal.png`` is byte-compatible with what ``svgd_endpoint.py --goal`` expects, and
every goal reuses the run's existing ``start_state.npy`` and ``manifest.json``,
so a chosen goal is run with nothing more than::

    python experiments/libero/svgd_endpoint.py --run-dir <run> \
        --goal <run>/goals/<goal_id>/goal.png ...

Each goal is scored for reachability (position/orientation tracking error) and
for scene contamination (how far the objects were nudged).  A goal that the arm
could not actually hold, or that knocked the mug over on the way, is a bad
optimization target -- ``index.json`` and the contact sheet flag both.

Usage::

    bash scripts/flux2/prepare_svgd_scene.sh multi-object     # writes the run dir
    python experiments/libero/prepare_multi_object_goals.py \
        --run-dir runs/multi_object_arm_preview
"""

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
for _path in (REPO_ROOT, REPO_ROOT / "third_party" / "LIBERO", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import robosuite.utils.transform_utils as T  # noqa: E402

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from sample_endpoint_trajectories import (  # noqa: E402
    _quaternion_error_degrees,
    _rollout_to_target,
    _synchronize_controllers_to_sim_state,
    _views_from_obs,
    _write_json,
    composed_right_view,
    env_from_manifest,
)

SCHEMA_VERSION = 1

# Tabletop height of the LIBERO `main_table` arena: body at z=0.875 with a
# 0.05 m thick top.  Overridden per run by the measured value when available.
DEFAULT_TABLE_SURFACE_Z = 0.90

# A goal whose terminal pose missed by more than this is not a pose the arm can
# hold, so optimizing an image of it would chase an unreachable target.
DEFAULT_POSITION_TOLERANCE_M = 0.02
DEFAULT_ORIENTATION_TOLERANCE_DEG = 10.0
# Objects are supposed to be landmarks, not things the arm shoves around on its
# way to the pose.  Anything past this and the goal image shows a different
# scene than the start image.
DEFAULT_DISTURBANCE_TOLERANCE_M = 0.01


def _object_geom_ids(model: Any, object_name: str) -> list[int]:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = model.geom_id2name(geom_id)
        if name and name.startswith(object_name):
            ids.append(geom_id)
    return ids


def _object_top_z(env: OffScreenRenderEnv, object_name: str) -> float | None:
    """Highest world-frame point of an object, from its geoms' local AABBs.

    ``geom_rbound`` is a bounding *sphere* and badly overestimates flat meshes
    (it puts the top of a plate 10 cm above the table), so this uses MuJoCo's
    per-geom axis-aligned box and rotates it into the world frame.
    """
    sim = getattr(getattr(env, "env", env), "sim", None)
    if sim is None:
        return None
    model, data = sim.model, sim.data
    raw_model = getattr(model, "_model", model)
    aabb = getattr(raw_model, "geom_aabb", None)
    geom_ids = _object_geom_ids(model, object_name)
    if aabb is None or not geom_ids:
        return None
    tops: list[float] = []
    for geom_id in geom_ids:
        center_local = np.asarray(aabb[geom_id][:3], dtype=np.float64)
        half_extent = np.asarray(aabb[geom_id][3:], dtype=np.float64)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        center_world = np.asarray(data.geom_xpos[geom_id], dtype=np.float64) + rotation @ center_local
        # Extent of the rotated box along world +z.
        tops.append(float(center_world[2] + np.abs(rotation[2, :]) @ half_extent))
    return max(tops) if tops else None


def _rotation_about(axis: str, degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    if axis == "y":
        return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    if axis == "z":
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unknown rotation axis: {axis}")


def _goal_quaternion(
    start_quat_xyzw: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """Start orientation pre-rotated in the world frame by yaw/pitch/roll.

    Angles are world-frame because that is the frame robosuite's OSC delta lives
    in, and because the start pose points the gripper straight down: yaw then
    reads as wrist spin, pitch as a tilt toward +x, roll as a tilt toward +y.
    """
    start = T.quat2mat(np.asarray(start_quat_xyzw, dtype=np.float64))
    rotation = _rotation_about("z", yaw_deg) @ _rotation_about("y", pitch_deg) @ _rotation_about("x", roll_deg)
    return np.asarray(T.mat2quat(rotation @ start), dtype=np.float64)


def _default_goal_specs(object_names: list[str]) -> list[dict[str, Any]]:
    """Hover poses over every object plus a few object-free table positions.

    Three variants per object -- straight down, wrist spun, tilted approach --
    because the interesting question is whether the optimizer can distinguish
    *orientation* at a fixed position, not only position.

    The tilt is a roll rather than a pitch on purpose.  Rolling tips the gripper
    toward +/-y, across the reach direction, and costs essentially nothing: 45
    degrees still settles within 2 mm.  Pitching tips it along +x, straight into
    the reach limit, and 25 degrees already misses by 23 mm at x = 0.06.
    """
    specs: list[dict[str, Any]] = []
    variants = (
        ("above", 0.10, {}),
        ("above_yaw90", 0.10, {"yaw_deg": 90.0}),
        ("tilt_roll30", 0.12, {"roll_deg": 30.0}),
    )
    for object_name in object_names:
        stem = object_name.removesuffix("_1")
        for suffix, clearance, rotation in variants:
            specs.append(
                {
                    "id": f"{stem}_{suffix}",
                    "site": {"kind": "object", "name": object_name},
                    "clearance_m": clearance,
                    **rotation,
                }
            )
    # Goals that name a place rather than an object.  All four sit at x <= -0.05,
    # which is both the best-conditioned part of the workspace and the only
    # region of the table with nothing on it -- the plate's 0.09 m radius reaches
    # to x = -0.03, and hovering near the 0.147 m mug at the 1.03 travel height
    # would put the gripper inside it.
    specs.extend(
        [
            {
                "id": "table_center",
                "site": {"kind": "table", "xy": [-0.05, 0.0]},
                "clearance_m": 0.13,
            },
            {
                "id": "table_far_side_yaw45",
                "site": {"kind": "table", "xy": [-0.05, -0.22]},
                "clearance_m": 0.13,
                "yaw_deg": 45.0,
            },
            {
                "id": "table_near_robot_yaw90",
                "site": {"kind": "table", "xy": [-0.14, 0.10]},
                "clearance_m": 0.13,
                "yaw_deg": 90.0,
            },
            {
                "id": "table_high_roll30",
                "site": {"kind": "table", "xy": [-0.05, -0.10]},
                "clearance_m": 0.25,
                "roll_deg": 30.0,
            },
        ]
    )
    return specs


def _resolve_goal_pose(
    spec: dict[str, Any],
    object_positions: dict[str, np.ndarray],
    object_tops: dict[str, float],
    table_surface_z: float,
    start_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    site = spec["site"]
    offset = np.asarray(spec.get("offset_xy_m", [0.0, 0.0]), dtype=np.float64)
    clearance = float(spec["clearance_m"])
    if site["kind"] == "object":
        name = site["name"]
        if name not in object_positions:
            raise KeyError(f"Goal '{spec['id']}' references unknown object '{name}'")
        base_xy = object_positions[name][:2]
        base_z = object_tops[name]
        reference = {"object": name, "object_top_z": base_z}
    elif site["kind"] == "table":
        base_xy = np.asarray(site["xy"], dtype=np.float64)
        base_z = table_surface_z
        reference = {"table_xy": base_xy.tolist(), "table_surface_z": table_surface_z}
    else:
        raise ValueError(f"Goal '{spec['id']}' has unknown site kind '{site['kind']}'")

    position = np.array([base_xy[0] + offset[0], base_xy[1] + offset[1], base_z + clearance])
    yaw = float(spec.get("yaw_deg", 0.0))
    pitch = float(spec.get("pitch_deg", 0.0))
    roll = float(spec.get("roll_deg", 0.0))
    quaternion = _goal_quaternion(start_quat_xyzw, yaw, pitch, roll)
    reference.update({"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll, "clearance_m": clearance})
    return position, quaternion, reference


def _contact_sheet(
    entries: list[dict[str, Any]],
    goals_dir: Path,
    columns: int,
    tile_size: int,
) -> Image.Image:
    # One short line per fact. The default PIL font is about 6 px per character,
    # so a 224 px tile holds roughly 36 -- long enough only if each line stays
    # terse, and a line that overruns silently paints over the next tile.
    line_height = 13
    lines_per_tile = 4
    label_height = lines_per_tile * line_height + 8
    rows = math.ceil(len(entries) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_size, rows * (tile_size + label_height)),
        (24, 24, 28),
    )
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(entries):
        column, row = index % columns, index // columns
        left = column * tile_size
        top = row * (tile_size + label_height)
        view = Image.open(goals_dir / entry["goal_id"] / "goal_agentview.png").convert("RGB")
        sheet.paste(view.resize((tile_size, tile_size), Image.LANCZOS), (left, top))
        # Red beats a footnote: a goal the arm could not hold is not a goal worth
        # running, and that has to be obvious at a glance.
        colour = (245, 245, 245) if entry["usable"] else (255, 120, 120)
        pose = entry["achieved_eef"]
        family = "xyz search" if entry["in_position_only_search_family"] else "needs 6-dof"
        lines = (
            entry["goal_id"],
            f"xyz {pose[0]:+.3f} {pose[1]:+.3f} {pose[2]:.3f}",
            f"rot {entry['rotation_label']} | {family}",
            f"err {entry['position_error_m'] * 1000:.0f}mm "
            f"{entry['orientation_error_deg']:.1f}deg "
            f"nudge {entry['max_object_displacement_m'] * 1000:.0f}mm",
        )
        for line_index, line in enumerate(lines):
            draw.text(
                (left + 5, top + tile_size + 3 + line_index * line_height),
                line,
                fill=colour,
            )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "multi_object_arm_preview"),
        help="Run directory written by sample_endpoint_trajectories.py.",
    )
    parser.add_argument(
        "--goal-specs",
        help="JSON file with a list of goal specs. Defaults to the built-in catalogue.",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated goal ids to (re)render instead of the whole catalogue.",
    )
    parser.add_argument("--move-steps", type=int, default=None, help="Defaults to the manifest value.")
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=24,
        help=(
            "Deliberately longer than the manifest's rollout budget: a goal image only has to show "
            "a pose the arm can hold, and OSC needs more than the usual 8 steps to converge a tilt."
        ),
    )
    parser.add_argument("--controller-gain", type=float, default=None, help="Defaults to the manifest value.")
    parser.add_argument(
        "--rotation-gain",
        type=float,
        default=1.0,
        help="Proportional gain on the commanded OSC orientation delta.",
    )
    parser.add_argument(
        "--arc-height",
        type=float,
        default=0.10,
        help="Meters added to the path midpoint so the arm lifts over the objects in transit.",
    )
    parser.add_argument("--position-tolerance", type=float, default=DEFAULT_POSITION_TOLERANCE_M)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=DEFAULT_ORIENTATION_TOLERANCE_DEG)
    parser.add_argument("--disturbance-tolerance", type=float, default=DEFAULT_DISTURBANCE_TOLERANCE_M)
    parser.add_argument("--contact-sheet-columns", type=int, default=4)
    parser.add_argument("--contact-sheet-tile", type=int, default=224)
    parser.add_argument(
        "--contact-sheet-only",
        action="store_true",
        help="Redraw the sheet from an existing index.json without re-running the simulator.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing goals/ directory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        parser.error(
            f"No manifest at {manifest_path}. Build the scene first:\n"
            f"  RUN_DIR={run_dir} BDDL=experiments/libero/multi_object_table_move_arm.bddl "
            f"bash scripts/flux2/prepare_svgd_scene.sh multi-object"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    goals_dir = run_dir / "goals"

    if args.contact_sheet_only:
        index_path = goals_dir / "index.json"
        if not index_path.exists():
            parser.error(f"--contact-sheet-only needs an existing catalogue at {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        sheet = _contact_sheet(
            index["goals"],
            goals_dir,
            max(args.contact_sheet_columns, 1),
            max(args.contact_sheet_tile, 64),
        )
        sheet.save(goals_dir / "contact_sheet.png")
        print(f"[review] {goals_dir / 'contact_sheet.png'}")
        return

    selected = {token.strip() for token in args.only.split(",") if token.strip()} if args.only else None
    if goals_dir.exists() and selected is None:
        if not args.force:
            parser.error(f"{goals_dir} already exists. Pass --force to replace it, or --only to re-render a subset.")
        shutil.rmtree(goals_dir)
    goals_dir.mkdir(parents=True, exist_ok=True)

    view_size = int(manifest["view_size"])
    move_steps = args.move_steps if args.move_steps is not None else int(manifest["move_steps"])
    settle_steps = args.settle_steps
    controller_gain = (
        args.controller_gain if args.controller_gain is not None else float(manifest["controller_gain"])
    )
    start_state = np.load(run_dir / "start_state.npy")
    start_gripper_actions = [
        np.asarray(action, dtype=np.float64) for action in manifest["start_gripper_controller_actions"]
    ]

    if args.goal_specs:
        specs = json.loads(Path(args.goal_specs).read_text(encoding="utf-8"))
        if not isinstance(specs, list):
            parser.error("--goal-specs must contain a JSON list of goal spec objects")
    else:
        specs = None  # built after the object names are known

    env = env_from_manifest(manifest)
    try:
        env.seed(int(manifest["sim_seed"]))
        env.reset()
        obs = env.set_init_state(start_state)
        _synchronize_controllers_to_sim_state(env, start_gripper_actions)

        start_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        start_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
        object_names = [str(name) for name in env.obj_of_interest]
        object_positions = {
            name: np.asarray(env.env.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64)
            for name in object_names
        }
        object_tops: dict[str, float] = {}
        for name in object_names:
            top = _object_top_z(env, name)
            if top is None:
                raise RuntimeError(f"Could not measure the top of '{name}'; cannot place a hover goal above it")
            object_tops[name] = float(top)
        table_surface_z = DEFAULT_TABLE_SURFACE_Z

        if specs is None:
            specs = _default_goal_specs(object_names)
        if selected is not None:
            known = {spec["id"] for spec in specs}
            unknown = selected - known
            if unknown:
                parser.error(f"Unknown goal ids for --only: {sorted(unknown)}")
            specs = [spec for spec in specs if spec["id"] in selected]

        print(f"[scene] start_eef={np.round(start_eef, 4)}")
        for name in object_names:
            position = object_positions[name]
            print(
                f"[scene] {name:<22s} xy=({position[0]:+.3f}, {position[1]:+.3f}) "
                f"top_z={object_tops[name]:.4f} height={object_tops[name] - table_surface_z:.3f}m"
            )

        entries: list[dict[str, Any]] = []
        for spec in specs:
            goal_id = str(spec["id"])
            goal_dir = goals_dir / goal_id
            goal_dir.mkdir(parents=True, exist_ok=True)

            target_position, target_quat, reference = _resolve_goal_pose(
                spec, object_positions, object_tops, table_surface_z, start_quat
            )

            env.reset()
            obs = env.set_init_state(start_state)
            _synchronize_controllers_to_sim_state(env, start_gripper_actions)
            restored_state = np.asarray(env.get_sim_state(), dtype=np.float64)
            state_drift = float(np.max(np.abs(restored_state - start_state)))
            if state_drift != 0.0:
                raise RuntimeError(f"Non-deterministic restore for '{goal_id}': max state diff={state_drift}")

            obs, actions, eef_path, _ = _rollout_to_target(
                env,
                obs,
                target_position,
                move_steps=move_steps,
                settle_steps=settle_steps,
                gain=controller_gain,
                target_quat_xyzw=target_quat,
                rotation_gain=args.rotation_gain,
                arc_height=float(spec.get("arc_height_m", args.arc_height)),
                view_size=view_size,
            )

            achieved_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
            achieved_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
            position_error = float(np.linalg.norm(achieved_position - target_position))
            orientation_error = _quaternion_error_degrees(achieved_quat, target_quat)
            displacements = {
                name: float(
                    np.linalg.norm(
                        np.asarray(env.env.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64)
                        - object_positions[name]
                    )
                )
                for name in object_names
            }
            max_displacement = max(displacements.values(), default=0.0)

            main_view, right_view_image, composed = _views_from_obs(obs, view_size)
            main_view.save(goal_dir / "goal_agentview.png")
            right_view_image.save(goal_dir / f"goal_{composed_right_view()}.png")
            composed.save(goal_dir / "goal.png")
            np.save(goal_dir / "terminal_state.npy", np.asarray(env.get_sim_state()))
            np.save(goal_dir / "actions.npy", actions.astype(np.float32))
            np.save(goal_dir / "eef_path.npy", eef_path.astype(np.float32))

            usable = (
                position_error <= args.position_tolerance
                and orientation_error <= args.orientation_tolerance_deg
                and max_displacement <= args.disturbance_tolerance
            )
            rotation_label = (
                f"y{reference['yaw_deg']:+.0f} p{reference['pitch_deg']:+.0f} r{reference['roll_deg']:+.0f}"
            )
            # svgd_endpoint.py optimizes a 3-D Cartesian endpoint and calls
            # _rollout_to_target without an orientation target, so every pose it
            # can produce keeps the start orientation.  A goal image showing a
            # rotated gripper is therefore outside its action family: useful as a
            # metric probe, not as something the current search can converge to.
            requested_rotation_deg = max(
                abs(reference["yaw_deg"]), abs(reference["pitch_deg"]), abs(reference["roll_deg"])
            )
            in_position_only_family = requested_rotation_deg == 0.0
            record = {
                "schema_version": SCHEMA_VERSION,
                "goal_id": goal_id,
                "spec": spec,
                "reference": reference,
                "start_eef": start_eef,
                "start_eef_quat_xyzw": start_quat,
                "target_eef": target_position,
                "target_eef_quat_xyzw": target_quat,
                "achieved_eef": achieved_position,
                "achieved_eef_quat_xyzw": achieved_quat,
                "position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "object_displacement_m": displacements,
                "max_object_displacement_m": max_displacement,
                "usable": usable,
                "in_position_only_search_family": in_position_only_family,
                "requested_rotation_deg": requested_rotation_deg,
                "rotation_label": rotation_label,
                "goal_image": f"goals/{goal_id}/goal.png",
                "libero_builtin_success": bool(env.check_success()),
                "move_steps": move_steps,
                "settle_steps": settle_steps,
                "controller_gain": controller_gain,
                "rotation_gain": args.rotation_gain,
                "arc_height_m": float(spec.get("arc_height_m", args.arc_height)),
            }
            _write_json(goal_dir / "metadata.json", record)
            entries.append(record)
            flag = "ok   " if usable else "CHECK"
            family = "xyz" if in_position_only_family else "6dof"
            print(
                f"[{flag}] {goal_id:<28s} pos_err={position_error * 1000:6.1f}mm "
                f"ori_err={orientation_error:5.1f}deg nudge={max_displacement * 1000:5.1f}mm "
                f"needs={family}"
            )

        # A --only re-render must not demote the catalogue to the subset it
        # touched: keep every goal whose images are still on disk, in the order
        # the full catalogue defines, and overwrite just the ones re-rendered.
        if selected is not None:
            rendered = {entry["goal_id"]: entry for entry in entries}
            previous_index = goals_dir / "index.json"
            if previous_index.exists():
                previous = json.loads(previous_index.read_text(encoding="utf-8"))
                merged = [
                    rendered.get(entry["goal_id"], entry)
                    for entry in previous.get("goals", [])
                    if entry["goal_id"] in rendered or (goals_dir / entry["goal_id"] / "goal.png").exists()
                ]
                known = {entry["goal_id"] for entry in merged}
                merged.extend(entry for entry in entries if entry["goal_id"] not in known)
                entries = merged

        sheet = _contact_sheet(
            entries, goals_dir, max(args.contact_sheet_columns, 1), max(args.contact_sheet_tile, 64)
        )
        sheet.save(goals_dir / "contact_sheet.png")

        index = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "bddl": manifest["bddl"],
            "start_image": manifest["start_image"],
            "start_eef": start_eef,
            "table_surface_z": table_surface_z,
            "objects": {
                name: {"position": object_positions[name], "top_z": object_tops[name]} for name in object_names
            },
            "tolerances": {
                "position_m": args.position_tolerance,
                "orientation_deg": args.orientation_tolerance_deg,
                "object_displacement_m": args.disturbance_tolerance,
            },
            "contact_sheet": "goals/contact_sheet.png",
            "goals": entries,
        }
        _write_json(goals_dir / "index.json", index)

        usable_ids = [entry["goal_id"] for entry in entries if entry["usable"]]
        runnable_now = [
            entry["goal_id"]
            for entry in entries
            if entry["usable"] and entry["in_position_only_search_family"]
        ]
        print(f"\n[done] {len(usable_ids)}/{len(entries)} goals within tolerance")
        print(
            f"[note] {len(runnable_now)} of those keep the start orientation and are reachable by "
            "svgd_endpoint.py's 3-D endpoint search; the rotated ones need a 6-DoF search to be "
            "reachable, and are otherwise only useful as latent-metric probes"
        )
        print(f"[review] {goals_dir / 'contact_sheet.png'}")
        print(f"[index]  {goals_dir / 'index.json'}")
        if runnable_now:
            example = runnable_now[0]
            print(
                "\n[next] run one with:\n"
                f"  python -u -B experiments/libero/svgd_endpoint.py \\\n"
                f"    --run-dir {run_dir} \\\n"
                f"    --out-dir {run_dir}/<sweep>/trials/<trial> \\\n"
                f"    --goal {goals_dir / example / 'goal.png'} --goal-latent-source reencode ..."
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
