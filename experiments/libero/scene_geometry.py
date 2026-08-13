"""World-frame scene geometry pulled out of a live LIBERO/robosuite env.

Rollout traces record where the end effector went; they say nothing about the
scene it moved through, so a 3D plot of them floats in empty space.  This module
supplies the missing landmarks -- the table top, the robot base, the arm links,
and the tracked objects -- in the same world frame as ``eef_path``.

Two products:

* :func:`capture_scene` -- one snapshot written next to a run as ``scene.json``;
* :func:`arm_link_positions` -- per-step link positions for the rollout trace,
  which is what lets the viewer draw the whole arm instead of one point.

Everything degrades to ``None``/empty rather than raising: a visualisation aid
must never be able to kill a multi-hour optimisation.

Runs that finished before this existed can be retrofitted without re-running the
optimiser -- the landmarks only depend on the start state::

    python experiments/libero/scene_geometry.py runs/living_room_mug_obstacle
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

# robosuite prefixes every model's bodies; these are the ones that make up the
# manipulator itself, in kinematic order.
ARM_BODY_PREFIXES = ("robot0_", "gripper0_")
# Bodies that exist for mounting/inertial reasons and only add noise to a skeleton.
ARM_BODY_EXCLUDE = ("_eef_visual", "_visual", "mount", "base_plate")


def _sim(env: Any) -> Any | None:
    """The MuJoCo sim behind a LIBERO ``OffScreenRenderEnv`` (or robosuite env)."""
    for holder in (env, getattr(env, "env", None)):
        sim = getattr(holder, "sim", None)
        if sim is not None and getattr(sim, "model", None) is not None:
            return sim
    return None


def _names(model: Any, kind: str) -> list[str]:
    names = getattr(model, f"{kind}_names", None)
    return [str(name) for name in names] if names is not None else []


def _body_id(model: Any, name: str) -> int | None:
    try:
        return int(model.body_name2id(name))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# arm
# --------------------------------------------------------------------------- #


def arm_link_names(env: Any) -> list[str]:
    """Body names forming the arm chain, ordered from base to gripper.

    MuJoCo numbers bodies in kinematic-tree order, so body id order *is* chain
    order for a serial manipulator.
    """
    sim = _sim(env)
    if sim is None:
        return []
    try:
        return [
            name
            for name in _names(sim.model, "body")
            if name.startswith(ARM_BODY_PREFIXES)
            and not any(token in name for token in ARM_BODY_EXCLUDE)
        ]
    except Exception:
        return []


def arm_link_parents(env: Any, names: list[str]) -> list[int]:
    """Index of each link's parent within ``names`` (-1 when the parent is outside).

    Drawing the skeleton from parent links rather than list order keeps the
    gripper fingers attached to the hand instead of to each other.
    """
    sim = _sim(env)
    if sim is None or not names:
        return [-1] * len(names)
    try:
        parent_ids = np.asarray(sim.model.body_parentid)
        position = {_body_id(sim.model, name): index for index, name in enumerate(names)}
        parents = []
        for name in names:
            body = _body_id(sim.model, name)
            parent = int(parent_ids[body]) if body is not None else None
            parents.append(position.get(parent, -1))
        return parents
    except Exception:
        return [-1] * len(names)


def arm_link_positions(env: Any, names: list[str]) -> np.ndarray:
    """World positions of ``names`` right now -- shape ``(len(names), 3)``."""
    sim = _sim(env)
    if sim is None or not names:
        return np.zeros((0, 3), dtype=np.float64)
    try:
        xpos = np.asarray(sim.data.body_xpos, dtype=np.float64)
        out = np.full((len(names), 3), np.nan, dtype=np.float64)
        for index, name in enumerate(names):
            body = _body_id(sim.model, name)
            if body is not None:
                out[index] = xpos[body]
        return out
    except Exception:
        return np.zeros((0, 3), dtype=np.float64)


# --------------------------------------------------------------------------- #
# static scene
# --------------------------------------------------------------------------- #


def _table_from_attributes(env: Any) -> dict[str, Any] | None:
    """robosuite table arenas expose the top-centre offset and the full size."""
    for holder in (env, getattr(env, "env", None)):
        offset = getattr(holder, "table_offset", None)
        full = getattr(holder, "table_full_size", None)
        if offset is None or full is None:
            continue
        try:
            offset = np.asarray(offset, dtype=np.float64).reshape(3)
            full = np.asarray(full, dtype=np.float64).reshape(3)
        except Exception:
            continue
        return {
            "top_z": float(offset[2]),
            "center": offset.tolist(),
            "half_extents": (full / 2.0).tolist(),
            "source": "env.table_offset",
        }
    return None


def _table_from_geoms(sim: Any) -> dict[str, Any] | None:
    """Largest box geom whose name mentions a table, in world coordinates.

    LIBERO names its table fixtures per scene (``living_room_table_...``), so the
    name is matched loosely and the widest candidate wins.
    """
    try:
        import mujoco

        box = int(mujoco.mjtGeom.mjGEOM_BOX)
    except Exception:
        box = 6  # mjGEOM_BOX
    try:
        names = _names(sim.model, "geom")
        sizes = np.asarray(sim.model.geom_size, dtype=np.float64)
        types = np.asarray(sim.model.geom_type)
        positions = np.asarray(sim.data.geom_xpos, dtype=np.float64)
    except Exception:
        return None
    best: dict[str, Any] | None = None
    best_area = 0.0
    for index, name in enumerate(names):
        if "table" not in name.lower() or int(types[index]) != box:
            continue
        half = sizes[index]
        area = float(4.0 * half[0] * half[1])
        if area <= best_area:
            continue
        center = positions[index]
        best_area = area
        best = {
            "top_z": float(center[2] + half[2]),
            "center": center.tolist(),
            "half_extents": half[:3].tolist(),
            "source": f"geom:{name}",
        }
    return best


def table_geometry(env: Any) -> dict[str, Any] | None:
    """Table top plane and extent, or ``None`` when the arena has no table."""
    from_attributes = _table_from_attributes(env)
    if from_attributes is not None:
        return from_attributes
    sim = _sim(env)
    return _table_from_geoms(sim) if sim is not None else None


def _body_radius(sim: Any, body: int) -> float | None:
    """Bounding radius of the geoms attached to one body."""
    try:
        owners = np.asarray(sim.model.geom_bodyid)
        bounds = np.asarray(sim.model.geom_rbound, dtype=np.float64)
        radii = bounds[owners == body]
        return float(radii.max()) if radii.size else None
    except Exception:
        return None


def object_geometry(env: Any) -> list[dict[str, Any]]:
    """Pose and rough size of every BDDL object of interest."""
    sim = _sim(env)
    objects: list[dict[str, Any]] = []
    try:
        names = [str(name) for name in env.obj_of_interest]
    except Exception:
        return objects
    for name in names:
        entry: dict[str, Any] = {"name": name}
        try:
            state = env.env.object_states_dict[name].get_geom_state()
            entry["position"] = np.asarray(state["pos"], dtype=np.float64).tolist()
            entry["quaternion_wxyz"] = np.asarray(state["quat"], dtype=np.float64).tolist()
        except Exception:
            pass
        if sim is not None:
            body = next(
                (
                    _body_id(sim.model, candidate)
                    for candidate in _names(sim.model, "body")
                    if candidate == name or candidate == f"{name}_main"
                ),
                None,
            )
            if body is not None:
                entry["radius"] = _body_radius(sim, body)
        objects.append(entry)
    return objects


def robot_base(env: Any) -> dict[str, Any] | None:
    """World position of the robot's base body -- the origin of the workspace."""
    sim = _sim(env)
    if sim is None:
        return None
    for name in ("robot0_base", "robot0_link0", "robot0_fixed_base_link"):
        body = _body_id(sim.model, name)
        if body is None:
            continue
        try:
            position = np.asarray(sim.data.body_xpos[body], dtype=np.float64)
        except Exception:
            return None
        return {"name": name, "position": position.tolist()}
    return None


def capture_scene(env: Any, **extras: Any) -> dict[str, Any]:
    """Snapshot every landmark the 3D viewer can draw.

    Call this once the environment is at the run's start state -- the arm pose
    and object poses recorded here are that state, and the viewer labels them so.
    """
    names = arm_link_names(env)
    positions = arm_link_positions(env, names)
    scene: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "table": table_geometry(env),
        "robot_base": robot_base(env),
        "arm": {
            "link_names": names,
            "link_parents": arm_link_parents(env, names),
            "start_positions": np.round(positions, 5).tolist() if len(positions) else [],
        },
        "objects": object_geometry(env),
    }
    scene.update(extras)
    return scene


def write_scene_json(path: Path, env: Any, **extras: Any) -> dict[str, Any] | None:
    """Write ``scene.json`` beside a run; never raises, returns what it wrote."""
    try:
        scene = capture_scene(env, **extras)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(scene), indent=2), encoding="utf-8")
        return scene
    except Exception as error:  # a missing landmark must not end a run
        print(f"[scene] capture skipped: {type(error).__name__}: {error}", flush=True)
        return None


def _load_start_state(run_dir: Path, manifest: dict[str, Any]) -> Any:
    candidates = [Path(str(manifest.get("start_state", ""))), run_dir / "start_state.npy"]
    for candidate in candidates:
        if candidate.name and candidate.is_file():
            return np.load(candidate)
    raise FileNotFoundError(f"no start_state.npy for {run_dir}")


def main() -> None:
    """Retrofit ``scene.json`` onto a run that finished before scene capture existed.

    Replays nothing: it rebuilds the environment from the run manifest, restores
    the same start state the optimiser used, and records the landmarks.  Cheap
    enough to run over every old run, and the 3D viewer picks the file up
    automatically for every trial underneath it.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir", type=Path, help="directory holding manifest.json and start_state.npy"
    )
    parser.add_argument("--out", type=Path, help="output path (default <run_dir>/scene.json)")
    arguments = parser.parse_args()

    run_dir = arguments.run_dir.expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from libero.libero.envs import OffScreenRenderEnv
    from sample_endpoint_trajectories import _synchronize_controllers_to_sim_state

    env = OffScreenRenderEnv(
        bddl_file_name=str(manifest["bddl"]),
        camera_heights=int(manifest.get("render_size", 256)),
        camera_widths=int(manifest.get("render_size", 256)),
    )
    try:
        env.seed(int(manifest.get("sim_seed", 0)))
        env.reset()
        obs = env.set_init_state(_load_start_state(run_dir, manifest))
        gripper = manifest.get("start_gripper_controller_actions")
        _synchronize_controllers_to_sim_state(
            env, [np.asarray(action) for action in gripper] if gripper else None
        )
        out = arguments.out or run_dir / "scene.json"
        scene = write_scene_json(
            out,
            env,
            start_eef=np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist(),
            goal_eef=manifest.get("physical_goal_eef"),
        )
    finally:
        env.close()
    if scene is None:
        raise SystemExit("scene capture failed")
    table = scene.get("table")
    print(
        f"wrote {out}\n"
        + (f"  table top z={table['top_z']:.3f} ({table['source']})\n" if table else "  no table\n")
        + f"  {len(scene['arm']['link_names'])} arm links, {len(scene['objects'])} objects"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


if __name__ == "__main__":
    main()
