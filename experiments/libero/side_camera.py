"""A pure side-profile camera for the LIBERO endpoint scenes.

``agentview`` sits in front of the robot on the ``+x`` axis and looks back along
``-x`` with a downward tilt.  The arm motion these scenes optimise runs along
``y``, which agentview resolves well -- but height and reach both project onto
roughly the same image direction there, so a goal image built from agentview
alone under-determines where the gripper is in the remaining two axes.

This module aims a second camera straight down the ``y`` axis at end-effector
height: the table is seen edge on, ``x`` runs across the image and ``z`` runs up
it.  Together the two views span all three world axes.

LIBERO's arenas already ship a camera named ``sideview``, but it is parked high
above the table on a three-quarter tilt.  Rather than rewrite the vendored arena
XML (``third_party/LIBERO`` is not tracked by this repo) the pose is overwritten
on the live ``mjModel``.  ``mjModel.cam_pos``/``cam_quat`` feed ``mj_camlight``
during ``mj_forward``, so a write plus a ``forward()`` is all a fixed camera
needs -- with one catch: robosuite rebuilds the model from XML on every hard
reset, which silently restores the stock pose.  :func:`install_side_camera`
therefore wraps ``env.reset`` so the pose is re-applied afterwards, and
:func:`open_env` is the constructor that should be used in place of a bare
``OffScreenRenderEnv``.

Typical use::

    from side_camera import open_env

    env = open_env(bddl_file_name=manifest["bddl"], render_size=256)

and, when reproducing a run whose framing was recorded at scene-prep time::

    env = open_env(..., side_camera=manifest["side_camera"])
"""

from __future__ import annotations

from typing import Any

import numpy as np

from libero.libero.envs import OffScreenRenderEnv

from scene_geometry import table_geometry

# The arena camera whose pose is overwritten.  Its observation key is
# ``f"{CAMERA_NAME}_image"``, which is what ``_views_from_obs`` reads.
CAMERA_NAME = "sideview"
IMAGE_KEY = f"{CAMERA_NAME}_image"

# Cameras every endpoint-pipeline env renders.  ``agentview`` and the wrist are
# LIBERO's stock pair; the third is the side profile.
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand", CAMERA_NAME]

# Defaults for :func:`side_camera_pose`, in meters / degrees.
DEFAULT_MARGIN = 1.10
DEFAULT_HEIGHT = 0.15
DEFAULT_ELEVATION_DEG = 0.0
DEFAULT_FOVY = 45.0

# Extents for an arena whose table ``scene_geometry`` cannot measure -- the
# living-room and study fixtures are meshes rather than named box geoms, so
# ``table_geometry`` returns None there.  The *height* is never guessed: LIBERO
# mounts the robot flush with its work surface, so the robot base z is the table
# top to within a couple of centimetres in every arena, and a hardcoded 0.90
# would put the camera half a metre above a living-room table.
_FALLBACK_HALF_X = 0.50
_FALLBACK_HALF_Y = 0.60


def _mat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """MuJoCo stores camera orientation as a ``(w, x, y, z)`` quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = np.array(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    return quat / np.linalg.norm(quat)


def _look_at_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orientation of a camera at ``eye`` pointed at ``target``, world ``+z`` up.

    A MuJoCo camera looks down its local ``-z`` with local ``+y`` up.  The stock
    LIBERO cameras are built the same way, so the resulting render needs the same
    ``[::-1, ::-1]`` correction that ``_views_from_obs`` already applies to
    agentview and the wrist.
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        raise ValueError("Side camera position and target coincide")
    backward = -forward / norm
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        raise ValueError("Side camera cannot look straight up or down")
    right = right / right_norm
    up = np.cross(backward, right)
    return _mat_to_quat_wxyz(np.stack([right, up, backward], axis=1))


def side_camera_pose(
    env: Any,
    *,
    margin: float = DEFAULT_MARGIN,
    height: float = DEFAULT_HEIGHT,
    elevation_deg: float = DEFAULT_ELEVATION_DEG,
    x_center: float | None = None,
    fovy: float = DEFAULT_FOVY,
) -> dict[str, Any]:
    """Derive a side-profile pose from the arena's own geometry.

    The camera is placed on the ``+y`` axis ``margin`` beyond the table edge,
    ``height`` above the work surface, and aimed horizontally at the point it is
    level with -- a pure profile.  ``elevation_deg`` raises the camera and tilts
    it back down onto the same aim point, which trades some of the edge-on look
    for a little visible tabletop.

    The work surface comes from ``scene_geometry.table_geometry`` when the arena
    has a measurable table and from the robot mount otherwise; the result records
    which, as ``table_top_z_source``.

    ``x_center`` defaults to the midpoint of the arm's reach across the table --
    from the robot base to the far table edge -- so the frame is centred on the
    span the runs actually work in rather than on the table or the robot alone.
    """
    base = _robot_base(env)
    table = table_geometry(env) or {}
    center = np.asarray(
        table.get("center", (0.0, 0.0, base[2])), dtype=np.float64
    ).reshape(3)
    half_extents = (
        np.asarray(table["half_extents"], dtype=np.float64).reshape(3)
        if table.get("half_extents") is not None
        else np.array([_FALLBACK_HALF_X, _FALLBACK_HALF_Y, 0.0])
    )
    half_x, half_y = float(half_extents[0]), float(half_extents[1])
    top_z = float(table["top_z"]) if "top_z" in table else float(base[2])
    surface_source = table.get("source", "robot0_base z (no measurable table)")

    if x_center is None:
        # The robot sits off the -x edge and reaches across, so the span worth
        # framing runs from its base to the far edge of the table.
        x_center = float((base[0] + center[0] + half_x) / 2.0)

    aim_z = top_z + float(height)
    distance = float(half_y) + float(margin)
    target = np.array([float(x_center), float(center[1]), aim_z], dtype=np.float64)

    elevation = np.radians(float(elevation_deg))
    position = target + np.array(
        [0.0, distance * np.cos(elevation), distance * np.sin(elevation)], dtype=np.float64
    )
    return {
        "camera": CAMERA_NAME,
        "position": position.tolist(),
        "target": target.tolist(),
        "quaternion_wxyz": _look_at_quat(position, target).tolist(),
        "fovy_deg": float(fovy),
        "table_top_z": top_z,
        "table_top_z_source": surface_source,
        "margin_m": float(margin),
        "height_above_table_m": float(height),
        "elevation_deg": float(elevation_deg),
    }


def _robot_base(env: Any) -> np.ndarray:
    """World position of the robot mount.

    LIBERO bolts the robot to the work surface, so this doubles as the surface
    height whenever ``table_geometry`` cannot measure the table itself.
    """
    for holder in (env, getattr(env, "env", None)):
        sim = getattr(holder, "sim", None)
        if sim is None:
            continue
        try:
            body = int(sim.model.body_name2id("robot0_base"))
            return np.asarray(sim.data.body_xpos[body], dtype=np.float64).copy()
        except Exception:
            continue
    raise RuntimeError(
        "Cannot locate 'robot0_base'; pass an explicit side-camera pose instead"
    )


def apply_side_camera(env: Any, pose: dict[str, Any]) -> None:
    """Write ``pose`` onto the live model and refresh the derived camera frame."""
    sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        raise RuntimeError("Environment exposes no MuJoCo sim")
    name = str(pose.get("camera", CAMERA_NAME))
    camera_id = int(sim.model.camera_name2id(name))
    quaternion = pose.get("quaternion_wxyz")
    if quaternion is None:
        quaternion = _look_at_quat(
            np.asarray(pose["position"], dtype=np.float64),
            np.asarray(pose["target"], dtype=np.float64),
        )
    sim.model.cam_pos[camera_id] = np.asarray(pose["position"], dtype=np.float64)
    sim.model.cam_quat[camera_id] = np.asarray(quaternion, dtype=np.float64)
    sim.model.cam_fovy[camera_id] = float(pose.get("fovy_deg", DEFAULT_FOVY))
    # cam_xpos/cam_xmat are recomputed from the model during mj_fwdPosition.
    sim.forward()


def _rerender(env: Any, fallback: Any) -> Any:
    """Rebuild the observation dict against the camera poses now in the model."""
    try:
        env._update_observables(force=True)
        return env.env._get_observations()
    except Exception:
        return fallback


def install_side_camera(env: Any, pose: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Apply the side-profile pose and keep it applied across ``env.reset()``.

    ``pose`` reuses a framing recorded in a run manifest; omit it to derive one
    from the arena via :func:`side_camera_pose`.  Returns the pose in use.
    """
    if pose is None:
        pose = side_camera_pose(env, **kwargs)
    elif kwargs:
        raise TypeError("Pass either an explicit pose or the arguments to derive one")
    apply_side_camera(env, pose)

    if getattr(env, "_side_camera_reset_wrapped", False):
        env._side_camera_pose = pose
        return pose

    inner_reset = env.reset

    def reset_with_side_camera(*args: Any, **reset_kwargs: Any) -> Any:
        # A hard reset rebuilds the sim from XML, restoring the arena's own pose.
        observation = inner_reset(*args, **reset_kwargs)
        apply_side_camera(env, env._side_camera_pose)
        # reset() already rendered, so that observation still holds the stock
        # framing; re-render it rather than hand back a stale side panel.
        return _rerender(env, observation)

    env._side_camera_pose = pose
    env._side_camera_reset_wrapped = True
    env.reset = reset_with_side_camera
    return pose


def installed_pose(env: Any) -> dict[str, Any] | None:
    """The pose :func:`install_side_camera` put on ``env``, if any."""
    return getattr(env, "_side_camera_pose", None)


def open_env(
    *,
    bddl_file_name: str,
    render_size: int,
    side_camera: dict[str, Any] | bool = True,
    **kwargs: Any,
) -> OffScreenRenderEnv:
    """``OffScreenRenderEnv`` that also renders the side-profile camera.

    ``side_camera`` may be ``True`` (derive a pose from the arena), a pose dict
    from a manifest, or ``False`` to build a stock two-camera env.
    """
    camera_names = list(kwargs.pop("camera_names", CAMERA_NAMES if side_camera else CAMERA_NAMES[:2]))
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file_name),
        camera_heights=int(render_size),
        camera_widths=int(render_size),
        camera_names=camera_names,
        **kwargs,
    )
    if side_camera is not False:
        install_side_camera(env, side_camera if isinstance(side_camera, dict) else None)
    return env
