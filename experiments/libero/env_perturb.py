"""Object-location perturbation for LIBERO init states (OOD layout testing).

A LIBERO init state is a *flattened MuJoCo sim state*:
    [ time(1) , qpos(nq) , qvel(nv) ]
Each free-body object contributes 7 qpos entries: [x, y, z, qw, qx, qy, qz].
So an object's x coordinate lives at flat index  1 + jnt_qposadr[joint]  (the +1
skips the leading time entry). This module edits those entries so we can move
objects to novel locations the model never saw during training.

Driven by an env var so it propagates cleanly to the parallel eval workers:
    export LIBERO_OBJECT_OVERRIDES=/path/to/spec.json

Spec format (JSON):
{
  "objects": {
    "plate_1":         {"x": 0.10, "y": -0.20, "yaw": 0.0},   # absolute pose
    "porcelain_mug_1": {"dx": 0.05, "dy": 0.15}               # relative shift
  },
  "trials": [0, 1, 2]        # optional; null / omitted = apply to every trial
}
Object names are the BDDL object ids (the free joint is "<name>_joint0").
"""

import json
import logging
import math

import numpy as np

_MJ_JNT_FREE = 0  # mjtJoint.mjJNT_FREE


def free_joint_offsets(env):
    """Map object name -> flat init-state index of its x coordinate."""
    m = env.sim.model
    offsets = {}
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != _MJ_JNT_FREE:
            continue
        name = m.joint_id2name(j)
        base = name[: -len("_joint0")] if name.endswith("_joint0") else name
        offsets[base] = 1 + int(m.jnt_qposadr[j])  # +1 for the leading time entry
    return offsets


def _yaw_to_quat(yaw):
    """Yaw about +z -> MuJoCo [w, x, y, z] quaternion."""
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _quat_mul(a, b):
    """Hamilton product of [w, x, y, z] quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def load_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_perturbation(env, initial_states, spec):
    """Return a NEW list of init states with object xy / yaw edited per ``spec``."""
    offsets = free_joint_offsets(env)
    objects = spec.get("objects", {})
    trials = spec.get("trials")  # None -> all trials

    out = []
    for idx, state in enumerate(initial_states):
        s = np.array(state, dtype=np.float64).copy()
        if trials is None or idx in trials:
            for name, mv in objects.items():
                if name not in offsets:
                    raise KeyError(
                        f"object '{name}' not in scene; available objects: {sorted(offsets)}"
                    )
                o = offsets[name]
                if "x" in mv:
                    s[o] = float(mv["x"])
                if "y" in mv:
                    s[o + 1] = float(mv["y"])
                if "z" in mv:
                    s[o + 2] = float(mv["z"])
                if "dx" in mv:
                    s[o] += float(mv["dx"])
                if "dy" in mv:
                    s[o + 1] += float(mv["dy"])
                if "yaw" in mv:  # absolute yaw (overwrites orientation)
                    s[o + 3 : o + 7] = _yaw_to_quat(float(mv["yaw"]))
                if "dyaw" in mv:  # relative yaw about +z, preserves current orientation
                    s[o + 3 : o + 7] = _quat_mul(
                        _yaw_to_quat(float(mv["dyaw"])), s[o + 3 : o + 7]
                    )
            logging.info("[perturb] trial %d: moved %s", idx, list(objects))
        out.append(s)
    return out
