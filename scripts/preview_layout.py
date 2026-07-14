#!/usr/bin/env python
"""Preview a LIBERO layout perturbation WITHOUT running the policy (no GPU needed).

Loads a task's first frozen init state, applies an object-override spec, and
saves before/after agentview PNGs so you can eyeball the layout before spending
a GPU rollout.

Usage:
    python scripts/preview_layout.py \
        --suite libero_10 --task-id 4 \
        --spec perturb_task4.json \
        --out-dir .

Requires (same as eval workers):
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
    PYTHONPATH must include  src  and  third_party/LIBERO  and  experiments/libero
"""

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "third_party" / "LIBERO", REPO / "experiments" / "libero"):
    sys.path.insert(0, str(p))

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark import get_benchmark_dict  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from env_perturb import apply_perturbation, free_joint_offsets, load_spec  # noqa: E402


def _agentview(obs):
    # match training preprocessing (rotate 180)
    return Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task-id", type=int, default=4)
    ap.add_argument("--spec", required=True, help="path to perturbation JSON")
    ap.add_argument("--trial", type=int, default=0, help="which frozen init state to preview")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--settle", type=int, default=20, help="no-op steps to check stability")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    suite = get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print(f"task: {task.language}")

    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=args.res, camera_widths=args.res)
    env.seed(0)
    env.reset()
    states = suite.get_task_init_states(args.task_id)

    offsets = free_joint_offsets(env)
    print("objects in scene (name -> flat x-index):")
    for name, idx in offsets.items():
        print(f"  {name:22s} {idx}")

    # BEFORE
    obs = env.set_init_state(states[args.trial])
    before = out / f"preview_{args.suite}_task{args.task_id}_trial{args.trial}_before.png"
    _agentview(obs).save(before)

    # AFTER
    spec = load_spec(args.spec)
    perturbed = apply_perturbation(env, states, spec)
    obs = env.set_init_state(perturbed[args.trial])
    after = out / f"preview_{args.suite}_task{args.task_id}_trial{args.trial}_after.png"
    _agentview(obs).save(after)

    m = env.sim.model
    def z_of(name):
        a = m.jnt_qposadr[m.joint_name2id(name + "_joint0")]
        return float(env.sim.data.qpos[a + 2])

    moved = list(spec.get("objects", {}).keys())

    # Table height differs per scene, so derive the reference from the UNPERTURBED
    # scene rather than hardcoding it: settle the original, record each object's z.
    env.set_init_state(states[args.trial])
    for _ in range(args.settle):
        env.step([0, 0, 0, 0, 0, 0, -1])
    ref_z = {n: z_of(n) for n in moved}

    # Now settle the perturbed scene and compare.
    env.set_init_state(perturbed[args.trial])
    for _ in range(args.settle):
        env.step([0, 0, 0, 0, 0, 0, -1])

    print(f"\nmoved objects: {moved}")
    ok = True
    for n in moved:
        z, r = z_of(n), ref_z[n]
        fell = z < r - 0.05  # dropped well below its normal resting height
        ok &= not fell
        print(f"  {n:22s} z={z:.3f}  (unperturbed rests at {r:.3f}) {'*** FELL ***' if fell else 'ok'}")
    print("RESULT:", "stable — all objects on the table" if ok else "UNSTABLE — object(s) fell; adjust the spec")
    print(f"\nsaved:\n  {before}\n  {after}")
    env.close()


if __name__ == "__main__":
    main()
