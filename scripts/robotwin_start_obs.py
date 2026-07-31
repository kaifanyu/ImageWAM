#!/usr/bin/env python
"""Grab a starting observation from a RoboTwin *test* episode -- no policy, no GPU DiT.

This is the RoboTwin counterpart of the LIBERO `preview_layout.py` path: it stands
up the same episode the evaluator would stand up, freezes it at t=0, and writes out
everything `run_image_edit_robotwin.py` needs to predict a goal frame:

    start.png        the 256x288 three-camera composite the checkpoint was trained on
    proprio.npy      the raw 14-dim joint_action vector at t=0 (un-normalized)
    instruction.txt  the sampled language instruction
    meta.json        task / seed / config provenance

Seed protocol -- this is the part that matters
----------------------------------------------
RoboTwin's evaluator does not roll out arbitrary seeds. It walks seeds from
`100000 * (1 + seed)` upward and keeps only those where the scripted expert can
plan AND succeed (`script/eval_policy.py::eval_policy`). A rejected seed is a
degenerate scene, not a hard one. We replicate that filter exactly, so the frame
you get is drawn from the same distribution the reported numbers come from.

Cost: each candidate seed runs a full expert rollout to completion, so expect
tens of seconds per accepted episode.

Usage
-----
python scripts/robotwin_start_obs.py \
    --task place_dual_shoes \
    --out-dir runs/robotwin_preview/place_dual_shoes
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = Path(
    os.environ.get("ROBOTWIN_DIR", REPO_ROOT / "third_party" / "RoboTwin")
).resolve()

# Camera composite geometry. Must stay in lockstep with
# experiments/robotwin/imagewam_policy/deploy_policy.py::_robotwin_camera_sizes --
# the checkpoint has no tolerance for a different layout.
CAMERA_SIZES = {
    "compact_288x256": ((256, 192), (128, 96), (128, 96)),
    "legacy_384x320": ((320, 256), (160, 128), (160, 128)),
}


@contextlib.contextmanager
def robotwin_cwd():
    """RoboTwin resolves ./task_config, ./assets, ./envs relative to its own root."""
    previous = Path.cwd()
    os.chdir(ROBOTWIN_ROOT)
    for p in (".", "./description/utils", "./script"):
        resolved = str((ROBOTWIN_ROOT / p).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    try:
        yield
    finally:
        os.chdir(previous)


def compose(obs, layout: str) -> Image.Image:
    """head on top, [left | right] wrist underneath -- deploy_policy's exact layout."""
    head_size, left_size, right_size = CAMERA_SIZES[layout]

    def resize(rgb, size_wh):
        return np.asarray(
            Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").resize(
                size_wh, resample=Image.BILINEAR
            ),
            dtype=np.uint8,
        )

    data = obs["observation"]
    head = resize(data["head_camera"]["rgb"], head_size)
    left = resize(data["left_camera"]["rgb"], left_size)
    right = resize(data["right_camera"]["rgb"], right_size)
    return Image.fromarray(
        np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
    )


def resolve_planner(requested: str) -> str:
    """RoboTwin's aloha-agilex config asks for curobo, which is an optional install.

    curobo is only used by the *scripted expert*, never by the policy or the
    renderer, so falling back to mplib_screw changes which seeds pass the expert
    check -- not what the camera sees.
    """
    if requested != "auto":
        return requested
    try:
        import curobo  # noqa: F401

        return "curobo"
    except ImportError:
        print("[planner] curobo not installed; falling back to mplib_screw")
        return "mplib_screw"


def patch_out_curobo() -> None:
    """Let a scene stand up without curobo installed.

    RoboTwin's `Robot.set_planner` builds a `CuroboPlanner` unconditionally -- the
    `planner:` key in the embodiment config only selects the mplib TOPP planner,
    it does not replace curobo. So on a machine without curobo, `setup_demo()`
    raises before any rendering happens, even though the planner is irrelevant to
    what the cameras see.

    This swaps in a set_planner that builds only the mplib planners and points
    `left_planner`/`right_planner` at them. Scene construction, object placement,
    domain randomization and rendering are untouched, so the start frame is
    byte-identical to what curobo would have produced. What you lose is the
    *scripted expert*: it cannot plan, so `--no-expert-check` becomes mandatory
    and seeds are no longer filtered for solvability.
    """
    from envs.robot import robot as robot_module
    from envs.robot.planner import MplibPlanner

    def set_planner(self, scene=None):
        self.communication_flag = False
        self.left_planner = MplibPlanner(
            self.left_urdf_path, self.left_srdf_path, self.left_move_group,
            self.left_entity_origion_pose, self.left_entity, self.left_planner_type, scene,
        )
        self.right_planner = MplibPlanner(
            self.right_urdf_path, self.right_srdf_path, self.right_move_group,
            self.right_entity_origion_pose, self.right_entity, self.right_planner_type, scene,
        )
        self.left_mplib_planner = self.left_planner
        self.right_mplib_planner = self.right_planner

    robot_module.Robot.set_planner = set_planner
    # reset() re-runs set_planner unless the existing planners are CuroboPlanner
    # instances; with curobo gone that check would loop us back here every reset,
    # which is harmless but wasteful -- make the isinstance test pass instead.
    robot_module.CuroboPlanner = MplibPlanner


def build_task_args(task_name: str, task_config: str, planner: str) -> dict:
    """Replicate script/eval_policy.py::main's config assembly."""
    from envs import CONFIGS_PATH

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["policy_name"] = "start_obs_probe"
    args["ckpt_setting"] = None
    args["eval_mode"] = True
    args["render_freq"] = 0
    # No rollout happens here, so there is nothing to record; leaving this on would
    # make the env look for an ffmpeg sink we never open.
    args["eval_video_log"] = False

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        cameras = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[head_camera_type]["h"]
    args["head_camera_w"] = cameras[head_camera_type]["w"]

    from eval_policy import get_embodiment_config

    embodiment_type = args["embodiment"]
    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiments[embodiment_type[0]]["file_path"]
        args["right_robot_file"] = embodiments[embodiment_type[0]]["file_path"]
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiments[embodiment_type[0]]["file_path"]
        args["right_robot_file"] = embodiments[embodiment_type[1]]["file_path"]
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["left_embodiment_config"]["planner"] = planner
    args["right_embodiment_config"]["planner"] = planner
    return args


def find_expert_verified_seed(task_env, args, start_seed: int, max_tries: int):
    """Walk seeds like eval_policy does; return (seed, episode_info) for the first keeper."""
    from envs.utils.create_actor import UnStableError

    seed = start_seed
    for attempt in range(max_tries):
        try:
            task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
            episode_info = task_env.play_once()
            accepted = bool(task_env.plan_success and task_env.check_success())
            task_env.close_env()
        except UnStableError:
            task_env.close_env()
            seed += 1
            continue
        except Exception as exc:  # a bad seed, not a bad setup -- keep walking
            print(f"[seed {seed}] rejected: {type(exc).__name__}: {exc}")
            with contextlib.suppress(Exception):
                task_env.close_env()
            seed += 1
            continue

        if accepted:
            print(f"[seed {seed}] accepted after {attempt + 1} candidate(s)")
            return seed, episode_info
        print(f"[seed {seed}] expert did not succeed; trying next")
        seed += 1

    raise RuntimeError(
        f"No expert-verified seed within {max_tries} tries from {start_seed}. "
        "Either the task is misnamed or its assets are missing."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", required=True, help="RoboTwin task name, e.g. place_dual_shoes.")
    ap.add_argument("--task-config", default="demo_randomized",
                    help="demo_randomized is the eval protocol; demo_clean is the easy variant.")
    ap.add_argument("--instruction-type", default="unseen",
                    choices=["seen", "unseen"],
                    help="Matches configs/sim_robotwin.yaml (unseen).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Evaluator seed index; the actual sim seed is 100000*(1+seed).")
    ap.add_argument("--max-seed-tries", type=int, default=20)
    ap.add_argument("--planner", default="auto",
                    choices=["auto", "curobo", "mplib_screw", "mplib_RRT"],
                    help="Scripted-expert motion planner. Only affects which seeds "
                         "pass the expert check, never the rendered frame.")
    ap.add_argument("--no-expert-check", action="store_true",
                    help="Take the first seed without running the expert rollout. "
                         "Fast, but the scene is no longer guaranteed solvable, so it "
                         "is NOT the same distribution the reported numbers use.")
    ap.add_argument("--layout", default="compact_288x256", choices=sorted(CAMERA_SIZES))
    ap.add_argument("--out-dir", required=True)
    args_cli = ap.parse_args()

    out_dir = Path(args_cli.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with robotwin_cwd():
        from eval_policy import class_decorator
        from generate_episode_instructions import generate_episode_descriptions

        planner = resolve_planner(args_cli.planner)
        if planner != "curobo":
            patch_out_curobo()
            if not args_cli.no_expert_check:
                ap.error(
                    "The scripted expert requires curobo, which is not installed. "
                    "Re-run with --no-expert-check to take the first seed as-is "
                    "(see this script's docstring for what that costs)."
                )
        args = build_task_args(args_cli.task, args_cli.task_config, planner)
        task_env = class_decorator(args_cli.task)

        start_seed = 100000 * (1 + args_cli.seed)
        if args_cli.no_expert_check:
            seed, episode_info = start_seed, None
        else:
            seed, episode_info = find_expert_verified_seed(
                task_env, args, start_seed, args_cli.max_seed_tries
            )

        # Re-stand the accepted scene and freeze it at t=0.
        task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        if episode_info is not None:
            results = generate_episode_descriptions(args_cli.task, [episode_info["info"]], 1)
            instruction = str(np.random.choice(results[0][args_cli.instruction_type]))
        else:
            # generate_episode_descriptions needs the expert's episode_info to fill the
            # {A}/{a}/<...> slots in the templates, so without it we fall back to the
            # task's generic full_description.
            template_path = (
                ROBOTWIN_ROOT / "description" / "task_instruction" / f"{args_cli.task}.json"
            )
            raw = json.loads(template_path.read_text(encoding="utf-8"))["full_description"]
            instruction = raw.replace("<", "").replace(">", "")
            print("[warn] --no-expert-check: using the generic task description, not a "
                  "per-episode sampled instruction")
        task_env.set_instruction(instruction=instruction)

        obs = task_env.get_obs()
        image = compose(obs, args_cli.layout)
        state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32).reshape(-1)
        task_env.close_env()

    image.save(out_dir / "start.png")
    np.save(out_dir / "proprio.npy", state)
    (out_dir / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "task": args_cli.task,
                "task_config": args_cli.task_config,
                "instruction_type": args_cli.instruction_type,
                "instruction": instruction,
                "seed_index": args_cli.seed,
                "sim_seed": int(seed),
                "planner": planner,
                "expert_checked": not args_cli.no_expert_check,
                "layout": args_cli.layout,
                "image_size_wh": list(image.size),
                "proprio_dim": int(state.shape[0]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n[instruction] {instruction!r}")
    print(f"[proprio] {state.shape[0]}-dim raw state")
    print(f"[saved] {out_dir}/start.png  ({image.size[0]}x{image.size[1]})")
    print(f"[saved] {out_dir}/proprio.npy, instruction.txt, meta.json")


if __name__ == "__main__":
    main()
