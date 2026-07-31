#!/usr/bin/env python
"""SVGD over minimum-jerk path shape for the upright-mug avoidance test.

Each particle is [midpoint_x, arc_height] in metres. The terminal EEF target is
fixed. The optimizer sees only terminal FLUX autoencoder distance to a reference
image where the mug remains upright; physical mug and endpoint measurements are
held-out diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (HERE, REPO_ROOT / "src", REPO_ROOT / "third_party" / "LIBERO"):
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
from score_endpoint_candidates import FluxAutoencoderMetric  # noqa: E402
from svgd_endpoint import (  # noqa: E402
    _cap_updates,
    _optimizer_latent_metrics,
    _svgd_step,
    _view_latent,
)


def _quaternion_delta_degrees(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    a = np.asarray(a_wxyz, dtype=np.float64)
    b = np.asarray(b_wxyz, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cosine_half_angle = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine_half_angle))


def _save_trace(
    path: Path,
    parameters: np.ndarray,
    target_eef: np.ndarray,
    actions: np.ndarray,
    eef_path: np.ndarray,
    trace: dict[str, list[Any]],
) -> None:
    np.savez(
        path,
        path_parameters=np.asarray(parameters, dtype=np.float64),
        midpoint_x_m=float(parameters[0]),
        arc_height_m=float(parameters[1]),
        target_eef=np.asarray(target_eef, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float32),
        eef_path=np.asarray(eef_path, dtype=np.float64),
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
        object_names=np.asarray(trace["tracked_object_names"], dtype=str),
        object_positions=np.asarray(trace["object_positions"], dtype=np.float64),
        object_quaternions_wxyz=np.asarray(
            trace["object_quaternions_wxyz"], dtype=np.float64
        ),
    )


class PathEnergy:
    def __init__(
        self,
        env: OffScreenRenderEnv,
        start_state: np.ndarray,
        gripper_actions: list[np.ndarray],
        fixed_target: np.ndarray,
        diagnostic_goal: np.ndarray,
        initial_object_quaternions: np.ndarray,
        encoder: FluxAutoencoderMetric,
        goal_latent: np.ndarray,
        *,
        move_steps: int,
        settle_steps: int,
        gain: float,
        view_size: int,
        latent_views: str,
        distance_metric: str,
        goal_tolerance: float,
        mug_displacement_tolerance: float,
        mug_orientation_tolerance_deg: float,
        out_dir: Path,
        trace_mode: str,
        verbose: bool,
    ) -> None:
        self.env = env
        self.start_state = start_state
        self.gripper_actions = gripper_actions
        self.fixed_target = fixed_target
        self.diagnostic_goal = diagnostic_goal
        self.initial_object_quaternions = initial_object_quaternions
        self.encoder = encoder
        self.goal_latent = _view_latent(goal_latent, view_size, latent_views)
        self.move_steps = move_steps
        self.settle_steps = settle_steps
        self.gain = gain
        self.view_size = view_size
        self.latent_views = latent_views
        self.distance_metric = distance_metric
        self.goal_tolerance = goal_tolerance
        self.mug_displacement_tolerance = mug_displacement_tolerance
        self.mug_orientation_tolerance_deg = mug_orientation_tolerance_deg
        self.out_dir = out_dir
        self.trace_mode = trace_mode
        self.verbose = verbose
        self.rollouts = 0
        self.events_path = out_dir / "evaluations.jsonl"
        self.events_path.write_text("", encoding="utf-8")

    def __call__(
        self,
        parameters: np.ndarray,
        *,
        iteration: int,
        particle: int,
        evaluation: str,
        capture_video: bool = False,
    ) -> dict[str, Any]:
        parameters = np.asarray(parameters, dtype=np.float64)
        self.env.reset()
        obs = self.env.set_init_state(self.start_state)
        _synchronize_controllers_to_sim_state(
            self.env, self.gripper_actions
        )
        collect_trace = (
            capture_video
            or evaluation in {"base", "base_repeat", "replay"}
            or self.trace_mode == "all"
        )
        trace: dict[str, list[Any]] | None = {} if collect_trace else None
        obs, actions, eef_path, frames = _rollout_to_target(
            self.env,
            obs,
            self.fixed_target,
            move_steps=self.move_steps,
            settle_steps=self.settle_steps,
            gain=self.gain,
            arc_height=float(parameters[1]),
            midpoint_x=float(parameters[0]),
            capture_video=capture_video,
            video_stride=2,
            view_size=self.view_size,
            trace=trace,
        )
        self.rollouts += 1
        _, _, terminal_image = _views_from_obs(obs, self.view_size)
        latent = _view_latent(
            self.encoder.encode(terminal_image),
            self.view_size,
            self.latent_views,
        )
        metrics = _optimizer_latent_metrics(latent, self.goal_latent)
        terminal_eef = np.asarray(eef_path[-1], dtype=np.float64)
        target_tracking_error = float(
            np.linalg.norm(terminal_eef - self.fixed_target)
        )
        goal_error = float(np.linalg.norm(terminal_eef - self.diagnostic_goal))
        object_motion: dict[str, dict[str, Any]] = {}
        orientation_deltas: dict[str, float] = {}
        mug_displacement: float | None = None
        mug_rotation: float | None = None
        mug_preserved = False
        if trace is not None and trace["tracked_object_names"]:
            object_motion = _object_motion_summary(trace, eef_path)
            terminal_quaternions = np.asarray(
                trace["object_quaternions_wxyz"][-1], dtype=np.float64
            )
            for index, name in enumerate(trace["tracked_object_names"]):
                orientation_deltas[name] = _quaternion_delta_degrees(
                    self.initial_object_quaternions[index],
                    terminal_quaternions[index],
                )
            mug_name = str(trace["tracked_object_names"][0])
            mug_displacement = float(
                object_motion[mug_name]["terminal_displacement_m"]
            )
            mug_rotation = float(orientation_deltas[mug_name])
            mug_preserved = bool(
                mug_displacement <= self.mug_displacement_tolerance
                and mug_rotation <= self.mug_orientation_tolerance_deg
            )
        success = bool(goal_error <= self.goal_tolerance and mug_preserved)

        should_save = (
            trace is not None
            and self.trace_mode != "none"
            and (
                self.trace_mode == "all"
                or evaluation in {"base", "base_repeat", "replay"}
            )
        )
        trace_file: str | None = None
        if should_save:
            trace_dir = self.out_dir / f"iter_{iteration:03d}" / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"particle_{particle:02d}_{evaluation}.npz"
            _save_trace(
                trace_path,
                parameters,
                self.fixed_target,
                actions,
                eef_path,
                trace,
            )
            trace_file = str(trace_path.relative_to(self.out_dir))

        event = {
            "rollout": self.rollouts,
            "iteration": iteration,
            "particle": particle,
            "evaluation": evaluation,
            "path_parameters": {
                "midpoint_x_m": float(parameters[0]),
                "arc_height_m": float(parameters[1]),
            },
            "objective": self.distance_metric,
            "energy": float(metrics[self.distance_metric]),
            "latent_metrics": metrics,
            "terminal_eef": terminal_eef,
            "commanded_target_eef": self.fixed_target,
            "target_tracking_error_m": target_tracking_error,
            "goal_error_m": goal_error,
            "object_motion": object_motion,
            "object_orientation_delta_deg": orientation_deltas,
            "mug_terminal_displacement_m": mug_displacement,
            "mug_orientation_delta_deg": mug_rotation,
            "mug_preserved": mug_preserved,
            "physical_success": success,
            "trace_file": trace_file,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    event,
                    default=lambda value: value.tolist()
                    if isinstance(value, np.ndarray)
                    else value,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
        if self.verbose:
            print(
                f"[eval {self.rollouts:05d}] iter={iteration:03d} "
                f"particle={particle:02d} kind={evaluation} "
                f"lateral={parameters[0]:+.4f} arc={parameters[1]:.4f} "
                f"E={metrics[self.distance_metric]:.7f} success={success}",
                flush=True,
            )
        return {
            "energy": float(metrics[self.distance_metric]),
            "latent_metrics": metrics,
            "terminal_image": terminal_image,
            "terminal_eef": terminal_eef,
            "target_tracking_error_m": target_tracking_error,
            "goal_error_m": goal_error,
            "mug_displacement_m": mug_displacement,
            "mug_rotation_deg": mug_rotation,
            "mug_preserved": mug_preserved,
            "success": success,
            "object_motion": object_motion,
            "orientation_deltas": orientation_deltas,
            "trace_file": trace_file,
            "actions": actions,
            "eef_path": eef_path,
            "frames": frames,
            "trace": trace,
        }


def _finite_difference_gradient(
    energy_fn: PathEnergy,
    parameters: np.ndarray,
    epsilons: np.ndarray,
    bounds: np.ndarray,
    *,
    iteration: int,
    particle: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    gradient = np.zeros_like(parameters)
    probes: list[dict[str, Any]] = []
    names = ("midpoint_x", "arc_height")
    for dimension, name in enumerate(names):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[dimension] = min(
            plus[dimension] + epsilons[dimension], bounds[dimension, 1]
        )
        minus[dimension] = max(
            minus[dimension] - epsilons[dimension], bounds[dimension, 0]
        )
        span = float(plus[dimension] - minus[dimension])
        plus_result = energy_fn(
            plus,
            iteration=iteration,
            particle=particle,
            evaluation=f"fd_{name}_plus",
        )
        minus_result = energy_fn(
            minus,
            iteration=iteration,
            particle=particle,
            evaluation=f"fd_{name}_minus",
        )
        gradient[dimension] = (
            plus_result["energy"] - minus_result["energy"]
        ) / span
        probes.append(
            {
                "parameter": name,
                "span_m": span,
                "plus": plus,
                "minus": minus,
                "plus_energy": plus_result["energy"],
                "minus_energy": minus_result["energy"],
                "gradient": float(gradient[dimension]),
            }
        )
    return gradient, probes


def _plot_history(
    history: list[dict[str, Any]],
    path: Path,
    safe_goal_parameters: dict[str, float],
) -> None:
    parameter_history = np.asarray(
        [record["parameters"] for record in history], dtype=np.float64
    )
    iterations = np.asarray([record["iteration"] for record in history])
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    path_axis, energy_axis, success_axis, motion_axis = axes.ravel()
    for particle in range(parameter_history.shape[1]):
        path_axis.plot(
            100.0 * parameter_history[:, particle, 0],
            100.0 * parameter_history[:, particle, 1],
            marker=".",
            alpha=0.65,
        )
    path_axis.scatter(
        100.0 * safe_goal_parameters["midpoint_x_m"],
        100.0 * safe_goal_parameters["arc_height_m"],
        marker="*",
        s=180,
        color="tab:green",
        label="reference safe path",
    )
    path_axis.set(
        title="Particle paths through trajectory-parameter space",
        xlabel="lateral midpoint offset (cm)",
        ylabel="vertical arc height (cm)",
    )
    path_axis.legend()
    energy_axis.plot(
        iterations,
        [record["energy_mean"] for record in history],
        marker="o",
        label="mean",
    )
    energy_axis.plot(
        iterations,
        [record["energy_min"] for record in history],
        marker=".",
        label="min",
    )
    energy_axis.set(
        title="Image-latent objective",
        xlabel="evaluated population",
        ylabel="energy",
    )
    energy_axis.legend()
    success_axis.plot(
        iterations,
        [100.0 * record["success_fraction"] for record in history],
        marker="o",
    )
    success_axis.set(
        title="Held-out physical success",
        xlabel="evaluated population",
        ylabel="successful particles (%)",
        ylim=(-2, 102),
    )
    motion_axis.plot(
        iterations,
        [
            100.0 * record["mug_displacement_mean_m"]
            for record in history
        ],
        marker="o",
        label="mean mug displacement",
    )
    motion_axis.plot(
        iterations,
        [100.0 * record["goal_error_mean_m"] for record in history],
        marker=".",
        label="mean endpoint error",
    )
    motion_axis.set(
        title="Physical diagnostics",
        xlabel="evaluated population",
        ylabel="distance (cm)",
    )
    motion_axis.legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.3)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_replay(replay: dict[str, Any], path: Path) -> None:
    trace = replay["trace"]
    if trace is None or not trace["tracked_object_names"]:
        raise RuntimeError("The selected replay is missing its mug trajectory")

    actual = np.asarray(replay["eef_path"], dtype=np.float64)
    desired = np.asarray(trace["desired_eefs"], dtype=np.float64)
    object_positions = np.asarray(
        trace["object_positions"], dtype=np.float64
    )
    mug = object_positions[:, 0]
    states = np.arange(actual.shape[0])
    center_distance = np.linalg.norm(actual - mug, axis=1)
    mug_displacement = np.linalg.norm(mug - mug[0], axis=1)

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    spatial = figure.add_subplot(2, 2, 1, projection="3d")
    top_down = figure.add_subplot(2, 2, 2)
    shape_axis = figure.add_subplot(2, 2, 3)
    distance_axis = figure.add_subplot(2, 2, 4)

    spatial.plot(
        desired[:, 0],
        desired[:, 1],
        desired[:, 2],
        linestyle="--",
        color="0.45",
        label="requested EEF path",
    )
    spatial.plot(
        actual[:, 0],
        actual[:, 1],
        actual[:, 2],
        color="tab:blue",
        linewidth=2.2,
        label="actual EEF path",
    )
    spatial.plot(
        mug[:, 0],
        mug[:, 1],
        mug[:, 2],
        color="tab:red",
        linewidth=1.8,
        label="mug center",
    )
    spatial.scatter(
        *actual[0], marker="*", s=150, color="tab:blue", label="start"
    )
    spatial.scatter(
        *actual[-1], marker="X", s=110, color="tab:green", label="terminal"
    )
    spatial.scatter(
        *mug[0], marker="D", s=80, color="tab:red", label="mug start"
    )
    spatial.set(
        title="3D end-effector and mug paths",
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
    )
    spatial.view_init(elev=24, azim=-58)
    spatial.legend(fontsize=8)

    top_down.plot(
        desired[:, 0],
        desired[:, 1],
        linestyle="--",
        color="0.45",
        label="requested EEF path",
    )
    top_down.plot(
        actual[:, 0],
        actual[:, 1],
        color="tab:blue",
        linewidth=2.2,
        label="actual EEF path",
    )
    top_down.plot(
        mug[:, 0],
        mug[:, 1],
        color="tab:red",
        linewidth=1.8,
        label="mug center",
    )
    top_down.scatter(actual[0, 0], actual[0, 1], marker="*", s=150)
    top_down.scatter(
        actual[-1, 0],
        actual[-1, 1],
        marker="X",
        s=110,
        color="tab:green",
    )
    top_down.scatter(
        mug[0, 0], mug[0, 1], marker="D", s=80, color="tab:red"
    )
    top_down.set(
        title="Top-down obstacle avoidance",
        xlabel="X (m)",
        ylabel="Y (m)",
    )
    top_down.set_aspect("equal", adjustable="box")
    top_down.legend(fontsize=8)

    shape_axis.plot(
        states,
        100.0 * (actual[:, 0] - actual[0, 0]),
        label="actual lateral X offset",
    )
    shape_axis.plot(
        states,
        100.0 * (actual[:, 2] - actual[0, 2]),
        label="actual vertical Z offset",
    )
    shape_axis.set(
        title="Executed path shape",
        xlabel="controller state index",
        ylabel="offset from start (cm)",
    )
    shape_axis.legend(fontsize=8)

    distance_axis.plot(
        states,
        100.0 * center_distance,
        color="tab:blue",
        label="EEF to mug center",
    )
    distance_axis.plot(
        states,
        100.0 * mug_displacement,
        color="tab:red",
        label="mug displacement",
    )
    distance_axis.set(
        title="Physical interaction diagnostics",
        xlabel="controller state index",
        ylabel="distance (cm)",
    )
    distance_axis.legend(fontsize=8)
    for axis in (top_down, shape_axis, distance_axis):
        axis.grid(alpha=0.3)

    figure.suptitle(
        "Objective-selected mug-avoidance trajectory "
        f"(physical success: {replay['success']})",
        fontsize=15,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-dir")
    parser.add_argument(
        "--target-eef",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Commanded terminal EEF inferred by an earlier endpoint search. "
            "Defaults to the test manifest goal."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--editor-ae", required=True)
    parser.add_argument("--flux2-src", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--init-mode",
        choices=["collision-cloud", "uniform"],
        default="collision-cloud",
    )
    parser.add_argument(
        "--path-bounds",
        type=float,
        nargs=4,
        default=[-0.14, 0.14, 0.0, 0.16],
        metavar=("LATERAL_MIN", "LATERAL_MAX", "ARC_MIN", "ARC_MAX"),
    )
    parser.add_argument(
        "--collision-cloud",
        type=float,
        nargs=4,
        default=[-0.04, 0.04, 0.0, 0.04],
        metavar=("LATERAL_MIN", "LATERAL_MAX", "ARC_MIN", "ARC_MAX"),
    )
    parser.add_argument("--fd-eps", type=float, nargs=2, default=[0.01, 0.01])
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--repulsion-weight", type=float, default=0.0)
    parser.add_argument(
        "--transport", choices=["svgd", "particle_gd"], default="svgd"
    )
    parser.add_argument("--max-update-norm", type=float, default=0.02)
    parser.add_argument(
        "--latent-distance",
        choices=["rms", "cosine", "token_cosine"],
        default="token_cosine",
    )
    parser.add_argument(
        "--latent-views",
        choices=["agentview", "wrist", "both"],
        default="agentview",
    )
    parser.add_argument(
        "--trace-mode", choices=["none", "base", "all"], default="base"
    )
    parser.add_argument("--save-all-particles", action="store_true")
    parser.add_argument("--verbose-evaluations", action="store_true")
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.particles < 2 or args.iterations <= 0:
        parser.error("--particles must be >=2 and --iterations must be positive")
    if args.transport == "particle_gd" and args.repulsion_weight != 0.0:
        parser.error("particle_gd requires --repulsion-weight 0")
    if min(args.fd_eps) <= 0.0 or args.temperature <= 0.0:
        parser.error("--fd-eps and --temperature must be positive")

    run_dir = Path(args.run_dir).expanduser().resolve()
    test_dir = (
        Path(args.test_dir).expanduser().resolve()
        if args.test_dir
        else run_dir / "mug_avoidance_test"
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    test_manifest = json.loads(
        (test_dir / "manifest.json").read_text(encoding="utf-8")
    )
    goal_image = Image.open(test_dir / test_manifest["goal_image"]).convert(
        "RGB"
    )
    goal_image.save(out_dir / "goal_reference.png")
    encoder = FluxAutoencoderMetric(
        Path(args.editor_ae).resolve(),
        Path(args.flux2_src).resolve(),
        args.device,
    )
    goal_latent = encoder.encode(goal_image)

    start_state = np.load(run_dir / run_manifest["start_state"])
    diagnostic_goal = np.asarray(
        test_manifest["physical_goal_eef"], dtype=np.float64
    )
    fixed_target = (
        np.asarray(args.target_eef, dtype=np.float64)
        if args.target_eef is not None
        else diagnostic_goal.copy()
    )
    gripper_actions = [
        np.asarray(action)
        for action in run_manifest["start_gripper_controller_actions"]
    ]
    bounds_flat = np.asarray(args.path_bounds, dtype=np.float64)
    bounds = np.asarray(
        [[bounds_flat[0], bounds_flat[1]], [bounds_flat[2], bounds_flat[3]]]
    )
    cloud_flat = np.asarray(args.collision_cloud, dtype=np.float64)
    collision_cloud = np.asarray(
        [[cloud_flat[0], cloud_flat[1]], [cloud_flat[2], cloud_flat[3]]]
    )
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        parser.error("Each path bound needs min < max")
    if np.any(collision_cloud[:, 0] < bounds[:, 0]) or np.any(
        collision_cloud[:, 1] > bounds[:, 1]
    ):
        parser.error("--collision-cloud must be inside --path-bounds")

    env = OffScreenRenderEnv(
        bddl_file_name=run_manifest["bddl"],
        camera_heights=int(run_manifest["render_size"]),
        camera_widths=int(run_manifest["render_size"]),
    )
    history: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None
    try:
        env.seed(int(run_manifest["sim_seed"]))
        env.reset()
        obs = env.set_init_state(start_state)
        _synchronize_controllers_to_sim_state(env, gripper_actions)
        object_names, _, initial_object_quaternions = _tracked_object_poses(env)
        if len(object_names) != 1:
            raise RuntimeError(f"Expected one tracked mug, found {object_names}")
        energy_fn = PathEnergy(
            env,
            start_state,
            gripper_actions,
            fixed_target,
            diagnostic_goal,
            initial_object_quaternions,
            encoder,
            goal_latent,
            move_steps=int(test_manifest["move_steps"]),
            settle_steps=int(test_manifest["settle_steps"]),
            gain=float(test_manifest["controller_gain"]),
            view_size=int(run_manifest["view_size"]),
            latent_views=args.latent_views,
            distance_metric=args.latent_distance,
            goal_tolerance=float(test_manifest["goal_tolerance_m"]),
            mug_displacement_tolerance=float(
                test_manifest["mug_displacement_tolerance_m"]
            ),
            mug_orientation_tolerance_deg=float(
                test_manifest["mug_orientation_tolerance_deg"]
            ),
            out_dir=out_dir,
            trace_mode=args.trace_mode,
            verbose=args.verbose_evaluations,
        )
        rng = np.random.default_rng(args.seed)
        initialization_bounds = (
            collision_cloud if args.init_mode == "collision-cloud" else bounds
        )
        particles = rng.uniform(
            initialization_bounds[:, 0],
            initialization_bounds[:, 1],
            size=(args.particles, 2),
        )
        initial_particles = particles.copy()
        print(
            f"[plan] particles={args.particles} iterations={args.iterations} "
            f"init={args.init_mode} objective={args.latent_distance} "
            f"views={args.latent_views} transport={args.transport}",
            flush=True,
        )
        print(
            f"[plan] fixed_target={fixed_target.tolist()} "
            f"path_bounds={bounds.tolist()}",
            flush=True,
        )

        def write_history() -> None:
            _write_json(
                out_dir / "history.json",
                {
                    "created_at_utc": dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat(),
                    "run_dir": run_dir,
                    "test_dir": test_dir,
                    "config": vars(args),
                    "fixed_target_eef": fixed_target,
                    "fixed_target_is_optimizer_input": True,
                    "diagnostic_goal_eef": diagnostic_goal,
                    "diagnostic_goal_is_optimizer_input": False,
                    "tracked_object": object_names[0],
                    "initial_particles": initial_particles,
                    "history": history,
                    "global_best": global_best,
                },
            )

        def evaluate_population(
            population: np.ndarray,
            iteration: int,
            compute_gradients: bool,
        ) -> dict[str, Any]:
            nonlocal global_best
            iteration_dir = out_dir / f"iter_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            energies = np.zeros(args.particles)
            metrics = {
                name: np.zeros(args.particles)
                for name in ("rms", "cosine", "token_cosine")
            }
            goal_errors = np.zeros(args.particles)
            mug_displacements = np.zeros(args.particles)
            mug_rotations = np.zeros(args.particles)
            successes = np.zeros(args.particles, dtype=bool)
            gradients = np.zeros_like(population)
            probes: list[list[dict[str, Any]]] = []
            trace_files: list[str | None] = []
            best_index = 0
            for index, parameters in enumerate(population):
                result = energy_fn(
                    parameters,
                    iteration=iteration,
                    particle=index,
                    evaluation="base",
                )
                energies[index] = result["energy"]
                for name in metrics:
                    metrics[name][index] = result["latent_metrics"][name]
                goal_errors[index] = result["goal_error_m"]
                mug_displacements[index] = result["mug_displacement_m"]
                mug_rotations[index] = result["mug_rotation_deg"]
                successes[index] = result["success"]
                trace_files.append(result["trace_file"])
                if args.save_all_particles:
                    result["terminal_image"].save(
                        iteration_dir / f"particle_{index:02d}.png"
                    )
                if energies[index] < energies[best_index]:
                    best_index = index
                if global_best is None or energies[index] < global_best["energy"]:
                    global_best = {
                        "energy": float(energies[index]),
                        "iteration": iteration,
                        "particle": index,
                        "parameters": parameters.copy(),
                        "physical_success": bool(successes[index]),
                        "mug_displacement_m": float(mug_displacements[index]),
                        "mug_rotation_deg": float(mug_rotations[index]),
                        "goal_error_m": float(goal_errors[index]),
                    }
                if compute_gradients:
                    gradients[index], particle_probes = (
                        _finite_difference_gradient(
                            energy_fn,
                            parameters,
                            np.asarray(args.fd_eps),
                            bounds,
                            iteration=iteration,
                            particle=index,
                        )
                    )
                    probes.append(particle_probes)
                else:
                    probes.append([])
            best_result = energy_fn(
                population[best_index],
                iteration=iteration,
                particle=best_index,
                evaluation="base_repeat",
            )
            best_result["terminal_image"].save(iteration_dir / "best.png")
            return {
                "parameters": population.copy(),
                "energies": energies,
                "metrics": metrics,
                "goal_errors": goal_errors,
                "mug_displacements": mug_displacements,
                "mug_rotations": mug_rotations,
                "successes": successes,
                "gradients": gradients,
                "probes": probes,
                "trace_files": trace_files,
                "best_index": best_index,
            }

        for iteration in range(args.iterations + 1):
            final_population = iteration == args.iterations
            evaluation = evaluate_population(
                particles.copy(), iteration, not final_population
            )
            record = {
                "iteration": iteration,
                "phase": "final_evaluation" if final_population else "update",
                "parameters": evaluation["parameters"],
                "energies": evaluation["energies"],
                "energy_mean": float(evaluation["energies"].mean()),
                "energy_min": float(evaluation["energies"].min()),
                "latent_metrics": evaluation["metrics"],
                "goal_errors_m": evaluation["goal_errors"],
                "goal_error_mean_m": float(evaluation["goal_errors"].mean()),
                "mug_displacements_m": evaluation["mug_displacements"],
                "mug_displacement_mean_m": float(
                    evaluation["mug_displacements"].mean()
                ),
                "mug_rotations_deg": evaluation["mug_rotations"],
                "physical_success": evaluation["successes"],
                "success_fraction": float(evaluation["successes"].mean()),
                "energy_gradients": evaluation["gradients"],
                "finite_difference_probes": evaluation["probes"],
                "trace_files": evaluation["trace_files"],
                "best_particle": evaluation["best_index"],
                "update_applied": not final_population,
            }
            if final_population:
                record["particles_after_update"] = particles.copy()
                history.append(record)
                write_history()
                print(
                    f"[final {iteration:03d}] E_mean={record['energy_mean']:.6f} "
                    f"success={100.0 * record['success_fraction']:.1f}% "
                    f"mug_move={100.0 * record['mug_displacement_mean_m']:.2f}cm",
                    flush=True,
                )
                break

            scores = -evaluation["gradients"] / args.temperature
            if args.transport == "particle_gd":
                direction = args.latent_weight * scores
                bandwidth = None
                latent_direction = scores
                repulsion_direction = np.zeros_like(scores)
            else:
                (
                    direction,
                    bandwidth,
                    latent_direction,
                    repulsion_direction,
                    _,
                ) = _svgd_step(
                    particles,
                    scores,
                    args.bandwidth_scale,
                    latent_weight=args.latent_weight,
                    repulsion_weight=args.repulsion_weight,
                )
            raw_update = args.step_size * direction
            capped_update, trust_scales = _cap_updates(
                raw_update, args.max_update_norm
            )
            unclipped = particles + capped_update
            particles = np.clip(unclipped, bounds[:, 0], bounds[:, 1])
            record.update(
                {
                    "scores": scores,
                    "kernel_bandwidth": bandwidth,
                    "latent_directions": latent_direction,
                    "repulsion_directions": repulsion_direction,
                    "raw_updates": raw_update,
                    "trust_region_scales": trust_scales,
                    "applied_updates": particles
                    - evaluation["parameters"],
                    "bounds_clipped": unclipped != particles,
                    "particles_after_update": particles.copy(),
                }
            )
            history.append(record)
            write_history()
            print(
                f"[iter {iteration:03d}] E_mean={record['energy_mean']:.6f} "
                f"success={100.0 * record['success_fraction']:.1f}% "
                f"mug_move={100.0 * record['mug_displacement_mean_m']:.2f}cm",
                flush=True,
            )

        if global_best is None:
            raise RuntimeError("No path particle was evaluated")
        selected_parameters = np.asarray(
            global_best["parameters"], dtype=np.float64
        )
        replay = energy_fn(
            selected_parameters,
            iteration=args.iterations + 1,
            particle=int(global_best["particle"]),
            evaluation="replay",
            capture_video=True,
        )
        replay["terminal_image"].save(out_dir / "best_terminal.png")
        np.save(
            out_dir / "best_actions.npy",
            replay["actions"].astype(np.float32),
        )
        np.save(
            out_dir / "best_eef_path.npy",
            replay["eef_path"].astype(np.float32),
        )
        _write_video(
            out_dir / "best_rollout.mp4", replay["frames"], args.video_fps
        )
        _plot_replay(replay, out_dir / "best_trajectory.png")
        best_metadata = {
            "selection_rule": "lowest image-latent objective; no physical filter",
            "selected_parameters": {
                "midpoint_x_m": float(selected_parameters[0]),
                "arc_height_m": float(selected_parameters[1]),
            },
            "selection": global_best,
            "replay": {
                "energy": replay["energy"],
                "latent_metrics": replay["latent_metrics"],
                "goal_error_m": replay["goal_error_m"],
                "target_tracking_error_m": replay[
                    "target_tracking_error_m"
                ],
                "mug_displacement_m": replay["mug_displacement_m"],
                "mug_rotation_deg": replay["mug_rotation_deg"],
                "mug_preserved": replay["mug_preserved"],
                "physical_success": replay["success"],
                "trace_file": replay["trace_file"],
            },
        }
        _write_json(out_dir / "best_metadata.json", best_metadata)
        _plot_history(
            history,
            out_dir / "progress.png",
            test_manifest["safe_goal_parameters"],
        )
        write_history()
    finally:
        env.close()

    print(f"[done] {out_dir}")
    print(f"[done] best: {out_dir / 'best_metadata.json'}")
    print(f"[done] plot: {out_dir / 'progress.png'}")


if __name__ == "__main__":
    main()
