#!/usr/bin/env python
"""SVGD over whole joint-setpoint trajectories, scored in image-feature space.

Where ``svgd_endpoint.py`` makes a particle a 3-D terminal end-effector position
and lets a fixed minimum-jerk helper invent the path, here **the particle is the
trajectory**:

    U_i in R^[H, 8]     u_t = [q1 ... q7, gripper]   (H = --horizon, default 300)

``u_t[:7]`` are *absolute desired joint positions in radians* at control step
``t``; ``u_t[7]`` is the gripper command.  The environment is built with
robosuite's ``JOINT_POSITION`` controller, whose action vector for a Panda is
exactly 8 wide, so a particle row is one controller command.  Because that
controller takes a *delta* scaled onto +/-``output_max`` rad per step, the
rollout converts each absolute setpoint into the delta that reaches it:

    a_t[:7] = clip((u_t[:7] - q_t) / output_max, -1, 1)

The objective is unchanged from the endpoint runs: encode the terminal
``[agentview | wrist]`` render with the frozen FLUX.2 (or DINOv3) encoder and
measure ``--latent-distance`` against the goal image's latent.

    E(U) = latent_distance(z(image_T), z_goal),   log p = -E / temperature

WHY THE GRADIENT IS NOT FINITE DIFFERENCES
------------------------------------------
A particle has ``H * 8 = 2400`` coordinates.  Central differences would cost
4800 rollouts per particle per iteration; one 300-step rollout takes ~36 s on
this machine, so a single iteration would take three weeks.  Instead the
gradient is assembled analytically through the same chain the Experiment-B
runs use (``latent_jacobian.py``), which needs **no extra rollouts at all**:

    dE/du_t = (dq_s/du_t)^T  J_zq^T  dL/dz

    dL/dz   autograd through the distance -- exact, free
    J_zq    dz/dq at the scored configuration, central differences over the 7
            joints via *kinematic* re-renders (14 renders, no physics)
    dq_s/du_t   a servo model: with tight tracking the arm reaches u_t within a
            step or two, so credit for the scored frame is spread backwards over
            the setpoints by --credit-mode.

Measured on this scene, the JOINT_POSITION servo lands within ~1e-3 rad of its
setpoint, which means ``last-only`` credit is the *literally* correct
sensitivity and also useless: it collapses a 2400-D search onto the last row.
``uniform`` credit (the default) instead treats U as one trajectory-valued
parameter and translates every free row along the same joint direction; the
start anchor and the slew-rate projection then reshape that translation into a
smooth ramp out of the true start pose.  ``--energy-mode eventually`` is the
honest fix if you want the *interior* of the trajectory to carry its own signal:
it scores K sampled frames and softmins them (the STL "eventually" operator), so
the scored step -- and therefore where credit lands -- moves along the horizon.

The probe in ``probe_latent_jacobian.py`` reports that ``J_zq`` has usable
direction but unreliable magnitude on this scene, which is why every update goes
through a trust region (``--max-joint-step-rad`` elementwise, then
``--max-update-norm`` on the flattened particle).

FEASIBILITY PROJECTION
----------------------
After every SVGD update each particle is projected back onto the set of
trajectories the controller can actually execute:

1. joint limits with ``--joint-limit-margin``,
2. slew rate ``|u_{t+1} - u_t| <= --max-setpoint-rate`` (a setpoint further than
   ``output_max`` ahead of the arm is simply saturated and thrown away),
3. the first ``--anchor-start-steps`` rows pinned to the measured start
   configuration, so every trajectory begins where the arm actually is.

CHECKPOINTS
-----------
``--checkpoint-every N`` writes ``checkpoints/iter_XXX.npz`` (particles, RNG
state, global best, per-particle Jacobians, rollout counter) plus
``checkpoints/latest.json``.  ``--resume auto`` picks up the newest checkpoint in
``--out-dir``, truncates ``history.json`` back to it, and continues -- so
re-running the identical command after a crash or a raised ``--max-iterations``
just carries on.

Written to <out-dir>/:
    goal_reference.png            the goal image being optimized toward
    scene.json                    static landmarks for the 3D viewer
    history.json                  per-iteration record (viewer input)
    iter_XXX/particle_NN.png      terminal render per particle
    iter_XXX/best.png             lowest-energy particle that iteration
    iter_XXX/traces/particle_NN_base.npz    rollout trace (viewer input)
    checkpoints/iter_XXX.npz      resume points
    evaluations.jsonl             one record per rollout
    best_*                        replay of the global lowest-energy trajectory
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from latent_jacobian import (  # noqa: E402
    JointLatentObserver,
    LatentJacobianEstimator,
)
from sample_endpoint_trajectories import (  # noqa: E402
    _synchronize_controllers_to_sim_state,
    _tracked_object_poses,
    _views_from_obs,
    _write_video,
    env_from_manifest,
)
from scene_geometry import arm_link_names, arm_link_positions, write_scene_json  # noqa: E402
from score_endpoint_candidates import (  # noqa: E402
    DinoV3FeatureMetric,
    FluxAutoencoderMetric,
)
from svgd_endpoint import (  # noqa: E402
    _cap_updates,
    _encode_view_features,
    _optimizer_latent_metrics,
    _svgd_step,
    _view_latent,
)

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

ARM_JOINTS = 7
ACTION_DIM = 8  # 7 arm joints + 1 gripper, matching JointPositionController


def _jsonable(value: Any) -> Any:
    """JSON-safe conversion that recurses through arrays *and* drops non-finites.

    The two existing helpers each do half of this: ``sample_endpoint_trajectories``
    handles ``Path`` but leaves a NaN inside a converted array, and
    ``latent_jacobian`` drops NaN but not ``Path``.  Jacobian prediction error is
    legitimately undefined on the first iteration, so ``allow_nan=False`` needs
    both halves at once.
    """
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def rebind_observer(observer: JointLatentObserver, env: OffScreenRenderEnv) -> None:
    """Point a ``JointLatentObserver`` back at the *live* simulator.

    LIBERO's ``reset()`` runs a hard reset, which rebuilds the model and hands
    back a brand-new ``MjSim``.  Every rollout does that, so the handles the
    observer cached when it was constructed are dangling by the time the
    Jacobian probes run -- ``sim.data`` on the dead object raises.  The joint
    *indices* are stable for a fixed model, so only the handles are refreshed.
    """
    inner = getattr(env, "env", env)
    observer._inner = inner
    observer.sim = inner.sim
    robot = inner.robots[0]
    observer._qpos_index = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    observer._qvel_index = np.asarray(robot._ref_joint_vel_indexes, dtype=int)


# --------------------------------------------------------------------------- #
# differentiable latent distances
# --------------------------------------------------------------------------- #


def _metric_loss_and_grad(
    tokens: np.ndarray,
    goal_tokens: np.ndarray,
    metric: str,
    *,
    device: torch.device,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """``L`` and ``dL/dz`` for the *same* three distances the runs report.

    ``svgd_endpoint._optimizer_latent_metrics`` defines the numbers; this is the
    torch transcription of whichever one drives the update, so the reported
    energy and the gradient can never drift apart.
    """
    current = torch.as_tensor(
        np.asarray(tokens, dtype=np.float32), device=device
    ).requires_grad_(True)
    goal = torch.as_tensor(np.asarray(goal_tokens, dtype=np.float32), device=device)
    if current.shape != goal.shape:
        raise ValueError(f"Latent shape mismatch: {current.shape} != {goal.shape}")

    if metric == "rms":
        loss = torch.sqrt(((current - goal) ** 2).mean() + eps)
    elif metric == "cosine":
        flat_current = current.reshape(-1)
        flat_goal = goal.reshape(-1)
        similarity = torch.nn.functional.cosine_similarity(
            flat_current[None], flat_goal[None], dim=-1, eps=eps
        ).clamp(-1.0, 1.0)
        loss = 1.0 - similarity.squeeze()
    elif metric == "token_cosine":
        normalized = torch.nn.functional.normalize(current, dim=-1, eps=eps)
        goal_normalized = torch.nn.functional.normalize(goal, dim=-1, eps=eps)
        similarity = (normalized * goal_normalized).sum(dim=-1).clamp(-1.0, 1.0)
        loss = 1.0 - similarity.mean()
    else:
        raise ValueError(f"Unknown latent distance {metric!r}")

    (gradient,) = torch.autograd.grad(loss, current)
    return (
        float(loss.detach()),
        gradient.detach().reshape(-1).cpu().numpy().astype(np.float64),
    )


# --------------------------------------------------------------------------- #
# trajectory feasibility
# --------------------------------------------------------------------------- #


def credit_weights(
    horizon: int,
    scored_step: int,
    mode: str,
    decay: float,
) -> np.ndarray:
    """``dq_s/du_t`` as a scalar per setpoint row, normalised to a peak of 1.

    ``uniform``     every row before the scored step shares the gradient equally.
                    U is treated as one trajectory-valued parameter.
    ``tail-decay``  ``kappa (1 - kappa)^(s - t)`` -- the first-order servo model,
                    so credit fades backwards over roughly ``1/kappa`` steps.
    ``last-only``   only the setpoint that produced the scored frame moves.
    """
    weights = np.zeros(horizon, dtype=np.float64)
    last = int(np.clip(scored_step, 0, horizon - 1))
    if mode == "uniform":
        weights[: last + 1] = 1.0
    elif mode == "last-only":
        weights[last] = 1.0
    elif mode == "tail-decay":
        kappa = float(np.clip(decay, 1e-4, 1.0))
        offsets = np.arange(last, -1, -1, dtype=np.float64)
        weights[: last + 1] = (1.0 - kappa) ** offsets
        weights /= max(float(weights.max()), 1e-12)
    else:
        raise ValueError(f"Unknown credit mode {mode!r}")
    return weights


def project_trajectories(
    particles: np.ndarray,
    *,
    joint_limits: np.ndarray,
    start_q: np.ndarray,
    anchor_steps: int,
    max_rate: float,
    gripper_command: float | None,
) -> np.ndarray:
    """Snap a population back onto executable setpoint trajectories.

    Order matters: anchor first (it defines where the ramp must start), then the
    forward/backward slew sweep, then joint limits last so nothing leaves the
    safe box.  The backward sweep is what stops the anchor from silently
    re-introducing a rate violation at the seam.
    """
    projected = np.array(particles, dtype=np.float64, copy=True)
    horizon = projected.shape[1]

    if anchor_steps > 0:
        projected[:, : min(anchor_steps, horizon), :ARM_JOINTS] = start_q[None, None, :]

    if max_rate is not None and max_rate > 0.0:
        arm = projected[:, :, :ARM_JOINTS]
        for step in range(1, horizon):
            delta = np.clip(arm[:, step] - arm[:, step - 1], -max_rate, max_rate)
            arm[:, step] = arm[:, step - 1] + delta
        for step in range(horizon - 2, max(anchor_steps - 1, 0) - 1, -1):
            delta = np.clip(arm[:, step] - arm[:, step + 1], -max_rate, max_rate)
            arm[:, step] = arm[:, step + 1] + delta
        if anchor_steps > 0:
            arm[:, : min(anchor_steps, horizon)] = start_q[None, None, :]
        projected[:, :, :ARM_JOINTS] = arm

    projected[:, :, :ARM_JOINTS] = np.clip(
        projected[:, :, :ARM_JOINTS],
        joint_limits[None, None, :, 0],
        joint_limits[None, None, :, 1],
    )
    if gripper_command is None:
        projected[:, :, ARM_JOINTS] = np.clip(projected[:, :, ARM_JOINTS], -1.0, 1.0)
    else:
        projected[:, :, ARM_JOINTS] = float(gripper_command)
    return projected


def _smooth_knot_noise(
    rng: np.random.Generator,
    count: int,
    horizon: int,
    knots: int,
    radius: np.ndarray,
) -> np.ndarray:
    """Low-frequency per-joint noise: ``knots`` random values, then interpolated.

    White noise over 300 steps is not a trajectory -- the slew projection would
    flatten it to nothing and every particle would come out identical.  Knots
    give initial populations that differ in *shape*, which is the only thing
    that keeps the SVGD kernel from collapsing on the first iteration.
    """
    knot_values = rng.normal(size=(count, knots, ARM_JOINTS))
    knot_values[:, 0] = 0.0  # every trajectory starts at the true configuration
    knot_positions = np.linspace(0.0, horizon - 1, knots)
    steps = np.arange(horizon, dtype=np.float64)
    out = np.empty((count, horizon, ARM_JOINTS), dtype=np.float64)
    for index in range(count):
        for joint in range(ARM_JOINTS):
            out[index, :, joint] = np.interp(
                steps, knot_positions, knot_values[index, :, joint]
            )
    return out * radius[None, None, :]


def initialize_particles(
    mode: str,
    rng: np.random.Generator,
    *,
    count: int,
    horizon: int,
    start_q: np.ndarray,
    radius: np.ndarray,
    knots: int,
    gripper_command: float,
) -> np.ndarray:
    """Seed a population of setpoint trajectories around the start configuration."""
    particles = np.zeros((count, horizon, ACTION_DIM), dtype=np.float64)
    particles[:, :, :ARM_JOINTS] = start_q[None, None, :]
    particles[:, :, ARM_JOINTS] = gripper_command

    if mode == "hold":
        return particles

    noise = _smooth_knot_noise(rng, count, horizon, knots, radius)
    if mode == "hold-cloud":
        particles[:, :, :ARM_JOINTS] += noise
        return particles
    if mode == "ramp-cloud":
        # A minimum-jerk ramp toward a random nearby configuration, so the
        # population already contains motion instead of having to discover that
        # moving at all is possible.
        u = np.linspace(0.0, 1.0, horizon)
        profile = (10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5)[None, :, None]
        goals = rng.normal(size=(count, 1, ARM_JOINTS)) * radius[None, None, :] * 4.0
        particles[:, :, :ARM_JOINTS] += profile * goals + 0.25 * noise
        return particles
    raise ValueError(f"Unknown init mode {mode!r}")


# --------------------------------------------------------------------------- #
# rollout + energy
# --------------------------------------------------------------------------- #


class TrajectoryEnergy:
    """U (setpoint trajectory) -> executed rollout -> latent distance to the goal.

    Every call restores the identical staged MuJoCo snapshot first, so the energy
    is a deterministic function of U up to renderer noise.
    """

    def __init__(
        self,
        env: OffScreenRenderEnv,
        start_state: np.ndarray,
        start_gripper_actions: list[np.ndarray],
        encoder: Any,
        goal_latent: np.ndarray,
        *,
        view_size: int,
        views: str,
        distance_metric: str,
        output_max: np.ndarray,
        energy_mode: str,
        waypoints: int,
        softmin_temperature: float,
        trace_root: Path | None,
        trace_stride: int,
        verbose: bool,
    ) -> None:
        self.env = env
        self.start_state = np.asarray(start_state, dtype=np.float64)
        self.start_gripper_actions = start_gripper_actions
        self.encoder = encoder
        self.goal_latent = np.asarray(goal_latent, dtype=np.float32)
        self.view_size = int(view_size)
        self.views = views
        self.distance_metric = distance_metric
        self.output_max = np.asarray(output_max, dtype=np.float64)
        self.energy_mode = energy_mode
        self.waypoints = int(waypoints)
        self.softmin_temperature = float(softmin_temperature)
        self.trace_root = trace_root
        self.trace_stride = max(int(trace_stride), 1)
        self.verbose = verbose
        self.rollouts = 0
        self.trace_index_path = (
            trace_root / "evaluations.jsonl" if trace_root is not None else None
        )

        inner = getattr(env, "env", env)
        self._inner = inner
        self._qpos_index = np.asarray(
            inner.robots[0]._ref_joint_pos_indexes, dtype=int
        )
        self._link_names = arm_link_names(env)

    def joint_positions(self) -> np.ndarray:
        return np.asarray(
            self.env.sim.data.qpos[self._qpos_index], dtype=np.float64
        ).copy()

    def scoring_steps(self, horizon: int) -> list[int]:
        """Rollout steps whose render is encoded and scored."""
        if self.energy_mode == "terminal" or self.waypoints <= 1:
            return [horizon - 1]
        # Always include the terminal step; spread the rest evenly behind it.
        raw = np.linspace(horizon - 1, horizon // 4, self.waypoints)
        return sorted({int(round(value)) for value in raw})

    def __call__(
        self,
        trajectory: np.ndarray,
        *,
        trace_context: dict[str, Any] | None = None,
        capture_video: bool = False,
        video_stride: int = 6,
        save_trace: bool = False,
    ) -> dict[str, Any]:
        trajectory = np.asarray(trajectory, dtype=np.float64)
        horizon = trajectory.shape[0]
        scoring_steps = self.scoring_steps(horizon)
        scoring_set = set(scoring_steps)

        self.env.reset()
        obs = self.env.set_init_state(self.start_state)
        _synchronize_controllers_to_sim_state(self.env, self.start_gripper_actions)

        object_names, object_positions, object_quaternions = _tracked_object_poses(
            self.env
        )
        eef_path = [np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()]
        joint_path = [self.joint_positions()]
        object_track = [object_positions]
        quaternion_track = [object_quaternions]
        arm_track = [arm_link_positions(self.env, self._link_names)]
        actions: list[np.ndarray] = []
        setpoint_errors: list[float] = []
        frames: list[Image.Image] = (
            [_views_from_obs(obs, self.view_size)[2]] if capture_video else []
        )
        scored: dict[int, dict[str, Any]] = {}

        started = time.time()
        for step in range(horizon):
            current_q = self.joint_positions()
            action = np.zeros(ACTION_DIM, dtype=np.float64)
            action[:ARM_JOINTS] = np.clip(
                (trajectory[step, :ARM_JOINTS] - current_q) / self.output_max,
                -1.0,
                1.0,
            )
            action[ARM_JOINTS] = float(np.clip(trajectory[step, ARM_JOINTS], -1.0, 1.0))
            obs, _, _, _ = self.env.step(action)

            actions.append(action)
            reached_q = self.joint_positions()
            joint_path.append(reached_q)
            eef_path.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy())
            setpoint_errors.append(
                float(np.linalg.norm(reached_q - trajectory[step, :ARM_JOINTS]))
            )
            _, positions, quaternions = _tracked_object_poses(self.env)
            object_track.append(positions)
            quaternion_track.append(quaternions)
            arm_track.append(arm_link_positions(self.env, self._link_names))
            if capture_video and step % max(video_stride, 1) == 0:
                frames.append(_views_from_obs(obs, self.view_size)[2])

            if step in scoring_set:
                image = _views_from_obs(obs, self.view_size)[2]
                latent = _encode_view_features(
                    self.encoder, image, self.view_size, self.views
                )
                metrics = _optimizer_latent_metrics(latent, self.goal_latent)
                scored[step] = {
                    "image": image,
                    "latent": latent,
                    "metrics": metrics,
                    "energy": float(metrics[self.distance_metric]),
                    "joint_configuration": reached_q.copy(),
                    "eef": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy(),
                }

        if capture_video and (horizon - 1) % max(video_stride, 1) != 0:
            frames.append(_views_from_obs(obs, self.view_size)[2])

        self.rollouts += 1
        elapsed = time.time() - started

        # Aggregate the scored frames.  "eventually" is a soft minimum, so a
        # trajectory that matches the goal at *any* scored moment is rewarded and
        # the gradient lands on the step that matched best.
        energies = np.array([scored[step]["energy"] for step in scoring_steps])
        if self.energy_mode == "eventually" and len(scoring_steps) > 1:
            temperature = max(self.softmin_temperature, 1e-6)
            logits = -(energies - energies.min()) / temperature
            weights = np.exp(logits)
            weights /= weights.sum()
            energy = float(np.sum(weights * energies))
            selected_step = int(scoring_steps[int(np.argmin(energies))])
        else:
            weights = np.ones(1)
            energy = float(energies[-1])
            selected_step = int(scoring_steps[-1])

        selected = scored[selected_step]
        terminal_state = np.asarray(self.env.get_sim_state(), dtype=np.float64).copy()
        result: dict[str, Any] = {
            "energy": energy,
            "energies_per_scored_step": energies.tolist(),
            "scoring_steps": list(scoring_steps),
            "softmin_weights": weights.tolist(),
            "selected_step": selected_step,
            "selected_latent": selected["latent"],
            "selected_metrics": selected["metrics"],
            "selected_joint_configuration": selected["joint_configuration"],
            "selected_eef": selected["eef"],
            "terminal_image": scored[scoring_steps[-1]]["image"],
            "terminal_metrics": scored[scoring_steps[-1]]["metrics"],
            "terminal_eef": np.asarray(eef_path[-1], dtype=np.float64),
            "terminal_joint_configuration": np.asarray(joint_path[-1]),
            "terminal_sim_state": terminal_state,
            "eef_path": np.asarray(eef_path, dtype=np.float64),
            "joint_path": np.asarray(joint_path, dtype=np.float64),
            "actions": np.asarray(actions, dtype=np.float64),
            "setpoint_tracking_error_mean_rad": float(np.mean(setpoint_errors)),
            "setpoint_tracking_error_final_rad": float(setpoint_errors[-1]),
            "seconds": elapsed,
            "frames": frames,
            "trace_file": None,
        }

        if save_trace and self.trace_root is not None and trace_context is not None:
            result["trace_file"] = self._write_trace(
                trajectory,
                result,
                object_names,
                np.asarray(object_track, dtype=np.float64),
                np.asarray(quaternion_track, dtype=np.float64),
                np.asarray(arm_track, dtype=np.float32),
                trace_context,
            )
        if trace_context is not None:
            self._log_evaluation(result, trace_context)
        return result

    def _write_trace(
        self,
        trajectory: np.ndarray,
        result: dict[str, Any],
        object_names: list[str],
        object_track: np.ndarray,
        quaternion_track: np.ndarray,
        arm_track: np.ndarray,
        trace_context: dict[str, Any],
    ) -> str:
        """Write the ``.npz`` schema ``svgd_traj3d.py`` reads, plus joint fields.

        ``target_eef`` is where the *final setpoint* puts the end effector, which
        is the trajectory analogue of the endpoint runs' particle: the viewer
        draws it as the particle marker and checks it against ``terminal_eef``.
        """
        assert self.trace_root is not None
        iteration = int(trace_context["iteration"])
        particle = int(trace_context["particle"])
        evaluation = str(trace_context["evaluation"])
        trace_dir = self.trace_root / f"iter_{iteration:03d}" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"particle_{particle:02d}_{evaluation}.npz"

        keep = np.arange(0, len(result["eef_path"]), self.trace_stride)
        if keep[-1] != len(result["eef_path"]) - 1:
            keep = np.append(keep, len(result["eef_path"]) - 1)
        np.savez_compressed(
            path,
            target_eef=np.asarray(trace_context["target_eef"], dtype=np.float64),
            terminal_eef=result["terminal_eef"],
            eef_path=result["eef_path"][keep],
            desired_eefs=result["eef_path"][keep],
            eef_before_actions=result["eef_path"][keep],
            position_errors=np.zeros_like(result["eef_path"][keep]),
            normalized_times=(keep / max(len(result["eef_path"]) - 1, 1)),
            minimum_jerk_progress=(keep / max(len(result["eef_path"]) - 1, 1)),
            phases=np.asarray(["setpoint"] * len(keep)),
            object_names=np.asarray(object_names, dtype=str),
            object_positions=object_track[keep],
            object_quaternions_wxyz=quaternion_track[keep],
            arm_link_names=np.asarray(self._link_names, dtype=str),
            arm_link_positions=arm_track[keep],
            trace_step_indices=keep,
            joint_setpoints=trajectory.astype(np.float32),
            joint_path=result["joint_path"][keep].astype(np.float32),
            actions=result["actions"].astype(np.float32),
            scoring_steps=np.asarray(result["scoring_steps"], dtype=int),
            energies_per_scored_step=np.asarray(
                result["energies_per_scored_step"], dtype=np.float64
            ),
        )
        return str(path.relative_to(self.trace_root))

    def _log_evaluation(
        self, result: dict[str, Any], trace_context: dict[str, Any]
    ) -> None:
        event = {
            "rollout": int(self.rollouts),
            "iteration": int(trace_context["iteration"]),
            "particle": int(trace_context["particle"]),
            "evaluation": str(trace_context["evaluation"]),
            "objective": self.distance_metric,
            "energy_mode": self.energy_mode,
            "energy": float(result["energy"]),
            "selected_step": int(result["selected_step"]),
            "latent_metrics": result["terminal_metrics"],
            "terminal_eef": result["terminal_eef"].tolist(),
            "target_eef": np.asarray(
                trace_context["target_eef"], dtype=np.float64
            ).tolist(),
            "setpoint_tracking_error_mean_rad": result[
                "setpoint_tracking_error_mean_rad"
            ],
            "seconds": result["seconds"],
            "trace_file": result["trace_file"],
        }
        if self.trace_index_path is not None:
            with self.trace_index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(event), allow_nan=False) + "\n")
                handle.flush()
        if self.verbose:
            print(
                f"[eval {self.rollouts:05d}] "
                f"iter={event['iteration']:03d} particle={event['particle']:02d} "
                f"{self.distance_metric}={event['energy']:.8f} "
                f"step={event['selected_step']} "
                f"track={event['setpoint_tracking_error_mean_rad']:.5f}rad "
                f"{event['seconds']:.1f}s",
                flush=True,
            )


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #


def save_checkpoint(
    directory: Path,
    *,
    iteration: int,
    particles: np.ndarray,
    rng: np.random.Generator,
    global_best: dict[str, Any] | None,
    jacobians: dict[int, np.ndarray],
    rollouts: int,
    config: dict[str, Any],
) -> Path:
    """One resumable snapshot.  ``iteration`` is the *next* iteration to run."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"iter_{iteration:03d}.npz"
    payload: dict[str, Any] = {
        "iteration": np.asarray(iteration),
        "particles": particles.astype(np.float64),
        "rollouts": np.asarray(rollouts),
        "rng_state": np.asarray(
            json.dumps(_jsonable(rng.bit_generator.state)), dtype=object
        ),
        "config": np.asarray(json.dumps(_jsonable(config)), dtype=object),
        "jacobian_keys": np.asarray(sorted(jacobians), dtype=int),
    }
    for index in sorted(jacobians):
        payload[f"jacobian_{index:02d}"] = jacobians[index].astype(np.float32)
    if global_best is not None:
        payload["best_energy"] = np.asarray(float(global_best["energy"]))
        payload["best_iteration"] = np.asarray(int(global_best["iteration"]))
        payload["best_particle"] = np.asarray(int(global_best["particle"]))
        payload["best_trajectory"] = np.asarray(
            global_best["trajectory"], dtype=np.float64
        )
    np.savez_compressed(path, **payload)
    (directory / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint": path.name,
                "iteration": int(iteration),
                "rollouts": int(rollouts),
                "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "global_best_energy": (
                    None if global_best is None else float(global_best["energy"])
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def find_checkpoint(out_dir: Path, resume: str) -> Path | None:
    if resume == "none":
        return None
    if resume != "auto":
        path = Path(resume)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = sorted((out_dir / "checkpoints").glob("iter_*.npz"))
    return candidates[-1] if candidates else None


def load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        state: dict[str, Any] = {
            "iteration": int(data["iteration"]),
            "particles": np.asarray(data["particles"], dtype=np.float64),
            "rollouts": int(data["rollouts"]),
            "rng_state": json.loads(str(data["rng_state"].item())),
            "config": json.loads(str(data["config"].item())),
            "jacobians": {
                int(index): np.asarray(
                    data[f"jacobian_{int(index):02d}"], dtype=np.float64
                )
                for index in np.asarray(data["jacobian_keys"], dtype=int)
            },
            "global_best": None,
        }
        if "best_energy" in data:
            state["global_best"] = {
                "energy": float(data["best_energy"]),
                "iteration": int(data["best_iteration"]),
                "particle": int(data["best_particle"]),
                "trajectory": np.asarray(data["best_trajectory"], dtype=np.float64),
            }
    return state


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _progress_plot(history: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [record["iteration"] for record in history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    energy_axis, error_axis, spread_axis, update_axis = axes.ravel()

    energy_axis.plot(iterations, [r["energy_min"] for r in history], label="min")
    energy_axis.plot(iterations, [r["energy_mean"] for r in history], label="mean")
    energy_axis.plot(iterations, [r["energy_max"] for r in history], label="max")
    energy_axis.set_title("latent energy")
    energy_axis.set_xlabel("iteration")
    energy_axis.legend()
    energy_axis.grid(alpha=0.3)

    error_axis.plot(iterations, [r["goal_error_min_m"] for r in history], label="min")
    error_axis.plot(iterations, [r["goal_error_mean_m"] for r in history], label="mean")
    error_axis.set_title("terminal EEF error to diagnostic goal (m)")
    error_axis.set_xlabel("iteration")
    error_axis.legend()
    error_axis.grid(alpha=0.3)

    spread_axis.plot(iterations, [r["particle_spread_rad"] for r in history])
    spread_axis.set_title("population spread (rad, mean std over setpoints)")
    spread_axis.set_xlabel("iteration")
    spread_axis.grid(alpha=0.3)

    update_axis.plot(
        iterations, [r["applied_update_norm_mean_rad"] for r in history], label="applied"
    )
    update_axis.plot(
        iterations, [r["gradient_norm_mean"] for r in history], label="|dE/dq|"
    )
    update_axis.set_yscale("symlog", linthresh=1e-6)
    update_axis.set_title("update and gradient magnitude")
    update_axis.set_xlabel("iteration")
    update_axis.legend()
    update_axis.grid(alpha=0.3)

    figure.savefig(path, dpi=140)
    plt.close(figure)


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    scene = parser.add_argument_group("scene and goal")
    scene.add_argument("--run-dir", required=True,
                       help="Endpoint run providing manifest.json and start_state.npy.")
    scene.add_argument("--out-dir", help="Defaults to <run-dir>/svgd_joint_traj.")
    scene.add_argument("--goal", help="Goal image. Defaults to <run-dir>/goal_oracle.png.")
    scene.add_argument("--goal-latent-source", choices=["editor", "reencode"],
                       default="reencode")
    scene.add_argument("--diagnostic-goal-eef", type=float, nargs=3,
                       metavar=("X", "Y", "Z"),
                       help="Known physical goal, used for plots only; never enters the energy.")
    scene.add_argument("--bounds", type=float, nargs=6,
                       default=[-0.18, 0.18, -0.32, 0.32, 0.98, 1.10],
                       metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                       help="Diagnostic Cartesian box drawn by the 3D viewer. Not a constraint.")

    encoder = parser.add_argument_group("encoder")
    encoder.add_argument("--feature-encoder", choices=["flux_ae", "dinov3"],
                         default="flux_ae")
    encoder.add_argument("--editor-ae", help="FLUX.2 ae.safetensors; required for flux_ae.")
    encoder.add_argument("--flux2-src", default=str(REPO_ROOT / "third_party" / "flux2"))
    encoder.add_argument("--dino-model", default="vit_base_patch16_dinov3.lvd1689m")
    encoder.add_argument("--device", default="auto")
    encoder.add_argument("--latent-views", choices=["both", "agentview", "right", "wrist"],
                         default="agentview")
    encoder.add_argument("--latent-distance",
                         choices=["rms", "cosine", "token_cosine"],
                         default="token_cosine")

    traj = parser.add_argument_group("trajectory particle")
    traj.add_argument("--horizon", type=int, default=300,
                      help="Control steps per particle; the particle is [H, 8].")
    traj.add_argument("--particles", type=int, default=10)
    traj.add_argument("--max-iterations", type=int, default=100)
    traj.add_argument("--init-mode", choices=["hold", "hold-cloud", "ramp-cloud"],
                      default="ramp-cloud")
    traj.add_argument("--init-joint-radius", type=float, default=0.02,
                      help="Per-joint init noise scale in radians.")
    traj.add_argument("--init-knots", type=int, default=6,
                      help="Random knots interpolated across the horizon at init.")
    traj.add_argument("--anchor-start-steps", type=int, default=10,
                      help="Leading setpoints pinned to the measured start configuration.")
    traj.add_argument("--max-setpoint-rate", type=float, default=0.04,
                      help="Slew cap |u_{t+1}-u_t| in rad; the controller saturates at 0.05.")
    traj.add_argument("--joint-limit-margin", type=float, default=0.05)
    traj.add_argument("--gripper-command", type=float, default=-1.0,
                      help="Held gripper command (-1 open). Ignored with --optimize-gripper.")
    traj.add_argument("--optimize-gripper", action="store_true",
                      help="Let SVGD move u[:, 7] too. Off by default: it has no gradient "
                           "here, since the arm never grasps.")

    energy = parser.add_argument_group("energy")
    energy.add_argument("--energy-mode", choices=["terminal", "eventually"],
                        default="terminal",
                        help="'terminal' scores the last frame; 'eventually' softmins over "
                             "--waypoints sampled frames, which is what gives the interior "
                             "of the trajectory its own gradient.")
    energy.add_argument("--waypoints", type=int, default=6)
    energy.add_argument("--waypoint-softmin-temp", type=float, default=0.02)

    grad = parser.add_argument_group("gradient")
    grad.add_argument("--credit-mode",
                      choices=["uniform", "tail-decay", "last-only"], default="uniform")
    grad.add_argument("--credit-decay", type=float, default=0.05,
                      help="kappa for --credit-mode tail-decay; credit fades over ~1/kappa steps.")
    grad.add_argument("--jacobian-mode", choices=["per-particle", "shared"],
                      default="per-particle")
    grad.add_argument("--jacobian-delta", type=float, default=0.005,
                      help="Central-difference half-step in rad; 0.005 is the probe's peak.")
    grad.add_argument("--jacobian-refresh-every", type=int, default=5,
                      help="Iterations between full re-estimates; Broyden updates in between.")

    transport = parser.add_argument_group("transport")
    transport.add_argument("--transport", choices=["svgd", "particle_gd"], default="svgd")
    transport.add_argument("--kernel-space", choices=["full", "terminal-joint", "terminal-eef"],
                           default="full",
                           help="Space the RBF kernel is computed in. 'full' is the faithful "
                                "choice; the terminal spaces are better conditioned but are a "
                                "deviation from SVGD theory.")
    transport.add_argument("--latent-weight", type=float, default=1.0)
    transport.add_argument("--repulsion-weight", type=float, default=0.01)
    transport.add_argument("--bandwidth-scale", type=float, default=1.0)
    transport.add_argument("--step-size", type=float, default=0.01)
    transport.add_argument("--temperature", type=float, default=0.10)
    transport.add_argument("--max-joint-step-rad", type=float, default=0.01,
                           help="Elementwise cap on one iteration's change to any setpoint.")
    transport.add_argument("--max-update-norm", type=float,
                           help="Optional L2 cap on the flattened per-particle update.")

    run = parser.add_argument_group("run control")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--checkpoint-every", type=int, default=25)
    run.add_argument("--resume", default="auto",
                     help="'auto' (newest checkpoint in --out-dir), 'none', or a .npz path.")
    run.add_argument("--rollout-trace-mode", choices=["none", "base"], default="base")
    run.add_argument("--trace-stride", type=int, default=3,
                     help="Keep every Nth rollout state in the trace files.")
    run.add_argument("--save-all-particles", action="store_true")
    run.add_argument("--verbose-evaluations", action="store_true")
    run.add_argument("--best-video-stride", type=int, default=6)
    run.add_argument("--best-video-fps", type=int, default=20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.feature_encoder == "flux_ae" and not args.editor_ae:
        parser.error("--editor-ae is required with --feature-encoder flux_ae")
    if args.particles < 2:
        parser.error("SVGD needs at least 2 particles for the repulsion term")
    if args.horizon < 8:
        parser.error("--horizon must be at least 8")
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.anchor_start_steps < 0 or args.anchor_start_steps >= args.horizon:
        parser.error("--anchor-start-steps must be in [0, horizon)")
    if args.transport == "particle_gd" and args.repulsion_weight != 0.0:
        parser.error("--transport particle_gd requires --repulsion-weight 0")

    run_dir = Path(args.run_dir).resolve()
    out_dir = (
        Path(args.out_dir).resolve() if args.out_dir else run_dir / "svgd_joint_traj"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    goal_path = (Path(args.goal) if args.goal else run_dir / "goal_oracle.png").resolve()
    if not goal_path.exists():
        parser.error(f"Goal image not found: {goal_path}")

    view_size = int(manifest["view_size"])
    bounds = np.asarray(args.bounds, dtype=np.float64).reshape(3, 2)
    start_state = np.load(run_dir / "start_state.npy")
    start_gripper_actions = [
        np.asarray(action, dtype=np.float64)
        for action in manifest["start_gripper_controller_actions"]
    ]
    actual_start_eef = np.asarray(manifest["actual_start_eef"], dtype=np.float64)
    physical_goal = (
        np.asarray(args.diagnostic_goal_eef, dtype=np.float64)
        if args.diagnostic_goal_eef is not None
        else np.asarray(manifest["physical_goal_eef"], dtype=np.float64)
    )

    # ---- encoder and goal latent ----------------------------------------- #
    goal_image = Image.open(goal_path).convert("RGB")
    if args.feature_encoder == "flux_ae":
        encoder = FluxAutoencoderMetric(
            Path(args.editor_ae).resolve(), Path(args.flux2_src).resolve(), args.device
        )
        full_goal_latent = encoder.encode(goal_image)
        goal_latent = _view_latent(full_goal_latent, view_size, args.latent_views)
    else:
        encoder = DinoV3FeatureMetric(args.dino_model, args.device)
        goal_latent = _encode_view_features(
            encoder, goal_image, view_size, args.latent_views
        )
    goal_source = f"encode({goal_path.name})"
    goal_image.save(out_dir / "goal_reference.png")
    torch_device = torch.device(getattr(encoder, "device", "cpu"))
    print(
        f"[goal] encoder={args.feature_encoder} source={goal_source} "
        f"shape={goal_latent.shape} metric={args.latent_distance}",
        flush=True,
    )

    # ---- simulator -------------------------------------------------------- #
    env = env_from_manifest(
        manifest,
        controller="JOINT_POSITION",
    )
    history: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None
    best_replay: dict[str, Any] | None = None
    try:
        env.seed(int(manifest["sim_seed"]))
        env.reset()
        env.set_init_state(start_state)
        _synchronize_controllers_to_sim_state(env, start_gripper_actions)
        try:
            write_scene_json(
                out_dir / "scene.json",
                env,
                start_eef=actual_start_eef,
                goal_eef=physical_goal,
                bounds=bounds,
            )
        except Exception as error:  # pragma: no cover - visualisation aid only
            print(f"[scene] capture skipped: {type(error).__name__}: {error}", flush=True)

        controller = env.robots[0].controller
        output_max = np.asarray(controller.output_max, dtype=np.float64)
        if int(controller.control_dim) != ARM_JOINTS:
            raise RuntimeError(
                f"Expected a {ARM_JOINTS}-joint JOINT_POSITION controller, got "
                f"{controller.control_dim}"
            )

        observer = JointLatentObserver(
            env,
            start_state,
            encoder,
            view_size=view_size,
            views=args.latent_views,
            joint_limit_margin=args.joint_limit_margin,
        )
        joint_limits = observer.joint_limits.copy()
        start_q = observer.home_q.copy()
        estimator = LatentJacobianEstimator(observer, delta_rad=args.jacobian_delta)

        energy_fn = TrajectoryEnergy(
            env,
            start_state,
            start_gripper_actions,
            encoder,
            goal_latent,
            view_size=view_size,
            views=args.latent_views,
            distance_metric=args.latent_distance,
            output_max=output_max,
            energy_mode=args.energy_mode,
            waypoints=args.waypoints,
            softmin_temperature=args.waypoint_softmin_temp,
            trace_root=out_dir if args.rollout_trace_mode != "none" else None,
            trace_stride=args.trace_stride,
            verbose=args.verbose_evaluations,
        )

        gripper_command = None if args.optimize_gripper else float(args.gripper_command)

        def project(population: np.ndarray) -> np.ndarray:
            return project_trajectories(
                population,
                joint_limits=joint_limits,
                start_q=start_q,
                anchor_steps=args.anchor_start_steps,
                max_rate=args.max_setpoint_rate,
                gripper_command=gripper_command,
            )

        # ---- initialise or resume ---------------------------------------- #
        rng = np.random.default_rng(args.seed)
        jacobians: dict[int, np.ndarray] = {}
        start_iteration = 0
        checkpoint_path = find_checkpoint(out_dir, args.resume)
        if checkpoint_path is not None:
            state = load_checkpoint(checkpoint_path)
            particles = state["particles"]
            if particles.shape != (args.particles, args.horizon, ACTION_DIM):
                parser.error(
                    f"Checkpoint {checkpoint_path} holds particles of shape "
                    f"{particles.shape}, incompatible with --particles "
                    f"{args.particles} --horizon {args.horizon}"
                )
            rng.bit_generator.state = state["rng_state"]
            global_best = state["global_best"]
            jacobians = state["jacobians"]
            energy_fn.rollouts = state["rollouts"]
            start_iteration = state["iteration"]
            history_path = out_dir / "history.json"
            if history_path.is_file():
                previous = json.loads(history_path.read_text(encoding="utf-8"))
                history = [
                    record
                    for record in previous.get("history", [])
                    if int(record["iteration"]) < start_iteration
                ]
            print(
                f"[resume] {checkpoint_path.name}: continuing at iteration "
                f"{start_iteration}/{args.max_iterations}, "
                f"{len(history)} history records kept, "
                f"{energy_fn.rollouts} rollouts already spent",
                flush=True,
            )
            if start_iteration >= args.max_iterations:
                print(
                    "[resume] checkpoint is already at --max-iterations; raise it to "
                    "continue optimizing",
                    flush=True,
                )
        else:
            particles = initialize_particles(
                args.init_mode,
                rng,
                count=args.particles,
                horizon=args.horizon,
                start_q=start_q,
                radius=np.full(ARM_JOINTS, args.init_joint_radius),
                knots=args.init_knots,
                gripper_command=float(args.gripper_command),
            )
            particles = project(particles)
            print(
                f"[init] mode={args.init_mode} particle_shape={particles.shape} "
                f"spread={float(np.mean(np.std(particles[:, :, :ARM_JOINTS], axis=0))):.5f} rad",
                flush=True,
            )

        seconds_per_rollout = 0.0
        print(
            f"[plan] {args.max_iterations - start_iteration} iterations x "
            f"{args.particles} rollouts x {args.horizon} steps; "
            f"energy_mode={args.energy_mode} credit={args.credit_mode} "
            f"jacobian={args.jacobian_mode}/{args.jacobian_refresh_every} "
            f"transport={args.transport} kernel={args.kernel_space} "
            f"checkpoint_every={args.checkpoint_every}",
            flush=True,
        )

        # ---- gradient assembly -------------------------------------------- #
        # Last (q, z) each Jacobian key was evaluated at, so the iterations
        # between full re-estimates can still maintain the model with the
        # rank-one Broyden update instead of drifting on a stale matrix.
        previous_observation: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def jacobian_for(
            index: int,
            configuration: np.ndarray,
            descriptor: np.ndarray,
            sim_state: np.ndarray,
            iteration: int,
        ) -> tuple[np.ndarray, float | None]:
            """``dz/dq`` at ``configuration``, re-estimated on the refresh schedule.

            The probes are kinematic (``sim.forward``), so they cost renders and
            not rollouts; the terminal sim state is restored first so the probe
            images differ from the rollout's terminal frame by the joint
            perturbation and nothing else.  Between scheduled refreshes the
            matrix is carried forward with a Broyden update from the step the
            optimizer just took, and the returned relative prediction error is
            how far the local linear model can still be trusted.
            """
            key = 0 if args.jacobian_mode == "shared" else index
            configuration = np.asarray(configuration, dtype=np.float64)
            descriptor = np.asarray(descriptor, dtype=np.float64).reshape(-1)
            relative_error: float | None = None

            previous = previous_observation.get(key)
            if previous is not None and key in jacobians:
                delta_q = configuration - previous[0]
                delta_z = descriptor - previous[1]
                if float(np.linalg.norm(delta_q)) > 1e-9:
                    estimator.matrix = jacobians[key]
                    relative_error = estimator.prediction_error(delta_q, delta_z)
                    estimator.broyden_update(delta_q, delta_z)
                    jacobians[key] = np.asarray(estimator.matrix, dtype=np.float64)
            previous_observation[key] = (configuration.copy(), descriptor.copy())

            stale = key not in jacobians
            scheduled = iteration % max(args.jacobian_refresh_every, 1) == 0
            if stale or scheduled:
                env.set_init_state(sim_state)
                rebind_observer(observer, env)
                estimator.matrix = jacobians.get(key)
                matrix, _ = estimator.central_difference(configuration)
                jacobians[key] = np.asarray(matrix, dtype=np.float64)
            return jacobians[key], relative_error

        def evaluate_population(
            population: np.ndarray, iteration: int
        ) -> dict[str, Any]:
            nonlocal global_best, seconds_per_rollout
            iteration_dir = out_dir / f"iter_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            energies = np.zeros(args.particles)
            goal_errors = np.zeros(args.particles)
            terminal_eefs = np.zeros((args.particles, 3))
            target_eefs = np.zeros((args.particles, 3))
            selected_steps = np.zeros(args.particles, dtype=int)
            tracking = np.zeros(args.particles)
            joint_gradients = np.zeros((args.particles, ARM_JOINTS))
            jacobian_errors = np.full(args.particles, np.nan)
            gradients = np.zeros_like(population)
            latent_metrics = {
                name: np.zeros(args.particles)
                for name in ("rms", "cosine", "token_cosine")
            }
            trace_files: list[str | None] = [None] * args.particles
            best_index, best_energy, best_image = 0, float("inf"), None

            shared_configuration = None
            if args.jacobian_mode == "shared":
                shared_configuration = population[:, -1, :ARM_JOINTS].mean(axis=0)

            for index in range(args.particles):
                trajectory = population[index]
                # Where the final setpoint puts the end effector: the trajectory
                # analogue of an endpoint particle, and what the viewer marks.
                # Kinematic placement, so this costs a render and not a rollout.
                rebind_observer(observer, env)
                target_eefs[index] = observer.observe(
                    trajectory[-1, :ARM_JOINTS]
                ).eef_pos

                result = energy_fn(
                    trajectory,
                    trace_context={
                        "iteration": iteration,
                        "particle": index,
                        "evaluation": "base",
                        "target_eef": target_eefs[index],
                    },
                    save_trace=args.rollout_trace_mode == "base",
                )
                seconds_per_rollout = result["seconds"]
                energies[index] = result["energy"]
                terminal_eefs[index] = result["terminal_eef"]
                selected_steps[index] = result["selected_step"]
                tracking[index] = result["setpoint_tracking_error_mean_rad"]
                for name in latent_metrics:
                    latent_metrics[name][index] = result["terminal_metrics"][name]
                goal_errors[index] = float(
                    np.linalg.norm(result["terminal_eef"] - physical_goal)
                )
                trace_files[index] = result["trace_file"]
                if args.save_all_particles:
                    result["terminal_image"].save(
                        iteration_dir / f"particle_{index:02d}.png"
                    )
                if result["energy"] < best_energy:
                    best_index = index
                    best_energy = float(result["energy"])
                    best_image = result["terminal_image"]
                if global_best is None or result["energy"] < float(
                    global_best["energy"]
                ):
                    global_best = {
                        "energy": float(result["energy"]),
                        "iteration": int(iteration),
                        "particle": int(index),
                        "trajectory": trajectory.copy(),
                    }

                if args.latent_weight > 0.0:
                    _, latent_gradient = _metric_loss_and_grad(
                        result["selected_latent"],
                        goal_latent,
                        args.latent_distance,
                        device=torch_device,
                    )
                    matrix, relative_error = jacobian_for(
                        index,
                        (
                            shared_configuration
                            if shared_configuration is not None
                            else result["selected_joint_configuration"]
                        ),
                        result["selected_latent"],
                        result["terminal_sim_state"],
                        iteration,
                    )
                    jacobian_errors[index] = (
                        np.nan if relative_error is None else float(relative_error)
                    )
                    joint_gradient = matrix.T @ latent_gradient  # dE/dq at the frame
                    joint_gradients[index] = joint_gradient
                    weights = credit_weights(
                        args.horizon,
                        int(result["selected_step"]),
                        args.credit_mode,
                        args.credit_decay,
                    )
                    gradients[index, :, :ARM_JOINTS] = (
                        weights[:, None] * joint_gradient[None, :]
                    )

            if best_image is not None:
                best_image.save(iteration_dir / "best.png")
            return {
                "energies": energies,
                "goal_errors": goal_errors,
                "terminal_eefs": terminal_eefs,
                "target_eefs": target_eefs,
                "selected_steps": selected_steps,
                "tracking": tracking,
                "joint_gradients": joint_gradients,
                "jacobian_errors": jacobian_errors,
                "gradients": gradients,
                "latent_metrics": latent_metrics,
                "trace_files": trace_files,
                "best_index": int(best_index),
            }

        def write_history() -> None:
            (out_dir / "history.json").write_text(
                json.dumps(
                    _jsonable(
                        {
                            "created_at_utc": dt.datetime.now(
                                dt.timezone.utc
                            ).isoformat(),
                            "run_dir": str(run_dir),
                            "goal_path": str(goal_path),
                            "goal_latent_source": goal_source,
                            "search_space": "joint_setpoint_trajectory",
                            "config": vars(args),
                            "horizon": int(args.horizon),
                            "action_dim": ACTION_DIM,
                            "controller": "JOINT_POSITION",
                            "controller_output_max_rad": output_max.tolist(),
                            "start_joint_configuration": start_q.tolist(),
                            "joint_limits": joint_limits.tolist(),
                            "actual_start_eef": actual_start_eef.tolist(),
                            "manifest_physical_goal_eef": manifest[
                                "physical_goal_eef"
                            ],
                            "diagnostic_goal_eef": physical_goal.tolist(),
                            "diagnostic_goal_is_optimizer_input": False,
                            "history": history,
                            "best_replay": best_replay,
                        }
                    ),
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )

        # ---- optimisation loop -------------------------------------------- #
        for iteration in range(start_iteration, args.max_iterations):
            evaluated = particles.copy()
            evaluation = evaluate_population(evaluated, iteration)

            # log p = -E / T  =>  grad log p = -grad E / T
            scores = -evaluation["gradients"].reshape(args.particles, -1) / args.temperature
            flat = evaluated.reshape(args.particles, -1)
            if args.kernel_space == "terminal-joint":
                kernel_points = evaluated[:, -1, :ARM_JOINTS]
            elif args.kernel_space == "terminal-eef":
                kernel_points = evaluation["terminal_eefs"]
            else:
                kernel_points = flat

            if args.transport == "particle_gd":
                latent_direction = scores.copy()
                repulsion_direction = np.zeros_like(scores)
                direction = args.latent_weight * latent_direction
                bandwidth = None
                kernel = np.eye(args.particles)
            elif args.kernel_space == "full":
                (
                    direction,
                    bandwidth,
                    latent_direction,
                    repulsion_direction,
                    kernel,
                ) = _svgd_step(
                    flat,
                    scores,
                    args.bandwidth_scale,
                    latent_weight=args.latent_weight,
                    repulsion_weight=args.repulsion_weight,
                )
            else:
                # Hybrid, and a deliberate deviation from SVGD theory: the kernel
                # *weights* come from a low-dimensional summary (2100-D pairwise
                # distances between 10 particles are nearly all equal, which
                # makes the full-space kernel useless), while both forces still
                # act in the full control space so the units stay radians.
                summary_differences = (
                    kernel_points[:, None, :] - kernel_points[None, :, :]
                )
                summary_squared = np.sum(summary_differences**2, axis=-1)
                summary_h = max(
                    args.bandwidth_scale
                    * max(float(np.median(summary_squared)), 1e-12)
                    / max(np.log(args.particles + 1.0), 1e-6),
                    1e-12,
                )
                kernel = np.exp(-summary_squared / summary_h)
                bandwidth = summary_h
                full_differences = flat[:, None, :] - flat[None, :, :]
                full_h = max(
                    args.bandwidth_scale
                    * max(float(np.median(np.sum(full_differences**2, axis=-1))), 1e-12)
                    / max(np.log(args.particles + 1.0), 1e-6),
                    1e-12,
                )
                latent_direction = (kernel.T @ scores) / args.particles
                repulsion_direction = (
                    (-2.0 / full_h)
                    * np.einsum("ji,jid->id", kernel, full_differences)
                ) / args.particles
                direction = (
                    args.latent_weight * latent_direction
                    + args.repulsion_weight * repulsion_direction
                )

            raw_update = (args.step_size * direction).reshape(evaluated.shape)
            if args.max_joint_step_rad is not None and args.max_joint_step_rad > 0.0:
                raw_update = np.clip(
                    raw_update, -args.max_joint_step_rad, args.max_joint_step_rad
                )
            capped, trust_scales = _cap_updates(
                raw_update.reshape(args.particles, -1), args.max_update_norm
            )
            particles = project(evaluated + capped.reshape(evaluated.shape))
            applied = particles - evaluated

            record = {
                "iteration": int(iteration),
                "phase": "update",
                "objective": args.latent_distance,
                "energy_mode": args.energy_mode,
                "energy_min": float(evaluation["energies"].min()),
                "energy_mean": float(evaluation["energies"].mean()),
                "energy_max": float(evaluation["energies"].max()),
                "energies": evaluation["energies"].tolist(),
                "latent_metrics": {
                    name: values.tolist()
                    for name, values in evaluation["latent_metrics"].items()
                },
                "goal_error_min_m": float(evaluation["goal_errors"].min()),
                "goal_error_mean_m": float(evaluation["goal_errors"].mean()),
                "goal_errors_m": evaluation["goal_errors"].tolist(),
                "best_particle": evaluation["best_index"],
                "best_image": f"iter_{iteration:03d}/best.png",
                "global_best_energy": float(global_best["energy"]),
                "global_best_iteration": int(global_best["iteration"]),
                "global_best_particle": int(global_best["particle"]),
                # The viewer keys "endpoint" search off this field and draws it as
                # the per-particle marker; for a trajectory particle the natural
                # marker is where its final setpoint lands the end effector.
                "particles_before_update": evaluation["target_eefs"].tolist(),
                "terminal_eefs": evaluation["terminal_eefs"].tolist(),
                "target_tracking_errors_m": np.linalg.norm(
                    evaluation["terminal_eefs"] - evaluation["target_eefs"], axis=1
                ).tolist(),
                "rollout_trace_files": evaluation["trace_files"],
                "selected_scoring_steps": evaluation["selected_steps"].tolist(),
                "setpoint_tracking_error_mean_rad": evaluation["tracking"].tolist(),
                "joint_gradients": evaluation["joint_gradients"].tolist(),
                "gradient_norm_mean": float(
                    np.mean(np.linalg.norm(evaluation["joint_gradients"], axis=1))
                ),
                # ||dz - J dq|| / ||dz|| from the step the optimizer just took.
                # Near 0 means the local linear latent model still holds; near or
                # above 1 means only the *direction* of dE/dq is meaningful,
                # which is what the trust region is there for.
                "jacobian_relative_prediction_error": evaluation[
                    "jacobian_errors"
                ].tolist(),
                "kernel_bandwidth": None if bandwidth is None else float(bandwidth),
                "kernel_column_mass": kernel.sum(axis=0).tolist(),
                "trust_region_scales": trust_scales.tolist(),
                "applied_update_norm_mean_rad": float(
                    np.mean(np.linalg.norm(applied.reshape(args.particles, -1), axis=1))
                ),
                "applied_update_max_rad": float(np.max(np.abs(applied))),
                "particle_spread_rad": float(
                    np.mean(np.std(particles[:, :, :ARM_JOINTS], axis=0))
                ),
                "terminal_setpoint_spread_rad": float(
                    np.mean(np.std(particles[:, -1, :ARM_JOINTS], axis=0))
                ),
                "seconds_per_rollout": seconds_per_rollout,
            }
            history.append(record)
            write_history()
            # All-NaN on the first iteration: there is no previous step to
            # predict from yet, and nanmean would warn about it every run.
            finite_errors = evaluation["jacobian_errors"][
                np.isfinite(evaluation["jacobian_errors"])
            ]
            jacobian_error = float(np.mean(finite_errors)) if finite_errors.size else float("nan")
            print(
                f"[iter {iteration:03d}/{args.max_iterations}] "
                f"E_min={record['energy_min']:.5f} E_mean={record['energy_mean']:.5f} "
                f"goal_err_min={record['goal_error_min_m']:.4f}m "
                f"|dE/dq|={record['gradient_norm_mean']:.3e} "
                f"J_err={jacobian_error:.3f} "
                f"spread={record['particle_spread_rad']:.5f}rad "
                f"rollouts={energy_fn.rollouts}",
                flush=True,
            )

            completed = iteration + 1
            if completed % args.checkpoint_every == 0 or completed == args.max_iterations:
                path = save_checkpoint(
                    checkpoint_dir,
                    iteration=completed,
                    particles=particles,
                    rng=rng,
                    global_best=global_best,
                    jacobians=jacobians,
                    rollouts=energy_fn.rollouts,
                    config=vars(args),
                )
                print(f"[checkpoint] {path}", flush=True)

        if global_best is None:
            raise RuntimeError("Optimization finished without evaluating a particle.")

        # ---- replay the global best --------------------------------------- #
        rebind_observer(observer, env)
        best_target_eef = np.asarray(
            observer.observe(
                np.asarray(global_best["trajectory"])[-1, :ARM_JOINTS]
            ).eef_pos,
            dtype=np.float64,
        )
        best = energy_fn(
            np.asarray(global_best["trajectory"]),
            trace_context={
                "iteration": args.max_iterations,
                "particle": int(global_best["particle"]),
                "evaluation": "best_replay",
                "target_eef": best_target_eef,
            },
            capture_video=True,
            video_stride=args.best_video_stride,
        )
        best["terminal_image"].save(out_dir / "best_terminal.png")
        np.save(out_dir / "best_trajectory.npy",
                np.asarray(global_best["trajectory"], dtype=np.float32))
        np.save(out_dir / "best_actions.npy", best["actions"].astype(np.float32))
        np.save(out_dir / "best_eef_path.npy", best["eef_path"].astype(np.float32))
        np.save(out_dir / "best_joint_path.npy", best["joint_path"].astype(np.float32))
        np.save(out_dir / "best_terminal_state.npy", best["terminal_sim_state"])
        np.save(out_dir / "best_terminal_latent.npy",
                np.asarray(best["selected_latent"], dtype=np.float32))
        video_path = out_dir / "best_rollout.mp4"
        _write_video(video_path, best["frames"], args.best_video_fps)

        best_replay = {
            "goal": {"path": str(goal_path), "latent_source": goal_source,
                     "feature_encoder": args.feature_encoder,
                     "encoder_provenance": encoder.provenance},
            "search_space": {
                "parameterization": "absolute joint setpoints per control step",
                "horizon": int(args.horizon),
                "action_dim": ACTION_DIM,
                "controller": "JOINT_POSITION",
                "anchor_start_steps": int(args.anchor_start_steps),
                "max_setpoint_rate_rad": float(args.max_setpoint_rate),
                "gripper": ("optimized" if args.optimize_gripper
                            else f"held at {args.gripper_command}"),
            },
            "selection": {
                "objective": f"{args.feature_encoder}_feature_{args.latent_distance}",
                "energy": float(global_best["energy"]),
                "iteration": int(global_best["iteration"]),
                "particle": int(global_best["particle"]),
                "target_eef": best_target_eef.tolist(),
            },
            "replay": {
                "objective_energy": float(best["energy"]),
                "objective_energy_delta_from_selection": float(
                    best["energy"] - float(global_best["energy"])
                ),
                "latent_metrics": best["terminal_metrics"],
                "actual_terminal_eef": best["terminal_eef"].tolist(),
                "terminal_joint_configuration": best[
                    "terminal_joint_configuration"
                ].tolist(),
                "physical_goal_error_m": float(
                    np.linalg.norm(best["terminal_eef"] - physical_goal)
                ),
                "physical_goal_error_is_diagnostic_only": True,
                "setpoint_tracking_error_mean_rad": best[
                    "setpoint_tracking_error_mean_rad"
                ],
                "num_actions": int(best["actions"].shape[0]),
                "optimization_rollouts": int(energy_fn.rollouts),
            },
            "artifacts": {
                "video": "best_rollout.mp4",
                "trajectory": "best_trajectory.npy",
                "actions": "best_actions.npy",
                "eef_path": "best_eef_path.npy",
                "joint_path": "best_joint_path.npy",
                "terminal_image": "best_terminal.png",
                "progress_plot": "progress.png",
            },
        }
        (out_dir / "best_metadata.json").write_text(
            json.dumps(_jsonable(best_replay), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_history()
    finally:
        env.close()

    if history:
        _progress_plot(history, out_dir / "progress.png")
    print(f"\n[done] {out_dir}")
    print(f"[done] history:       {out_dir / 'history.json'}")
    print(f"[done] checkpoints:   {checkpoint_dir}")
    if best_replay is not None:
        print(f"[done] best rollout:  {out_dir / 'best_rollout.mp4'}")
        print(f"[done] best metadata: {out_dir / 'best_metadata.json'}")
    print(
        "[done] visualise:     python experiments/libero/svgd_traj3d.py "
        f"--runs-root {out_dir.parent.parent}"
    )


if __name__ == "__main__":
    main()
