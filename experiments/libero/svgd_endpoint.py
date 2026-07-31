#!/usr/bin/env python
"""SVGD over terminal end-effector poses, scored in image-feature space.

Baseline for the loop:

    supplied/edited goal image --> z_goal
    particles theta -> MuJoCo rollout -> terminal render -> encode -> z(theta)
    E(theta) = latent_distance(z(theta), z_goal),  log p = -E / temperature
    SVGD update with an RBF kernel and the median bandwidth heuristic

Two things about this are deliberately crude, and both are why it may not work:

1. MuJoCo + rendering + the image encoder are not differentiable end to end, so
   ``grad log p`` is estimated by central finite differences.  That costs
   ``2 * 3`` extra rollouts per particle per iteration and is only meaningful if
   the energy actually varies faster than render noise across ``--fd-eps``.
   Run ``probe_latent_landscape.py`` first -- it measures exactly that.
2. The default particle initialization is a goal-agnostic uniform prior over
   the reachable box. ``--init-mode start-cloud`` places a deterministic,
   symmetric cloud around the actual starting end-effector pose, while
   ``random-start-cloud`` samples a seeded local uniform cloud. These local
   modes test whether the image-latent objective can pull endpoints away from
   the start and toward a held-out simulator goal.

theta is the 3-D terminal end-effector position only. The evaluator currently
uses the rollout helper's zero-arc path. Because redundant robot posture and the
wrist view can retain some path dependence after settling, a goal image created
with a different path can have a nonzero achievable energy floor; use matching
goal/evaluation rollouts for a controlled endpoint baseline.

Debug images written to <out-dir>/:
    goal_reference.png            the goal image being optimized toward
    goal_latent_decoded.png       z_goal pushed through the decoder (FLUX AE only)
    iter_000/particle_00.png ...  every particle's terminal render per iteration
    iter_000/best.png             lowest-energy particle that iteration
    progress.png                  energy and physical error vs iteration
    particle_motion.png           particle/terminal positions across iterations
    latent_pull_summary.json      start-to-goal progress and term decomposition
    evaluations.jsonl             one precise record per base/FD rollout
    iter_000/traces/*.npz         actions and desired/actual EEF at every step

Best evaluated trajectory artifacts written after all iterations:
    best_rollout.mp4              replay of the global lowest-energy particle
    best_actions.npy              7-D controller actions used by that replay
    best_eef_path.npy             actual end-effector position after every action
    best_terminal.png             terminal [agentview | wrist] render
    best_terminal_state.npy       final flattened MuJoCo state
    best_terminal_latent.npy      selected encoder tokens for the terminal render
    best_metadata.json            selection and deterministic replay details
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sample_endpoint_trajectories import (  # noqa: E402
    _jsonable,
    _object_motion_summary,
    _rollout_to_target,
    _synchronize_controllers_to_sim_state,
    _views_from_obs,
    _write_video,
)
from score_endpoint_candidates import (  # noqa: E402
    DinoV3FeatureMetric,
    FluxAutoencoderMetric,
    _latent_metrics,
)

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def _view_latent(latent: np.ndarray, view_size: int, views: str) -> np.ndarray:
    """Restrict a two-view latent to one camera half.

    The terminal render is ``[agentview | wrist]`` side by side, so the FLUX token
    grid splits down its middle column.  This mirrors the split in
    ``probe_latent_path._per_view_latent_rms``.  The wrist half stays nearly
    constant until the last few millimetres of approach, so restricting the energy
    to ``agentview`` removes a term that contributes magnitude but no gradient.
    """
    if views == "both":
        return latent
    array = np.asarray(latent, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Expected a (batch, tokens, dim) latent, got shape {array.shape}")
    height, width = view_size, 2 * view_size
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"View size {view_size} does not tile into 16-pixel latent cells")
    latent_height, latent_width = height // 16, width // 16
    if array.shape[1] != latent_height * latent_width:
        raise ValueError(
            f"Latent has {array.shape[1]} tokens, expected {latent_height * latent_width} "
            f"for a {width}x{height} render"
        )
    grid = array.reshape(array.shape[0], latent_height, latent_width, array.shape[2])
    midpoint = latent_width // 2
    half = grid[:, :, :midpoint] if views == "agentview" else grid[:, :, midpoint:]
    return half.reshape(array.shape[0], -1, array.shape[2])


def _encode_view_features(
    encoder: Any,
    image: Image.Image,
    view_size: int,
    views: str,
) -> np.ndarray:
    """Return spatial tokens for the requested camera view or views."""
    encode_views = getattr(encoder, "encode_views", None)
    if callable(encode_views):
        return np.asarray(
            encode_views(image, view_size, views), dtype=np.float32
        )
    return _view_latent(encoder.encode(image), view_size, views)


def _optimizer_latent_metrics(
    candidate: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Distances used by the three controlled optimization trials.

    ``cosine`` treats the complete packed latent as one vector. ``token_cosine``
    preserves spatial correspondence and averages channel-wise cosine distance
    independently at every latent token.
    """
    base_metrics = _latent_metrics(candidate, target)
    metrics = {
        "rms": base_metrics["rms"],
        "cosine": base_metrics["cosine_distance"],
    }
    candidate_array = np.asarray(candidate, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if candidate_array.shape != target_array.shape:
        raise ValueError(
            f"Latent shape mismatch: {candidate_array.shape} != {target_array.shape}"
        )
    if candidate_array.ndim < 2:
        raise ValueError(
            "Token cosine distance needs a latent with a channel dimension"
        )

    dot = np.sum(candidate_array * target_array, axis=-1)
    candidate_norm = np.linalg.norm(candidate_array, axis=-1)
    target_norm = np.linalg.norm(target_array, axis=-1)
    denominator = candidate_norm * target_norm
    similarity = np.zeros_like(dot, dtype=np.float32)
    nonzero = denominator > 0.0
    similarity[nonzero] = dot[nonzero] / denominator[nonzero]
    both_zero = (candidate_norm == 0.0) & (target_norm == 0.0)
    similarity[both_zero] = 1.0
    metrics["token_cosine"] = float(
        np.mean(1.0 - np.clip(similarity, -1.0, 1.0))
    )
    return metrics


class EndpointEnergy:
    """theta (terminal eef xyz) -> latent distance to the edited goal.

    Every evaluation restores the identical staged MuJoCo snapshot first, so the
    energy is a deterministic function of theta up to renderer noise.
    """

    def __init__(
        self,
        env: OffScreenRenderEnv,
        start_state: np.ndarray,
        start_gripper_actions: list[np.ndarray],
        encoder: Any,
        goal_latent: np.ndarray,
        *,
        move_steps: int,
        settle_steps: int,
        gain: float,
        view_size: int,
        fixed_arc_height: float = 0.0,
        fixed_midpoint_x: float = 0.0,
        views: str = "both",
        distance_metric: str = "rms",
        trace_root: Path | None = None,
        trace_mode: str = "all",
        verbose_evaluations: bool = False,
    ) -> None:
        self.env = env
        self.start_state = start_state
        self.start_gripper_actions = start_gripper_actions
        self.encoder = encoder
        self.views = views
        self.goal_latent = np.asarray(goal_latent, dtype=np.float32)
        self.move_steps = move_steps
        self.settle_steps = settle_steps
        self.gain = gain
        self.view_size = view_size
        self.fixed_arc_height = fixed_arc_height
        self.fixed_midpoint_x = fixed_midpoint_x
        self.distance_metric = distance_metric
        self.trace_root = trace_root
        self.trace_mode = trace_mode
        self.trace_index_path = (
            trace_root / "evaluations.jsonl" if trace_root is not None else None
        )
        self.verbose_evaluations = verbose_evaluations
        self.rollouts = 0
        if self.trace_index_path is not None:
            self.trace_root.mkdir(parents=True, exist_ok=True)
            self.trace_index_path.write_text("", encoding="utf-8")

    def __call__(
        self,
        theta: np.ndarray,
        *,
        trace_context: dict[str, Any] | None = None,
    ) -> tuple[float, Image.Image, np.ndarray, dict[str, float], str | None]:
        self.env.reset()
        obs = self.env.set_init_state(self.start_state)
        _synchronize_controllers_to_sim_state(self.env, self.start_gripper_actions)
        evaluation_kind = (
            str(trace_context["evaluation"]) if trace_context is not None else None
        )
        should_save_trace = (
            self.trace_root is not None
            and trace_context is not None
            and (
                self.trace_mode == "all"
                or (
                    self.trace_mode == "base"
                    and evaluation_kind in {"base", "base_repeat"}
                )
            )
        )
        rollout_trace: dict[str, list[Any]] | None = {} if should_save_trace else None
        obs, actions, eef_path, _ = _rollout_to_target(
            self.env,
            obs,
            np.asarray(theta, dtype=np.float64),
            move_steps=self.move_steps,
            settle_steps=self.settle_steps,
            gain=self.gain,
            arc_height=self.fixed_arc_height,
            midpoint_x=self.fixed_midpoint_x,
            view_size=self.view_size,
            trace=rollout_trace,
        )
        self.rollouts += 1
        terminal_image = _views_from_obs(obs, self.view_size)[2]
        latent = _encode_view_features(
            self.encoder, terminal_image, self.view_size, self.views
        )
        metrics = _optimizer_latent_metrics(latent, self.goal_latent)
        energy = metrics[self.distance_metric]
        terminal_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        trace_file: str | None = None

        if rollout_trace is not None and trace_context is not None:
            iteration = int(trace_context["iteration"])
            particle = int(trace_context["particle"])
            evaluation = str(trace_context["evaluation"])
            trace_dir = self.trace_root / f"iter_{iteration:03d}" / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"particle_{particle:02d}_{evaluation}.npz"
            trace_payload = {
                "target_eef": np.asarray(theta, dtype=np.float64),
                "terminal_eef": terminal_eef,
                "actions": actions,
                "eef_path": eef_path,
                "desired_eefs": np.asarray(
                    rollout_trace["desired_eefs"], dtype=np.float64
                ),
                "eef_before_actions": np.asarray(
                    rollout_trace["eef_before_actions"], dtype=np.float64
                ),
                "position_errors": np.asarray(
                    rollout_trace["position_errors"], dtype=np.float64
                ),
                "normalized_times": np.asarray(
                    rollout_trace["normalized_times"], dtype=np.float64
                ),
                "minimum_jerk_progress": np.asarray(
                    rollout_trace["minimum_jerk_progress"], dtype=np.float64
                ),
                "phases": np.asarray(rollout_trace["phases"]),
                "object_names": np.asarray(
                    rollout_trace["tracked_object_names"], dtype=str
                ),
                "object_positions": np.asarray(
                    rollout_trace["object_positions"], dtype=np.float64
                ),
                "object_quaternions_wxyz": np.asarray(
                    rollout_trace["object_quaternions_wxyz"], dtype=np.float64
                ),
            }
            np.savez(trace_path, **trace_payload)
            trace_file = str(trace_path.relative_to(self.trace_root))
        if trace_context is not None:
            iteration = int(trace_context["iteration"])
            particle = int(trace_context["particle"])
            evaluation = str(trace_context["evaluation"])
            event = {
                "rollout": int(self.rollouts),
                "iteration": iteration,
                "particle": particle,
                "evaluation": evaluation,
                "target_eef": np.asarray(theta, dtype=np.float64).tolist(),
                "terminal_eef": terminal_eef.tolist(),
                "target_tracking_error_m": float(
                    np.linalg.norm(terminal_eef - np.asarray(theta, dtype=np.float64))
                ),
                "fixed_arc_height_m": float(self.fixed_arc_height),
                "fixed_midpoint_x_m": float(self.fixed_midpoint_x),
                "objective": self.distance_metric,
                "energy": float(energy),
                "latent_metrics": metrics,
                "object_motion": (
                    _object_motion_summary(rollout_trace, eef_path)
                    if rollout_trace is not None
                    else {}
                ),
                "actions": int(actions.shape[0]),
                "trace_file": trace_file,
            }
            if self.trace_index_path is not None:
                with self.trace_index_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(_jsonable(event), allow_nan=False) + "\n"
                    )
                    handle.flush()
            if self.verbose_evaluations:
                print(
                    f"[eval {self.rollouts:05d}] iter={iteration:03d} "
                    f"particle={particle:02d} kind={evaluation} "
                    f"{self.distance_metric}={energy:.8f} "
                    f"tracking={event['target_tracking_error_m']:.6f}m",
                    flush=True,
                )

        return energy, terminal_image, terminal_eef, metrics, trace_file


def _finite_difference_grad(
    energy_fn: EndpointEnergy,
    theta: np.ndarray,
    epsilon: np.ndarray,
    bounds: np.ndarray,
    *,
    iteration: int,
    particle: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Central differences of E w.r.t. theta, one probe pair per dimension."""
    epsilon = np.asarray(epsilon, dtype=np.float64)
    if epsilon.shape != theta.shape:
        raise ValueError(
            f"Finite-difference epsilon shape {epsilon.shape} != particle shape {theta.shape}"
        )
    gradient = np.zeros_like(theta)
    probes: list[dict[str, Any]] = []
    dimension_names = ("x", "y", "z")
    for dim in range(theta.size):
        step = np.zeros_like(theta)
        step[dim] = epsilon[dim]
        plus = np.clip(theta + step, bounds[:, 0], bounds[:, 1])
        minus = np.clip(theta - step, bounds[:, 0], bounds[:, 1])
        span = float(plus[dim] - minus[dim])
        if span <= 0.0:
            continue
        plus_result = energy_fn(
            plus,
            trace_context={
                "iteration": iteration,
                "particle": particle,
                "evaluation": f"fd_{dimension_names[dim]}_plus",
            },
        )
        minus_result = energy_fn(
            minus,
            trace_context={
                "iteration": iteration,
                "particle": particle,
                "evaluation": f"fd_{dimension_names[dim]}_minus",
            },
        )
        energy_plus = plus_result[0]
        energy_minus = minus_result[0]
        gradient[dim] = (energy_plus - energy_minus) / span
        probes.append(
            {
                "dimension": dimension_names[dim],
                "span_m": span,
                "plus_target_eef": plus.tolist(),
                "minus_target_eef": minus.tolist(),
                "plus_energy": float(energy_plus),
                "minus_energy": float(energy_minus),
                "plus_latent_metrics": plus_result[3],
                "minus_latent_metrics": minus_result[3],
                "plus_trace_file": plus_result[4],
                "minus_trace_file": minus_result[4],
                "gradient": float(gradient[dim]),
            }
        )
    return gradient, probes


def _svgd_step(
    particles: np.ndarray,
    score: np.ndarray,
    bandwidth_scale: float,
    *,
    latent_weight: float = 1.0,
    repulsion_weight: float = 1.0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """One SVGD direction with an RBF kernel and the median bandwidth heuristic.

    phi(theta_i) = 1/n * sum_j [ k(theta_j, theta_i) * score_j
                                 + grad_{theta_j} k(theta_j, theta_i) ]
    with k(x, y) = exp(-||x - y||^2 / h) and grad_x k = -2 (x - y) / h * k.
    """
    count = particles.shape[0]
    differences = particles[:, None, :] - particles[None, :, :]
    squared = np.sum(differences**2, axis=-1)
    median = float(np.median(squared))
    # Fall back to a small positive bandwidth once the particles collapse.
    h = bandwidth_scale * max(median, 1e-12) / max(np.log(count + 1.0), 1e-6)
    h = max(h, 1e-12)
    kernel = np.exp(-squared / h)
    # sum_j k(j, i) * score_j
    driving = (kernel.T @ score) / count
    # sum_j grad_{theta_j} k(theta_j, theta_i) = sum_j -2 (theta_j - theta_i)/h * k
    repulsion = ((-2.0 / h) * np.einsum("ji,jid->id", kernel, differences)) / count
    direction = latent_weight * driving + repulsion_weight * repulsion
    return direction, h, driving, repulsion, kernel


def _start_cloud(
    center: np.ndarray,
    radius: np.ndarray,
    count: int,
    bounds: np.ndarray,
) -> np.ndarray:
    """A deterministic, centered local cloud that does not use the goal pose.

    A small nonzero spread is essential: identical SVGD particles have a
    collapsed kernel and cannot expose the distinction between attraction and
    repulsion.  Fibonacci-sphere directions give a reproducible cloud for any
    particle count; centering and per-axis normalization keep it inside the
    requested radius box.
    """
    indices = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / float(count)
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * indices
    radial = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    unit_cloud = np.stack(
        [radial * np.cos(azimuth), radial * np.sin(azimuth), z], axis=1
    )
    unit_cloud -= unit_cloud.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(unit_cloud), axis=0)
    unit_cloud /= np.where(max_abs > 0.0, max_abs, 1.0)
    return np.clip(center[None, :] + unit_cloud * radius[None, :],
                   bounds[:, 0], bounds[:, 1])


def _goal_axis_diagnostics(
    positions: np.ndarray,
    start_eef: np.ndarray,
    diagnostic_goal_eef: np.ndarray,
) -> dict[str, Any]:
    """Physical diagnostics only; these values never enter the optimizer."""
    positions = np.asarray(positions, dtype=np.float64)
    centroid = positions.mean(axis=0)
    goal_delta = diagnostic_goal_eef - start_eef
    goal_distance = float(np.linalg.norm(goal_delta))
    if goal_distance <= 0.0:
        goal_unit = np.zeros(3, dtype=np.float64)
        progress_m = 0.0
        progress_fraction = 0.0
    else:
        goal_unit = goal_delta / goal_distance
        progress_m = float(np.dot(centroid - start_eef, goal_unit))
        progress_fraction = progress_m / goal_distance
    return {
        "centroid_eef": centroid.tolist(),
        "centroid_goal_error_m": float(np.linalg.norm(centroid - diagnostic_goal_eef)),
        "centroid_goal_axis_progress_m": progress_m,
        "centroid_goal_axis_fraction": progress_fraction,
        "particle_goal_error_mean_m": float(
            np.mean(np.linalg.norm(positions - diagnostic_goal_eef[None, :], axis=1))
        ),
    }


def _mean_goal_projection(
    vectors: np.ndarray,
    start_eef: np.ndarray,
    diagnostic_goal_eef: np.ndarray,
) -> float:
    goal_delta = diagnostic_goal_eef - start_eef
    goal_norm = float(np.linalg.norm(goal_delta))
    if goal_norm <= 0.0:
        return 0.0
    return float(np.mean(np.asarray(vectors) @ (goal_delta / goal_norm)))


def _cap_updates(updates: np.ndarray, max_norm: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Apply a per-particle trust region and return the applied scale factors."""
    if max_norm is None:
        return updates, np.ones(updates.shape[0], dtype=np.float64)
    norms = np.linalg.norm(updates, axis=1)
    scales = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return updates * scales[:, None], scales


def _progress_plot(history: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [record["iteration"] for record in history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    energy_axis, error_axis, progress_axis, update_axis = axes.ravel()
    energy_axis.plot(iterations, [r["energy_min"] for r in history], marker="o", label="min")
    energy_axis.plot(iterations, [r["energy_mean"] for r in history], marker=".", label="mean")
    energy_axis.set_xlabel("evaluated population")
    objective = history[0].get("objective", "rms")
    energy_axis.set_ylabel(f"latent {objective} energy")
    energy_axis.set_title(f"Latent objective: {objective}")
    energy_axis.legend()
    energy_axis.grid(alpha=0.3)

    error_axis.plot(
        iterations, [r["goal_error_min_m"] for r in history], marker="o", label="best particle"
    )
    error_axis.plot(
        iterations, [r["goal_error_mean_m"] for r in history], marker=".", label="particle mean"
    )
    error_axis.plot(
        iterations,
        [r["terminal_diagnostics"]["centroid_goal_error_m"] for r in history],
        marker="s",
        label="centroid",
    )
    error_axis.set_xlabel("evaluated population")
    error_axis.set_ylabel("physical goal error (m)")
    error_axis.set_title("Oracle geometry (diagnostic only)")
    error_axis.legend()
    error_axis.grid(alpha=0.3)

    progress_axis.plot(
        iterations,
        [r["terminal_diagnostics"]["centroid_goal_axis_fraction"] for r in history],
        marker="o",
    )
    progress_axis.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    progress_axis.axhline(1.0, color="green", linewidth=1, alpha=0.4)
    progress_axis.set_xlabel("evaluated population")
    progress_axis.set_ylabel("centroid goal-axis fraction")
    progress_axis.set_title("0 = start, 1 = physical goal")
    progress_axis.grid(alpha=0.3)

    update_records = [record for record in history if record["update_applied"]]
    update_iterations = [record["iteration"] for record in update_records]
    update_axis.plot(
        update_iterations,
        [record["latent_update_goal_projection_m"] for record in update_records],
        marker="o",
        label="latent attraction",
    )
    update_axis.plot(
        update_iterations,
        [record["repulsion_update_goal_projection_m"] for record in update_records],
        marker=".",
        label="repulsion",
    )
    update_axis.plot(
        update_iterations,
        [record["applied_update_goal_projection_m"] for record in update_records],
        marker="s",
        label="total applied",
    )
    update_axis.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    update_axis.set_xlabel("optimizer update")
    update_axis.set_ylabel("mean projection toward goal (m)")
    update_axis.set_title("Why particles moved (diagnostic only)")
    update_axis.legend()
    update_axis.grid(alpha=0.3)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _particle_motion_plot(
    history: list[dict[str, Any]],
    path: Path,
    start_eef: np.ndarray,
    diagnostic_goal_eef: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terminal_positions = np.asarray(
        [record["terminal_eefs"] for record in history], dtype=np.float64
    )
    iterations = np.asarray([record["iteration"] for record in history])
    goal_delta = diagnostic_goal_eef - start_eef
    denominator = max(float(np.dot(goal_delta, goal_delta)), 1e-12)
    fractions = np.einsum(
        "tnd,d->tn", terminal_positions - start_eef[None, None, :], goal_delta
    ) / denominator

    figure = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    spatial_axis = figure.add_subplot(1, 3, 1, projection="3d")
    xy_axis = figure.add_subplot(1, 3, 2)
    progress_axis = figure.add_subplot(1, 3, 3)
    for particle_index in range(terminal_positions.shape[1]):
        spatial_axis.plot(
            terminal_positions[:, particle_index, 0],
            terminal_positions[:, particle_index, 1],
            terminal_positions[:, particle_index, 2],
            marker="o",
            markersize=3,
            alpha=0.75,
        )
        xy_axis.plot(
            terminal_positions[:, particle_index, 0],
            terminal_positions[:, particle_index, 1],
            marker="o",
            markersize=3,
            alpha=0.75,
        )
        progress_axis.plot(
            iterations, fractions[:, particle_index], marker=".", alpha=0.55
        )
    spatial_axis.scatter(
        *start_eef, marker="*", s=180, color="tab:blue", label="start"
    )
    spatial_axis.scatter(
        *diagnostic_goal_eef,
        marker="X",
        s=130,
        color="tab:green",
        label="goal",
    )
    spatial_axis.set(
        xlabel="EEF x (m)",
        ylabel="EEF y (m)",
        zlabel="EEF z (m)",
        title="Particle endpoint paths in 3D",
    )
    spatial_axis.view_init(elev=24, azim=-58)
    spatial_axis.legend()
    spatial_axis.grid(alpha=0.25)

    xy_axis.scatter(
        start_eef[0], start_eef[1], marker="*", s=180, label="start"
    )
    xy_axis.scatter(
        diagnostic_goal_eef[0], diagnostic_goal_eef[1], marker="X", s=130, label="goal"
    )
    xy_axis.set(
        xlabel="EEF x (m)",
        ylabel="EEF y (m)",
        title="Top-down endpoint paths",
    )
    xy_axis.legend()
    xy_axis.grid(alpha=0.3)

    progress_axis.plot(
        iterations,
        fractions.mean(axis=1),
        color="black",
        marker="o",
        linewidth=2.5,
        label="particle mean",
    )
    progress_axis.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    progress_axis.axhline(1.0, color="green", linewidth=1, alpha=0.4)
    progress_axis.set(
        xlabel="evaluated population",
        ylabel="goal-axis fraction",
        title="0 = start, 1 = physical goal",
    )
    progress_axis.legend()
    progress_axis.grid(alpha=0.3)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Endpoint run providing manifest.json and start_state.npy.",
    )
    parser.add_argument("--out-dir", help="Defaults to <run-dir>/svgd.")
    parser.add_argument("--goal", help="Defaults to <run-dir>/goal_edit.png.")
    parser.add_argument(
        "--feature-encoder",
        choices=["flux_ae", "dinov3"],
        default="flux_ae",
        help="Frozen image encoder used to score terminal renders.",
    )
    parser.add_argument(
        "--editor-ae",
        help="FLUX.2 ae.safetensors; required with --feature-encoder flux_ae.",
    )
    parser.add_argument("--flux2-src", default=str(REPO_ROOT / "third_party" / "flux2"))
    parser.add_argument(
        "--dino-model",
        default="vit_base_patch16_dinov3.lvd1689m",
        help="Pretrained timm DINOv3 backbone.",
    )
    parser.add_argument("--device", default="auto", help="Device for the image encoder.")
    parser.add_argument(
        "--goal-latent-source", choices=["editor", "reencode"], default="reencode",
        help=(
            "'reencode' encodes the selected goal PNG (safe for any goal); "
            "'editor' uses the paired FLUX goal_editor_latent.npy ablation."
        ),
    )
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--init-mode",
        choices=["uniform", "start-cloud", "random-start-cloud"],
        default="uniform",
        help=(
            "'uniform' samples the full bounds; 'start-cloud' uses a deterministic "
            "local cloud; 'random-start-cloud' uses a seeded local uniform cloud "
            "around manifest.actual_start_eef."
        ),
    )
    parser.add_argument(
        "--init-radius",
        type=float,
        nargs=3,
        default=[0.005, 0.005, 0.003],
        metavar=("X", "Y", "Z"),
        help=("Per-axis local-cloud half-width in metres "
              "(unused for uniform initialization)."),
    )
    parser.add_argument("--step-size", type=float, default=0.02,
                        help="SVGD step size in metres per unit score.")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="log p = -E / temperature. Lower = sharper, noisier.")
    parser.add_argument(
        "--fd-eps",
        type=float,
        nargs="+",
        default=[0.01],
        metavar="METRES",
        help=(
            "Central-difference half-step in metres. Supply one value for all "
            "XYZ axes or three values for independent X Y Z probes."
        ),
    )
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument(
        "--latent-views",
        choices=["both", "agentview", "wrist"],
        default="both",
        help=(
            "Which camera half of the two-view latent drives the energy. The wrist "
            "half is nearly constant until the final few millimetres, so 'agentview' "
            "gives a smoother, better-conditioned objective."
        ),
    )
    parser.add_argument(
        "--latent-distance",
        choices=["rms", "cosine", "token_cosine"],
        default="rms",
        help=(
            "Latent distance optimized by finite differences. All three distances "
            "are logged regardless of which one drives the update."
        ),
    )
    parser.add_argument(
        "--latent-weight",
        type=float,
        default=1.0,
        help="Scale the kernel-smoothed latent score; set to 0 for a control run.",
    )
    parser.add_argument(
        "--repulsion-weight",
        type=float,
        default=1.0,
        help="Scale SVGD particle repulsion; use 0 for the clean latent-only test.",
    )
    parser.add_argument(
        "--transport",
        choices=["svgd", "particle_gd"],
        default="svgd",
        help=(
            "'svgd' kernel-averages particle scores and optionally adds repulsion; "
            "'particle_gd' applies each particle's own finite-difference score."
        ),
    )
    parser.add_argument(
        "--max-update-norm",
        type=float,
        help="Optional per-particle update cap in metres; 0.02 is recommended for start-cloud.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bounds", type=float, nargs=6,
                        default=[-0.18, 0.18, -0.32, 0.32, 0.98, 1.10],
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="Goal-agnostic uniform prior over the reachable box.")
    parser.add_argument(
        "--diagnostic-goal-eef",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Known physical goal used only for plots/diagnostics; defaults to the manifest goal. "
            "It never affects energy, gradients, or updates."
        ),
    )
    parser.add_argument("--save-all-particles", action="store_true",
                        help="Write every particle render, not just the best one.")
    parser.add_argument(
        "--save-rollout-traces",
        action="store_true",
        help=(
            "Save actions, desired/actual EEF paths, position errors, and minimum-jerk "
            "progress for every base and finite-difference rollout."
        ),
    )
    parser.add_argument(
        "--rollout-trace-mode",
        choices=["none", "base", "all"],
        help=(
            "Trajectory files to save. 'base' saves base/repeat particles while "
            "still logging scalar metrics for finite-difference evaluations. "
            "--save-rollout-traces is a backwards-compatible alias for 'all'."
        ),
    )
    parser.add_argument(
        "--verbose-evaluations",
        action="store_true",
        help="Print one flushed log line for every simulator/encoder evaluation.",
    )
    parser.add_argument(
        "--repeatability-particles",
        type=int,
        default=0,
        help=(
            "Repeat the first N base particles in every population to measure the "
            "same-target render/encoding noise floor. Repeats never affect updates."
        ),
    )
    parser.add_argument(
        "--move-steps",
        type=int,
        help="Override manifest move_steps for a controlled action-horizon ablation.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        help="Override manifest settle_steps for a controlled settling ablation.",
    )
    parser.add_argument(
        "--controller-gain",
        type=float,
        help="Override manifest controller_gain.",
    )
    parser.add_argument(
        "--fixed-arc-height",
        type=float,
        default=0.0,
        help=(
            "Fixed vertical midpoint arc used for every endpoint-search rollout. "
            "This is useful for isolating endpoint inference from obstacle contact."
        ),
    )
    parser.add_argument(
        "--fixed-midpoint-x",
        type=float,
        default=0.0,
        help="Fixed lateral midpoint offset used for every endpoint-search rollout.",
    )
    parser.add_argument(
        "--best-video-stride",
        type=int,
        help="Capture every Nth action in best_rollout.mp4; defaults to manifest video_stride.",
    )
    parser.add_argument(
        "--best-video-fps",
        type=int,
        help="Frame rate for best_rollout.mp4; defaults to manifest video_fps.",
    )
    args = parser.parse_args()

    if args.feature_encoder == "flux_ae" and not args.editor_ae:
        parser.error("--editor-ae is required with --feature-encoder flux_ae")
    if args.feature_encoder != "flux_ae" and args.goal_latent_source == "editor":
        parser.error(
            "--goal-latent-source editor is only available with "
            "--feature-encoder flux_ae"
        )
    if args.particles < 2:
        parser.error("SVGD needs at least 2 particles for the repulsion term")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if len(args.fd_eps) not in {1, 3}:
        parser.error("--fd-eps needs one value or three values for X Y Z")
    if any(value <= 0.0 for value in args.fd_eps) or args.temperature <= 0.0:
        parser.error("--fd-eps and --temperature must be positive")
    init_radius = np.asarray(args.init_radius, dtype=np.float64)
    if np.any(init_radius < 0.0):
        parser.error("--init-radius values must be non-negative")
    if args.init_mode in {"start-cloud", "random-start-cloud"} and not np.any(
        init_radius > 0.0
    ):
        parser.error("A local-cloud mode needs at least one positive --init-radius value")
    if args.latent_weight < 0.0 or args.repulsion_weight < 0.0:
        parser.error("--latent-weight and --repulsion-weight must be non-negative")
    if args.transport == "particle_gd" and args.repulsion_weight != 0.0:
        parser.error("--transport particle_gd requires --repulsion-weight 0")
    if args.max_update_norm is not None and args.max_update_norm <= 0.0:
        parser.error("--max-update-norm must be positive")
    if args.best_video_stride is not None and args.best_video_stride <= 0:
        parser.error("--best-video-stride must be positive")
    if args.best_video_fps is not None and args.best_video_fps <= 0:
        parser.error("--best-video-fps must be positive")
    if args.move_steps is not None and args.move_steps <= 0:
        parser.error("--move-steps must be positive")
    if args.settle_steps is not None and args.settle_steps < 0:
        parser.error("--settle-steps must be non-negative")
    if args.controller_gain is not None and args.controller_gain <= 0.0:
        parser.error("--controller-gain must be positive")
    if args.fixed_arc_height < 0.0:
        parser.error("--fixed-arc-height must be non-negative")
    if args.repeatability_particles < 0:
        parser.error("--repeatability-particles must be non-negative")
    trace_mode = (
        args.rollout_trace_mode
        if args.rollout_trace_mode is not None
        else ("all" if args.save_rollout_traces else "none")
    )

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "svgd"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    goal_path = (
        Path(args.goal) if args.goal else run_dir / "goal_edit.png"
    ).resolve()
    if not goal_path.exists():
        parser.error(f"Goal image not found: {goal_path}. Run the edit stage first.")

    bounds = np.asarray(args.bounds, dtype=np.float64).reshape(3, 2)
    fd_eps = np.asarray(args.fd_eps, dtype=np.float64)
    if fd_eps.size == 1:
        fd_eps = np.repeat(fd_eps, 3)
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        parser.error("Each --bounds pair must be (min, max) with min < max")

    goal_image = Image.open(goal_path).convert("RGB")
    if args.feature_encoder == "flux_ae":
        encoder = FluxAutoencoderMetric(
            Path(args.editor_ae).resolve(),
            Path(args.flux2_src).resolve(),
            args.device,
        )
    else:
        encoder = DinoV3FeatureMetric(args.dino_model, args.device)
    editor_latent = run_dir / "goal_editor_latent.npy"
    if args.goal_latent_source == "editor":
        expected_editor_goal = (run_dir / "goal_edit.png").resolve()
        if goal_path != expected_editor_goal:
            parser.error(
                "--goal-latent-source editor is only valid with the paired "
                f"goal image {expected_editor_goal}; use reencode for {goal_path}."
            )
        if not editor_latent.exists():
            parser.error(
                f"Editor latent not found: {editor_latent}. Use reencode or run the edit stage."
            )
        full_goal_latent = np.load(editor_latent)
        goal_latent = _view_latent(
            full_goal_latent,
            int(manifest["view_size"]),
            args.latent_views,
        )
        goal_source = "goal_editor_latent.npy"
    else:
        if args.feature_encoder == "flux_ae":
            full_goal_latent = encoder.encode(goal_image)
            goal_latent = _view_latent(
                full_goal_latent,
                int(manifest["view_size"]),
                args.latent_views,
            )
        else:
            full_goal_latent = None
            goal_latent = _encode_view_features(
                encoder,
                goal_image,
                int(manifest["view_size"]),
                args.latent_views,
            )
        goal_source = f"encode({goal_path.name})"

    # Debug: what is the search actually aiming at?
    goal_image.save(out_dir / "goal_reference.png")
    if args.feature_encoder == "flux_ae":
        encoder.decode(
            full_goal_latent,
            height=goal_image.height,
            width=goal_image.width,
        ).save(out_dir / "goal_latent_decoded.png")
    print(
        f"[goal] encoder={args.feature_encoder} source={goal_source} "
        f"shape={goal_latent.shape}"
    )
    print(f"[goal] reference image: {out_dir / 'goal_reference.png'}")
    if args.feature_encoder == "flux_ae":
        print(f"[goal] decoded latent: {out_dir / 'goal_latent_decoded.png'}")

    start_state = np.load(run_dir / "start_state.npy")
    start_gripper_actions = [
        np.asarray(action, dtype=np.float64)
        for action in manifest["start_gripper_controller_actions"]
    ]
    actual_start_eef = np.asarray(manifest["actual_start_eef"], dtype=np.float64)
    manifest_physical_goal = np.asarray(manifest["physical_goal_eef"], dtype=np.float64)
    physical_goal = (
        np.asarray(args.diagnostic_goal_eef, dtype=np.float64)
        if args.diagnostic_goal_eef is not None
        else manifest_physical_goal.copy()
    )
    if actual_start_eef.shape != (3,) or physical_goal.shape != (3,):
        parser.error("Start and diagnostic goal EEF positions must each have three values")
    if args.init_mode in {"start-cloud", "random-start-cloud"} and np.any(
        (actual_start_eef < bounds[:, 0]) | (actual_start_eef > bounds[:, 1])
    ):
        parser.error(
            f"actual_start_eef {actual_start_eef.tolist()} lies outside --bounds"
        )
    view_size = int(manifest["view_size"])
    move_steps = int(
        args.move_steps if args.move_steps is not None else manifest["move_steps"]
    )
    settle_steps = int(
        args.settle_steps
        if args.settle_steps is not None
        else manifest["settle_steps"]
    )
    controller_gain = float(
        args.controller_gain
        if args.controller_gain is not None
        else manifest["controller_gain"]
    )
    best_video_stride = int(
        args.best_video_stride
        if args.best_video_stride is not None
        else manifest.get("video_stride", 2)
    )
    best_video_fps = int(
        args.best_video_fps
        if args.best_video_fps is not None
        else manifest.get("video_fps", 12)
    )

    env = OffScreenRenderEnv(
        bddl_file_name=str(manifest["bddl"]),
        camera_heights=int(manifest["render_size"]),
        camera_widths=int(manifest["render_size"]),
    )
    history: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None
    best_replay: dict[str, Any] | None = None
    pull_summary: dict[str, Any] | None = None
    try:
        env.seed(int(manifest["sim_seed"]))
        energy_fn = EndpointEnergy(
            env,
            start_state,
            start_gripper_actions,
            encoder,
            goal_latent,
            move_steps=move_steps,
            settle_steps=settle_steps,
            gain=controller_gain,
            view_size=view_size,
            fixed_arc_height=args.fixed_arc_height,
            fixed_midpoint_x=args.fixed_midpoint_x,
            views=args.latent_views,
            distance_metric=args.latent_distance,
            trace_root=out_dir if trace_mode != "none" else None,
            trace_mode=trace_mode,
            verbose_evaluations=args.verbose_evaluations,
        )

        rng = np.random.default_rng(args.seed)
        if args.init_mode == "start-cloud":
            particles = _start_cloud(
                actual_start_eef, init_radius, args.particles, bounds
            ).astype(np.float64)
        elif args.init_mode == "random-start-cloud":
            local_offsets = rng.uniform(
                -init_radius, init_radius, size=(args.particles, 3)
            )
            particles = np.clip(
                actual_start_eef[None, :] + local_offsets,
                bounds[:, 0],
                bounds[:, 1],
            ).astype(np.float64)
        else:
            particles = rng.uniform(
                bounds[:, 0], bounds[:, 1], size=(args.particles, 3)
            ).astype(np.float64)
        initial_particles = particles.copy()
        print(
            f"[init] mode={args.init_mode} center={particles.mean(axis=0).tolist()} "
            f"spread={np.std(particles, axis=0).tolist()}"
        )

        gradient_rollouts = 2 * 3 if args.latent_weight > 0.0 else 0
        repeatability_count = min(args.repeatability_particles, args.particles)
        per_update_rollouts = (
            args.particles * (1 + gradient_rollouts) + repeatability_count
        )
        final_rollouts = args.particles + repeatability_count
        total_planned_rollouts = args.iterations * per_update_rollouts + final_rollouts
        print(
            f"[plan] {args.iterations} updates x {per_update_rollouts} rollouts "
            f"+ {final_rollouts} final evaluations = {total_planned_rollouts} total"
        )
        print(
            f"[plan] encoder={args.feature_encoder} "
            f"model={args.dino_model if args.feature_encoder == 'dinov3' else 'flux_ae'} "
            f"objective={args.latent_distance} transport={args.transport} "
            f"views={args.latent_views} trace_mode={trace_mode} "
            f"actions={move_steps} move + {settle_steps} settle gain={controller_gain}"
        )

        def evaluate_population(
            evaluated_particles: np.ndarray,
            iteration: int,
            *,
            compute_gradients: bool,
        ) -> dict[str, Any]:
            nonlocal global_best
            iteration_dir = out_dir / f"iter_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            energies = np.zeros(args.particles, dtype=np.float64)
            goal_errors = np.zeros(args.particles, dtype=np.float64)
            terminal_eefs = np.zeros_like(evaluated_particles)
            energy_gradients = np.zeros_like(evaluated_particles)
            scores = np.zeros_like(evaluated_particles)
            latent_metrics = {
                name: np.zeros(args.particles, dtype=np.float64)
                for name in ("rms", "cosine", "token_cosine")
            }
            rollout_trace_files: list[str | None] = [None] * args.particles
            finite_difference_probes: list[list[dict[str, Any]]] = [
                [] for _ in range(args.particles)
            ]
            repeatability: list[dict[str, Any] | None] = [
                None for _ in range(args.particles)
            ]
            best_index, best_energy, best_image = 0, float("inf"), None

            for index, theta in enumerate(evaluated_particles):
                energy, image, terminal_eef, metrics, trace_file = energy_fn(
                    theta,
                    trace_context={
                        "iteration": iteration,
                        "particle": index,
                        "evaluation": "base",
                    },
                )
                energies[index] = energy
                terminal_eefs[index] = terminal_eef
                for name in latent_metrics:
                    latent_metrics[name][index] = metrics[name]
                rollout_trace_files[index] = trace_file
                goal_errors[index] = float(np.linalg.norm(terminal_eef - physical_goal))
                if index < repeatability_count:
                    repeat_result = energy_fn(
                        theta,
                        trace_context={
                            "iteration": iteration,
                            "particle": index,
                            "evaluation": "base_repeat",
                        },
                    )
                    repeatability[index] = {
                        "energy": float(repeat_result[0]),
                        "energy_delta": float(repeat_result[0] - energy),
                        "energy_abs_delta": float(abs(repeat_result[0] - energy)),
                        "latent_metrics": repeat_result[3],
                        "latent_metric_deltas": {
                            name: float(repeat_result[3][name] - metrics[name])
                            for name in latent_metrics
                        },
                        "terminal_eef": repeat_result[2].tolist(),
                        "terminal_eef_delta_m": (
                            repeat_result[2] - terminal_eef
                        ).tolist(),
                        "terminal_eef_delta_norm_m": float(
                            np.linalg.norm(repeat_result[2] - terminal_eef)
                        ),
                        "trace_file": repeat_result[4],
                    }
                if args.save_all_particles:
                    image.save(iteration_dir / f"particle_{index:02d}.png")
                if energy < best_energy:
                    best_index, best_energy, best_image = index, energy, image
                if global_best is None or energy < float(global_best["energy"]):
                    global_best = {
                        "energy": float(energy),
                        "iteration": int(iteration),
                        "particle": int(index),
                        "target_eef": np.asarray(theta, dtype=np.float64).copy(),
                        "evaluated_terminal_eef": terminal_eef.copy(),
                    }
                if compute_gradients and args.latent_weight > 0.0:
                    gradient, probes = _finite_difference_grad(
                        energy_fn,
                        theta,
                        fd_eps,
                        bounds,
                        iteration=iteration,
                        particle=index,
                    )
                    energy_gradients[index] = gradient
                    finite_difference_probes[index] = probes
                    # log p = -E / T  =>  grad log p = -grad E / T
                    scores[index] = -gradient / args.temperature

            if best_image is not None:
                best_image.save(iteration_dir / "best.png")
            return {
                "particles": evaluated_particles.copy(),
                "energies": energies,
                "goal_errors": goal_errors,
                "terminal_eefs": terminal_eefs,
                "energy_gradients": energy_gradients,
                "scores": scores,
                "latent_metrics": latent_metrics,
                "rollout_trace_files": rollout_trace_files,
                "finite_difference_probes": finite_difference_probes,
                "repeatability": repeatability,
                "best_index": int(best_index),
            }

        def base_record(evaluation: dict[str, Any], iteration: int) -> dict[str, Any]:
            evaluated_particles = evaluation["particles"]
            energies = evaluation["energies"]
            goal_errors = evaluation["goal_errors"]
            best_index = int(evaluation["best_index"])
            return {
                "iteration": int(iteration),
                "energy_min": float(energies.min()),
                "energy_mean": float(energies.mean()),
                "energy_max": float(energies.max()),
                "energies": energies.tolist(),
                "objective": args.latent_distance,
                "latent_metrics": {
                    name: values.tolist()
                    for name, values in evaluation["latent_metrics"].items()
                },
                "goal_error_min_m": float(goal_errors.min()),
                "goal_error_mean_m": float(goal_errors.mean()),
                "goal_errors_m": goal_errors.tolist(),
                "best_particle": best_index,
                "best_particle_target_eef": evaluated_particles[best_index].tolist(),
                "best_image": f"iter_{iteration:03d}/best.png",
                "global_best_energy": float(global_best["energy"]),
                "global_best_iteration": int(global_best["iteration"]),
                "global_best_particle": int(global_best["particle"]),
                "global_best_target_eef": np.asarray(
                    global_best["target_eef"], dtype=np.float64
                ).tolist(),
                "terminal_eefs": evaluation["terminal_eefs"].tolist(),
                "target_tracking_errors_m": np.linalg.norm(
                    evaluation["terminal_eefs"] - evaluated_particles, axis=1
                ).tolist(),
                "rollout_trace_files": evaluation["rollout_trace_files"],
                "finite_difference_probes": evaluation[
                    "finite_difference_probes"
                ],
                "repeatability": evaluation["repeatability"],
                "target_diagnostics": _goal_axis_diagnostics(
                    evaluated_particles, actual_start_eef, physical_goal
                ),
                "terminal_diagnostics": _goal_axis_diagnostics(
                    evaluation["terminal_eefs"], actual_start_eef, physical_goal
                ),
                "particles_before_update": evaluated_particles.tolist(),
            }

        def write_history() -> None:
            (out_dir / "history.json").write_text(
                json.dumps(
                    {
                        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "run_dir": str(run_dir),
                        "goal_path": str(goal_path),
                        "goal_latent_source": goal_source,
                        "config": vars(args),
                        "effective_rollout": {
                            "move_steps": move_steps,
                            "settle_steps": settle_steps,
                            "controller_gain": controller_gain,
                            "fixed_arc_height_m": args.fixed_arc_height,
                            "fixed_midpoint_x_m": args.fixed_midpoint_x,
                            "actions_per_rollout": move_steps + settle_steps,
                            "trace_mode": trace_mode,
                        },
                        "actual_start_eef": actual_start_eef.tolist(),
                        "manifest_physical_goal_eef": manifest_physical_goal.tolist(),
                        "diagnostic_goal_eef": physical_goal.tolist(),
                        "diagnostic_goal_is_optimizer_input": False,
                        "initial_particles": initial_particles.tolist(),
                        "history": history,
                        "latent_pull_summary": pull_summary,
                        "best_replay": best_replay,
                    },
                    indent=2,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )

        for iteration in range(args.iterations):
            evaluated_particles = particles.copy()
            evaluation = evaluate_population(
                evaluated_particles, iteration, compute_gradients=True
            )
            if args.transport == "particle_gd":
                latent_direction = evaluation["scores"].copy()
                repulsion_direction = np.zeros_like(latent_direction)
                direction = args.latent_weight * latent_direction
                bandwidth = None
                kernel = np.eye(args.particles, dtype=np.float64)
            else:
                (
                    direction,
                    bandwidth,
                    latent_direction,
                    repulsion_direction,
                    kernel,
                ) = _svgd_step(
                    evaluated_particles,
                    evaluation["scores"],
                    args.bandwidth_scale,
                    latent_weight=args.latent_weight,
                    repulsion_weight=args.repulsion_weight,
                )
            raw_latent_update = args.step_size * args.latent_weight * latent_direction
            raw_repulsion_update = (
                args.step_size * args.repulsion_weight * repulsion_direction
            )
            raw_update = args.step_size * direction
            capped_update, trust_scales = _cap_updates(
                raw_update, args.max_update_norm
            )
            latent_update = raw_latent_update * trust_scales[:, None]
            repulsion_update = raw_repulsion_update * trust_scales[:, None]
            unclipped_particles = evaluated_particles + capped_update
            particles = np.clip(
                unclipped_particles, bounds[:, 0], bounds[:, 1]
            )
            applied_update = particles - evaluated_particles
            bounds_clipped = unclipped_particles != particles

            record = base_record(evaluation, iteration)
            record.update(
                {
                    "phase": "update",
                    "transport": args.transport,
                    "update_applied": True,
                    "gradients_computed": bool(args.latent_weight > 0.0),
                    "energy_gradients": evaluation["energy_gradients"].tolist(),
                    "scores": evaluation["scores"].tolist(),
                    "score_norms": np.linalg.norm(
                        evaluation["scores"], axis=1
                    ).tolist(),
                    "energy_gradient_norms": np.linalg.norm(
                        evaluation["energy_gradients"], axis=1
                    ).tolist(),
                    "kernel_bandwidth": (
                        float(bandwidth) if bandwidth is not None else None
                    ),
                    "kernel_matrix": kernel.tolist(),
                    "kernel_column_mass": kernel.sum(axis=0).tolist(),
                    "kernel_column_mass_divided_by_particles": (
                        kernel.sum(axis=0) / args.particles
                    ).tolist(),
                    "latent_directions": latent_direction.tolist(),
                    "repulsion_directions": repulsion_direction.tolist(),
                    "total_directions": direction.tolist(),
                    "raw_updates": raw_update.tolist(),
                    "trust_region_scales": trust_scales.tolist(),
                    "latent_updates": latent_update.tolist(),
                    "repulsion_updates": repulsion_update.tolist(),
                    "applied_updates": applied_update.tolist(),
                    "bounds_clipped": bounds_clipped.tolist(),
                    "latent_update_goal_projection_m": _mean_goal_projection(
                        latent_update, actual_start_eef, physical_goal
                    ),
                    "repulsion_update_goal_projection_m": _mean_goal_projection(
                        repulsion_update, actual_start_eef, physical_goal
                    ),
                    "applied_update_goal_projection_m": _mean_goal_projection(
                        applied_update, actual_start_eef, physical_goal
                    ),
                    "latent_update_norm_mean_m": float(
                        np.mean(np.linalg.norm(latent_update, axis=1))
                    ),
                    "repulsion_update_norm_mean_m": float(
                        np.mean(np.linalg.norm(repulsion_update, axis=1))
                    ),
                    "applied_update_norm_mean_m": float(
                        np.mean(np.linalg.norm(applied_update, axis=1))
                    ),
                    "particle_spread_m": float(
                        np.mean(np.std(particles, axis=0))
                    ),
                    "particles_after_update": particles.tolist(),
                }
            )
            history.append(record)
            write_history()
            print(
                f"[iter {iteration:03d}] E_mean={record['energy_mean']:.4f} "
                f"goal_fraction={record['terminal_diagnostics']['centroid_goal_axis_fraction']:+.3f} "
                f"latent_push={record['latent_update_goal_projection_m']:+.4f}m "
                f"repulsion_push={record['repulsion_update_goal_projection_m']:+.4f}m "
                f"rollouts={energy_fn.rollouts}"
            )

        # Score the population produced by the final update.  This gives N+1
        # measured populations for N optimizer updates and makes the last move
        # visible instead of silently leaving it unevaluated.
        final_iteration = args.iterations
        final_evaluation = evaluate_population(
            particles.copy(), final_iteration, compute_gradients=False
        )
        final_record = base_record(final_evaluation, final_iteration)
        zeros = np.zeros_like(particles)
        final_record.update(
            {
                "phase": "final_evaluation",
                "transport": args.transport,
                "update_applied": False,
                "gradients_computed": False,
                "energy_gradients": zeros.tolist(),
                "scores": zeros.tolist(),
                "kernel_bandwidth": None,
                "kernel_matrix": None,
                "kernel_column_mass": None,
                "kernel_column_mass_divided_by_particles": None,
                "score_norms": np.zeros(args.particles).tolist(),
                "energy_gradient_norms": np.zeros(args.particles).tolist(),
                "latent_directions": zeros.tolist(),
                "repulsion_directions": zeros.tolist(),
                "total_directions": zeros.tolist(),
                "raw_updates": zeros.tolist(),
                "trust_region_scales": np.ones(args.particles).tolist(),
                "latent_updates": zeros.tolist(),
                "repulsion_updates": zeros.tolist(),
                "applied_updates": zeros.tolist(),
                "bounds_clipped": np.zeros_like(particles, dtype=bool).tolist(),
                "latent_update_goal_projection_m": 0.0,
                "repulsion_update_goal_projection_m": 0.0,
                "applied_update_goal_projection_m": 0.0,
                "latent_update_norm_mean_m": 0.0,
                "repulsion_update_norm_mean_m": 0.0,
                "applied_update_norm_mean_m": 0.0,
                "particle_spread_m": float(np.mean(np.std(particles, axis=0))),
                "particles_after_update": particles.tolist(),
            }
        )
        history.append(final_record)
        write_history()
        print(
            f"[final {final_iteration:03d}] E_mean={final_record['energy_mean']:.4f} "
            f"goal_fraction="
            f"{final_record['terminal_diagnostics']['centroid_goal_axis_fraction']:+.3f} "
            f"rollouts={energy_fn.rollouts}"
        )

        transition_deltas = [
            np.asarray(next_record["energies"], dtype=np.float64)
            - np.asarray(record["energies"], dtype=np.float64)
            for record, next_record in zip(history[:-1], history[1:])
        ]
        all_transition_deltas = np.concatenate(transition_deltas)
        update_records = [record for record in history if record["update_applied"]]
        initial_diagnostics = history[0]["terminal_diagnostics"]
        final_diagnostics = history[-1]["terminal_diagnostics"]
        energy_mean_delta = float(history[-1]["energy_mean"] - history[0]["energy_mean"])
        goal_fraction_delta = float(
            final_diagnostics["centroid_goal_axis_fraction"]
            - initial_diagnostics["centroid_goal_axis_fraction"]
        )
        repeatability_records = [
            repeat
            for record in history
            for repeat in record["repeatability"]
            if repeat is not None
        ]
        repeatability_abs_deltas = {
            metric: [
                abs(float(repeat["latent_metric_deltas"][metric]))
                for repeat in repeatability_records
            ]
            for metric in ("rms", "cosine", "token_cosine")
        }
        pull_summary = {
            "schema_version": 2,
            "purpose": (
                "Measure whether image-latent attraction moves terminal-pose particles "
                "from the actual start toward a diagnostic physical goal."
            ),
            "diagnostic_goal_is_optimizer_input": False,
            "init_mode": args.init_mode,
            "actual_start_eef": actual_start_eef.tolist(),
            "diagnostic_goal_eef": physical_goal.tolist(),
            "optimizer_updates": int(args.iterations),
            "evaluated_populations": int(len(history)),
            "latent_weight": float(args.latent_weight),
            "repulsion_weight": float(args.repulsion_weight),
            "transport": args.transport,
            "feature_encoder": args.feature_encoder,
            "encoder_provenance": encoder.provenance,
            "latent_distance": args.latent_distance,
            "latent_views": args.latent_views,
            "initial_latent_metric_means": {
                name: float(np.mean(values))
                for name, values in history[0]["latent_metrics"].items()
            },
            "final_latent_metric_means": {
                name: float(np.mean(values))
                for name, values in history[-1]["latent_metrics"].items()
            },
            "repeatability_evaluations": len(repeatability_records),
            "repeatability_latent_metric_abs_delta_mean": {
                metric: (
                    float(np.mean(values)) if values else None
                )
                for metric, values in repeatability_abs_deltas.items()
            },
            "repeatability_terminal_eef_delta_mean_m": (
                float(
                    np.mean(
                        [
                            repeat["terminal_eef_delta_norm_m"]
                            for repeat in repeatability_records
                        ]
                    )
                )
                if repeatability_records
                else None
            ),
            "initial_energy_mean": float(history[0]["energy_mean"]),
            "final_energy_mean": float(history[-1]["energy_mean"]),
            "energy_mean_delta": energy_mean_delta,
            "initial_centroid_goal_error_m": float(
                initial_diagnostics["centroid_goal_error_m"]
            ),
            "final_centroid_goal_error_m": float(
                final_diagnostics["centroid_goal_error_m"]
            ),
            "initial_centroid_goal_axis_fraction": float(
                initial_diagnostics["centroid_goal_axis_fraction"]
            ),
            "final_centroid_goal_axis_fraction": float(
                final_diagnostics["centroid_goal_axis_fraction"]
            ),
            "centroid_goal_axis_fraction_delta": goal_fraction_delta,
            "mean_latent_update_goal_projection_m": float(
                np.mean(
                    [record["latent_update_goal_projection_m"] for record in update_records]
                )
            ),
            "mean_repulsion_update_goal_projection_m": float(
                np.mean(
                    [record["repulsion_update_goal_projection_m"] for record in update_records]
                )
            ),
            "mean_applied_update_goal_projection_m": float(
                np.mean(
                    [record["applied_update_goal_projection_m"] for record in update_records]
                )
            ),
            "particle_transition_energy_delta_mean": float(
                np.mean(all_transition_deltas)
            ),
            "particle_transitions_lower_energy_fraction": float(
                np.mean(all_transition_deltas < 0.0)
            ),
            "bounds_clipped_coordinate_count": int(
                sum(
                    np.count_nonzero(np.asarray(record["bounds_clipped"], dtype=bool))
                    for record in update_records
                )
            ),
            "latent_pull_observed": bool(
                args.latent_weight > 0.0
                and energy_mean_delta < 0.0
                and goal_fraction_delta > 0.0
            ),
        }
        (out_dir / "latent_pull_summary.json").write_text(
            json.dumps(pull_summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_history()

        if global_best is None:
            raise RuntimeError("SVGD completed without evaluating a particle.")

        # Re-run the lowest-energy particle seen over the entire optimization.
        # SVGD's repulsion can make the final population worse than an earlier
        # one, so selecting only the last iteration would discard valid work.
        env.reset()
        obs = env.set_init_state(start_state)
        _synchronize_controllers_to_sim_state(env, start_gripper_actions)
        obs, best_actions, best_eef_path, best_frames = _rollout_to_target(
            env,
            obs,
            np.asarray(global_best["target_eef"], dtype=np.float64),
            move_steps=move_steps,
            settle_steps=settle_steps,
            gain=controller_gain,
            arc_height=args.fixed_arc_height,
            midpoint_x=args.fixed_midpoint_x,
            capture_video=True,
            video_stride=best_video_stride,
            view_size=view_size,
        )
        best_main, best_wrist, best_terminal = _views_from_obs(obs, view_size)
        best_terminal_latent = _encode_view_features(
            encoder,
            best_terminal,
            view_size,
            args.latent_views,
        )
        best_replay_metrics = _optimizer_latent_metrics(
            best_terminal_latent,
            goal_latent,
        )
        best_replay_energy = best_replay_metrics[args.latent_distance]
        best_actual_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        best_terminal_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()

        best_main.save(out_dir / "best_terminal_agentview.png")
        best_wrist.save(out_dir / "best_terminal_wrist.png")
        best_terminal.save(out_dir / "best_terminal.png")
        np.save(out_dir / "best_actions.npy", best_actions.astype(np.float32))
        np.save(out_dir / "best_eef_path.npy", best_eef_path.astype(np.float32))
        np.save(out_dir / "best_terminal_state.npy", best_terminal_state)
        np.save(
            out_dir / "best_terminal_latent.npy",
            best_terminal_latent.astype(np.float32),
        )
        best_video_path = out_dir / "best_rollout.mp4"
        _write_video(best_video_path, best_frames, best_video_fps)
        if not best_video_path.is_file() or best_video_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write non-empty best-rollout video: {best_video_path}")

        best_replay = {
            "goal": {
                "path": str(goal_path.resolve()),
                "latent_source": goal_source,
                "feature_encoder": args.feature_encoder,
                "encoder_provenance": encoder.provenance,
            },
            "optimization": {
                "init_mode": args.init_mode,
                "init_radius_m": init_radius.tolist(),
                "feature_encoder": args.feature_encoder,
                "latent_views": args.latent_views,
                "latent_distance": args.latent_distance,
                "transport": args.transport,
                "latent_weight": float(args.latent_weight),
                "repulsion_weight": float(args.repulsion_weight),
                "max_update_norm_m": args.max_update_norm,
                "actual_start_eef": actual_start_eef.tolist(),
                "diagnostic_goal_eef": physical_goal.tolist(),
                "diagnostic_goal_is_optimizer_input": False,
            },
            "selection": {
                "objective": (
                    f"{args.feature_encoder}_feature_{args.latent_distance}"
                ),
                "energy": float(global_best["energy"]),
                "iteration": int(global_best["iteration"]),
                "particle": int(global_best["particle"]),
                "target_eef": np.asarray(
                    global_best["target_eef"], dtype=np.float64
                ).tolist(),
                "evaluated_terminal_eef": np.asarray(
                    global_best["evaluated_terminal_eef"], dtype=np.float64
                ).tolist(),
            },
            "replay": {
                "objective_energy": float(best_replay_energy),
                "latent_metrics": best_replay_metrics,
                "latent_rms": float(best_replay_metrics["rms"]),
                "objective_energy_delta_from_selection": float(
                    best_replay_energy - float(global_best["energy"])
                ),
                "actual_terminal_eef": best_actual_eef.tolist(),
                "physical_goal_error_m": float(
                    np.linalg.norm(best_actual_eef - physical_goal)
                ),
                "physical_goal_error_is_diagnostic_only": True,
                "target_tracking_error_m": float(
                    np.linalg.norm(
                        best_actual_eef
                        - np.asarray(global_best["target_eef"], dtype=np.float64)
                    )
                ),
                "num_actions": int(best_actions.shape[0]),
                "action_dim": int(best_actions.shape[1]),
                "num_eef_states": int(best_eef_path.shape[0]),
                "num_video_frames": int(len(best_frames)),
                "video_stride": int(best_video_stride),
                "video_fps": int(best_video_fps),
                "move_steps": move_steps,
                "settle_steps": settle_steps,
                "controller_gain": controller_gain,
                "fixed_arc_height_m": float(args.fixed_arc_height),
                "fixed_midpoint_x_m": float(args.fixed_midpoint_x),
                "optimization_rollouts": int(energy_fn.rollouts),
                "final_replay_rollouts": 1,
            },
            "simulator": {
                "bddl": str(Path(manifest["bddl"]).resolve()),
                "sim_seed": int(manifest["sim_seed"]),
                "start_state": str((run_dir / "start_state.npy").resolve()),
                "render_size": int(manifest["render_size"]),
                "view_size": int(view_size),
            },
            "artifacts": {
                "video": "best_rollout.mp4",
                "actions": "best_actions.npy",
                "eef_path": "best_eef_path.npy",
                "terminal_state": "best_terminal_state.npy",
                "terminal_image": "best_terminal.png",
                "terminal_agentview": "best_terminal_agentview.png",
                "terminal_wrist": "best_terminal_wrist.png",
                "terminal_latent": "best_terminal_latent.npy",
                "progress_plot": "progress.png",
                "particle_motion_plot": "particle_motion.png",
                "latent_pull_summary": "latent_pull_summary.json",
            },
            "latent_pull_summary": pull_summary,
        }
        (out_dir / "best_metadata.json").write_text(
            json.dumps(best_replay, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        # Append the selected replay to the completed history. Per-iteration
        # writes above remain useful if a long optimization is interrupted.
        write_history()
    finally:
        env.close()

    if history:
        _progress_plot(history, out_dir / "progress.png")
        _particle_motion_plot(
            history, out_dir / "particle_motion.png", actual_start_eef, physical_goal
        )
    print(f"\n[done] {out_dir}")
    print(f"[done] goal reference: {out_dir / 'goal_reference.png'}")
    print(f"[done] progress plot:  {out_dir / 'progress.png'}")
    print(f"[done] motion plot:    {out_dir / 'particle_motion.png'}")
    print(f"[done] pull summary:   {out_dir / 'latent_pull_summary.json'}")
    if best_replay is not None:
        print(f"[done] best rollout:   {out_dir / 'best_rollout.mp4'}")
        print(f"[done] best metadata:  {out_dir / 'best_metadata.json'}")


if __name__ == "__main__":
    main()
