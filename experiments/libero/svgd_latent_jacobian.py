#!/usr/bin/env python
"""Closed-loop visual servoing on an estimated joint-space latent Jacobian.

Experiment B of the ImageSTL Visual-SVPIO spec (sections 10.1-10.8), run against
simulator observations with every differentiable path deliberately disabled: the
simulator is a black box that turns a joint configuration into an image, exactly
as a physical camera would be.

    capture -> frozen encoder -> z -> L(z, z_goal) -> dL/dz   (autograd)
            -> J_zq estimated from image perturbations
            -> dL/dq = J_zq^T dL/dz
            -> SVGD (or direct servo) -> bounded joint command -> repeat

Two planners share that gradient:

``--planner direct``
    The section 10.7 baseline, ``dq = -eta J^T dL/dz`` with a norm cap, an
    element cap, and joint-limit projection.  One particle, no population.

``--planner svgd_local_linear``
    Section 10.8.  Particles are short joint-increment sequences
    ``DeltaQ [horizon, n_joints]`` rolled out through the *local linear* latent
    model ``z_{k+1} = z_k + J dq_k``.  That model is differentiable in torch, so
    ``dE/dDeltaQ`` is exact for the model (not for the world), and the existing
    RBF/median-bandwidth SVGD update transports the population.  Only the first
    increment is executed before re-observing -- section 10.1 forbids running a
    long sequence open-loop off a local Jacobian.

Why joint space rather than the 3-D endpoint the other runs optimize: a wrist
roll moves the end-effector by zero and the image by a lot.  The endpoint
parameterization holds gripper orientation at its start value and so cannot
represent the ``_roll30``/``_yaw45`` goals at all; ``dz/dq`` can.

``--iterations`` counts closed-loop control cycles here, not optimizer sweeps.
Each cycle runs ``--svgd-iters-per-step`` SVGD updates on the local model, which
cost no rollouts -- the simulator is touched once per cycle plus a Jacobian
refresh every ``--refresh-every-steps`` cycles.

Written to <out-dir>/:
    goal_reference.png            the goal image being optimized toward
    initial.png / best.png        first and lowest-loss captures
    metrics.json                  episode summary (spec section 16)
    history.json                  per-cycle record
    iteration_metrics.jsonl       one line per control cycle
    jacobian_initial.npy          first central-difference estimate
    jacobian_latest.npy           after the last Broyden update
    jacobian_diagnostics.jsonl    prediction error / refresh events
    command_log.jsonl             every commanded joint delta
    camera_frames/step_XXX.png    what the controller saw
    joint_path.npy                commanded configuration after every cycle
    progress.png                  loss and held-out physical error vs cycle
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import sys
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
    JacobianRefreshMonitor,
    JointLatentObserver,
    LatentJacobianEstimator,
    latent_loss_and_grad,
    latent_loss_terms,
)
from svgd_endpoint import (  # noqa: E402
    _cap_updates,
    _encode_view_features,
    _optimizer_latent_metrics,
    _svgd_step,
)
from sample_endpoint_trajectories import env_from_manifest  # noqa: E402


def _jsonable(value: Any) -> Any:
    """Replace non-finite floats with null.

    A singular Jacobian gives an infinite condition number, and ``allow_nan=False``
    turns that into a crash after the run has already spent its rollouts.
    """
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n", "utf-8"
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), allow_nan=False) + "\n")


# --------------------------------------------------------------------------- #
# local linear latent model (spec 10.8)
# --------------------------------------------------------------------------- #


def _local_linear_cost(
    increments: torch.Tensor,
    *,
    z0: torch.Tensor,
    q0: torch.Tensor,
    jacobian: torch.Tensor,
    goal_tokens: torch.Tensor,
    token_shape: tuple[int, int],
    limits_low: torch.Tensor,
    limits_high: torch.Tensor,
    cosine_weight: float,
    l2_weight: float,
    softmin_temperature: float,
    temporal_mode: str,
    safety_weight: float,
    safety_temperature: float,
    smooth_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cost of one particle's increment sequence under ``z_{k+1} = z_k + J dq_k``.

    ``increments`` is ``[H, n_joints]``.  Lower is better, matching the sign
    convention the SVGD step already assumes.
    """
    horizon = increments.shape[0]
    # Cumulative sums give every intermediate state in one shot: the model is
    # linear, so z_k = z_0 + J (sum_{i<k} dq_i) and q_k = q_0 + sum_{i<k} dq_i.
    cumulative = torch.cumsum(increments, dim=0)  # [H, n_joints]
    states = q0.unsqueeze(0) + cumulative  # [H, n_joints]
    latents = z0.unsqueeze(0) + cumulative @ jacobian.T  # [H, d]

    tokens = latents.reshape(horizon, token_shape[0], token_shape[1])
    frame_loss, frame_cos, frame_l2 = latent_loss_terms(
        tokens,
        goal_tokens.unsqueeze(0),
        cosine_weight=cosine_weight,
        l2_weight=l2_weight,
    )

    if temporal_mode == "terminal":
        visual = frame_loss[-1]
    elif temporal_mode == "mean":
        visual = frame_loss.mean()
    else:
        # Normalized soft minimum (spec 7.4): the -log(T) term keeps the cost
        # from moving just because the horizon changed.
        tau = softmin_temperature
        visual = -tau * (
            torch.logsumexp(-frame_loss / tau, dim=0)
            - torch.log(torch.tensor(float(horizon), device=frame_loss.device))
        )

    slack = torch.minimum(states - limits_low, limits_high - states)
    safety = safety_temperature * torch.nn.functional.softplus(
        -slack / safety_temperature
    )
    safety_cost = safety.sum(dim=-1).mean()

    if horizon > 1:
        smooth_cost = ((increments[1:] - increments[:-1]) ** 2).sum(dim=-1).mean()
    else:
        smooth_cost = torch.zeros((), device=increments.device)

    total = visual + safety_weight * safety_cost + smooth_weight * smooth_cost
    return total, {
        "visual": visual.detach(),
        "safety": safety_cost.detach(),
        "smooth": smooth_cost.detach(),
        "frame_loss_min": frame_loss.detach().min(),
        "frame_cosine_min": frame_cos.detach().min(),
        "frame_l2_min": frame_l2.detach().min(),
    }


def _local_linear_value_and_grad(
    particles: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The spec's ``GradientOracle.value_and_grad_batch`` for the local model.

    Returns ``costs [N]`` and ``grads [N, H, n_joints]`` with lower cost better,
    so the caller can form ``score = -grad / temperature`` unchanged.
    """
    device = kwargs["z0"].device
    costs = np.zeros(particles.shape[0], dtype=np.float64)
    gradients = np.zeros_like(particles, dtype=np.float64)
    terms: dict[str, list[float]] = {}
    for index, particle in enumerate(particles):
        increments = torch.as_tensor(
            particle, dtype=torch.float32, device=device
        ).requires_grad_(True)
        cost, parts = _local_linear_cost(increments, **kwargs)
        (gradient,) = torch.autograd.grad(cost, increments)
        costs[index] = float(cost.detach())
        gradients[index] = gradient.detach().cpu().numpy().astype(np.float64)
        for name, value in parts.items():
            terms.setdefault(name, []).append(float(value))
    return costs, gradients, terms


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #


def _progress_plot(history: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [record["step"] for record in history]
    losses = [record["observed_loss"] for record in history]
    errors = [record["physical_goal_error_m"] for record in history]
    relative = [
        record["jacobian_relative_prediction_error"]
        if record["jacobian_relative_prediction_error"] is not None
        else np.nan
        for record in history
    ]

    figure, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True)
    axes[0].plot(steps, losses, marker="o", markersize=3, color="tab:blue")
    axes[0].set_ylabel("observed latent loss")
    axes[0].set_title(path.parent.name)
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, errors, marker="o", markersize=3, color="tab:red")
    axes[1].set_ylabel("held-out EEF error (m)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, relative, marker="o", markersize=3, color="tab:green")
    axes[2].axhline(0.5, linestyle="--", color="gray", linewidth=1.0)
    axes[2].set_ylabel("||dz - J dq|| / ||dz||")
    axes[2].set_xlabel("control cycle")
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument(
        "--feature-encoder", choices=["flux_ae", "dinov3"], default="flux_ae"
    )
    parser.add_argument("--editor-ae")
    parser.add_argument("--flux2-src", default=str(REPO_ROOT / "third_party" / "flux2"))
    parser.add_argument("--dino-model", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--latent-views", choices=["both", "agentview", "right", "wrist"], default="agentview"
    )
    parser.add_argument(
        "--goal-latent-source",
        choices=["reencode"],
        default="reencode",
        help="Kept explicit so the goal path matches the endpoint runs.",
    )
    parser.add_argument(
        "--diagnostic-goal-eef",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Known physical goal used only for plots/diagnostics; defaults to the "
            "manifest goal. It never affects the loss, the Jacobian, or a command."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=45,
        help="Closed-loop control cycles (one capture and one command each).",
    )
    parser.add_argument(
        "--planner",
        choices=["svgd_local_linear", "direct"],
        default="svgd_local_linear",
    )

    latent = parser.add_argument_group("latent objective")
    latent.add_argument("--cosine-weight", type=float, default=1.0)
    latent.add_argument(
        "--l2-weight",
        type=float,
        default=0.0,
        help="Spec default is 0.10; 0 keeps the reported loss equal to token_cosine.",
    )
    latent.add_argument(
        "--temporal-mode", choices=["softmin", "terminal", "mean"], default="softmin"
    )
    latent.add_argument("--time-softmin-temperature", type=float, default=0.05)

    jacobian = parser.add_argument_group("latent Jacobian")
    jacobian.add_argument("--fd-delta-rad", type=float, default=0.005)
    jacobian.add_argument("--broyden-eps", type=float, default=1e-8)
    jacobian.add_argument("--refresh-every-steps", type=int, default=25)
    jacobian.add_argument("--max-relative-prediction-error", type=float, default=0.5)
    jacobian.add_argument("--bad-prediction-patience", type=int, default=3)
    jacobian.add_argument(
        "--no-broyden",
        action="store_true",
        help="Ablation: re-probe every cycle instead of updating online.",
    )

    planner = parser.add_argument_group("planner")
    planner.add_argument("--horizon", type=int, default=10)
    planner.add_argument("--particles", type=int, default=10)
    planner.add_argument("--svgd-iters-per-step", type=int, default=20)
    planner.add_argument("--temperature", type=float, default=0.10)
    planner.add_argument("--repulsion-weight", type=float, default=0.01)
    planner.add_argument("--latent-weight", type=float, default=1.0)
    planner.add_argument("--bandwidth-scale", type=float, default=1.0)
    planner.add_argument("--step-size", type=float, default=0.01)
    planner.add_argument("--max-update-norm", type=float, default=0.02)
    planner.add_argument("--init-scale-rad", type=float, default=0.002)
    planner.add_argument("--safety-weight", type=float, default=0.25)
    planner.add_argument("--safety-temperature", type=float, default=0.02)
    planner.add_argument("--smooth-weight", type=float, default=0.001)

    control = parser.add_argument_group("bounded execution")
    control.add_argument("--gradient-step-size", type=float, default=0.05)
    control.add_argument("--max-joint-step-rad", type=float, default=0.02)
    control.add_argument("--max-joint-step-norm-rad", type=float, default=0.04)
    control.add_argument("--joint-limit-margin-rad", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-camera-frames", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _bounded_step(
    raw: np.ndarray, *, max_norm: float, max_element: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Norm cap then element cap, in that order (spec 10.7)."""
    norm = float(np.linalg.norm(raw))
    scale = 1.0 if norm <= max_norm else max_norm / max(norm, 1e-12)
    stepped = raw * scale
    clipped = np.clip(stepped, -max_element, max_element)
    return clipped, {
        "raw_norm_rad": norm,
        "norm_trust_scale": scale,
        "element_clipped": float(np.abs(stepped - clipped).max()),
        "applied_norm_rad": float(np.linalg.norm(clipped)),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.feature_encoder == "flux_ae" and not args.editor_ae:
        parser.error("--editor-ae is required with --feature-encoder flux_ae")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.planner == "svgd_local_linear" and args.particles < 2:
        parser.error("--planner svgd_local_linear needs at least 2 particles")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.temperature <= 0.0 or args.time_softmin_temperature <= 0.0:
        parser.error("Temperatures must be positive")

    run_dir = Path(args.run_dir).resolve()
    goal_path = Path(args.goal).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not (run_dir / "manifest.json").is_file() or not goal_path.is_file():
        parser.error("The run manifest or goal image does not exist")
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(
        f"physical_gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n"
        f"command={shlex.join([sys.executable, *sys.argv])}\n",
        encoding="utf-8",
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    view_size = int(manifest["view_size"])
    start_state = np.load(run_dir / "start_state.npy")
    physical_goal = (
        np.asarray(args.diagnostic_goal_eef, dtype=np.float64)
        if args.diagnostic_goal_eef is not None
        else np.asarray(manifest["physical_goal_eef"], dtype=np.float64)
    )

    device_string = args.device
    if device_string == "auto":
        device_string = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)

    from score_endpoint_candidates import DinoV3FeatureMetric, FluxAutoencoderMetric

    if args.feature_encoder == "flux_ae":
        encoder: Any = FluxAutoencoderMetric(
            Path(args.editor_ae), Path(args.flux2_src), args.device
        )
    else:
        encoder = DinoV3FeatureMetric(args.dino_model, args.device)

    goal_image = Image.open(goal_path).convert("RGB")
    goal_image.save(out_dir / "goal_reference.png")
    goal_tokens = np.asarray(
        _encode_view_features(encoder, goal_image, view_size, args.latent_views),
        dtype=np.float32,
    )
    if goal_tokens.ndim == 3:
        goal_tokens = goal_tokens[0]
    token_shape = (int(goal_tokens.shape[0]), int(goal_tokens.shape[1]))
    descriptor_dim = token_shape[0] * token_shape[1]

    env = env_from_manifest(manifest)
    history: list[dict[str, Any]] = []
    diagnostics_path = out_dir / "jacobian_diagnostics.jsonl"
    commands_path = out_dir / "command_log.jsonl"
    iteration_path = out_dir / "iteration_metrics.jsonl"
    frames_dir = out_dir / "camera_frames"
    if args.save_camera_frames:
        frames_dir.mkdir(exist_ok=True)

    try:
        env.seed(int(manifest["sim_seed"]))
        observer = JointLatentObserver(
            env,
            start_state,
            encoder,
            view_size=view_size,
            views=args.latent_views,
            joint_limit_margin=args.joint_limit_margin_rad,
        )
        estimator = LatentJacobianEstimator(
            observer, delta_rad=args.fd_delta_rad, broyden_eps=args.broyden_eps
        )
        monitor = JacobianRefreshMonitor(
            max_relative_prediction_error=args.max_relative_prediction_error,
            patience=args.bad_prediction_patience,
            refresh_every_steps=args.refresh_every_steps,
        )

        goal_torch = torch.as_tensor(goal_tokens, device=device)
        limits_low = torch.as_tensor(
            observer.joint_limits[:, 0], dtype=torch.float32, device=device
        )
        limits_high = torch.as_tensor(
            observer.joint_limits[:, 1], dtype=torch.float32, device=device
        )

        q = observer.home_q.copy()
        observation = observer.observe(q)
        observation.image.save(out_dir / "initial.png")
        initial_loss, _, initial_terms = latent_loss_and_grad(
            observation.tokens,
            goal_tokens,
            device=device,
            cosine_weight=args.cosine_weight,
            l2_weight=args.l2_weight,
        )

        # Section 10.3: one central-difference estimate before any control.
        jacobian, columns = estimator.central_difference(q)
        np.save(out_dir / "jacobian_initial.npy", jacobian.astype(np.float32))
        _append_jsonl(
            diagnostics_path,
            {
                "step": -1,
                "event": "central_difference",
                "reason": "initialization",
                "columns": [vars(record) for record in columns],
                "summary": estimator.summary(),
            },
        )
        monitor.mark_refreshed()

        particles = (
            np.random.default_rng(args.seed).normal(
                scale=args.init_scale_rad,
                size=(args.particles, args.horizon, observer.num_joints),
            )
            if args.planner == "svgd_local_linear"
            else np.zeros((1, args.horizon, observer.num_joints))
        )

        best = {
            "loss": float(initial_loss),
            "step": -1,
            "q": q.copy(),
            "eef": observation.eef_pos.copy(),
            "image": observation.image,
        }
        nan_events = 0

        for step in range(args.iterations):
            metrics = _optimizer_latent_metrics(
                observation.tokens[None], goal_tokens[None]
            )
            loss, grad_z, loss_terms = latent_loss_and_grad(
                observation.tokens,
                goal_tokens,
                device=device,
                cosine_weight=args.cosine_weight,
                l2_weight=args.l2_weight,
            )
            if args.save_camera_frames:
                observation.image.save(frames_dir / f"step_{step:03d}.png")
            if loss < best["loss"]:
                best = {
                    "loss": float(loss),
                    "step": step,
                    "q": q.copy(),
                    "eef": observation.eef_pos.copy(),
                    "image": observation.image,
                }

            planner_terms: dict[str, float] = {}
            if args.planner == "direct":
                # dL/dq = J^T dL/dz  (spec 10.7)
                joint_gradient = estimator.matrix.T @ grad_z
                raw_step = -args.gradient_step_size * joint_gradient
            else:
                z0 = torch.as_tensor(
                    observation.descriptor.astype(np.float32), device=device
                )
                jacobian_torch = torch.as_tensor(
                    estimator.matrix.astype(np.float32), device=device
                )
                q0 = torch.as_tensor(q.astype(np.float32), device=device)
                cost_kwargs = dict(
                    z0=z0,
                    q0=q0,
                    jacobian=jacobian_torch,
                    goal_tokens=goal_torch,
                    token_shape=token_shape,
                    limits_low=limits_low,
                    limits_high=limits_high,
                    cosine_weight=args.cosine_weight,
                    l2_weight=args.l2_weight,
                    softmin_temperature=args.time_softmin_temperature,
                    temporal_mode=args.temporal_mode,
                    safety_weight=args.safety_weight,
                    safety_temperature=args.safety_temperature,
                    smooth_weight=args.smooth_weight,
                )
                flat_shape = (args.particles, args.horizon * observer.num_joints)
                for _ in range(args.svgd_iters_per_step):
                    costs, gradients, terms = _local_linear_value_and_grad(
                        particles, **cost_kwargs
                    )
                    bad = ~np.isfinite(gradients)
                    if bad.any():
                        nan_events += 1
                        gradients[bad] = 0.0
                    # log p = -E / T  =>  grad log p = -grad E / T
                    scores = (-gradients / args.temperature).reshape(flat_shape)
                    direction, _, _, _, _ = _svgd_step(
                        particles.reshape(flat_shape),
                        scores,
                        args.bandwidth_scale,
                        latent_weight=args.latent_weight,
                        repulsion_weight=args.repulsion_weight,
                    )
                    update, _ = _cap_updates(
                        args.step_size * direction, args.max_update_norm
                    )
                    particles = np.clip(
                        particles + update.reshape(particles.shape),
                        -args.max_joint_step_rad,
                        args.max_joint_step_rad,
                    )
                costs, _, terms = _local_linear_value_and_grad(particles, **cost_kwargs)
                elite = int(np.argmin(costs))
                planner_terms = {
                    "particle_cost_min": float(costs.min()),
                    "particle_cost_mean": float(costs.mean()),
                    "best_particle": elite,
                    "predicted_visual": float(terms["visual"][elite]),
                    "predicted_safety": float(terms["safety"][elite]),
                    "predicted_smooth": float(terms["smooth"][elite]),
                }
                raw_step = particles[elite, 0].copy()

            delta_q, step_diagnostics = _bounded_step(
                raw_step,
                max_norm=args.max_joint_step_norm_rad,
                max_element=args.max_joint_step_rad,
            )
            q_command = observer.project_joint_limits(q + delta_q)
            applied = q_command - q

            next_observation = observer.observe(q_command)
            delta_z = (
                next_observation.descriptor.astype(np.float64)
                - observation.descriptor.astype(np.float64)
            )
            relative_error = estimator.prediction_error(applied, delta_z)

            _append_jsonl(
                commands_path,
                {
                    "step": step,
                    "q": q.tolist(),
                    "raw_step_rad": np.asarray(raw_step, dtype=float).tolist(),
                    "commanded_delta_rad": applied.tolist(),
                    "q_command": q_command.tolist(),
                    "limit_projected": bool(
                        not np.allclose(q + delta_q, q_command, atol=1e-12)
                    ),
                    **step_diagnostics,
                },
            )

            record = {
                "step": step,
                "observed_loss": float(loss),
                "observed_cosine": float(loss_terms["cosine"]),
                "observed_l2": float(loss_terms["l2"]),
                "latent_metrics": metrics,
                "grad_z_norm": float(np.linalg.norm(grad_z)),
                "joint_gradient_norm": float(
                    np.linalg.norm(estimator.matrix.T @ grad_z)
                ),
                "commanded_delta_norm_rad": float(np.linalg.norm(applied)),
                "physical_goal_error_m": float(
                    np.linalg.norm(observation.eef_pos - physical_goal)
                ),
                "eef_pos": observation.eef_pos.tolist(),
                "q": q.tolist(),
                "jacobian_relative_prediction_error": relative_error,
                "jacobian_condition_number": estimator.condition_number(),
                "planner": planner_terms,
                "simulator_captures": observer.captures,
            }
            history.append(record)
            _append_jsonl(iteration_path, record)
            if args.verbose:
                print(
                    f"[{step:03d}] loss={loss:.6f} "
                    f"|dL/dz|={np.linalg.norm(grad_z):.4e} "
                    f"|dq|={np.linalg.norm(applied):.5f} "
                    f"r={relative_error if relative_error is not None else float('nan'):.3f} "
                    f"eef_err={record['physical_goal_error_m']:.4f}",
                    flush=True,
                )

            # Section 10.5 / 10.6: maintain the model, then refresh if it stopped
            # predicting.  A refresh costs 2*n_joints captures, so it is the
            # exception rather than the per-cycle default.
            if not args.no_broyden:
                estimator.broyden_update(applied, delta_z)
            needs_refresh, reason = monitor.update(relative_error)
            if args.no_broyden:
                needs_refresh, reason = True, "no_broyden_ablation"
            if needs_refresh and step < args.iterations - 1:
                # central_difference restores the arm to q_command on exit, and
                # q -> image is deterministic, so next_observation stays valid.
                _, columns = estimator.central_difference(q_command)
                monitor.mark_refreshed()
                _append_jsonl(
                    diagnostics_path,
                    {
                        "step": step,
                        "event": "central_difference",
                        "reason": reason,
                        "relative_prediction_error": relative_error,
                        "columns": [vars(item) for item in columns],
                        "summary": estimator.summary(),
                    },
                )
            else:
                _append_jsonl(
                    diagnostics_path,
                    {
                        "step": step,
                        "event": "broyden",
                        "relative_prediction_error": relative_error,
                        "condition_number": estimator.condition_number(),
                        "delta_q_norm": float(np.linalg.norm(applied)),
                        "delta_z_norm": float(np.linalg.norm(delta_z)),
                    },
                )

            q = q_command
            observation = next_observation
            if args.planner == "svgd_local_linear":
                # Warm start: shift executed increments off the front.
                particles = np.concatenate(
                    [particles[:, 1:], np.zeros_like(particles[:, :1])], axis=1
                )

        final_loss, _, final_terms = latent_loss_and_grad(
            observation.tokens,
            goal_tokens,
            device=device,
            cosine_weight=args.cosine_weight,
            l2_weight=args.l2_weight,
        )
        final_metrics = _optimizer_latent_metrics(
            observation.tokens[None], goal_tokens[None]
        )
        observation.image.save(out_dir / "final.png")
        best["image"].save(out_dir / "best.png")
        np.save(out_dir / "jacobian_latest.npy", estimator.matrix.astype(np.float32))
        np.save(
            out_dir / "joint_path.npy",
            np.asarray([record["q"] for record in history], dtype=np.float64),
        )
        np.save(out_dir / "best_q.npy", best["q"])

        reduction = (
            (float(initial_loss) - float(best["loss"])) / float(initial_loss)
            if initial_loss > 0.0
            else 0.0
        )
        metrics_payload = {
            "schema_version": 1,
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "planner": args.planner,
            "feature_encoder": args.feature_encoder,
            "gradient_source": "finite_difference_latent_jacobian"
            + ("" if args.no_broyden else "_with_broyden"),
            "descriptor_dim": descriptor_dim,
            "token_shape": list(token_shape),
            "num_joints": observer.num_joints,
            "control_cycles": args.iterations,
            "initial_visual_loss": float(initial_loss),
            "initial_visual_cosine_loss": float(initial_terms["cosine"]),
            "final_visual_loss": float(final_loss),
            "final_visual_cosine_loss": float(final_terms["cosine"]),
            "final_visual_l2_loss": float(final_terms["l2"]),
            "best_visual_loss": float(best["loss"]),
            "best_step": int(best["step"]),
            "visual_loss_reduction_fraction": float(reduction),
            "final_latent_metrics": final_metrics,
            "num_nan_gradient_events": int(nan_events),
            "simulator_captures": int(observer.captures),
            "jacobian": estimator.summary(),
            "held_out_diagnostics": {
                "note": "never enters the objective, the Jacobian, or a command",
                "diagnostic_goal_eef": physical_goal.tolist(),
                "initial_eef_error_m": float(history[0]["physical_goal_error_m"]),
                "final_eef_error_m": float(
                    np.linalg.norm(observation.eef_pos - physical_goal)
                ),
                "best_eef_error_m": float(
                    np.linalg.norm(best["eef"] - physical_goal)
                ),
            },
        }
        _write_json(out_dir / "metrics.json", metrics_payload)
        _write_json(out_dir / "history.json", history)
        _write_json(
            out_dir / "config_resolved.yaml.json",
            {key: value for key, value in sorted(vars(args).items())},
        )
        _progress_plot(history, out_dir / "progress.png")
        print(
            f"[done] initial={initial_loss:.6f} best={best['loss']:.6f} "
            f"final={final_loss:.6f} reduction={reduction:.3%} "
            f"captures={observer.captures}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
