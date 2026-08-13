#!/usr/bin/env python
"""Joint-space visual latent Jacobian: ``J_zq = dz/dq``.

This is the Experiment B machinery from the ImageSTL Visual-SVPIO spec, sections
10.2-10.6, ported to the LIBERO/robosuite scene used by the endpoint runs.

The identity being implemented is

    dL/dq = J_zq^T  dL/dz

where ``dL/dz`` comes from autograd through the frozen encoder and ``J_zq`` is
*estimated from observations only* -- the simulator is treated as a black box
that turns a joint configuration into an image.  Nothing here differentiates
MuJoCo or the renderer, which is the whole point: the same estimator works
against a physical camera.

Two deliberate choices about the observation model:

1. ``observe`` sets the arm's ``qpos`` and calls ``sim.forward()`` rather than
   stepping physics toward ``q``.  A finite-difference column is only meaningful
   if the two probe images differ by the joint perturbation and nothing else; a
   controller transient or a nudged object would contaminate it.  Kinematic
   placement makes ``q -> image`` an exact deterministic function, which is what
   a real robot approximates by servoing to ``q`` and holding still.
2. robosuite caches observables, so every capture forces an observable refresh.
   Without ``force_update=True`` the renders come back byte-identical no matter
   what ``qpos`` says, and every Jacobian column silently estimates as zero.

The descriptor is the encoder's native token grid for the selected view (14x14
tokens for a 224px agentview render), which already sits at the resolution the
spec's ``spatial_pool_h/w: 16`` suggestion is aiming for, so no extra pooling is
applied.  Goal and current observations go through the identical path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from svgd_endpoint import _encode_view_features  # noqa: E402

__all__ = [
    "JointLatentObserver",
    "LatentJacobianEstimator",
    "JacobianRefreshMonitor",
    "latent_loss_terms",
    "latent_loss_and_grad",
]


# --------------------------------------------------------------------------- #
# observation model
# --------------------------------------------------------------------------- #


@dataclass
class JointObservation:
    """One (configuration, image, descriptor) triple."""

    q: np.ndarray
    image: Image.Image
    tokens: np.ndarray  # [P, C]
    eef_pos: np.ndarray
    eef_quat_xyzw: np.ndarray

    @property
    def descriptor(self) -> np.ndarray:
        """Flat ``d = P*C`` vector, the space the Jacobian maps into."""
        return self.tokens.reshape(-1)


class JointLatentObserver:
    """``q -> rendered image -> frozen-encoder tokens`` for a staged LIBERO scene.

    The scene is restored from the same flat MuJoCo snapshot the endpoint runs
    stage from, so the objects, table, and camera are identical across
    experiments and the descriptors are directly comparable.
    """

    def __init__(
        self,
        env: Any,
        start_state: np.ndarray,
        encoder: Any,
        *,
        view_size: int,
        views: str = "agentview",
        joint_limit_margin: float = 0.05,
    ) -> None:
        self.env = env
        self.encoder = encoder
        self.view_size = int(view_size)
        self.views = views
        self.captures = 0

        env.reset()
        env.set_init_state(np.asarray(start_state, dtype=np.float64))
        inner = getattr(env, "env", env)
        self._inner = inner
        self.sim = inner.sim
        robot = inner.robots[0]
        self._qpos_index = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        self._qvel_index = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
        joint_index = np.asarray(robot._ref_joint_indexes, dtype=int)

        limits = np.asarray(self.sim.model.jnt_range[joint_index], dtype=np.float64)
        limited = np.asarray(self.sim.model.jnt_limited[joint_index], dtype=bool)
        # An unlimited joint reports [0, 0]; treat it as unconstrained rather
        # than as a zero-width box that rejects every perturbation.
        limits[~limited, 0] = -np.inf
        limits[~limited, 1] = np.inf
        self.raw_joint_limits = limits
        margin = float(joint_limit_margin)
        self.joint_limits = np.stack(
            [limits[:, 0] + margin, limits[:, 1] - margin], axis=1
        )
        self.num_joints = int(self._qpos_index.size)
        self.home_q = self.joint_positions()

    # -- state ------------------------------------------------------------- #

    def joint_positions(self) -> np.ndarray:
        return np.asarray(self.sim.data.qpos[self._qpos_index], dtype=np.float64).copy()

    def project_joint_limits(self, q: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(q, dtype=np.float64),
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )

    def within_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=np.float64)
        return bool(
            np.all(q >= self.joint_limits[:, 0]) and np.all(q <= self.joint_limits[:, 1])
        )

    def limit_slack(self, q: np.ndarray) -> np.ndarray:
        """Signed distance to the nearer safe bound; negative means outside."""
        q = np.asarray(q, dtype=np.float64)
        return np.minimum(q - self.joint_limits[:, 0], self.joint_limits[:, 1] - q)

    # -- observation ------------------------------------------------------- #

    def observe(self, q: np.ndarray) -> JointObservation:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.num_joints,):
            raise ValueError(f"Expected {self.num_joints} joint values, got {q.shape}")
        self.sim.data.qpos[self._qpos_index] = q
        self.sim.data.qvel[self._qvel_index] = 0.0
        self.sim.forward()
        # robosuite caches observables between steps; without force_update every
        # capture returns the previous render and the Jacobian estimates as zero.
        obs = self._inner._get_observations(force_update=True)
        image = _views_from_obs(obs, self.view_size)[2]
        tokens = np.asarray(
            _encode_view_features(self.encoder, image, self.view_size, self.views),
            dtype=np.float32,
        )
        if tokens.ndim == 3:
            tokens = tokens[0]
        self.captures += 1
        return JointObservation(
            q=q.copy(),
            image=image,
            tokens=tokens,
            eef_pos=np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy(),
            eef_quat_xyzw=np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy(),
        )


def _views_from_obs(obs: dict[str, np.ndarray], view_size: int):
    # Imported lazily so this module can be read without the sampling helpers.
    from sample_endpoint_trajectories import _views_from_obs as impl

    return impl(obs, view_size)


# --------------------------------------------------------------------------- #
# latent objective (spec 7.1 / 7.2)
# --------------------------------------------------------------------------- #


def latent_loss_terms(
    tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
    *,
    cosine_weight: float = 1.0,
    l2_weight: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token cosine distance and raw latent MSE, both differentiable.

    ``tokens``/``goal_tokens`` are ``[..., P, C]``.  The cosine term keeps
    spatial correspondence -- it averages channel cosine distance independently
    at every token -- so it matches ``token_cosine`` in the endpoint runs and the
    two experiments report the same number.
    """
    normalized = torch.nn.functional.normalize(tokens, dim=-1, eps=eps)
    goal_normalized = torch.nn.functional.normalize(goal_tokens, dim=-1, eps=eps)
    similarity = (normalized * goal_normalized).sum(dim=-1).clamp(-1.0, 1.0)
    loss_cos = 1.0 - similarity.mean(dim=-1)
    loss_l2 = ((tokens - goal_tokens) ** 2).mean(dim=(-1, -2))
    return cosine_weight * loss_cos + l2_weight * loss_l2, loss_cos, loss_l2


def latent_loss_and_grad(
    tokens: np.ndarray,
    goal_tokens: np.ndarray,
    *,
    device: torch.device,
    cosine_weight: float = 1.0,
    l2_weight: float = 0.0,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """``L(z, z_goal)`` and ``dL/dz`` as a flat ``d``-vector.

    The encoder is frozen but the current descriptor must stay outside
    ``no_grad`` or ``dL/dz`` is destroyed (spec 5.4).  Here the descriptor is
    already detached numerics, so the leaf is created explicitly.
    """
    current = torch.as_tensor(
        np.asarray(tokens, dtype=np.float32), device=device
    ).requires_grad_(True)
    goal = torch.as_tensor(np.asarray(goal_tokens, dtype=np.float32), device=device)
    if current.shape != goal.shape:
        raise ValueError(f"Descriptor shape mismatch: {current.shape} != {goal.shape}")
    loss, loss_cos, loss_l2 = latent_loss_terms(
        current, goal, cosine_weight=cosine_weight, l2_weight=l2_weight, eps=eps
    )
    (gradient,) = torch.autograd.grad(loss, current)
    return (
        float(loss.detach()),
        gradient.detach().reshape(-1).cpu().numpy().astype(np.float64),
        {
            "loss": float(loss.detach()),
            "cosine": float(loss_cos.detach()),
            "l2": float(loss_l2.detach()),
        },
    )


# --------------------------------------------------------------------------- #
# Jacobian estimation (spec 10.3 / 10.5 / 10.6)
# --------------------------------------------------------------------------- #


@dataclass
class JacobianColumnRecord:
    joint: int
    span_rad: float
    scheme: str  # "central" | "one_sided_low" | "one_sided_high" | "skipped"
    column_norm: float
    note: str = ""


@dataclass
class JacobianRefreshMonitor:
    """Trips a refresh when the local linear model stops predicting (spec 10.6)."""

    max_relative_prediction_error: float = 0.5
    patience: int = 3
    refresh_every_steps: int = 25
    consecutive_bad: int = 0
    steps_since_refresh: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def update(self, relative_error: float | None) -> tuple[bool, str]:
        self.steps_since_refresh += 1
        if relative_error is None or not np.isfinite(relative_error):
            self.consecutive_bad = 0
        elif relative_error > self.max_relative_prediction_error:
            self.consecutive_bad += 1
        else:
            self.consecutive_bad = 0
        if self.consecutive_bad >= self.patience:
            return True, "prediction_error"
        if self.steps_since_refresh >= self.refresh_every_steps:
            return True, "scheduled"
        return False, ""

    def mark_refreshed(self) -> None:
        self.consecutive_bad = 0
        self.steps_since_refresh = 0


class LatentJacobianEstimator:
    """Central-difference initialization plus rank-one Broyden maintenance."""

    def __init__(
        self,
        observer: JointLatentObserver,
        *,
        delta_rad: float = 0.005,
        broyden_eps: float = 1e-8,
    ) -> None:
        self.observer = observer
        self.delta_rad = float(delta_rad)
        self.broyden_eps = float(broyden_eps)
        self.matrix: np.ndarray | None = None
        self.num_central_difference_refreshes = 0
        self.num_broyden_updates = 0

    # -- initialization ---------------------------------------------------- #

    def central_difference(
        self, q: np.ndarray, *, joints: list[int] | None = None
    ) -> tuple[np.ndarray, list[JacobianColumnRecord]]:
        """``J[:, j] = (z(q + d e_j) - z(q - d e_j)) / (2 d)``, limit-aware.

        A perturbation that would leave the safe box is replaced by a one-sided
        difference and logged; a joint with no room at all keeps whatever column
        it already had rather than silently estimating zero.
        """
        q = np.asarray(q, dtype=np.float64)
        observer = self.observer
        columns: list[np.ndarray | None] = []
        records: list[JacobianColumnRecord] = []
        target_joints = list(range(observer.num_joints)) if joints is None else joints
        if joints is not None and self.matrix is None:
            raise ValueError("Partial refresh needs an existing Jacobian to patch")

        width = None if self.matrix is None else int(self.matrix.shape[0])
        for joint in range(observer.num_joints):
            if joint not in target_joints:
                columns.append(self.matrix[:, joint].copy())  # type: ignore[index]
                records.append(
                    JacobianColumnRecord(joint, 0.0, "skipped", 0.0, "not requested")
                )
                continue

            low = observer.joint_limits[joint, 0]
            high = observer.joint_limits[joint, 1]
            plus_q = q.copy()
            minus_q = q.copy()
            plus_q[joint] = min(q[joint] + self.delta_rad, high)
            minus_q[joint] = max(q[joint] - self.delta_rad, low)
            span = float(plus_q[joint] - minus_q[joint])
            if span <= 0.0:
                # Keep a previously estimated column rather than replacing a real
                # sensitivity with a zero the optimizer would read as "this joint
                # does nothing".
                columns.append(
                    self.matrix[:, joint].copy() if self.matrix is not None else None
                )
                records.append(
                    JacobianColumnRecord(
                        joint, 0.0, "skipped", 0.0, "no safe travel at this joint"
                    )
                )
                continue

            scheme = "central"
            if plus_q[joint] < q[joint] + self.delta_rad:
                scheme = "one_sided_low"
            elif minus_q[joint] > q[joint] - self.delta_rad:
                scheme = "one_sided_high"

            plus = observer.observe(plus_q).descriptor
            minus = observer.observe(minus_q).descriptor
            column = (plus.astype(np.float64) - minus.astype(np.float64)) / span
            columns.append(column)
            width = column.size
            records.append(
                JacobianColumnRecord(
                    joint,
                    span,
                    scheme,
                    float(np.linalg.norm(column)),
                    "" if scheme == "central" else "clipped by joint limit margin",
                )
            )

        if width is None:
            raise RuntimeError(
                "No joint had safe travel for a finite-difference probe; "
                "the start configuration is on every limit at once"
            )
        matrix = np.stack(
            [np.zeros(width, dtype=np.float64) if c is None else c for c in columns],
            axis=1,
        )
        self.matrix = matrix
        self.num_central_difference_refreshes += 1
        # Leave the arm exactly where the caller had it (spec 10.3).
        observer.observe(q)
        return matrix, records

    # -- maintenance ------------------------------------------------------- #

    def prediction_error(
        self, delta_q: np.ndarray, delta_z: np.ndarray
    ) -> float | None:
        """``r_t = ||dz - J dq|| / (||dz|| + eps)`` (spec 10.6)."""
        if self.matrix is None:
            return None
        residual = np.asarray(delta_z, dtype=np.float64) - self.matrix @ np.asarray(
            delta_q, dtype=np.float64
        )
        denominator = float(np.linalg.norm(delta_z)) + self.broyden_eps
        if denominator <= self.broyden_eps:
            return None
        return float(np.linalg.norm(residual) / denominator)

    def broyden_update(self, delta_q: np.ndarray, delta_z: np.ndarray) -> bool:
        """``J <- J + ((dz - J dq) dq^T) / (dq^T dq + eps)`` (spec 10.5)."""
        if self.matrix is None:
            return False
        delta_q = np.asarray(delta_q, dtype=np.float64)
        delta_z = np.asarray(delta_z, dtype=np.float64)
        denominator = float(delta_q @ delta_q)
        if denominator <= self.broyden_eps:
            # A stationary step carries no information about any column.
            return False
        residual = delta_z - self.matrix @ delta_q
        self.matrix = self.matrix + np.outer(residual, delta_q) / (
            denominator + self.broyden_eps
        )
        self.num_broyden_updates += 1
        return True

    # -- diagnostics ------------------------------------------------------- #

    def condition_number(self) -> float:
        if self.matrix is None:
            return float("nan")
        singular = np.linalg.svd(self.matrix, compute_uv=False)
        smallest = float(singular[-1])
        if smallest <= 0.0:
            return float("inf")
        return float(singular[0] / smallest)

    def summary(self) -> dict[str, Any]:
        if self.matrix is None:
            return {"available": False}
        singular = np.linalg.svd(self.matrix, compute_uv=False)
        return {
            "available": True,
            "shape": list(self.matrix.shape),
            "frobenius_norm": float(np.linalg.norm(self.matrix)),
            "column_norms": np.linalg.norm(self.matrix, axis=0).tolist(),
            "singular_values": singular.tolist(),
            "condition_number": self.condition_number(),
            "central_difference_refreshes": self.num_central_difference_refreshes,
            "broyden_updates": self.num_broyden_updates,
        }
