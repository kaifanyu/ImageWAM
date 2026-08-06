#!/usr/bin/env python
"""Hybrid SVGD with learned differentiable terminal-feature dynamics.

This experiment replaces the six MuJoCo finite-difference probes used for each
particle by a differentiable surrogate:

    terminal target theta -> learned terminal feature -> distance to goal feature

The surrogate is trained from terminal images and then differentiated with
``torch.autograd``.  MuJoCo is still used once per particle and population for
ground-truth scoring, best-particle selection, and online surrogate refinement.
It is never used to estimate a gradient.

The model is deliberately scoped to the controlled endpoint experiment: the
start snapshot, controller, camera setup, and action horizon are fixed, so its
learned map is F(theta) -> projected terminal image feature.  Use an
action-conditioned recurrent model instead when any of those inputs vary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _path in (_HERE, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sample_endpoint_trajectories import (  # noqa: E402
    _rollout_to_target,
    _synchronize_controllers_to_sim_state,
    _views_from_obs,
    _write_video,
)
from svgd_endpoint import (  # noqa: E402
    DinoV3FeatureMetric,
    EndpointEnergy,
    FluxAutoencoderMetric,
    _cap_updates,
    _encode_view_features,
    _finite_difference_grad,
    _goal_axis_diagnostics,
    _mean_goal_projection,
    _optimizer_latent_metrics,
    _particle_motion_plot,
    _progress_plot,
    _start_cloud,
    _svgd_step,
    _view_latent,
)

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class CapturingEncoder:
    """Delegate to an existing encoder while retaining its latest features."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.last_full: np.ndarray | None = None
        self.last_selected: np.ndarray | None = None
        if callable(getattr(base, "encode_views", None)):
            # Only expose encode_views when the wrapped encoder implements it.
            self.encode_views = self._encode_views  # type: ignore[method-assign]

    @property
    def provenance(self) -> dict[str, Any]:
        return self.base.provenance

    def encode(self, image: Image.Image) -> np.ndarray:
        features = np.asarray(self.base.encode(image), dtype=np.float32)
        self.last_full = features
        self.last_selected = None
        return features

    def _encode_views(
        self, image: Image.Image, view_size: int, views: str
    ) -> np.ndarray:
        features = np.asarray(
            self.base.encode_views(image, view_size, views), dtype=np.float32
        )
        self.last_full = None
        self.last_selected = features
        return features

    def captured_features(self, view_size: int, views: str) -> np.ndarray:
        if self.last_selected is not None:
            return self.last_selected.copy()
        if self.last_full is None:
            raise RuntimeError("The encoder has not produced features yet")
        return _view_latent(self.last_full, view_size, views).copy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


class CountSketchProjector:
    """Memory-efficient fixed projection of image features to a compact vector."""

    def __init__(
        self,
        feature_shape: tuple[int, ...],
        output_dim: int,
        metric: str,
        seed: int,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("Projection dimension must be positive")
        self.feature_shape = tuple(int(value) for value in feature_shape)
        self.input_dim = int(np.prod(self.feature_shape))
        self.output_dim = int(output_dim)
        self.metric = str(metric)
        rng = np.random.default_rng(seed)
        self.buckets = rng.integers(
            0, self.output_dim, size=self.input_dim, dtype=np.int32
        )
        self.signs = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32), size=self.input_dim
        )

    def _canonical_vector(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim > len(self.feature_shape) and array.shape[0] == 1:
            array = array[0]
        if tuple(array.shape) != self.feature_shape:
            raise ValueError(
                f"Feature shape {tuple(array.shape)} != expected {self.feature_shape}"
            )
        if self.metric == "token_cosine":
            if array.ndim < 2:
                raise ValueError("token_cosine needs tokens with a channel dimension")
            norms = np.linalg.norm(array, axis=-1, keepdims=True)
            array = array / np.maximum(norms, 1e-12)
            vector = array.reshape(-1) / np.sqrt(float(array.shape[-2]))
        elif self.metric == "cosine":
            vector = array.reshape(-1)
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        else:
            vector = array.reshape(-1) / np.sqrt(float(array.size))
        return vector.astype(np.float32, copy=False)

    def transform(self, features: np.ndarray) -> np.ndarray:
        vector = self._canonical_vector(features)
        projected = np.bincount(
            self.buckets,
            weights=(self.signs * vector).astype(np.float64),
            minlength=self.output_dim,
        ).astype(np.float32)
        if self.metric in {"cosine", "token_cosine"}:
            projected /= max(float(np.linalg.norm(projected)), 1e-12)
        return projected

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "count_sketch",
            "feature_shape": list(self.feature_shape),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "metric": self.metric,
        }


class TerminalFeatureDynamics(nn.Module):
    """Smooth fixed-start dynamics map from terminal EEF target to features."""

    def __init__(
        self,
        bounds: np.ndarray,
        output_dim: int,
        hidden_dim: int,
        feature_scale: float,
    ) -> None:
        super().__init__()
        bounds_tensor = torch.as_tensor(bounds, dtype=torch.float32)
        self.register_buffer("theta_center", bounds_tensor.mean(dim=1))
        self.register_buffer(
            "theta_scale", (bounds_tensor[:, 1] - bounds_tensor[:, 0]) / 2.0
        )
        self.register_buffer(
            "feature_scale", torch.tensor(max(float(feature_scale), 1e-6))
        )
        input_dim = 21
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _basis(self, theta: torch.Tensor) -> torch.Tensor:
        x = (theta - self.theta_center) / self.theta_scale.clamp_min(1e-6)
        return torch.cat(
            [
                x,
                x.square(),
                x.pow(3),
                torch.sin(torch.pi * x),
                torch.cos(torch.pi * x),
                torch.sin(2.0 * torch.pi * x),
                torch.cos(2.0 * torch.pi * x),
            ],
            dim=-1,
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        normalized_prediction = self.network(self._basis(theta))
        return normalized_prediction * self.feature_scale


def _projected_energy(
    predicted: torch.Tensor,
    goal: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    if goal.ndim == 1:
        goal = goal.unsqueeze(0)
    if goal.shape[0] == 1 and predicted.shape[0] != 1:
        goal = goal.expand(predicted.shape[0], -1)
    if metric in {"cosine", "token_cosine"}:
        return 1.0 - F.cosine_similarity(predicted, goal, dim=-1, eps=1e-8)
    return torch.linalg.vector_norm(predicted - goal, dim=-1)


@dataclass
class SurrogateDataset:
    theta: np.ndarray
    terminal_eef: np.ndarray
    feature: np.ndarray
    energy: np.ndarray

    def append(
        self,
        theta: np.ndarray,
        terminal_eef: np.ndarray,
        feature: np.ndarray,
        energy: np.ndarray,
    ) -> None:
        self.theta = np.concatenate([self.theta, np.asarray(theta)], axis=0)
        self.terminal_eef = np.concatenate(
            [self.terminal_eef, np.asarray(terminal_eef)], axis=0
        )
        self.feature = np.concatenate([self.feature, np.asarray(feature)], axis=0)
        self.energy = np.concatenate([self.energy, np.asarray(energy)], axis=0)

    def subset(self, indices: np.ndarray) -> "SurrogateDataset":
        return SurrogateDataset(
            theta=self.theta[indices],
            terminal_eef=self.terminal_eef[indices],
            feature=self.feature[indices],
            energy=self.energy[indices],
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            theta=self.theta.astype(np.float32),
            terminal_eef=self.terminal_eef.astype(np.float32),
            feature=self.feature.astype(np.float32),
            energy=self.energy.astype(np.float32),
        )


def _empty_dataset(feature_dim: int) -> SurrogateDataset:
    return SurrogateDataset(
        theta=np.empty((0, 3), dtype=np.float32),
        terminal_eef=np.empty((0, 3), dtype=np.float32),
        feature=np.empty((0, feature_dim), dtype=np.float32),
        energy=np.empty((0,), dtype=np.float32),
    )


def _latin_hypercube(
    count: int, bounds: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    unit = np.empty((count, 3), dtype=np.float64)
    for dimension in range(3):
        unit[:, dimension] = (
            rng.permutation(count) + rng.uniform(size=count)
        ) / float(count)
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def _features_and_energy(
    encoder: Any,
    image: Image.Image,
    goal_latent: np.ndarray,
    projector: CountSketchProjector,
    view_size: int,
    views: str,
    metric: str,
) -> tuple[np.ndarray, float]:
    latent = _encode_view_features(encoder, image, view_size, views)
    energy = _optimizer_latent_metrics(latent, goal_latent)[metric]
    return projector.transform(latent), float(energy)


def _load_bootstrap_history(
    history_path: Path,
    encoder: Any,
    goal_latent: np.ndarray,
    projector: CountSketchProjector,
    view_size: int,
    views: str,
    metric: str,
    max_samples: int,
) -> SurrogateDataset:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    records = payload.get("history", [])
    candidates: list[tuple[np.ndarray, np.ndarray, Path]] = []
    for record in records:
        particles = record.get("particles_before_update", record.get("particles"))
        terminal_eefs = record.get("terminal_eefs")
        if particles is None or terminal_eefs is None:
            continue
        iteration = int(record["iteration"])
        for index, (theta, terminal_eef) in enumerate(
            zip(particles, terminal_eefs, strict=True)
        ):
            image_path = history_path.parent / f"iter_{iteration:03d}" / f"particle_{index:02d}.png"
            if image_path.is_file():
                candidates.append(
                    (
                        np.asarray(theta, dtype=np.float32),
                        np.asarray(terminal_eef, dtype=np.float32),
                        image_path,
                    )
                )
    if not candidates:
        raise RuntimeError(
            f"No saved particle images referenced by bootstrap history {history_path}"
        )
    if max_samples > 0 and len(candidates) > max_samples:
        indices = np.linspace(0, len(candidates) - 1, max_samples).round().astype(int)
        candidates = [candidates[index] for index in indices]

    dataset = _empty_dataset(projector.output_dim)
    print(f"[bootstrap] encoding {len(candidates)} saved terminal images", flush=True)
    for index, (theta, terminal_eef, image_path) in enumerate(candidates):
        with Image.open(image_path) as image:
            feature, energy = _features_and_energy(
                encoder,
                image.convert("RGB"),
                goal_latent,
                projector,
                view_size,
                views,
                metric,
            )
        dataset.append(
            theta[None], terminal_eef[None], feature[None], np.asarray([energy])
        )
        if (index + 1) % 25 == 0 or index + 1 == len(candidates):
            print(f"[bootstrap] encoded {index + 1}/{len(candidates)}", flush=True)
    return dataset


def _collect_surrogate_data(
    count: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
    energy_fn: EndpointEnergy,
    encoder: CapturingEncoder,
    projector: CountSketchProjector,
    view_size: int,
    views: str,
) -> SurrogateDataset:
    points = _latin_hypercube(count, bounds, rng)
    dataset = _empty_dataset(projector.output_dim)
    print(f"[collect] running {count} fixed-start terminal rollouts", flush=True)
    for index, theta in enumerate(points):
        energy, _, terminal_eef, _, _ = energy_fn(theta)
        latent = encoder.captured_features(view_size, views)
        feature = projector.transform(latent)
        dataset.append(
            theta[None].astype(np.float32),
            terminal_eef[None].astype(np.float32),
            feature[None],
            np.asarray([energy], dtype=np.float32),
        )
        if (index + 1) % 10 == 0 or index + 1 == count:
            print(f"[collect] rollout {index + 1}/{count}", flush=True)
    return dataset


def _fit_surrogate(
    model: TerminalFeatureDynamics,
    optimizer: torch.optim.Optimizer,
    train_data: SurrogateDataset,
    validation_data: SurrogateDataset,
    goal_feature: torch.Tensor,
    metric: str,
    steps: int,
    batch_size: int,
    energy_weight: float,
    device: torch.device,
    label: str,
) -> dict[str, float]:
    model.train()
    theta = torch.as_tensor(train_data.theta, device=device, dtype=torch.float32)
    feature = torch.as_tensor(train_data.feature, device=device, dtype=torch.float32)
    energy = torch.as_tensor(train_data.energy, device=device, dtype=torch.float32)
    effective_batch = min(int(batch_size), len(train_data.theta))
    if effective_batch <= 0:
        raise ValueError("Cannot train a surrogate with an empty dataset")

    for _ in range(steps):
        indices = torch.randint(0, len(theta), (effective_batch,), device=device)
        predicted = model(theta[indices])
        feature_loss = F.mse_loss(
            predicted / model.feature_scale,
            feature[indices] / model.feature_scale,
        )
        predicted_energy = _projected_energy(predicted, goal_feature, metric)
        energy_loss = F.mse_loss(predicted_energy, energy[indices])
        loss = feature_loss + float(energy_weight) * energy_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()

    def metrics(data: SurrogateDataset) -> tuple[float, float, float]:
        if len(data.theta) == 0:
            return float("nan"), float("nan"), float("nan")
        with torch.no_grad():
            data_theta = torch.as_tensor(data.theta, device=device, dtype=torch.float32)
            data_feature = torch.as_tensor(data.feature, device=device, dtype=torch.float32)
            data_energy = torch.as_tensor(data.energy, device=device, dtype=torch.float32)
            predicted = model(data_theta)
            feature_rmse = torch.sqrt(
                F.mse_loss(
                    predicted / model.feature_scale,
                    data_feature / model.feature_scale,
                )
            )
            predicted_energy = _projected_energy(predicted, goal_feature, metric)
            energy_rmse = torch.sqrt(F.mse_loss(predicted_energy, data_energy))
            if len(data.theta) > 1:
                centered_x = predicted_energy - predicted_energy.mean()
                centered_y = data_energy - data_energy.mean()
                correlation = (
                    (centered_x * centered_y).sum()
                    / (
                        torch.linalg.vector_norm(centered_x)
                        * torch.linalg.vector_norm(centered_y)
                    ).clamp_min(1e-12)
                )
            else:
                correlation = torch.tensor(0.0, device=device)
        return (
            float(feature_rmse.cpu()),
            float(energy_rmse.cpu()),
            float(correlation.cpu()),
        )

    train_feature_rmse, train_energy_rmse, train_correlation = metrics(train_data)
    val_feature_rmse, val_energy_rmse, val_correlation = metrics(validation_data)
    result = {
        "train_feature_rmse": train_feature_rmse,
        "train_energy_rmse": train_energy_rmse,
        "train_energy_correlation": train_correlation,
        "validation_feature_rmse": val_feature_rmse,
        "validation_energy_rmse": val_energy_rmse,
        "validation_energy_correlation": val_correlation,
    }
    print(
        f"[surrogate:{label}] train_energy_rmse={train_energy_rmse:.6f} "
        f"val_energy_rmse={val_energy_rmse:.6f} val_corr={val_correlation:.4f}",
        flush=True,
    )
    return result


def _surrogate_energy_and_gradient(
    model: TerminalFeatureDynamics,
    particles: np.ndarray,
    goal_feature: torch.Tensor,
    metric: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    theta = torch.tensor(
        particles, device=device, dtype=torch.float32, requires_grad=True
    )
    predicted_features = model(theta)
    energies = _projected_energy(predicted_features, goal_feature, metric)
    gradient = torch.autograd.grad(energies.sum(), theta, create_graph=False)[0]
    return (
        energies.detach().cpu().numpy().astype(np.float64),
        gradient.detach().cpu().numpy().astype(np.float64),
    )


def _gradient_audit(
    model: TerminalFeatureDynamics,
    validation_data: SurrogateDataset,
    goal_feature: torch.Tensor,
    metric: str,
    energy_fn: EndpointEnergy,
    bounds: np.ndarray,
    epsilon: np.ndarray,
    sample_count: int,
    device: torch.device,
) -> dict[str, Any]:
    """Compare held-out surrogate derivatives with simulator finite differences."""
    count = min(int(sample_count), len(validation_data.theta))
    if count <= 0:
        return {"samples": 0, "finite_difference_rollouts": 0}
    indices = np.linspace(0, len(validation_data.theta) - 1, count).round().astype(int)
    points = validation_data.theta[indices].astype(np.float64)
    _, surrogate_gradients = _surrogate_energy_and_gradient(
        model, points, goal_feature, metric, device
    )
    true_gradients = np.zeros_like(points)
    rollouts_before = energy_fn.rollouts
    for index, point in enumerate(points):
        true_gradients[index], _ = _finite_difference_grad(
            energy_fn,
            point,
            epsilon,
            bounds,
            iteration=-1,
            particle=index,
        )
    finite_difference_rollouts = int(energy_fn.rollouts - rollouts_before)
    true_norms = np.linalg.norm(true_gradients, axis=1)
    surrogate_norms = np.linalg.norm(surrogate_gradients, axis=1)
    usable = (true_norms > 1e-8) & (surrogate_norms > 1e-8)
    cosines = np.full(count, np.nan, dtype=np.float64)
    cosines[usable] = np.sum(
        true_gradients[usable] * surrogate_gradients[usable], axis=1
    ) / (true_norms[usable] * surrogate_norms[usable])
    usable_cosines = cosines[usable]
    return {
        "samples": count,
        "usable_samples": int(usable.sum()),
        "finite_difference_rollouts": finite_difference_rollouts,
        "fd_epsilon_m": epsilon.tolist(),
        "mean_cosine": float(usable_cosines.mean()) if usable_cosines.size else 0.0,
        "median_cosine": (
            float(np.median(usable_cosines)) if usable_cosines.size else 0.0
        ),
        "positive_cosine_fraction": (
            float(np.mean(usable_cosines > 0.0)) if usable_cosines.size else 0.0
        ),
        "surrogate_to_true_norm_ratio": float(
            surrogate_norms.mean() / max(float(true_norms.mean()), 1e-12)
        ),
        "cosines": [float(value) if np.isfinite(value) else None for value in cosines],
        "true_gradient_norms": true_norms.tolist(),
        "surrogate_gradient_norms": surrogate_norms.tolist(),
        "points": points.tolist(),
        "true_gradients": true_gradients.tolist(),
        "surrogate_gradients": surrogate_gradients.tolist(),
    }


def _split_dataset(
    dataset: SurrogateDataset,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[SurrogateDataset, SurrogateDataset]:
    count = len(dataset.theta)
    if count < 4:
        raise ValueError("At least four bootstrap samples are required")
    validation_count = max(1, int(round(count * validation_fraction)))
    validation_count = min(validation_count, count - 2)
    permutation = rng.permutation(count)
    return (
        dataset.subset(permutation[validation_count:]),
        dataset.subset(permutation[:validation_count]),
    )


def _initialize_particles(
    mode: str,
    start: np.ndarray,
    radius: np.ndarray,
    count: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "start-cloud":
        return _start_cloud(start, radius, count, bounds).astype(np.float64)
    if mode == "random-start-cloud":
        offsets = rng.uniform(-radius, radius, size=(count, 3))
        return np.clip(start[None] + offsets, bounds[:, 0], bounds[:, 1])
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, 3))


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
    parser.add_argument(
        "--dino-model", default="vit_base_patch16_dinov3.lvd1689m"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--particles", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--init-mode",
        choices=["uniform", "start-cloud", "random-start-cloud"],
        default="start-cloud",
    )
    parser.add_argument(
        "--init-radius", type=float, nargs=3, default=[0.005, 0.005, 0.003]
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        default=[-0.06, -0.04, -0.32, 0.32, 1.02, 1.045],
    )
    parser.add_argument(
        "--latent-views", choices=["both", "agentview", "wrist"], default="agentview"
    )
    parser.add_argument(
        "--latent-distance",
        choices=["rms", "cosine", "token_cosine"],
        default="token_cosine",
    )
    parser.add_argument("--transport", choices=["svgd", "particle_gd"], default="svgd")
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--repulsion-weight", type=float, default=0.01)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--max-update-norm", type=float, default=0.02)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--move-steps", type=int, default=40)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--controller-gain", type=float, default=15.0)
    parser.add_argument("--fixed-arc-height", type=float, default=0.0)
    parser.add_argument("--fixed-midpoint-x", type=float, default=0.0)
    parser.add_argument("--repeatability-particles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-all-particles", action="store_true")
    parser.add_argument("--verbose-evaluations", action="store_true")
    parser.add_argument("--best-video-stride", type=int, default=2)
    parser.add_argument("--best-video-fps", type=int, default=12)

    bootstrap = parser.add_argument_group("surrogate bootstrap")
    bootstrap.add_argument(
        "--bootstrap-history",
        help="Existing history.json with saved particle images; avoids bootstrap rollouts.",
    )
    bootstrap.add_argument("--bootstrap-max-samples", type=int, default=0)
    bootstrap.add_argument("--surrogate-samples", type=int, default=192)
    bootstrap.add_argument("--validation-fraction", type=float, default=0.10)
    bootstrap.add_argument("--projection-dim", type=int, default=256)
    bootstrap.add_argument("--projection-seed", type=int, default=1729)

    training = parser.add_argument_group("surrogate training")
    training.add_argument("--surrogate-hidden-dim", type=int, default=256)
    training.add_argument("--surrogate-train-steps", type=int, default=2500)
    training.add_argument("--online-train-steps", type=int, default=100)
    training.add_argument("--surrogate-batch-size", type=int, default=128)
    training.add_argument("--surrogate-lr", type=float, default=1e-3)
    training.add_argument("--surrogate-weight-decay", type=float, default=1e-5)
    training.add_argument("--surrogate-energy-weight", type=float, default=10.0)
    audit = parser.add_argument_group("surrogate gradient audit")
    audit.add_argument("--gradient-audit-samples", type=int, default=0)
    audit.add_argument(
        "--gradient-audit-fd-eps", type=float, nargs=3, default=[0.01, 0.04, 0.01]
    )
    audit.add_argument(
        "--minimum-gradient-audit-cosine",
        type=float,
        default=-1.0,
        help="Abort before optimization when the held-out mean cosine is lower.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.feature_encoder == "flux_ae" and not args.editor_ae:
        parser.error("--editor-ae is required with --feature-encoder flux_ae")
    if args.particles < 2 or args.iterations <= 0:
        parser.error("--particles must be at least 2 and --iterations must be positive")
    if args.surrogate_samples < 4 and not args.bootstrap_history:
        parser.error("--surrogate-samples must be at least 4 without bootstrap history")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must lie between 0 and 0.5")
    if args.transport == "particle_gd" and args.repulsion_weight != 0.0:
        parser.error("--transport particle_gd requires --repulsion-weight 0")
    if args.temperature <= 0.0 or args.step_size <= 0.0:
        parser.error("--temperature and --step-size must be positive")
    if args.gradient_audit_samples < 0:
        parser.error("--gradient-audit-samples must be non-negative")
    if np.any(np.asarray(args.gradient_audit_fd_eps) <= 0.0):
        parser.error("--gradient-audit-fd-eps values must be positive")
    if not -1.0 <= args.minimum_gradient_audit_cosine <= 1.0:
        parser.error("--minimum-gradient-audit-cosine must lie in [-1, 1]")

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

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bounds = np.asarray(args.bounds, dtype=np.float64).reshape(3, 2)
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        parser.error("Each bounds pair must have min < max")
    init_radius = np.asarray(args.init_radius, dtype=np.float64)
    actual_start_eef = np.asarray(manifest["actual_start_eef"], dtype=np.float64)
    physical_goal = np.asarray(manifest["physical_goal_eef"], dtype=np.float64)
    view_size = int(manifest["view_size"])
    start_state = np.load(run_dir / "start_state.npy")
    start_gripper_actions = [
        np.asarray(action, dtype=np.float64)
        for action in manifest["start_gripper_controller_actions"]
    ]
    if np.any((actual_start_eef < bounds[:, 0]) | (actual_start_eef > bounds[:, 1])):
        parser.error("The actual start EEF lies outside --bounds")

    if args.feature_encoder == "flux_ae":
        base_encoder = FluxAutoencoderMetric(
            Path(args.editor_ae).resolve(), Path(args.flux2_src).resolve(), args.device
        )
    else:
        base_encoder = DinoV3FeatureMetric(args.dino_model, args.device)
    encoder = CapturingEncoder(base_encoder)
    device = base_encoder.device

    goal_image = Image.open(goal_path).convert("RGB")
    goal_image.save(out_dir / "goal_reference.png")
    goal_latent = _encode_view_features(
        encoder, goal_image, view_size, args.latent_views
    )
    feature_shape = tuple(int(value) for value in goal_latent.shape[1:])
    projector = CountSketchProjector(
        feature_shape,
        args.projection_dim,
        args.latent_distance,
        args.projection_seed,
    )
    goal_projected_np = projector.transform(goal_latent)
    goal_projected = torch.as_tensor(
        goal_projected_np, device=device, dtype=torch.float32
    )
    np.save(out_dir / "goal_projected_feature.npy", goal_projected_np)
    np.save(out_dir / "projection_buckets.npy", projector.buckets)
    np.save(out_dir / "projection_signs.npy", projector.signs)
    print(
        f"[goal] encoder={args.feature_encoder} latent_shape={goal_latent.shape} "
        f"projected_dim={args.projection_dim}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    particles = _initialize_particles(
        args.init_mode,
        actual_start_eef,
        init_radius,
        args.particles,
        bounds,
        rng,
    )
    initial_particles = particles.copy()

    env = OffScreenRenderEnv(
        bddl_file_name=str(manifest["bddl"]),
        camera_heights=int(manifest["render_size"]),
        camera_widths=int(manifest["render_size"]),
    )
    history: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None
    try:
        env.seed(int(manifest["sim_seed"]))
        energy_fn = EndpointEnergy(
            env,
            start_state,
            start_gripper_actions,
            encoder,
            goal_latent,
            move_steps=args.move_steps,
            settle_steps=args.settle_steps,
            gain=args.controller_gain,
            view_size=view_size,
            fixed_arc_height=args.fixed_arc_height,
            fixed_midpoint_x=args.fixed_midpoint_x,
            views=args.latent_views,
            distance_metric=args.latent_distance,
            trace_root=out_dir,
            trace_mode="base",
            verbose_evaluations=args.verbose_evaluations,
        )

        if args.bootstrap_history:
            all_bootstrap = _load_bootstrap_history(
                Path(args.bootstrap_history).resolve(),
                encoder,
                goal_latent,
                projector,
                view_size,
                args.latent_views,
                args.latent_distance,
                args.bootstrap_max_samples,
            )
            bootstrap_source = str(Path(args.bootstrap_history).resolve())
            bootstrap_rollouts = 0
        else:
            all_bootstrap = _collect_surrogate_data(
                args.surrogate_samples,
                bounds,
                rng,
                energy_fn,
                encoder,
                projector,
                view_size,
                args.latent_views,
            )
            bootstrap_source = "fresh_fixed_start_rollouts"
            bootstrap_rollouts = args.surrogate_samples

        train_data, validation_data = _split_dataset(
            all_bootstrap, args.validation_fraction, rng
        )
        all_bootstrap.save(out_dir / "surrogate_bootstrap_dataset.npz")
        feature_scale = float(np.sqrt(np.mean(np.square(train_data.feature))))
        model = TerminalFeatureDynamics(
            bounds,
            args.projection_dim,
            args.surrogate_hidden_dim,
            feature_scale,
        ).to(device)
        model_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.surrogate_lr,
            weight_decay=args.surrogate_weight_decay,
        )
        training_history: list[dict[str, Any]] = []
        initial_fit = _fit_surrogate(
            model,
            model_optimizer,
            train_data,
            validation_data,
            goal_projected,
            args.latent_distance,
            args.surrogate_train_steps,
            args.surrogate_batch_size,
            args.surrogate_energy_weight,
            device,
            "initial",
        )
        training_history.append({"iteration": -1, **initial_fit})
        gradient_audit = _gradient_audit(
            model,
            validation_data,
            goal_projected,
            args.latent_distance,
            energy_fn,
            bounds,
            np.asarray(args.gradient_audit_fd_eps, dtype=np.float64),
            args.gradient_audit_samples,
            device,
        )
        _write_json(out_dir / "gradient_audit.json", gradient_audit)
        audit_rollouts = int(gradient_audit["finite_difference_rollouts"])
        if gradient_audit["samples"]:
            print(
                "[gradient-audit] "
                f"mean_cosine={gradient_audit['mean_cosine']:.4f} "
                f"median_cosine={gradient_audit['median_cosine']:.4f} "
                f"norm_ratio={gradient_audit['surrogate_to_true_norm_ratio']:.4f} "
                f"rollouts={audit_rollouts}",
                flush=True,
            )
            if (
                gradient_audit["mean_cosine"]
                < args.minimum_gradient_audit_cosine
            ):
                raise RuntimeError(
                    "Surrogate gradient audit failed: mean cosine "
                    f"{gradient_audit['mean_cosine']:.4f} < required "
                    f"{args.minimum_gradient_audit_cosine:.4f}. See "
                    f"{out_dir / 'gradient_audit.json'}"
                )

        repeat_count = min(args.repeatability_particles, args.particles)

        def evaluate_population(
            evaluated_particles: np.ndarray, iteration: int
        ) -> dict[str, Any]:
            nonlocal global_best
            iteration_dir = out_dir / f"iter_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            energies = np.zeros(args.particles, dtype=np.float64)
            terminal_eefs = np.zeros_like(evaluated_particles)
            features = np.zeros(
                (args.particles, args.projection_dim), dtype=np.float32
            )
            best_index = 0
            for index, theta in enumerate(evaluated_particles):
                energy, image, terminal_eef, _, trace_file = energy_fn(
                    theta,
                    trace_context={
                        "iteration": iteration,
                        "particle": index,
                        "evaluation": "base",
                    },
                )
                energies[index] = energy
                terminal_eefs[index] = terminal_eef
                features[index] = projector.transform(
                    encoder.captured_features(view_size, args.latent_views)
                )
                if args.save_all_particles:
                    image.save(iteration_dir / f"particle_{index:02d}.png")
                if energy < energies[best_index] or index == 0:
                    best_index = index
                if global_best is None or energy < float(global_best["energy"]):
                    global_best = {
                        "energy": float(energy),
                        "iteration": int(iteration),
                        "particle": int(index),
                        "target_eef": np.asarray(theta, dtype=np.float64).copy(),
                        "evaluated_terminal_eef": terminal_eef.copy(),
                        "trace_file": trace_file,
                    }
                if index < repeat_count:
                    energy_fn(
                        theta,
                        trace_context={
                            "iteration": iteration,
                            "particle": index,
                            "evaluation": "base_repeat",
                        },
                    )
            if args.save_all_particles:
                source = iteration_dir / f"particle_{best_index:02d}.png"
                if source.is_file():
                    with Image.open(source) as image:
                        image.save(iteration_dir / "best.png")
            return {
                "energies": energies,
                "terminal_eefs": terminal_eefs,
                "features": features,
                "best_index": best_index,
            }

        def base_record(
            evaluation: dict[str, Any], evaluated_particles: np.ndarray, iteration: int
        ) -> dict[str, Any]:
            energies = evaluation["energies"]
            terminal_eefs = evaluation["terminal_eefs"]
            goal_errors = np.linalg.norm(terminal_eefs - physical_goal[None], axis=1)
            return {
                "iteration": int(iteration),
                "objective": args.latent_distance,
                "particles_before_update": evaluated_particles.tolist(),
                "energies": energies.tolist(),
                "energy_min": float(energies.min()),
                "energy_mean": float(energies.mean()),
                "energy_max": float(energies.max()),
                "goal_errors_m": goal_errors.tolist(),
                "goal_error_min_m": float(goal_errors.min()),
                "goal_error_mean_m": float(goal_errors.mean()),
                "terminal_eefs": terminal_eefs.tolist(),
                "target_diagnostics": _goal_axis_diagnostics(
                    evaluated_particles, actual_start_eef, physical_goal
                ),
                "terminal_diagnostics": _goal_axis_diagnostics(
                    terminal_eefs, actual_start_eef, physical_goal
                ),
                "best_particle": int(evaluation["best_index"]),
            }

        def write_history() -> None:
            _write_json(
                out_dir / "history.json",
                {
                    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "run_dir": str(run_dir),
                    "goal_path": str(goal_path),
                    "config": vars(args),
                    "gradient_source": "learned_terminal_feature_dynamics_autograd",
                    "bootstrap_source": bootstrap_source,
                    "projector": projector.metadata(),
                    "gradient_audit": gradient_audit,
                    "actual_start_eef": actual_start_eef.tolist(),
                    "diagnostic_goal_eef": physical_goal.tolist(),
                    "diagnostic_goal_is_optimizer_input": False,
                    "initial_particles": initial_particles.tolist(),
                    "surrogate_training": training_history,
                    "history": history,
                },
            )

        print(
            f"[plan] {args.iterations} updates, {args.particles} real base rollouts/update, "
            f"zero finite-difference optimizer rollouts, {audit_rollouts} audit rollouts",
            flush=True,
        )
        for iteration in range(args.iterations):
            evaluated_particles = particles.copy()
            evaluation = evaluate_population(evaluated_particles, iteration)
            train_data.append(
                evaluated_particles.astype(np.float32),
                evaluation["terminal_eefs"].astype(np.float32),
                evaluation["features"].astype(np.float32),
                evaluation["energies"].astype(np.float32),
            )
            online_fit = _fit_surrogate(
                model,
                model_optimizer,
                train_data,
                validation_data,
                goal_projected,
                args.latent_distance,
                args.online_train_steps,
                args.surrogate_batch_size,
                args.surrogate_energy_weight,
                device,
                f"iter_{iteration:03d}",
            )
            training_history.append({"iteration": iteration, **online_fit})
            surrogate_energies, energy_gradients = _surrogate_energy_and_gradient(
                model,
                evaluated_particles,
                goal_projected,
                args.latent_distance,
                device,
            )
            scores = -energy_gradients / args.temperature
            if args.transport == "particle_gd":
                latent_direction = scores.copy()
                repulsion_direction = np.zeros_like(scores)
                direction = args.latent_weight * latent_direction
                bandwidth = None
            else:
                direction, bandwidth, latent_direction, repulsion_direction, _ = _svgd_step(
                    evaluated_particles,
                    scores,
                    args.bandwidth_scale,
                    latent_weight=args.latent_weight,
                    repulsion_weight=args.repulsion_weight,
                )
            raw_latent_update = args.step_size * args.latent_weight * latent_direction
            raw_repulsion_update = args.step_size * args.repulsion_weight * repulsion_direction
            raw_update = args.step_size * direction
            capped_update, trust_scales = _cap_updates(raw_update, args.max_update_norm)
            latent_update = raw_latent_update * trust_scales[:, None]
            repulsion_update = raw_repulsion_update * trust_scales[:, None]
            unclipped = evaluated_particles + capped_update
            particles = np.clip(unclipped, bounds[:, 0], bounds[:, 1])
            applied_update = particles - evaluated_particles

            record = base_record(evaluation, evaluated_particles, iteration)
            record.update(
                {
                    "phase": "update",
                    "transport": args.transport,
                    "gradient_source": "surrogate_autograd",
                    "update_applied": True,
                    "gradients_computed": True,
                    "surrogate_energies": surrogate_energies.tolist(),
                    "surrogate_population_energy_rmse": float(
                        np.sqrt(np.mean(np.square(surrogate_energies - evaluation["energies"])))
                    ),
                    "energy_gradients": energy_gradients.tolist(),
                    "scores": scores.tolist(),
                    "score_norms": np.linalg.norm(scores, axis=1).tolist(),
                    "energy_gradient_norms": np.linalg.norm(
                        energy_gradients, axis=1
                    ).tolist(),
                    "kernel_bandwidth": float(bandwidth) if bandwidth is not None else None,
                    "latent_directions": latent_direction.tolist(),
                    "repulsion_directions": repulsion_direction.tolist(),
                    "total_directions": direction.tolist(),
                    "raw_updates": raw_update.tolist(),
                    "trust_region_scales": trust_scales.tolist(),
                    "latent_updates": latent_update.tolist(),
                    "repulsion_updates": repulsion_update.tolist(),
                    "applied_updates": applied_update.tolist(),
                    "bounds_clipped": (unclipped != particles).tolist(),
                    "latent_update_goal_projection_m": _mean_goal_projection(
                        latent_update, actual_start_eef, physical_goal
                    ),
                    "repulsion_update_goal_projection_m": _mean_goal_projection(
                        repulsion_update, actual_start_eef, physical_goal
                    ),
                    "applied_update_goal_projection_m": _mean_goal_projection(
                        applied_update, actual_start_eef, physical_goal
                    ),
                    "particles_after_update": particles.tolist(),
                }
            )
            history.append(record)
            write_history()
            print(
                f"[iter {iteration:03d}] true_E_mean={record['energy_mean']:.6f} "
                f"surrogate_rmse={record['surrogate_population_energy_rmse']:.6f} "
                f"grad_norm={np.mean(np.linalg.norm(energy_gradients, axis=1)):.5f} "
                f"rollouts={energy_fn.rollouts}",
                flush=True,
            )

        final_evaluation = evaluate_population(particles.copy(), args.iterations)
        final_record = base_record(final_evaluation, particles.copy(), args.iterations)
        zeros = np.zeros_like(particles)
        final_record.update(
            {
                "phase": "final_evaluation",
                "transport": args.transport,
                "gradient_source": None,
                "update_applied": False,
                "gradients_computed": False,
                "energy_gradients": zeros.tolist(),
                "scores": zeros.tolist(),
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
                "particles_after_update": particles.tolist(),
            }
        )
        history.append(final_record)
        write_history()

        train_data.save(out_dir / "surrogate_final_training_dataset.npz")
        validation_data.save(out_dir / "surrogate_validation_dataset.npz")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "bounds": bounds.astype(np.float32),
                "goal_projected_feature": goal_projected_np,
                "projection_buckets": projector.buckets,
                "projection_signs": projector.signs,
                "projector": projector.metadata(),
                "config": vars(args),
            },
            out_dir / "surrogate_model.pt",
        )

        if global_best is None:
            raise RuntimeError("No particle was evaluated")
        env.reset()
        obs = env.set_init_state(start_state)
        _synchronize_controllers_to_sim_state(env, start_gripper_actions)
        obs, best_actions, best_eef_path, best_frames = _rollout_to_target(
            env,
            obs,
            np.asarray(global_best["target_eef"], dtype=np.float64),
            move_steps=args.move_steps,
            settle_steps=args.settle_steps,
            gain=args.controller_gain,
            arc_height=args.fixed_arc_height,
            midpoint_x=args.fixed_midpoint_x,
            capture_video=True,
            video_stride=args.best_video_stride,
            view_size=view_size,
        )
        best_main, best_wrist, best_terminal = _views_from_obs(obs, view_size)
        best_latent = _encode_view_features(
            encoder, best_terminal, view_size, args.latent_views
        )
        best_metrics = _optimizer_latent_metrics(best_latent, goal_latent)
        best_actual_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        best_main.save(out_dir / "best_terminal_agentview.png")
        best_wrist.save(out_dir / "best_terminal_wrist.png")
        best_terminal.save(out_dir / "best_terminal.png")
        np.save(out_dir / "best_actions.npy", best_actions.astype(np.float32))
        np.save(out_dir / "best_eef_path.npy", best_eef_path.astype(np.float32))
        np.save(out_dir / "best_terminal_latent.npy", best_latent.astype(np.float32))
        np.save(
            out_dir / "best_terminal_state.npy",
            np.asarray(env.get_sim_state(), dtype=np.float64),
        )
        _write_video(out_dir / "best_rollout.mp4", best_frames, args.best_video_fps)
        _write_json(
            out_dir / "best_metadata.json",
            {
                "goal": {
                    "path": str(goal_path),
                    "feature_encoder": args.feature_encoder,
                    "encoder_provenance": encoder.provenance,
                },
                "optimization": {
                    "gradient_source": "learned_terminal_feature_dynamics_autograd",
                    "bootstrap_source": bootstrap_source,
                    "bootstrap_rollouts": bootstrap_rollouts,
                    "gradient_audit": gradient_audit,
                    "gradient_audit_rollouts": audit_rollouts,
                    "online_ground_truth_rollouts": int(
                        energy_fn.rollouts - bootstrap_rollouts - audit_rollouts
                    ),
                    "finite_difference_rollouts": audit_rollouts,
                    "optimization_finite_difference_rollouts": 0,
                    "iterations": args.iterations,
                    "particles": args.particles,
                    "latent_distance": args.latent_distance,
                    "latent_views": args.latent_views,
                    "transport": args.transport,
                },
                "selection": {
                    "energy": float(global_best["energy"]),
                    "iteration": int(global_best["iteration"]),
                    "particle": int(global_best["particle"]),
                    "target_eef": np.asarray(global_best["target_eef"]).tolist(),
                    "evaluated_terminal_eef": np.asarray(
                        global_best["evaluated_terminal_eef"]
                    ).tolist(),
                },
                "replay": {
                    "objective_energy": float(best_metrics[args.latent_distance]),
                    "latent_metrics": best_metrics,
                    "actual_terminal_eef": best_actual_eef.tolist(),
                    "physical_goal_error_m": float(
                        np.linalg.norm(best_actual_eef - physical_goal)
                    ),
                    "target_tracking_error_m": float(
                        np.linalg.norm(
                            best_actual_eef - np.asarray(global_best["target_eef"])
                        )
                    ),
                },
            },
        )
    finally:
        env.close()

    _progress_plot(history, out_dir / "progress.png")
    _particle_motion_plot(
        history, out_dir / "particle_motion.png", actual_start_eef, physical_goal
    )
    print(f"[done] output: {out_dir}", flush=True)
    print(
        f"[done] finite-difference rollouts: {audit_rollouts} (gradient audit only)",
        flush=True,
    )
    print(f"[done] best rollout: {out_dir / 'best_rollout.mp4'}", flush=True)
    (out_dir / "complete.status").write_text(
        dt.datetime.now(dt.timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
