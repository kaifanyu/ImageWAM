#!/usr/bin/env python
"""Rank simulator terminal images against one edited endpoint image.

Pixel scores are always available.  If the FLUX.2 autoencoder weights are
provided, this also computes deterministic editor-VAE distances.  A captured
final denoising latent from ``scripts/run_image_edit.py --latent-output`` can be
used as the direct image-editor latent target.

The scorer intentionally treats dynamics-VAE features as an explicit adapter:
pass a goal ``.npy`` and place one same-shaped feature file in every candidate
directory.  That avoids silently substituting a different representation when
the intended "Dyno" / dynamics encoder is not known.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from imagewam.utils.video_metrics import (  # noqa: E402
    pil_frames_to_video_tensor,
    video_ssim,
)


# FLUX.2 AE maps a HxW image to (H/16)x(W/16) latent positions with 128 channels.
LATENT_DOWNSAMPLE = 16


def _read_image(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        raise ValueError(
            f"Image size mismatch for {path}: got {image.size}, expected {size}. "
            "Evaluation never resizes terminal or goal images."
        )
    return image


def _normalized_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_metadata_hash(
    metadata: dict[str, Any],
    key: str,
    path: Path,
    *,
    required: bool,
) -> None:
    expected = metadata.get(key)
    if expected is None:
        if required:
            raise ValueError(f"Editor metadata is missing required hash field: {key}")
        return
    actual = _sha256(path)
    if not isinstance(expected, str) or expected.lower() != actual:
        raise ValueError(
            f"Editor artifact hash mismatch for {path}: metadata {key} does not match"
        )


def _change_mask(
    start: Image.Image,
    goal: Image.Image,
    *,
    threshold: int,
    dilation: int,
) -> tuple[np.ndarray, bool]:
    delta = np.max(
        np.abs(
            np.asarray(goal, dtype=np.int16)
            - np.asarray(start, dtype=np.int16)
        ),
        axis=2,
    )
    mask = torch.from_numpy((delta >= threshold).astype(np.float32))[None, None]
    if dilation > 0:
        kernel = 2 * dilation + 1
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
    result = mask[0, 0].numpy() > 0.5
    used_fallback = False
    if int(result.sum()) < 32:
        result[:] = True
        used_fallback = True
    return result, used_fallback


def _safe_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def _pixel_metrics(
    candidate: Image.Image,
    goal: Image.Image,
    mask: np.ndarray,
) -> dict[str, float | None]:
    x = _normalized_array(candidate)
    y = _normalized_array(goal)
    error = x - y
    abs_error = np.abs(error)
    squared_error = error**2
    mse = float(squared_error.mean())
    masked_mse = float(squared_error[mask].mean())
    background = ~mask
    ssim = video_ssim(
        pil_frames_to_video_tensor([candidate]),
        pil_frames_to_video_tensor([goal]),
    )
    return {
        "pixel_mae": float(abs_error.mean()),
        "pixel_mse": mse,
        "pixel_psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
        "pixel_ssim": float(ssim),
        "masked_mae": float(abs_error[mask].mean()),
        "masked_mse": masked_mse,
        "masked_psnr_db": 10.0 * math.log10(1.0 / max(masked_mse, 1e-12)),
        "background_mae": _safe_mean(abs_error[background]),
    }


def _split_two_view_image(image: Image.Image) -> dict[str, Image.Image] | None:
    """Split the expected horizontal [agentview | wrist] square-view composite."""
    if image.width != 2 * image.height:
        return None
    midpoint = image.width // 2
    return {
        "agentview": image.crop((0, 0, midpoint, image.height)),
        "wrist": image.crop((midpoint, 0, image.width, image.height)),
    }


def _per_view_pixel_metrics(
    candidate: Image.Image,
    goal_views: dict[str, Image.Image],
    masks: dict[str, np.ndarray],
) -> dict[str, float | None]:
    candidate_views = _split_two_view_image(candidate)
    if candidate_views is None:
        raise ValueError(
            "Per-view metrics require a horizontal [agentview | wrist] composite "
            f"with width == 2 * height, got {candidate.size}"
        )
    result: dict[str, float | None] = {}
    for view_name in ("agentview", "wrist"):
        metrics = _pixel_metrics(candidate_views[view_name], goal_views[view_name], masks[view_name])
        result.update(
            {
                f"{view_name}_mae": metrics["pixel_mae"],
                f"{view_name}_mse": metrics["pixel_mse"],
                f"{view_name}_psnr_db": metrics["pixel_psnr_db"],
                f"{view_name}_ssim": metrics["pixel_ssim"],
                f"{view_name}_masked_mae": metrics["masked_mae"],
                f"{view_name}_masked_mse": metrics["masked_mse"],
                f"{view_name}_masked_psnr_db": metrics["masked_psnr_db"],
                f"{view_name}_background_mae": metrics["background_mae"],
            }
        )
    return result


def _latent_metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    candidate_array = np.asarray(candidate, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if candidate_array.shape != target_array.shape:
        raise ValueError(
            f"Latent shape mismatch: {candidate_array.shape} != {target_array.shape}"
        )
    if candidate_array.size == 0:
        raise ValueError("Latent arrays must be non-empty")
    if not np.isfinite(candidate_array).all() or not np.isfinite(target_array).all():
        raise ValueError("Latent arrays must contain only finite values")
    x = candidate_array.reshape(-1)
    y = target_array.reshape(-1)
    rms = float(np.sqrt(np.mean((x - y) ** 2)))
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 and y_norm == 0.0:
        cosine = 0.0
    elif x_norm == 0.0 or y_norm == 0.0:
        cosine = 1.0
    else:
        similarity = float(np.dot(x, y) / (x_norm * y_norm))
        cosine = 1.0 - float(np.clip(similarity, -1.0, 1.0))
    return {"rms": rms, "cosine_distance": cosine}


class FluxAutoencoderMetric:
    """Deterministic FLUX.2 AE encoder without loading the 4B transformer."""

    def __init__(self, weights: Path, flux2_src: Path, device: str):
        for path in (flux2_src, flux2_src / "src"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from flux2.autoencoder import AutoEncoder, AutoEncoderParams
        from safetensors.torch import load_file

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
        self.device = torch.device(device)
        # Keep metric precision identical on CPU and CUDA. The 4B transformer is
        # not loaded here, so fp32 AE memory is modest enough for this scorer.
        self.dtype = torch.float32

        with torch.device("meta"):
            autoencoder = AutoEncoder(AutoEncoderParams())
        state = load_file(str(weights), device="cpu")
        autoencoder.load_state_dict(state, strict=True, assign=True)
        del state
        gc.collect()
        self.model = autoencoder.to(device=self.device, dtype=self.dtype).eval()

    @property
    def provenance(self) -> dict[str, str]:
        return {"device": str(self.device), "dtype": str(self.dtype)}

    @torch.inference_mode()
    def encode(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * (2.0 / 255.0) - 1.0
        latent = self.model.encode(tensor.to(device=self.device, dtype=self.dtype))
        # Match Flux2VideoExpert.pack_latents: [B,C,H,W] -> [B,H*W,C].
        tokens = latent.permute(0, 2, 3, 1).reshape(latent.shape[0], -1, latent.shape[1])
        return tokens.detach().float().cpu().numpy()

    @torch.inference_mode()
    def decode(self, tokens: np.ndarray, height: int, width: int) -> Image.Image:
        """Inverse of ``encode``: packed tokens -> PIL image.

        Only used for visual debugging -- it answers "is the latent I am
        optimizing toward actually the image I think it is?".
        """
        array = np.asarray(tokens, dtype=np.float32)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3:
            raise ValueError(f"Expected packed tokens [B,N,C], got shape {array.shape}")
        latent_h = height // LATENT_DOWNSAMPLE
        latent_w = width // LATENT_DOWNSAMPLE
        if array.shape[1] != latent_h * latent_w:
            raise ValueError(
                f"Token count {array.shape[1]} does not match {latent_h}x{latent_w} "
                f"for a {height}x{width} image at downsample {LATENT_DOWNSAMPLE}"
            )
        tensor = torch.from_numpy(array).to(device=self.device, dtype=self.dtype)
        latent = tensor.reshape(1, latent_h, latent_w, -1).permute(0, 3, 1, 2)
        decoded = self.model.decode(latent)[0]
        decoded = decoded.detach().float().clamp(-1.0, 1.0)
        pixels = ((decoded + 1.0) * (255.0 / 2.0)).round().clamp(0, 255).byte()
        return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())


class DinoV3FeatureMetric:
    """Frozen DINOv3 patch-token encoder loaded through timm."""

    def __init__(
        self,
        model_name: str,
        device: str,
        *,
        pretrained: bool = True,
    ) -> None:
        import timm

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
        if not timm.is_model(model_name):
            raise ValueError(f"Unknown timm model: {model_name}")

        self.device = torch.device(device)
        self.dtype = torch.float32
        self.model_name = model_name
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        ).to(device=self.device, dtype=self.dtype).eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.input_height = int(data_config["input_size"][-2])
        self.input_width = int(data_config["input_size"][-1])
        self.mean = torch.tensor(
            data_config["mean"], dtype=self.dtype, device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            data_config["std"], dtype=self.dtype, device=self.device
        ).view(1, 3, 1, 1)
        patch_size = self.model.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "encoder": "dinov3",
            "model": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "input_size": [self.input_height, self.input_width],
            "patch_size": list(self.patch_size),
            "num_prefix_tokens_removed": self.num_prefix_tokens,
        }

    @torch.inference_mode()
    def encode(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize(
            (self.input_width, self.input_height),
            Image.Resampling.BICUBIC,
        )
        array = np.array(resized, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.dtype) / 255.0
        tensor = (tensor - self.mean) / self.std
        tokens = self.model.forward_features(tensor)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError(
                f"DINOv3 forward_features returned unsupported output "
                f"{type(tokens).__name__}"
            )
        tokens = tokens[:, self.num_prefix_tokens :, :]
        expected_tokens = (
            self.input_height // self.patch_size[0]
        ) * (
            self.input_width // self.patch_size[1]
        )
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"DINOv3 returned {tokens.shape[1]} patch tokens, "
                f"expected {expected_tokens}"
            )
        return tokens.detach().float().cpu().numpy()

    def encode_views(
        self,
        image: Image.Image,
        view_size: int,
        views: str,
    ) -> np.ndarray:
        """Encode camera views separately to avoid distorting the joined image."""
        rgb = image.convert("RGB")
        expected_size = (2 * view_size, view_size)
        if rgb.size != expected_size:
            raise ValueError(
                f"DINOv3 expected joined [agentview | wrist] image size "
                f"{expected_size}, got {rgb.size}"
            )
        crops = {
            "agentview": rgb.crop((0, 0, view_size, view_size)),
            "wrist": rgb.crop((view_size, 0, 2 * view_size, view_size)),
        }
        selected = (
            ("agentview", "wrist") if views == "both" else (views,)
        )
        return np.concatenate(
            [self.encode(crops[name]) for name in selected],
            axis=1,
        )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks, including correct handling of exact ties."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Rank values must be one-dimensional, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Rank values must contain only finite numbers")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def _rank_rows(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    field_prefix: str,
) -> None:
    available = [(index, float(row[metric])) for index, row in enumerate(rows) if row.get(metric) is not None]
    if not available:
        return
    ranks = _average_ranks(np.asarray([value for _, value in available]))
    for (index, _), rank in zip(available, ranks):
        rows[index][f"{field_prefix}_{metric}"] = float(rank)


def _spearman(rows: list[dict[str, Any]], metric: str) -> float | None:
    pairs = [
        (float(row[metric]), float(row["goal_error_m"]))
        for row in rows
        if row.get(metric) is not None and row.get("goal_error_m") is not None
    ]
    if len(pairs) < 3:
        return None
    values = np.asarray(pairs, dtype=np.float64)
    metric_rank = _average_ranks(values[:, 0])
    physical_rank = _average_ranks(values[:, 1])
    if float(np.std(metric_rank)) == 0.0 or float(np.std(physical_rank)) == 0.0:
        return None
    correlation = np.corrcoef(metric_rank, physical_rank)[0, 1]
    return float(correlation)


def _label(image: Image.Image, text: str) -> Image.Image:
    header = 30
    result = Image.new("RGB", (image.width, image.height + header), (18, 18, 18))
    result.paste(image, (0, header))
    ImageDraw.Draw(result).text((5, 5), text, fill=(245, 245, 245))
    return result


def _contact_sheet(
    run_dir: Path,
    start: Image.Image,
    goal: Image.Image,
    rows: list[dict[str, Any]],
    metric: str,
    output: Path,
) -> None:
    ordered = [row for row in rows if row.get(metric) is not None]
    ordered.sort(key=lambda row: float(row[metric]))
    strips: list[Image.Image] = []
    for row in ordered:
        candidate = _read_image(run_dir / row["terminal_image"], start.size)
        diff = Image.fromarray(
            np.abs(
                np.asarray(candidate, dtype=np.int16)
                - np.asarray(goal, dtype=np.int16)
            ).clip(0, 255).astype(np.uint8)
        )
        tiles = [
            _label(start, "start"),
            _label(goal, "edited goal"),
            _label(
                candidate,
                f"{row['id']} {row['kind']} | {metric}={float(row[metric]):.4f}",
            ),
            _label(diff, f"abs diff | EE error={float(row['goal_error_m']):.3f}m"),
        ]
        strip = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)))
        x = 0
        for tile in tiles:
            strip.paste(tile, (x, 0))
            x += tile.width
        strips.append(strip)
    if not strips:
        return
    sheet = Image.new("RGB", (max(s.width for s in strips), sum(s.height for s in strips)), (18, 18, 18))
    y = 0
    for strip in strips:
        sheet.paste(strip, (0, y))
        y += strip.height
    sheet.save(output)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row if not isinstance(row[key], (list, dict))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    available = [row for row in rows if row.get(metric) is not None]
    return min(available, key=lambda row: float(row[metric])) if available else None


def _best_by_metric(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for metric in metric_names:
        selected = _best(rows, metric)
        if selected is None:
            continue
        value = float(selected[metric])
        tied = [row for row in rows if row.get(metric) is not None and float(row[metric]) == value]
        result[metric] = {
            "candidate": selected["id"],
            "tied_candidates": [row["id"] for row in tied],
            "tied_candidate_details": [
                {
                    "candidate": row["id"],
                    "kind": row["kind"],
                    "goal_error_m": row.get("goal_error_m"),
                    "physical_success": row.get("physical_success"),
                }
                for row in tied
            ],
            "kind": selected["kind"],
            "value": value,
            "goal_error_m": selected.get("goal_error_m"),
            "physical_success": selected.get("physical_success"),
        }
    return result


def _spearman_by_metric(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, float | None]:
    return {
        metric: _spearman(rows, metric)
        for metric in metric_names
        if _best(rows, metric) is not None
    }


def _path_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    exists = resolved.exists()
    is_file = exists and resolved.is_file()
    stat = resolved.stat() if exists else None
    return {
        "path": str(resolved),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": stat.st_size if is_file and stat is not None else None,
        "mtime_ns": stat.st_mtime_ns if stat is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--goal", help="Edited goal image; defaults to <run-dir>/goal_edit.png.")
    parser.add_argument("--mask-threshold", type=int, default=12, help="RGB level in [0,255].")
    parser.add_argument("--mask-dilation", type=int, default=9, help="Pixels around the edit region.")
    parser.add_argument("--editor-ae", help="FLUX.2 ae.safetensors for deterministic VAE metrics.")
    parser.add_argument("--flux2-src", default=str(REPO_ROOT / "third_party" / "flux2"))
    parser.add_argument(
        "--latent-device",
        default="cpu",
        help="cpu (reproducible default), auto, cuda, or cuda:N",
    )
    parser.add_argument("--goal-editor-latent", help="Final denoising tokens saved by run_image_edit.py.")
    parser.add_argument(
        "--editor-metadata",
        help="Editor provenance JSON; defaults to <goal-stem>_metadata.json when present.",
    )
    parser.add_argument(
        "--goal-dynamics-latent",
        help="Optional .npy from the intended dynamics/Dyno encoder.",
    )
    parser.add_argument(
        "--dynamics-metadata",
        help=(
            "Encoder provenance/hash JSON; defaults beside the goal latent "
            "as <stem-without-_latent>_metadata.json."
        ),
    )
    parser.add_argument("--candidate-dynamics-filename", default="dynamics_latent.npy")
    args = parser.parse_args()
    if not 0 <= args.mask_threshold <= 255:
        parser.error("--mask-threshold must be in [0, 255]")
    if args.mask_dilation < 0:
        parser.error("--mask-dilation must be non-negative")

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        parser.error(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    start_path = run_dir / manifest.get("start_image", "start.png")
    goal_path = Path(args.goal).resolve() if args.goal else run_dir / "goal_edit.png"
    if not goal_path.exists():
        parser.error(f"Goal image not found: {goal_path}")
    start = _read_image(start_path)
    goal = _read_image(goal_path, start.size)
    mask, mask_fallback = _change_mask(
        start,
        goal,
        threshold=args.mask_threshold,
        dilation=args.mask_dilation,
    )
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(run_dir / "edit_mask.png")

    start_views = _split_two_view_image(start)
    goal_views = _split_two_view_image(goal)
    view_masks: dict[str, np.ndarray] | None = None
    view_mask_fallbacks: dict[str, bool] = {}
    if start_views is not None and goal_views is not None:
        view_masks = {}
        for view_name in ("agentview", "wrist"):
            view_mask, view_fallback = _change_mask(
                start_views[view_name],
                goal_views[view_name],
                threshold=args.mask_threshold,
                dilation=args.mask_dilation,
            )
            view_masks[view_name] = view_mask
            view_mask_fallbacks[view_name] = view_fallback
            Image.fromarray(view_mask.astype(np.uint8) * 255, mode="L").save(
                run_dir / f"edit_mask_{view_name}.png"
            )

    default_editor_metadata_path = goal_path.with_name(goal_path.stem + "_metadata.json")
    editor_metadata_path = (
        Path(args.editor_metadata).resolve()
        if args.editor_metadata
        else default_editor_metadata_path.resolve()
    )
    if args.editor_metadata and not editor_metadata_path.exists():
        parser.error(f"Editor metadata not found: {editor_metadata_path}")
    editor_metadata = None
    if editor_metadata_path.exists():
        loaded_editor_metadata = json.loads(editor_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_editor_metadata, dict):
            parser.error("Editor metadata JSON must contain an object")
        editor_metadata = loaded_editor_metadata
        try:
            _verify_metadata_hash(
                editor_metadata,
                "output_image_sha256",
                goal_path,
                required=False,
            )
        except ValueError as error:
            parser.error(str(error))

    flux_metric: FluxAutoencoderMetric | None = None
    goal_flux_tokens: np.ndarray | None = None
    goal_editor_tokens: np.ndarray | None = None
    ae_path: Path | None = None
    goal_editor_latent_path: Path | None = None
    if args.editor_ae:
        ae_path = Path(args.editor_ae).resolve()
        if not ae_path.exists():
            parser.error(f"FLUX.2 AE weights not found: {ae_path}")
        flux_metric = FluxAutoencoderMetric(
            ae_path,
            Path(args.flux2_src).resolve(),
            args.latent_device,
        )
        goal_flux_tokens = flux_metric.encode(goal)
        if args.goal_editor_latent:
            goal_editor_latent_path = Path(args.goal_editor_latent).resolve()
            if not goal_editor_latent_path.exists():
                parser.error(f"Goal editor latent not found: {goal_editor_latent_path}")
            if editor_metadata is None:
                parser.error(
                    "--goal-editor-latent requires editor metadata so the goal image "
                    "and latent can be verified as one edit run"
                )
            try:
                _verify_metadata_hash(
                    editor_metadata,
                    "output_image_sha256",
                    goal_path,
                    required=True,
                )
                _verify_metadata_hash(
                    editor_metadata,
                    "final_latent_sha256",
                    goal_editor_latent_path,
                    required=True,
                )
            except ValueError as error:
                parser.error(str(error))
            goal_editor_tokens = np.load(goal_editor_latent_path)
            if goal_editor_tokens.shape != goal_flux_tokens.shape:
                parser.error(
                    "Goal editor latent shape does not match FLUX tokens: "
                    f"{goal_editor_tokens.shape} != {goal_flux_tokens.shape}"
                )
    elif args.goal_editor_latent:
        parser.error("--goal-editor-latent also requires --editor-ae to encode candidates.")

    goal_dynamics: np.ndarray | None = None
    goal_dynamics_path: Path | None = None
    dynamics_metadata_path: Path | None = None
    dynamics_metadata: dict[str, Any] | None = None
    if args.goal_dynamics_latent:
        goal_dynamics_path = Path(args.goal_dynamics_latent).resolve()
        if not goal_dynamics_path.exists():
            parser.error(f"Goal dynamics latent not found: {goal_dynamics_path}")
        goal_dynamics = np.load(goal_dynamics_path)
        stem = goal_dynamics_path.stem
        metadata_stem = stem[: -len("_latent")] if stem.endswith("_latent") else stem
        default_dynamics_metadata = goal_dynamics_path.with_name(
            metadata_stem + "_metadata.json"
        )
        dynamics_metadata_path = (
            Path(args.dynamics_metadata).resolve()
            if args.dynamics_metadata
            else default_dynamics_metadata
        )
        if not dynamics_metadata_path.exists():
            parser.error(f"Dynamics metadata not found: {dynamics_metadata_path}")
        loaded_metadata = json.loads(dynamics_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_metadata, dict):
            parser.error("Dynamics metadata JSON must contain an object")
        dynamics_metadata = loaded_metadata
        required_dynamics_fields = {
            "encoder_name",
            "checkpoint",
            "preprocessing",
            "latent_shape",
            "goal_latent_sha256",
            "candidate_latent_sha256",
        }
        missing_fields = sorted(required_dynamics_fields - dynamics_metadata.keys())
        if missing_fields:
            parser.error(
                "Dynamics metadata is missing required fields: " + ", ".join(missing_fields)
            )
        if list(goal_dynamics.shape) != list(dynamics_metadata["latent_shape"]):
            parser.error(
                "Goal dynamics latent shape does not match metadata: "
                f"{list(goal_dynamics.shape)} != {dynamics_metadata['latent_shape']}"
            )
        expected_goal_hash = dynamics_metadata["goal_latent_sha256"]
        if (
            not isinstance(expected_goal_hash, str)
            or expected_goal_hash.lower() != _sha256(goal_dynamics_path)
        ):
            parser.error("Goal dynamics latent hash does not match dynamics metadata")
        if not isinstance(dynamics_metadata["candidate_latent_sha256"], dict):
            parser.error("candidate_latent_sha256 must be an object keyed by candidate id")
    elif args.dynamics_metadata:
        parser.error("--dynamics-metadata requires --goal-dynamics-latent")

    rows: list[dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        terminal_path = run_dir / candidate["terminal_image"]
        terminal = _read_image(terminal_path, start.size)
        row = {
            "id": candidate["id"],
            "kind": candidate["kind"],
            "trajectory_seed": candidate.get("trajectory_seed"),
            "terminal_image": candidate["terminal_image"],
            "goal_error_m": candidate.get("goal_error_m"),
            "height_error_m": candidate.get("height_error_m"),
            "orientation_error_deg": candidate.get("orientation_error_deg"),
            "gripper_error": candidate.get("gripper_error"),
            "target_tracking_error_m": candidate.get("target_tracking_error_m"),
            "goal_progress": candidate.get("goal_progress"),
            "physical_success": candidate.get("physical_success"),
            "libero_builtin_success": candidate.get("libero_builtin_success"),
            "reset_state_max_abs": candidate.get("reset_state_max_abs"),
            "reset_rgb_max_abs": candidate.get("reset_rgb_max_abs"),
            "reset_rgb_mae": candidate.get("reset_rgb_mae"),
            "reset_rgb_fraction_gt8": candidate.get("reset_rgb_fraction_gt8"),
        }
        row.update(_pixel_metrics(terminal, goal, mask))
        if goal_views is not None and view_masks is not None:
            row.update(_per_view_pixel_metrics(terminal, goal_views, view_masks))

        if flux_metric is not None and goal_flux_tokens is not None:
            candidate_tokens = flux_metric.encode(terminal)
            vae_metrics = _latent_metrics(candidate_tokens, goal_flux_tokens)
            row["flux_vae_rms"] = vae_metrics["rms"]
            row["flux_vae_cosine"] = vae_metrics["cosine_distance"]
            if goal_editor_tokens is not None:
                editor_metrics = _latent_metrics(candidate_tokens, goal_editor_tokens)
                row["editor_final_rms"] = editor_metrics["rms"]
                row["editor_final_cosine"] = editor_metrics["cosine_distance"]

        if goal_dynamics is not None:
            candidate_feature = terminal_path.parent / args.candidate_dynamics_filename
            if not candidate_feature.exists():
                parser.error(f"Missing candidate dynamics latent: {candidate_feature}")
            assert dynamics_metadata is not None
            expected_candidate_hash = dynamics_metadata["candidate_latent_sha256"].get(
                candidate["id"]
            )
            if (
                not isinstance(expected_candidate_hash, str)
                or expected_candidate_hash.lower() != _sha256(candidate_feature)
            ):
                parser.error(
                    f"Dynamics latent hash is missing or mismatched for {candidate['id']}"
                )
            dynamics_metrics = _latent_metrics(np.load(candidate_feature), goal_dynamics)
            row["dynamics_vae_rms"] = dynamics_metrics["rms"]
            row["dynamics_vae_cosine"] = dynamics_metrics["cosine_distance"]
        rows.append(row)

    metric_names = [
        "agentview_masked_mae",
        "wrist_masked_mae",
        "masked_mae",
        "pixel_mse",
        "flux_vae_rms",
        "flux_vae_cosine",
        "editor_final_rms",
        "editor_final_cosine",
        "dynamics_vae_rms",
        "dynamics_vae_cosine",
    ]
    if not rows:
        parser.error("Manifest contains no candidates to score.")

    sampled_rows = [row for row in rows if row["kind"] == "sampled"]
    for metric in metric_names:
        _rank_rows(rows, metric, field_prefix="rank_all")
        _rank_rows(sampled_rows, metric, field_prefix="rank_sampled")

    best_by_metric_all = _best_by_metric(rows, metric_names)
    best_by_metric_sampled = _best_by_metric(sampled_rows, metric_names)
    spearman_all = _spearman_by_metric(rows, metric_names)
    spearman_sampled = _spearman_by_metric(sampled_rows, metric_names)

    per_view_summary: dict[str, Any] = {"enabled": view_masks is not None}
    if view_masks is not None:
        for view_name in ("agentview", "wrist"):
            per_view_summary[view_name] = {
                "mask_pixels": int(view_masks[view_name].sum()),
                "mask_fraction": float(view_masks[view_name].mean()),
                "mask_fallback_to_full_frame": view_mask_fallbacks[view_name],
                "mask_image": str((run_dir / f"edit_mask_{view_name}.png").resolve()),
            }

    scorer_provenance = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "mask_threshold_rgb": args.mask_threshold,
        "mask_dilation_pixels": args.mask_dilation,
        "latent_device_requested": args.latent_device,
        "flux_autoencoder_runtime": flux_metric.provenance if flux_metric is not None else None,
        "candidate_dynamics_filename": args.candidate_dynamics_filename,
        "files": {
            "manifest": _path_provenance(manifest_path),
            "start_image": _path_provenance(start_path),
            "goal_image": _path_provenance(goal_path),
            "flux_autoencoder": _path_provenance(ae_path),
            "flux2_source": _path_provenance(Path(args.flux2_src)),
            "goal_editor_latent": _path_provenance(goal_editor_latent_path),
            "goal_dynamics_latent": _path_provenance(goal_dynamics_path),
            "dynamics_metadata": _path_provenance(dynamics_metadata_path),
        },
        "dynamics_encoder_metadata": dynamics_metadata,
        "latents": {
            "goal_editor_shape": (
                list(goal_editor_tokens.shape) if goal_editor_tokens is not None else None
            ),
            "goal_editor_dtype": (
                str(goal_editor_tokens.dtype) if goal_editor_tokens is not None else None
            ),
            "goal_dynamics_shape": (
                list(goal_dynamics.shape) if goal_dynamics is not None else None
            ),
            "goal_dynamics_dtype": (
                str(goal_dynamics.dtype) if goal_dynamics is not None else None
            ),
            "goal_flux_vae_shape": (
                list(goal_flux_tokens.shape) if goal_flux_tokens is not None else None
            ),
            "goal_flux_vae_dtype": (
                str(goal_flux_tokens.dtype) if goal_flux_tokens is not None else None
            ),
        },
    }
    summary: dict[str, Any] = {
        "schema_version": 2,
        "goal_image": str(goal_path),
        "mask_pixels": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
        "mask_fallback_to_full_frame": mask_fallback,
        "per_view_pixel_diagnostics": per_view_summary,
        "num_candidates": len(rows),
        "best_of_all_physical_success": any(bool(row["physical_success"]) for row in rows),
        "minimum_all_goal_error_m": min(float(row["goal_error_m"]) for row in rows),
        "num_sampled_candidates": len(sampled_rows),
        "best_of_sampled_physical_success": (
            any(bool(row["physical_success"]) for row in sampled_rows) if sampled_rows else None
        ),
        "minimum_sampled_goal_error_m": (
            min(float(row["goal_error_m"]) for row in sampled_rows) if sampled_rows else None
        ),
        "median_sampled_goal_error_m": (
            float(np.median([float(row["goal_error_m"]) for row in sampled_rows]))
            if sampled_rows
            else None
        ),
        "best_by_metric": {
            "all_candidates": best_by_metric_all,
            "sampled_only": best_by_metric_sampled,
        },
        "spearman_with_physical_error": {
            "all_candidates": spearman_all,
            "sampled_only": spearman_sampled,
        },
        "provenance": {
            "editor_metadata_file": _path_provenance(editor_metadata_path),
            "editor": editor_metadata,
            "scorer": scorer_provenance,
        },
        "rows": rows,
    }

    if goal_flux_tokens is not None and goal_editor_tokens is not None:
        summary["goal_decode_reencode"] = _latent_metrics(goal_flux_tokens, goal_editor_tokens)

    oracle_path = run_dir / manifest.get("goal_oracle_image", "goal_oracle.png")
    if oracle_path.exists():
        oracle = _read_image(oracle_path, start.size)
        summary["edited_goal_vs_oracle"] = _pixel_metrics(goal, oracle, mask)
        summary["start_vs_oracle"] = _pixel_metrics(start, oracle, mask)

    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(run_dir / "metrics.csv", rows)
    _contact_sheet(run_dir, start, goal, rows, "masked_mae", run_dir / "contact_sheet_pixel.png")
    for metric, filename in [
        ("flux_vae_rms", "contact_sheet_flux_vae.png"),
        ("editor_final_rms", "contact_sheet_editor_final.png"),
        ("dynamics_vae_rms", "contact_sheet_dynamics_vae.png"),
    ]:
        if _best(rows, metric) is not None:
            _contact_sheet(run_dir, start, goal, rows, metric, run_dir / filename)

    lines = [
        "# Endpoint baseline summary",
        "",
        f"Goal: `{goal_path}`",
        f"Candidates: {len(rows)}",
        f"Best of all candidates (including oracle control): {summary['best_of_all_physical_success']}",
        f"Minimum all-candidate physical error: {summary['minimum_all_goal_error_m']:.4f} m",
        f"Sampled candidates: {summary['num_sampled_candidates']}",
        f"Best-of-sampled physical success: {summary['best_of_sampled_physical_success']}",
        (
            "Minimum sampled physical error: n/a"
            if summary["minimum_sampled_goal_error_m"] is None
            else f"Minimum sampled physical error: {summary['minimum_sampled_goal_error_m']:.4f} m"
        ),
        (
            "Median sampled physical error: n/a"
            if summary["median_sampled_goal_error_m"] is None
            else f"Median sampled physical error: {summary['median_sampled_goal_error_m']:.4f} m"
        ),
        "",
        "## Metric selection: all candidates",
        "",
        "This table includes diagnostic controls such as the physical oracle.",
        "",
        "| metric | selected (ties shown) | value | EE error (m) | physically successful |",
        "|---|---|---:|---:|---:|",
    ]
    for metric, selected in best_by_metric_all.items():
        selected_ids = ", ".join(selected["tied_candidates"])
        error_text = ", ".join(
            "n/a" if detail["goal_error_m"] is None else f"{float(detail['goal_error_m']):.4f}"
            for detail in selected["tied_candidate_details"]
        )
        success_text = ", ".join(
            str(detail["physical_success"])
            for detail in selected["tied_candidate_details"]
        )
        lines.append(
            f"| {metric} | {selected_ids} | {float(selected['value']):.6f} | "
            f"{error_text} | {success_text} |"
        )
    lines.extend(
        [
            "",
            "## Metric selection: sampled candidates only",
            "",
        ]
    )
    if best_by_metric_sampled:
        lines.extend(
            [
                "| metric | selected (ties shown) | value | EE error (m) | physically successful |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for metric, selected in best_by_metric_sampled.items():
            selected_ids = ", ".join(selected["tied_candidates"])
            error_text = ", ".join(
                "n/a"
                if detail["goal_error_m"] is None
                else f"{float(detail['goal_error_m']):.4f}"
                for detail in selected["tied_candidate_details"]
            )
            success_text = ", ".join(
                str(detail["physical_success"])
                for detail in selected["tied_candidate_details"]
            )
            lines.append(
                f"| {metric} | {selected_ids} | {float(selected['value']):.6f} | "
                f"{error_text} | {success_text} |"
            )
    else:
        lines.append("No sampled candidates were present in the manifest.")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] metrics: {run_dir / 'metrics.json'}")
    for metric, selected in best_by_metric_all.items():
        print(
            f"[all:{metric}] {selected['candidate']} value={float(selected['value']):.6f} "
            f"EE_error={float(selected['goal_error_m']):.4f}m "
            f"success={selected['physical_success']}"
        )
    for metric, selected in best_by_metric_sampled.items():
        print(
            f"[sampled:{metric}] {selected['candidate']} value={float(selected['value']):.6f} "
            f"EE_error={float(selected['goal_error_m']):.4f}m "
            f"success={selected['physical_success']}"
        )


if __name__ == "__main__":
    main()
