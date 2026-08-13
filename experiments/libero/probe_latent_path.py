#!/usr/bin/env python
"""Validate FLUX AE distance along a known simulator start-to-goal path.

For evenly spaced fractions ``t`` from zero to one, this script:

1. restores the exact saved MuJoCo start state,
2. rolls the physical simulator arm to ``start + t * (goal - start)``,
3. saves the actual terminal camera images and controller trajectory,
4. encodes the terminal image with the same FLUX.2 autoencoder as SVGD, and
5. measures latent RMS to the supplied goal image.

The physical goal is used only to construct this diagnostic line scan.  The
result answers whether raw FLUX AE distance is a smooth, correctly ordered
objective along the path we know is correct before testing an optimizer.

When the goal is ``goal_oracle.png``, the script recovers the oracle rollout's
arc/midpoint settings from ``manifest.json``.  The excursion is scaled by the
requested fraction, so the zero-fraction probe remains a true no-op while the
full-fraction probe reproduces the trajectory that created the oracle image.

Example:

    python experiments/libero/probe_latent_path.py \
        --run-dir runs/empty_arm_preview \
        --goal runs/empty_arm_preview/goal_oracle.png \
        --goal-latent-source reencode \
        --editor-ae /path/to/ae.safetensors \
        --device cuda:0 \
        --samples 11 --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sample_endpoint_trajectories import (  # noqa: E402
    _rollout_to_target,
    _synchronize_controllers_to_sim_state,
    _views_from_obs,
    _write_video,
    composed_right_view,
    env_from_manifest,
)
from score_endpoint_candidates import (  # noqa: E402
    FluxAutoencoderMetric,
    _latent_metrics,
    _spearman,
    _verify_metadata_hash,
)



def _goal_latent(
    run_dir: Path,
    goal_path: Path,
    goal_image: Image.Image,
    encoder: FluxAutoencoderMetric,
    source: str,
) -> tuple[np.ndarray, str]:
    if source == "reencode":
        return encoder.encode(goal_image), f"encode({goal_path.name})"
    expected_goal = (run_dir / "goal_edit.png").resolve()
    editor_latent = run_dir / "goal_editor_latent.npy"
    if goal_path.resolve() != expected_goal:
        raise ValueError(
            "--goal-latent-source editor is only valid for the paired "
            f"{expected_goal}; use reencode for {goal_path}."
        )
    if not editor_latent.is_file():
        raise ValueError(f"Editor latent not found: {editor_latent}")
    metadata_path = run_dir / "goal_edit_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(
            "Editor latent requires goal_edit_metadata.json so the image/latent pair "
            "can be verified"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("goal_edit_metadata.json must contain a JSON object")
    _verify_metadata_hash(metadata, "output_image_sha256", goal_path, required=True)
    _verify_metadata_hash(metadata, "final_latent_sha256", editor_latent, required=True)
    tokens = np.load(editor_latent)
    encoded_shape = encoder.encode(goal_image).shape
    if tokens.shape != encoded_shape:
        raise ValueError(
            f"Editor latent shape {tokens.shape} does not match re-encoded goal shape "
            f"{encoded_shape}"
        )
    if not np.isfinite(tokens).all():
        raise ValueError("Editor latent contains non-finite values")
    return tokens, "goal_editor_latent.npy (metadata hashes verified)"


def _change_mask(
    start: Image.Image,
    goal: Image.Image,
    threshold: int,
    dilation: int,
) -> tuple[np.ndarray, bool]:
    if start.size != goal.size:
        raise ValueError(f"Start and goal image sizes differ: {start.size} != {goal.size}")
    start_array = np.asarray(start.convert("RGB"), dtype=np.int16)
    goal_array = np.asarray(goal.convert("RGB"), dtype=np.int16)
    changed = np.max(np.abs(goal_array - start_array), axis=2) >= threshold
    mask_image = Image.fromarray((changed.astype(np.uint8) * 255), mode="L")
    if dilation > 0:
        mask_image = mask_image.filter(ImageFilter.MaxFilter(2 * dilation + 1))
    mask = np.asarray(mask_image) > 0
    used_fallback = int(mask.sum()) < 32
    if used_fallback:
        mask[:] = True
    return mask, used_fallback


def _residual_image(a: Image.Image, b: Image.Image, gain: float = 4.0) -> Image.Image:
    difference = np.abs(
        np.asarray(a.convert("RGB"), dtype=np.float32)
        - np.asarray(b.convert("RGB"), dtype=np.float32)
    )
    return Image.fromarray(np.clip(gain * difference, 0, 255).astype(np.uint8))


def _reconstruction_metrics(
    reference: Image.Image,
    decoded: Image.Image,
) -> dict[str, float]:
    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32)
    decoded_array = np.asarray(decoded.convert("RGB"), dtype=np.float32)
    if reference_array.shape != decoded_array.shape:
        raise ValueError(
            f"Reference/decoded image shapes differ: {reference_array.shape} != "
            f"{decoded_array.shape}"
        )
    error = decoded_array - reference_array
    return {
        "pixel_rms_255": float(np.sqrt(np.mean(error**2))),
        "pixel_mae_255": float(np.mean(np.abs(error))),
        "pixel_max_abs_255": float(np.max(np.abs(error))),
    }


def _optional_float_array(values: list[float | None]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )


def _format_optional(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def _oracle_path_shape(
    manifest: dict[str, Any],
    run_dir: Path,
    goal_path: Path,
    arc_override: float | None,
    midpoint_override: float | None,
) -> tuple[float, float, str]:
    arc = arc_override
    midpoint = midpoint_override
    source = "command_line"
    expected_oracle = (run_dir / manifest.get("goal_oracle_image", "goal_oracle.png")).resolve()
    if goal_path.resolve() == expected_oracle and (arc is None or midpoint is None):
        oracle = next(
            (candidate for candidate in manifest.get("candidates", [])
             if candidate.get("kind") == "oracle"),
            None,
        )
        if oracle is not None:
            if arc is None:
                arc = float(oracle.get("arc_height_m", 0.0))
            if midpoint is None:
                midpoint = float(oracle.get("midpoint_x_m", 0.0))
            source = "manifest_oracle_candidate"
    if arc is None:
        arc = 0.0
    if midpoint is None:
        midpoint = 0.0
    if source == "command_line" and arc_override is None and midpoint_override is None:
        source = "zero_default"
    elif source == "manifest_oracle_candidate" and (
        arc_override is not None or midpoint_override is not None
    ):
        source = "manifest_oracle_with_command_line_override"
    return float(arc), float(midpoint), source


def _pixel_metrics(
    candidate: Image.Image,
    goal: Image.Image,
    change_mask: np.ndarray,
) -> dict[str, float]:
    candidate_array = np.asarray(candidate.convert("RGB"), dtype=np.float32) / 255.0
    goal_array = np.asarray(goal.convert("RGB"), dtype=np.float32) / 255.0
    if candidate_array.shape != goal_array.shape:
        raise ValueError(
            f"Candidate and goal image shapes differ: {candidate_array.shape} != "
            f"{goal_array.shape}"
        )
    error = candidate_array - goal_array
    squared = error**2
    absolute = np.abs(error)
    return {
        "pixel_rms": float(np.sqrt(np.mean(squared))),
        "pixel_mae": float(np.mean(absolute)),
        "masked_pixel_rms": float(np.sqrt(np.mean(squared[change_mask]))),
        "masked_pixel_mae": float(np.mean(absolute[change_mask])),
    }


def _per_view_latent_rms(
    candidate: np.ndarray,
    goal: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, float | None]:
    width, height = image_size
    if width != 2 * height or width % 16 != 0 or height % 16 != 0:
        return {"agentview_latent_rms": None, "wrist_latent_rms": None}
    latent_height, latent_width = height // 16, width // 16
    candidate_array = np.asarray(candidate, dtype=np.float32)
    goal_array = np.asarray(goal, dtype=np.float32)
    if candidate_array.ndim == 2:
        candidate_array = candidate_array[None]
    if goal_array.ndim == 2:
        goal_array = goal_array[None]
    expected_tokens = latent_height * latent_width
    if (
        candidate_array.shape != goal_array.shape
        or candidate_array.ndim != 3
        or candidate_array.shape[1] != expected_tokens
    ):
        return {"agentview_latent_rms": None, "wrist_latent_rms": None}
    candidate_grid = candidate_array.reshape(
        candidate_array.shape[0], latent_height, latent_width, candidate_array.shape[2]
    )
    goal_grid = goal_array.reshape(
        goal_array.shape[0], latent_height, latent_width, goal_array.shape[2]
    )
    midpoint = latent_width // 2
    return {
        "agentview_latent_rms": float(
            np.sqrt(np.mean((candidate_grid[:, :, :midpoint] - goal_grid[:, :, :midpoint]) ** 2))
        ),
        "wrist_latent_rms": float(
            np.sqrt(np.mean((candidate_grid[:, :, midpoint:] - goal_grid[:, :, midpoint:]) ** 2))
        ),
    }


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _group_waypoint(
    index: int,
    fraction: float,
    target: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = (
        "latent_rms",
        "latent_cosine_distance",
        "agentview_latent_rms",
        "wrist_latent_rms",
        "pixel_rms",
        "pixel_mae",
        "masked_pixel_rms",
        "masked_pixel_mae",
        "actual_progress_fraction",
        "goal_error_m",
        "tracking_error_m",
        "off_axis_error_m",
        "reset_state_max_abs",
        "reset_render_mae_255",
        "reset_render_fraction_gt8",
    )
    summary: dict[str, Any] = {
        "index": int(index),
        "requested_fraction": float(fraction),
        "target_eef": target.tolist(),
        "terminal_image": rows[0]["terminal_image"],
        "repeats": rows,
    }
    for metric in metrics:
        if rows[0][metric] is None:
            summary[f"{metric}_mean"] = None
            summary[f"{metric}_std"] = None
        else:
            mean, std = _mean_std(rows, metric)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
    return summary


def _contact_sheet(
    waypoints: list[dict[str, Any]],
    out_dir: Path,
    output: Path,
) -> None:
    columns = min(4, len(waypoints))
    header_height = 38
    tiles: list[Image.Image] = []
    for waypoint in waypoints:
        image = Image.open(out_dir / waypoint["terminal_image"]).convert("RGB")
        tile = Image.new("RGB", (image.width, image.height + header_height), (18, 18, 18))
        tile.paste(image, (0, header_height))
        label = (
            f"t={waypoint['requested_fraction']:.2f}  "
            f"RMS={waypoint['latent_rms_mean']:.4f}±{waypoint['latent_rms_std']:.4f}\n"
            f"actual={waypoint['actual_progress_fraction_mean']:.3f}  "
            f"goal_err={waypoint['goal_error_m_mean']:.3f}m"
        )
        ImageDraw.Draw(tile).multiline_text((5, 3), label, fill=(245, 245, 245), spacing=1)
        tiles.append(tile)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * tiles[0].width, rows * tiles[0].height), (18, 18, 18)
    )
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    sheet.save(output)


def _normalized(values: np.ndarray) -> np.ndarray:
    denominator = float(values[0])
    if abs(denominator) <= 1e-12:
        return values.copy()
    return values / denominator


def _plot(
    waypoints: list[dict[str, Any]],
    summary: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fraction = np.asarray([row["requested_fraction"] for row in waypoints])
    actual = np.asarray([row["actual_progress_fraction_mean"] for row in waypoints])
    goal_error = np.asarray([row["goal_error_m_mean"] for row in waypoints])
    latent = np.asarray([row["latent_rms_mean"] for row in waypoints])
    latent_std = np.asarray([row["latent_rms_std"] for row in waypoints])
    agent = _optional_float_array(
        [row["agentview_latent_rms_mean"] for row in waypoints]
    )
    wrist = _optional_float_array(
        [row["wrist_latent_rms_mean"] for row in waypoints]
    )
    pixel = np.asarray([row["pixel_rms_mean"] for row in waypoints])
    masked = np.asarray([row["masked_pixel_rms_mean"] for row in waypoints])
    tracking = np.asarray([row["tracking_error_m_mean"] for row in waypoints])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle("FLUX latent metric along the known physical path", fontsize=15)

    axis = axes[0, 0]
    axis.errorbar(fraction, latent, yerr=latent_std, marker="o", capsize=3,
                  label="full two-view latent")
    if np.isfinite(agent).any():
        axis.plot(fraction, agent, marker=".", label="agent-view latent")
    if np.isfinite(wrist).any():
        axis.plot(fraction, wrist, marker=".", label="wrist latent")
    axis.set(xlabel="requested path fraction", ylabel="latent RMS to goal",
             title="Should decrease toward the goal re-render floor")
    axis.grid(alpha=0.3)
    axis.legend()

    axis = axes[0, 1]
    axis.errorbar(goal_error, latent, yerr=latent_std, marker="o", capsize=3)
    for x, y, value in zip(goal_error, latent, fraction):
        axis.annotate(f"{value:.1f}", (x, y), xytext=(3, 3),
                      textcoords="offset points", fontsize=7)
    axis.set(
        xlabel="actual physical goal error (m)",
        ylabel="latent RMS to goal",
        title=(
            "Distance alignment: Spearman="
            + _format_optional(summary["spearman_goal_error_vs_latent_rms"], "+.3f")
        ),
    )
    axis.grid(alpha=0.3)

    axis = axes[1, 0]
    axis.plot(fraction, _normalized(latent), marker="o", label="FLUX latent RMS")
    axis.plot(fraction, _normalized(pixel), marker=".", label="full-image pixel RMS")
    axis.plot(fraction, _normalized(masked), marker=".", label="changed-region pixel RMS")
    axis.set(xlabel="requested path fraction", ylabel="distance / distance at start",
             title="Representation comparison")
    axis.grid(alpha=0.3)
    axis.legend()

    axis = axes[1, 1]
    axis.plot(fraction, actual, marker="o", label="actual terminal progress")
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", label="ideal")
    second = axis.twinx()
    second.plot(fraction, 1000.0 * tracking, marker=".", color="tab:red",
                label="tracking error")
    axis.set(xlabel="requested path fraction", ylabel="actual path fraction",
             title="Did the simulator reach each requested point?")
    second.set_ylabel("target tracking error (mm)", color="tab:red")
    axis.grid(alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    handles_2, labels_2 = second.get_legend_handles_labels()
    axis.legend(handles + handles_2, labels + labels_2, loc="upper left")

    figure.savefig(output, dpi=150)
    plt.close(figure)


def _metric_summary(waypoints: list[dict[str, Any]]) -> dict[str, Any]:
    latent = np.asarray([row["latent_rms_mean"] for row in waypoints])
    latent_std = np.asarray([row["latent_rms_std"] for row in waypoints])
    pixel = np.asarray([row["pixel_rms_mean"] for row in waypoints])
    masked = np.asarray([row["masked_pixel_rms_mean"] for row in waypoints])
    spearman_rows = [
        {
            "goal_error_m": row["goal_error_m_mean"],
            "latent_rms": row["latent_rms_mean"],
            "pixel_rms": row["pixel_rms_mean"],
            "masked_pixel_rms": row["masked_pixel_rms_mean"],
        }
        for row in waypoints
    ]
    latent_spearman = _spearman(spearman_rows, "latent_rms")
    pixel_spearman = _spearman(spearman_rows, "pixel_rms")
    masked_spearman = _spearman(spearman_rows, "masked_pixel_rms")
    latent_spearman_without_endpoint = _spearman(
        spearman_rows[:-1], "latent_rms"
    ) if len(spearman_rows) > 3 else None
    monotonic_fraction = float(np.mean(np.diff(latent) <= 0.0))
    monotonic_without_endpoint = float(np.mean(np.diff(latent[:-1]) <= 0.0))
    pixel_monotonic_fraction = float(np.mean(np.diff(pixel) <= 0.0))
    masked_monotonic_fraction = float(np.mean(np.diff(masked) <= 0.0))
    noise = float(np.mean(latent_std))
    adjacent_signal = float(np.median(np.abs(np.diff(latent))))
    signal_to_repeat_noise = adjacent_signal / noise if noise > 1e-12 else None
    assessment_spearman = (
        latent_spearman_without_endpoint
        if latent_spearman_without_endpoint is not None
        else latent_spearman
    )
    assessment_monotonic = monotonic_without_endpoint
    if (
        assessment_spearman is not None
        and assessment_spearman >= 0.9
        and assessment_monotonic >= 0.8
    ):
        verdict = "strongly_aligned"
    elif (
        assessment_spearman is not None
        and assessment_spearman >= 0.7
        and assessment_monotonic >= 0.6
    ):
        verdict = "partially_aligned"
    else:
        verdict = "not_reliably_aligned"
    return {
        "verdict": verdict,
        "spearman_goal_error_vs_latent_rms": latent_spearman,
        "spearman_goal_error_vs_latent_rms_excluding_final_waypoint": (
            latent_spearman_without_endpoint
        ),
        "spearman_goal_error_vs_pixel_rms": pixel_spearman,
        "spearman_goal_error_vs_masked_pixel_rms": masked_spearman,
        "latent_monotonic_nonincreasing_fraction": monotonic_fraction,
        "latent_monotonic_nonincreasing_fraction_excluding_final_waypoint": (
            monotonic_without_endpoint
        ),
        "pixel_monotonic_nonincreasing_fraction": pixel_monotonic_fraction,
        "masked_pixel_monotonic_nonincreasing_fraction": masked_monotonic_fraction,
        "latent_rms_start": float(latent[0]),
        "latent_rms_goal_endpoint": float(latent[-1]),
        "latent_rms_dynamic_range": float(latent[0] - latent[-1]),
        "mean_repeat_latent_rms_std": noise,
        "median_adjacent_latent_rms_change": adjacent_signal,
        "adjacent_signal_to_repeat_noise": signal_to_repeat_noise,
        "first_step_latent_rms_delta": float(latent[1] - latent[0]),
        "largest_upward_latent_rms_reversal": float(max(np.max(np.diff(latent)), 0.0)),
        "minimum_latent_requested_fraction": float(
            waypoints[int(np.argmin(latent))]["requested_fraction"]
        ),
        "goal_endpoint_is_latent_minimum": bool(int(np.argmin(latent)) == len(latent) - 1),
        "maximum_tracking_error_m": float(
            max(row["tracking_error_m_mean"] for row in waypoints)
        ),
        "maximum_off_axis_error_m": float(
            max(row["off_axis_error_m_mean"] for row in waypoints)
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "waypoint_index",
        "requested_fraction",
        "repeat",
        "target_x",
        "target_y",
        "target_z",
        "terminal_x",
        "terminal_y",
        "terminal_z",
        "actual_progress_fraction",
        "goal_error_m",
        "tracking_error_m",
        "off_axis_error_m",
        "reset_state_max_abs",
        "reset_render_mae_255",
        "reset_render_fraction_gt8",
        "rollout_arc_height_m",
        "rollout_midpoint_x_m",
        "latent_rms",
        "latent_cosine_distance",
        "agentview_latent_rms",
        "wrist_latent_rms",
        "pixel_rms",
        "masked_pixel_rms",
        "terminal_image",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True,
                        help="Run bundle containing manifest.json and start_state.npy.")
    parser.add_argument("--out-dir", help="Defaults to <run-dir>/latent_path_probe.")
    parser.add_argument("--goal", help="Defaults to <run-dir>/goal_oracle.png.")
    parser.add_argument("--goal-eef", type=float, nargs=3,
                        help="Physical line-scan endpoint; defaults to manifest physical_goal_eef.")
    parser.add_argument(
        "--arc-height", type=float,
        help=("Arc height for the full-goal rollout. Partial probes scale it by their "
              "path fraction. Defaults to the manifest oracle value for goal_oracle.png, "
              "otherwise zero."),
    )
    parser.add_argument(
        "--midpoint-x", type=float,
        help=("X excursion for the full-goal rollout. Partial probes scale it by their "
              "path fraction. Defaults to the manifest oracle value for goal_oracle.png, "
              "otherwise zero."),
    )
    parser.add_argument("--editor-ae", required=True, help="FLUX.2 ae.safetensors.")
    parser.add_argument("--flux2-src", default=str(REPO_ROOT / "third_party" / "flux2"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--goal-latent-source", choices=["reencode", "editor"],
                        default="reencode")
    parser.add_argument("--samples", type=int, default=11,
                        help="Number of evenly spaced endpoints including start and goal.")
    parser.add_argument(
        "--fraction-start",
        type=float,
        default=0.0,
        help="First global start-to-goal fraction to probe (default: 0).",
    )
    parser.add_argument(
        "--fraction-end",
        type=float,
        default=1.0,
        help="Last global start-to-goal fraction to probe (default: 1).",
    )
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeated exact resets per endpoint for a render/noise estimate.")
    parser.add_argument("--mask-threshold", type=int, default=8,
                        help="RGB change threshold used for the start-vs-goal pixel mask.")
    parser.add_argument("--mask-dilation", type=int, default=5, help="Mask dilation in pixels.")
    parser.add_argument("--save-all-videos", action="store_true",
                        help="Save a rollout MP4 for every repeat, not only the full goal rollout.")
    parser.add_argument("--video-stride", type=int)
    parser.add_argument("--video-fps", type=int)
    parser.add_argument("--force", action="store_true",
                        help="Replace artifacts from an earlier latent-path probe in --out-dir.")
    args = parser.parse_args()

    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if not (
        np.isfinite(args.fraction_start)
        and np.isfinite(args.fraction_end)
        and 0.0 <= args.fraction_start < args.fraction_end <= 1.0
    ):
        parser.error(
            "--fraction-start and --fraction-end must satisfy "
            "0 <= start < end <= 1"
        )
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not 0 <= args.mask_threshold <= 255 or args.mask_dilation < 0:
        parser.error("--mask-threshold must be in [0,255] and --mask-dilation non-negative")
    if args.video_stride is not None and args.video_stride <= 0:
        parser.error("--video-stride must be positive")
    if args.video_fps is not None and args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if args.arc_height is not None and not np.isfinite(args.arc_height):
        parser.error("--arc-height must be finite")
    if args.midpoint_x is not None and not np.isfinite(args.midpoint_x):
        parser.error("--midpoint-x must be finite")

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "latent_path_probe"
    report_path = out_dir / "path_probe.json"
    if report_path.exists() and not args.force:
        parser.error(f"Probe already exists: {report_path}. Use --force or a new --out-dir.")
    if args.force and out_dir.is_dir():
        for point_dir in out_dir.glob("point_*_t_*"):
            if point_dir.is_dir():
                shutil.rmtree(point_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    goal_path = (Path(args.goal) if args.goal else run_dir / "goal_oracle.png").resolve()
    if not goal_path.is_file():
        parser.error(f"Goal image not found: {goal_path}")
    start_path = run_dir / "start.png"
    if not start_path.is_file():
        parser.error(f"Start image not found: {start_path}")
    start_image = Image.open(start_path).convert("RGB")
    goal_image = Image.open(goal_path).convert("RGB")
    if start_image.size != goal_image.size:
        parser.error(f"Start/goal image sizes differ: {start_image.size} != {goal_image.size}")

    start_state = np.load(run_dir / "start_state.npy")
    start_eef = np.asarray(manifest["actual_start_eef"], dtype=np.float64)
    goal_eef = (
        np.asarray(args.goal_eef, dtype=np.float64)
        if args.goal_eef is not None
        else np.asarray(manifest["physical_goal_eef"], dtype=np.float64)
    )
    displacement = goal_eef - start_eef
    denominator = float(np.dot(displacement, displacement))
    if (
        start_eef.shape != (3,)
        or goal_eef.shape != (3,)
        or not np.isfinite(start_eef).all()
        or not np.isfinite(goal_eef).all()
        or denominator <= 1e-12
    ):
        parser.error("Start and goal EEF must be distinct three-vectors")
    start_gripper_actions = [
        np.asarray(action, dtype=np.float64)
        for action in manifest["start_gripper_controller_actions"]
    ]
    view_size = int(manifest["view_size"])
    video_stride = int(
        args.video_stride if args.video_stride is not None else manifest.get("video_stride", 2)
    )
    video_fps = int(
        args.video_fps if args.video_fps is not None else manifest.get("video_fps", 12)
    )
    tracking_tolerance = float(manifest.get("stage_tolerance_m", 0.01))
    reset_render_mae_tolerance = float(manifest.get("reset_render_mae_tolerance", 0.25))
    reset_render_outlier_tolerance = float(
        manifest.get("reset_render_outlier_fraction", 0.01)
    )
    full_arc_height, full_midpoint_x, path_shape_source = _oracle_path_shape(
        manifest, run_dir, goal_path, args.arc_height, args.midpoint_x
    )

    encoder = FluxAutoencoderMetric(
        Path(args.editor_ae).resolve(), Path(args.flux2_src).resolve(), args.device
    )
    try:
        goal_latent, goal_source = _goal_latent(
            run_dir, goal_path, goal_image, encoder, args.goal_latent_source
        )
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(goal_latent).all():
        parser.error("Goal latent contains non-finite values")
    np.save(out_dir / "goal_latent.npy", goal_latent.astype(np.float32))
    start_latent = encoder.encode(start_image)
    np.save(out_dir / "start_latent.npy", start_latent.astype(np.float32))
    start_image.save(out_dir / "start_reference.png")
    goal_image.save(out_dir / "goal_reference.png")
    start_decoded = encoder.decode(
        start_latent, height=start_image.height, width=start_image.width
    )
    start_decoded.save(out_dir / "start_latent_decoded.png")
    _residual_image(start_image, start_decoded).save(
        out_dir / "start_latent_residual.png"
    )
    goal_decoded = encoder.decode(
        goal_latent, height=goal_image.height, width=goal_image.width
    )
    goal_decoded.save(out_dir / "goal_latent_decoded.png")
    _residual_image(goal_image, goal_decoded).save(out_dir / "goal_latent_residual.png")
    change_mask, mask_fallback = _change_mask(
        start_image, goal_image, args.mask_threshold, args.mask_dilation
    )
    Image.fromarray((change_mask.astype(np.uint8) * 255), mode="L").save(
        out_dir / "change_mask.png"
    )

    env = env_from_manifest(manifest)
    raw_rows: list[dict[str, Any]] = []
    waypoints: list[dict[str, Any]] = []
    try:
        env.seed(int(manifest["sim_seed"]))
        fractions = np.linspace(
            args.fraction_start, args.fraction_end, args.samples
        )
        for waypoint_index, fraction in enumerate(fractions):
            target = start_eef + float(fraction) * displacement
            rollout_arc_height = float(fraction) * full_arc_height
            rollout_midpoint_x = float(fraction) * full_midpoint_x
            point_dir = out_dir / f"point_{waypoint_index:03d}_t_{fraction:.3f}"
            point_rows: list[dict[str, Any]] = []
            for repeat in range(args.repeats):
                repeat_dir = point_dir / f"repeat_{repeat:02d}"
                repeat_dir.mkdir(parents=True, exist_ok=True)
                env.reset()
                obs = env.set_init_state(start_state)
                _synchronize_controllers_to_sim_state(env, start_gripper_actions)
                restored_state = np.asarray(env.get_sim_state(), dtype=np.float64)
                reset_state_max_abs = float(np.max(np.abs(restored_state - start_state)))
                if reset_state_max_abs != 0.0:
                    raise RuntimeError(
                        f"Simulator reset mismatch at waypoint {waypoint_index}, repeat {repeat}: "
                        f"max state difference={reset_state_max_abs}"
                    )
                restored_image = _views_from_obs(obs, view_size)[2]
                reset_rgb_abs = np.abs(
                    np.asarray(restored_image, dtype=np.int16)
                    - np.asarray(start_image, dtype=np.int16)
                )
                reset_render_mae = float(reset_rgb_abs.mean())
                reset_render_fraction_gt8 = float(np.mean(reset_rgb_abs > 8))
                if (
                    reset_render_mae > reset_render_mae_tolerance
                    or reset_render_fraction_gt8 > reset_render_outlier_tolerance
                ):
                    raise RuntimeError(
                        f"Restored render drift at waypoint {waypoint_index}, repeat {repeat}: "
                        f"RGB MAE={reset_render_mae:.4f}, "
                        f"fraction(|diff|>8)={reset_render_fraction_gt8:.6f}"
                    )
                capture_video = args.save_all_videos or (
                    waypoint_index == args.samples - 1 and repeat == 0
                )
                obs, actions, eef_path, frames = _rollout_to_target(
                    env,
                    obs,
                    target,
                    move_steps=int(manifest["move_steps"]),
                    settle_steps=int(manifest["settle_steps"]),
                    gain=float(manifest["controller_gain"]),
                    arc_height=rollout_arc_height,
                    midpoint_x=rollout_midpoint_x,
                    capture_video=capture_video,
                    video_stride=video_stride,
                    view_size=view_size,
                )
                main_image, right_image, terminal_image = _views_from_obs(obs, view_size)
                if terminal_image.size != goal_image.size:
                    raise RuntimeError(
                        f"Terminal/goal image sizes differ: {terminal_image.size} != "
                        f"{goal_image.size}"
                    )
                terminal_latent = encoder.encode(terminal_image)
                latent_metrics = _latent_metrics(terminal_latent, goal_latent)
                view_metrics = _per_view_latent_rms(
                    terminal_latent, goal_latent, terminal_image.size
                )
                pixel_metrics = _pixel_metrics(terminal_image, goal_image, change_mask)
                terminal_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                actual_progress = float(
                    np.dot(terminal_eef - start_eef, displacement) / denominator
                )
                off_axis = float(
                    np.linalg.norm(
                        (terminal_eef - start_eef) - actual_progress * displacement
                    )
                )
                goal_error = float(np.linalg.norm(terminal_eef - goal_eef))
                tracking_error = float(np.linalg.norm(terminal_eef - target))
                if tracking_error > tracking_tolerance:
                    raise RuntimeError(
                        f"Endpoint tracking failed at waypoint {waypoint_index}, repeat {repeat}: "
                        f"error={tracking_error:.4f}m > tolerance={tracking_tolerance:.4f}m"
                    )
                expected_actions = int(manifest["move_steps"]) + int(manifest["settle_steps"])
                if actions.shape != (expected_actions, 7):
                    raise RuntimeError(
                        f"Unexpected action shape at waypoint {waypoint_index}: {actions.shape}"
                    )
                if eef_path.shape != (expected_actions + 1, 3):
                    raise RuntimeError(
                        f"Unexpected EEF path shape at waypoint {waypoint_index}: {eef_path.shape}"
                    )
                if not np.isfinite(actions).all() or not np.isfinite(eef_path).all():
                    raise RuntimeError(f"Non-finite rollout at waypoint {waypoint_index}")

                main_image.save(repeat_dir / "terminal_agentview.png")
                right_image.save(repeat_dir / f"terminal_{composed_right_view()}.png")
                terminal_image.save(repeat_dir / "terminal.png")
                np.save(repeat_dir / "terminal_latent.npy", terminal_latent.astype(np.float32))
                np.save(repeat_dir / "actions.npy", actions.astype(np.float32))
                np.save(repeat_dir / "eef_path.npy", eef_path.astype(np.float32))
                np.save(repeat_dir / "terminal_state.npy", np.asarray(env.get_sim_state()))
                if capture_video and args.save_all_videos:
                    _write_video(repeat_dir / "rollout.mp4", frames, video_fps)
                if waypoint_index == args.samples - 1 and repeat == 0:
                    _write_video(out_dir / "start_to_goal_rollout.mp4", frames, video_fps)

                relative_terminal = str((repeat_dir / "terminal.png").relative_to(out_dir))
                row = {
                    "waypoint_index": int(waypoint_index),
                    "requested_fraction": float(fraction),
                    "repeat": int(repeat),
                    "target_eef": target.tolist(),
                    "terminal_eef": terminal_eef.tolist(),
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                    "target_z": float(target[2]),
                    "terminal_x": float(terminal_eef[0]),
                    "terminal_y": float(terminal_eef[1]),
                    "terminal_z": float(terminal_eef[2]),
                    "actual_progress_fraction": actual_progress,
                    "goal_error_m": goal_error,
                    "tracking_error_m": tracking_error,
                    "off_axis_error_m": off_axis,
                    "reset_state_max_abs": reset_state_max_abs,
                    "reset_render_mae_255": reset_render_mae,
                    "reset_render_fraction_gt8": reset_render_fraction_gt8,
                    "rollout_arc_height_m": rollout_arc_height,
                    "rollout_midpoint_x_m": rollout_midpoint_x,
                    "latent_rms": latent_metrics["rms"],
                    "latent_cosine_distance": latent_metrics["cosine_distance"],
                    **view_metrics,
                    **pixel_metrics,
                    "terminal_image": relative_terminal,
                }
                point_rows.append(row)
                raw_rows.append(row)
            waypoints.append(
                _group_waypoint(waypoint_index, float(fraction), target, point_rows)
            )
            latest = waypoints[-1]
            print(
                f"[point {waypoint_index:02d}] t={fraction:.2f} "
                f"actual={latest['actual_progress_fraction_mean']:.3f} "
                f"goal_err={latest['goal_error_m_mean']:.4f}m "
                f"latent_rms={latest['latent_rms_mean']:.5f}"
                f"±{latest['latent_rms_std']:.5f}"
            )
    finally:
        env.close()

    summary = _metric_summary(waypoints)
    payload = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "goal_image": str(goal_path),
        "goal_latent_source": goal_source,
        "encoder": encoder.provenance,
        "reference_latent_metrics": {
            "start_to_goal": _latent_metrics(start_latent, goal_latent),
            "start_reconstruction": _reconstruction_metrics(start_image, start_decoded),
            "goal_reconstruction": _reconstruction_metrics(goal_image, goal_decoded),
        },
        "start_eef": start_eef.tolist(),
        "goal_eef": goal_eef.tolist(),
        "physical_path_is_diagnostic_only": True,
        "config": {
            "samples": args.samples,
            "fraction_start": args.fraction_start,
            "fraction_end": args.fraction_end,
            "repeats": args.repeats,
            "move_steps": int(manifest["move_steps"]),
            "settle_steps": int(manifest["settle_steps"]),
            "controller_gain": float(manifest["controller_gain"]),
            "tracking_tolerance_m": tracking_tolerance,
            "reset_render_mae_tolerance_255": reset_render_mae_tolerance,
            "reset_render_outlier_fraction_tolerance": reset_render_outlier_tolerance,
            "full_goal_arc_height_m": full_arc_height,
            "full_goal_midpoint_x_m": full_midpoint_x,
            "partial_path_shape_scaling": "multiply full-goal excursion by requested_fraction",
            "path_shape_source": path_shape_source,
            "video_stride": video_stride,
            "video_fps": video_fps,
            "mask_threshold": args.mask_threshold,
            "mask_dilation": args.mask_dilation,
            "mask_fraction": float(change_mask.mean()),
            "mask_used_fallback": mask_fallback,
        },
        "summary": summary,
        "waypoints": waypoints,
        "artifacts": {
            "plot": "path_metrics.png",
            "contact_sheet": "contact_sheet.png",
            "csv": "path_probe.csv",
            "goal_rollout_video": "start_to_goal_rollout.mp4",
            "change_mask": "change_mask.png",
            "start_latent_decoded": "start_latent_decoded.png",
            "start_latent_residual": "start_latent_residual.png",
            "goal_latent_decoded": "goal_latent_decoded.png",
            "goal_latent_residual": "goal_latent_residual.png",
        },
    }
    _write_csv(out_dir / "path_probe.csv", raw_rows)
    _contact_sheet(waypoints, out_dir, out_dir / "contact_sheet.png")
    _plot(waypoints, summary, out_dir / "path_metrics.png")
    report_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(f"\nVERDICT: {summary['verdict']}")
    print(
        f"Spearman(physical goal error, latent RMS): "
        f"{_format_optional(summary['spearman_goal_error_vs_latent_rms'], '+.3f')}"
    )
    print(
        f"Monotonic non-increasing steps: "
        f"{100.0 * summary['latent_monotonic_nonincreasing_fraction']:.1f}%"
    )
    print(f"[done] report:        {report_path}")
    print(f"[done] plot:          {out_dir / 'path_metrics.png'}")
    print(f"[done] contact sheet: {out_dir / 'contact_sheet.png'}")
    print(f"[done] goal rollout:  {out_dir / 'start_to_goal_rollout.mp4'}")


if __name__ == "__main__":
    main()
