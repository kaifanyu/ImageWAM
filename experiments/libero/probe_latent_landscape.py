#!/usr/bin/env python
"""Is the editor latent distance a usable objective at all?

Before running any search over trajectories, this answers the prerequisite
question: does ``||AE(terminal) - z_goal||`` actually decrease as the arm gets
physically closer to the goal?  It encodes every candidate an endpoint run
already produced, pairs the latent distance with the ``goal_error_m`` recorded
in the manifest, and reports the rank correlation between them.

Because both axes are distances, a usable metric has a strongly positive
correlation: candidates farther from the physical goal should also be farther
from the goal latent.  A flat or negative relationship means the metric needs
work before putting an optimizer on top of it.

Debug images written to <run-dir>/probe/:
    goal_reference.png        the edited goal image the search aims at
    goal_latent_decoded.png   z_goal pushed back through the AE decoder
    goal_latent_residual.png  |goal_reference - goal_latent_decoded|, amplified
    landscape.png             latent distance vs physical goal error
    contact_sheet.png         candidates ordered by latent distance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_endpoint_candidates import (  # noqa: E402
    FluxAutoencoderMetric,
    _latent_metrics,
    _spearman,
)


def _load_goal_latent(
    run_dir: Path,
    encoder: FluxAutoencoderMetric,
    goal_image: Image.Image,
    prefer_editor_latent: bool,
) -> tuple[np.ndarray, str]:
    """Prefer the DiT's own final tokens; fall back to re-encoding the PNG.

    These are not identical: the saved editor latent is what the model actually
    produced, while re-encoding the PNG adds a decode/encode round trip.  Which
    one you optimize toward matters, so the choice is recorded.
    """
    editor_latent = run_dir / "goal_editor_latent.npy"
    if prefer_editor_latent and editor_latent.exists():
        return np.load(editor_latent), "goal_editor_latent.npy"
    return encoder.encode(goal_image), "encode(goal_edit.png)"


def _residual_image(a: Image.Image, b: Image.Image, gain: float = 4.0) -> Image.Image:
    diff = np.abs(
        np.asarray(a.convert("RGB"), dtype=np.float32)
        - np.asarray(b.convert("RGB"), dtype=np.float32)
    )
    return Image.fromarray(np.clip(diff * gain, 0, 255).astype(np.uint8))


def _landscape_plot(rows: list[dict[str, Any]], path: Path, spearman: dict[str, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errors = [row["goal_error_m"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, metric in zip(axes, ("latent_rms", "latent_cosine_distance")):
        values = [row[metric] for row in rows]
        axis.scatter(errors, values, s=42, zorder=3)
        for row, x, y in zip(rows, errors, values):
            axis.annotate(row["kind"], (x, y), fontsize=7, xytext=(4, 4),
                          textcoords="offset points")
        axis.set_xlabel("physical goal error (m)")
        axis.set_ylabel(metric)
        axis.set_title(f"{metric}   spearman={spearman[metric]:+.3f}")
        axis.grid(alpha=0.3, zorder=0)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _contact_sheet(rows: list[dict[str, Any]], run_dir: Path, path: Path) -> None:
    tiles = []
    for row in rows:
        image = Image.open(run_dir / row["terminal_image"]).convert("RGB")
        canvas = Image.new("RGB", (image.width, image.height + 22), (16, 16, 16))
        canvas.paste(image, (0, 22))
        ImageDraw.Draw(canvas).text(
            (4, 6),
            f"{row['id']} {row['kind']} rms={row['latent_rms']:.4f} err={row['goal_error_m']:.3f}m",
            fill=(235, 235, 235),
        )
        tiles.append(canvas)
    if not tiles:
        return
    sheet = Image.new("RGB", (tiles[0].width, tiles[0].height * len(tiles)), (16, 16, 16))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, index * tile.height))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                        help="An endpoint run that already has candidates and goal_edit.png.")
    parser.add_argument("--goal", help="Defaults to <run-dir>/goal_edit.png.")
    parser.add_argument("--editor-ae", required=True, help="FLUX.2 ae.safetensors.")
    parser.add_argument("--flux2-src", default=str(REPO_ROOT / "third_party" / "flux2"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--goal-latent-source",
        choices=["editor", "reencode"],
        default="editor",
        help="'editor' uses the DiT's saved final tokens; 'reencode' re-encodes the PNG.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    goal_path = Path(args.goal) if args.goal else run_dir / "goal_edit.png"
    if not goal_path.exists():
        parser.error(f"Goal image not found: {goal_path}. Run the edit stage first.")

    out_dir = run_dir / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    goal_image = Image.open(goal_path).convert("RGB")
    encoder = FluxAutoencoderMetric(
        Path(args.editor_ae).resolve(), Path(args.flux2_src).resolve(), args.device
    )
    goal_latent, goal_source = _load_goal_latent(
        run_dir, encoder, goal_image, args.goal_latent_source == "editor"
    )
    print(f"[goal] latent source: {goal_source}  shape={goal_latent.shape}")

    # Visual proof that the optimization target is the image we think it is.
    goal_image.save(out_dir / "goal_reference.png")
    decoded = encoder.decode(goal_latent, height=goal_image.height, width=goal_image.width)
    decoded.save(out_dir / "goal_latent_decoded.png")
    _residual_image(goal_image, decoded).save(out_dir / "goal_latent_residual.png")
    round_trip = float(
        np.mean(
            np.abs(
                np.asarray(goal_image, dtype=np.float32)
                - np.asarray(decoded, dtype=np.float32)
            )
        )
    )
    print(f"[goal] decode round-trip MAE vs goal image: {round_trip:.2f}/255")

    # Disambiguates an inverted result: did the edit actually move the arm, or
    # did it hand back something close to its own input?
    goal_reference_mae: dict[str, float] | None = None
    oracle_path = run_dir / "goal_oracle.png"
    start_path = run_dir / "start.png"
    if oracle_path.exists() and start_path.exists():
        goal_array = np.asarray(goal_image, dtype=np.float32)
        goal_reference_mae = {
            "start": float(
                np.mean(np.abs(goal_array - np.asarray(
                    Image.open(start_path).convert("RGB"), dtype=np.float32)))
            ),
            "oracle": float(
                np.mean(np.abs(goal_array - np.asarray(
                    Image.open(oracle_path).convert("RGB"), dtype=np.float32)))
            ),
        }

    rows: list[dict[str, Any]] = []
    for record in manifest["candidates"]:
        terminal = Image.open(run_dir / record["terminal_image"]).convert("RGB")
        metrics = _latent_metrics(encoder.encode(terminal), goal_latent)
        rows.append(
            {
                "id": record["id"],
                "kind": record["kind"],
                "goal_error_m": float(record["goal_error_m"]),
                "terminal_image": record["terminal_image"],
                "latent_rms": metrics["rms"],
                "latent_cosine_distance": metrics["cosine_distance"],
            }
        )

    rows.sort(key=lambda row: row["goal_error_m"])
    spearman = {
        metric: (_spearman(rows, metric) or float("nan"))
        for metric in ("latent_rms", "latent_cosine_distance")
    }

    print(f"\n{'candidate':<14}{'kind':<17}{'goal_err_m':>11}{'rms':>11}{'cos_dist':>11}")
    for row in rows:
        print(
            f"{row['id']:<14}{row['kind']:<17}{row['goal_error_m']:>11.4f}"
            f"{row['latent_rms']:>11.4f}{row['latent_cosine_distance']:>11.5f}"
        )

    _landscape_plot(rows, out_dir / "landscape.png", spearman)
    _contact_sheet(
        sorted(rows, key=lambda row: row["latent_rms"]), run_dir, out_dir / "contact_sheet.png"
    )
    (out_dir / "probe.json").write_text(
        json.dumps(
            {"goal_latent_source": goal_source, "decode_round_trip_mae": round_trip,
             "goal_reference_mae": goal_reference_mae,
             "spearman": spearman, "candidates": rows},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nspearman(goal_error, rms)      = {spearman['latent_rms']:+.3f}")
    print(f"spearman(goal_error, cos_dist) = {spearman['latent_cosine_distance']:+.3f}")

    # Anti-correlation and no correlation are different diagnoses with different
    # fixes, so separate them instead of lumping both into "weak".
    correlation = spearman["latent_rms"]
    spread = float(np.std([row["latent_rms"] for row in rows]))
    if correlation > 0.7:
        verdict = "USABLE -- latent distance tracks physical error. SVGD is worth trying."
    elif correlation < -0.3:
        verdict = (
            "INVERTED -- candidates FARTHER from the goal score BETTER. The edited "
            "goal image resembles the start more than the true goal, so the search "
            "target itself is wrong. Fix the edit; the metric is not the problem."
        )
    else:
        verdict = (
            f"FLAT -- no usable signal (latent rms std={spread:.4f} across candidates). "
            "Search on top of this metric will wander."
        )
    print(f"\nVERDICT: {verdict}")

    if goal_reference_mae is not None:
        print(
            f"\n[cross-check] goal_edit vs start.png        MAE={goal_reference_mae['start']:.2f}/255\n"
            f"[cross-check] goal_edit vs goal_oracle.png  MAE={goal_reference_mae['oracle']:.2f}/255"
        )
        if goal_reference_mae["start"] < goal_reference_mae["oracle"]:
            print("[cross-check] The edit stayed closer to its input than to the true goal.")
    print(f"[done] debug images: {out_dir}")


if __name__ == "__main__":
    main()
