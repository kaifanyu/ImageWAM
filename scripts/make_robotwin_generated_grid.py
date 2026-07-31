#!/usr/bin/env python3
"""Build a contact sheet from RoboTwin ImageWAM generated images."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def sort_key(path: Path) -> tuple[int, int, int, int, str]:
    match = re.search(
        r"episode_(\d+)_step_(\d+)_infer_(\d+)_generated_(\d+)\.png$",
        path.name,
    )
    if match is None:
        return (10**9, 10**9, 10**9, 10**9, path.name)
    return tuple(int(group) for group in match.groups()) + (path.name,)


def resolve_model_outputs(path: Path) -> Path:
    if path.name == "model_outputs":
        return path
    candidate = path / "model_outputs"
    if candidate.is_dir():
        return candidate
    return path


def make_grid(
    image_paths: list[Path],
    output_path: Path,
    *,
    cols: int,
    thumb_width: int,
    label: bool,
) -> None:
    if not image_paths:
        raise ValueError("No generated images found. Expected files matching '*_generated_*.png'.")

    thumbs: list[tuple[Path, Image.Image]] = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        scale = thumb_width / image.width
        thumb_size = (thumb_width, max(1, int(round(image.height * scale))))
        thumbs.append((path, image.resize(thumb_size, Image.Resampling.LANCZOS)))

    label_height = 22 if label else 0
    tile_width = max(thumb.width for _, thumb in thumbs)
    tile_height = max(thumb.height for _, thumb in thumbs) + label_height
    rows = math.ceil(len(thumbs) / cols)

    grid = Image.new("RGB", (cols * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for idx, (path, thumb) in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = col * tile_width
        y = row * tile_height
        grid.paste(thumb, (x, y))
        if label:
            draw.text((x + 4, y + thumb.height + 4), path.stem, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Task output dir, e.g. evaluate_results/.../open_laptop, or its model_outputs dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <model_outputs>/generated_grid.png.",
    )
    parser.add_argument("--cols", type=int, default=6, help="Number of columns in the grid.")
    parser.add_argument("--thumb-width", type=int, default=320, help="Width of each thumbnail.")
    parser.add_argument("--no-label", action="store_true", help="Do not draw filenames under images.")
    args = parser.parse_args()

    model_outputs = resolve_model_outputs(args.path.expanduser())
    image_paths = sorted(model_outputs.glob("*_generated_*.png"), key=sort_key)
    output_path = args.output or (model_outputs / "generated_grid.png")

    make_grid(
        image_paths,
        output_path,
        cols=max(1, args.cols),
        thumb_width=max(1, args.thumb_width),
        label=not args.no_label,
    )
    print(f"Wrote {output_path} from {len(image_paths)} generated images")


if __name__ == "__main__":
    main()
