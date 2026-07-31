#!/usr/bin/env python
"""Standalone ImageWAM (FLUX.2) image-editing / next-frame prediction.

This bypasses the whole LIBERO/RoboTwin simulator harness and just runs the
image-editing model directly:

    starting image  +  text prompt   -->   predicted future image

It uses the released FLUX.2-4B ImageWAM LIBERO checkpoint. The model was trained
on LIBERO with a 224 (H) x 448 (W) input made of TWO camera views concatenated
side by side: [ main/agentview | wrist ]. So for in-distribution results you
should feed a 224x448 two-view image. The script can build one for you either
from two image files or straight from the LIBERO dataset videos.

Examples
--------
# 1) Easiest: pull a real starting frame + its instruction from the LIBERO dataset
python scripts/run_image_edit.py --from-dataset libero_goal --episode 0

# 2) Your own two views + your own prompt
python scripts/run_image_edit.py \
    --main-image main.png --wrist-image wrist.png \
    --prompt "open the middle drawer of the cabinet"

# 3) A single already-composed 224x448 image
python scripts/run_image_edit.py --image obs_224x448.png --prompt "put the bowl on the plate"

Notes
-----
* VRAM: the DiT + action expert sit on the GPU (~8.5 GB); the VAE and the Qwen3
  text encoder are kept on the CPU and only touched on demand, so this fits on a
  single 11 GB card (e.g. RTX 2080 Ti). The Qwen3 text pass runs on CPU (a few
  tens of seconds); the 20-step DiT denoise runs on GPU.
* proprio: the LIBERO checkpoint is proprioception-conditioned (8-dim robot
  state). Without a simulator we pass zeros by default, which is fine for a
  qualitative look. Real proprio would be slightly more faithful.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------------------------------
# Paths / imports
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env_local(repo_root: Path) -> dict:
    """Parse simple KEY=VALUE and ${VAR:-default} entries from .env.local."""
    env = {}
    env_file = repo_root / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            default_expression = re.fullmatch(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}",
                val,
            )
            if default_expression:
                variable_name, fallback = default_expression.groups()
                val = os.environ.get(variable_name) or fallback
            elif len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
                val = val[1:-1]
            env[key] = val
    return env


ENV = load_env_local(REPO_ROOT)


def resolve(name: str, default: str | None = None) -> str | None:
    """Prefer a real shell env var, then .env.local, then default."""
    return os.environ.get(name) or ENV.get(name) or default


# FLUX.2 source must be importable before we import the model code.
FLUX2_SRC = resolve("FLUX2_SRC", str(REPO_ROOT / "third_party" / "flux2"))
for p in (str(REPO_ROOT / "src"), FLUX2_SRC, str(Path(FLUX2_SRC) / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from imagewam.models.backbones.imagewam import ImageWAM  # noqa: E402


# ----------------------------------------------------------------------------
# Image preparation
# ----------------------------------------------------------------------------
def center_crop_resize(img: Image.Image, width: int, height: int) -> Image.Image:
    """Match eval_libero_single._center_crop_resize (aspect-preserving)."""
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    resized = img.resize((round(src_w * scale), round(src_h * scale)), Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    return resized.crop((left, top, left + width, top + height))


def compose_two_views(main: Image.Image, wrist: Image.Image, view_hw=(224, 224)) -> Image.Image:
    """Resize each view to 224x224 and concat horizontally -> 224x448."""
    h, w = view_hw
    m = center_crop_resize(main.convert("RGB"), width=w, height=h)
    r = center_crop_resize(wrist.convert("RGB"), width=w, height=h)
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(m, (0, 0))
    canvas.paste(r, (w, 0))
    return canvas


def read_first_mp4_frame(path: Path) -> Image.Image:
    import imageio.v3 as iio

    frame = iio.imread(str(path), index=0)  # HxWxC uint8
    return Image.fromarray(np.asarray(frame)).convert("RGB")


def build_input_from_dataset(suite: str, episode: int, frame_idx: int, data_root: Path):
    """Grab main + wrist frame for one LIBERO episode and its task instruction."""
    import json

    ds_dir = data_root / f"{suite}_no_noops_lerobot"
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_dir}")
    chunk = "chunk-000"
    main_mp4 = ds_dir / "videos" / chunk / "observation.images.image" / f"episode_{episode:06d}.mp4"
    wrist_mp4 = ds_dir / "videos" / chunk / "observation.images.wrist_image" / f"episode_{episode:06d}.mp4"

    import imageio.v3 as iio

    main = Image.fromarray(np.asarray(iio.imread(str(main_mp4), index=frame_idx))).convert("RGB")
    wrist = Image.fromarray(np.asarray(iio.imread(str(wrist_mp4), index=frame_idx))).convert("RGB")

    # Task instruction from meta/episodes.jsonl
    instruction = None
    ep_file = ds_dir / "meta" / "episodes.jsonl"
    if ep_file.exists():
        for line in ep_file.read_text().splitlines():
            rec = json.loads(line)
            if int(rec.get("episode_index", -1)) == int(episode):
                tasks = rec.get("tasks") or []
                instruction = tasks[0] if tasks else None
                break
    return compose_two_views(main, wrist), instruction


def pil_to_model_tensor(img: Image.Image, height: int, width: int, device, dtype) -> torch.Tensor:
    """PIL RGB -> [1,3,H,W] normalized to [-1, 1] (matches eval)."""
    img = center_crop_resize(img.convert("RGB"), width=width, height=height)
    arr = np.asarray(img, dtype=np.float32)
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = x * (2.0 / 255.0) - 1.0
    return x.to(device=device, dtype=dtype)


def model_tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """[3,H,W] in [-1,1] -> PIL RGB."""
    x = x.detach().float().clamp(-1, 1)
    arr = ((x + 1.0) * (255.0 / 2.0)).round().clamp(0, 255).byte()
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_image(image: Image.Image, path: Path) -> None:
    image_format = Image.registered_extensions().get(path.suffix.lower())
    if image_format is None:
        raise ValueError(f"Cannot infer image format from output suffix: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        image.save(temporary, format=image_format)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npy(array: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_libero_proprio(raw: np.ndarray, stats_path: Path) -> np.ndarray:
    """Apply the checkpoint's global min/max state normalization."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))["state"]["default"]
    low = np.asarray(stats["global_min"], dtype=np.float32)
    high = np.asarray(stats["global_max"], dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw.shape != low.shape or raw.shape != high.shape:
        raise ValueError(
            f"Proprio/stats shape mismatch: raw={raw.shape}, min={low.shape}, max={high.shape}"
        )
    span = high - low
    ignored = span < 1e-4
    safe_span = span.copy()
    safe_span[ignored] = 2.0
    normalized = 2.0 * (raw - low) / safe_span - 1.0
    normalized[ignored] = raw[ignored] - low[ignored]
    return np.clip(normalized, -5.0, 5.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(args, device, dtype):
    action_dit_config = {
        "action_dim": 7,          # LIBERO action_output_dim
        "hidden_dim": 1024,
        "mlp_ratio": 4.0,
        "max_action_horizon": 64,
        "use_gradient_checkpointing": False,
    }
    model = ImageWAM.from_flux2_klein_pretrained(
        flux2_model_path=resolve("FLUX2_MODEL_PATH"),
        ae_model_path=resolve("FLUX2_AE_MODEL_PATH"),
        action_dit_config=action_dit_config,
        flux2_src_path=FLUX2_SRC,
        variant="klein-base-4b",
        proprio_dim=8,                 # LIBERO checkpoint is proprio-conditioned
        load_text_encoder=True,        # kept on CPU, moved to GPU embeds on demand
        qwen3_model_spec=resolve("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B"),
        qwen_context_len=512,
        pack_proprio_after_text=True,
        device=device,
        torch_dtype=dtype,
    )
    model.load_checkpoint(args.ckpt)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("input image (choose one)")
    src.add_argument("--from-dataset", metavar="SUITE",
                     help="Pull a starting frame + instruction from a LIBERO suite "
                          "(libero_goal, libero_object, libero_spatial, libero_10).")
    src.add_argument("--episode", type=int, default=0, help="Episode index for --from-dataset.")
    src.add_argument("--frame", type=int, default=0, help="Frame index for --from-dataset.")
    src.add_argument("--main-image", help="Path to the main/agentview image.")
    src.add_argument("--wrist-image", help="Path to the wrist image (pairs with --main-image).")
    src.add_argument("--image", help="A single, already-composed 224x448 (or any) image.")

    ap.add_argument("--prompt", help="Task instruction. Optional if --from-dataset supplies one.")
    ap.add_argument("--output", default=str(REPO_ROOT / "image_edit_output.png"),
                    help="Where to save the result (a side-by-side input|prediction is also saved).")
    ap.add_argument("--steps", type=int, default=20, help="Denoising steps (fewer = faster/lower quality).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index for the DiT.")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--ckpt", default=resolve("CKPT_PATH"), help="Path to model.pt.")
    ap.add_argument("--data-root", default=resolve("DATA_ROOT"),
                    help="LIBERO data root (for --from-dataset).")
    ap.add_argument("--proprio-npy",
                    help="Optional 8-D LIBERO proprio vector captured with the input image.")
    ap.add_argument("--proprio-normalized", action="store_true",
                    help="Treat --proprio-npy as already normalized for the checkpoint.")
    ap.add_argument("--dataset-stats", default=resolve("DATASET_STATS_PATH"),
                    help="Stats JSON used to normalize raw --proprio-npy values.")
    ap.add_argument("--latent-output",
                    help="Optional .npy path for the final FLUX denoising tokens.")
    ap.add_argument("--metadata-output",
                    help="Optional JSON path; defaults beside --output.")
    args = ap.parse_args()

    if args.steps <= 0:
        ap.error("--steps must be positive")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        ap.error("--height and --width must be positive multiples of 16")
    if args.proprio_normalized and not args.proprio_npy:
        ap.error("--proprio-normalized requires --proprio-npy")

    planned_output = Path(args.output)
    planned_paths = {
        "output": planned_output.resolve(),
        "comparison": planned_output.with_name(
            planned_output.stem + "_compare.png"
        ).resolve(),
        "metadata": (
            Path(args.metadata_output)
            if args.metadata_output
            else planned_output.with_name(planned_output.stem + "_metadata.json")
        ).resolve(),
    }
    if args.latent_output:
        planned_paths["latent"] = Path(args.latent_output).resolve()
    if len(set(planned_paths.values())) != len(planned_paths):
        ap.error(f"Output artifact paths must be distinct: {planned_paths}")

    if not args.ckpt or not Path(args.ckpt).exists():
        ap.error(f"Checkpoint not found: {args.ckpt!r}. Set CKPT_PATH or pass --ckpt.")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # ---- Resolve the input image + prompt ----
    prompt = args.prompt
    if args.from_dataset:
        if not args.data_root:
            ap.error("--from-dataset needs DATA_ROOT (set it or pass --data-root).")
        input_img, ds_instruction = build_input_from_dataset(
            args.from_dataset, args.episode, args.frame, Path(args.data_root)
        )
        prompt = prompt or ds_instruction
        print(f"[input] dataset={args.from_dataset} episode={args.episode} frame={args.frame}")
    elif args.main_image and args.wrist_image:
        input_img = compose_two_views(Image.open(args.main_image), Image.open(args.wrist_image))
        print(f"[input] composed two views: {args.main_image} | {args.wrist_image}")
    elif args.image:
        input_img = Image.open(args.image)
        print(f"[input] single image: {args.image}")
    else:
        ap.error("Provide an input via --from-dataset, or --main-image/--wrist-image, or --image.")

    if not prompt:
        ap.error("No prompt given and none available from the dataset. Pass --prompt.")

    print(f"[prompt] {prompt!r}")
    print(f"[device] {device}  [dtype] {args.dtype}  [steps] {args.steps}  [seed] {args.seed}")

    # ---- Build model ----
    print("[model] building FLUX.2-4B ImageWAM and loading checkpoint (this takes a bit)...")
    model = build_model(args, device, dtype)

    x = pil_to_model_tensor(input_img, height=args.height, width=args.width, device=device, dtype=dtype)
    if args.proprio_npy:
        proprio_array = np.asarray(np.load(args.proprio_npy), dtype=np.float32).reshape(-1)
        if proprio_array.shape != (8,):
            ap.error(f"--proprio-npy must contain 8 values, got {proprio_array.shape}")
        if not args.proprio_normalized:
            if not args.dataset_stats or not Path(args.dataset_stats).exists():
                ap.error(
                    "Raw --proprio-npy requires --dataset-stats (or pass --proprio-normalized)."
                )
            proprio_array = normalize_libero_proprio(proprio_array, Path(args.dataset_stats))
        proprio = torch.from_numpy(proprio_array).unsqueeze(0).to(device=device, dtype=dtype)
        print(
            f"[proprio] loaded {'normalized' if args.proprio_normalized else 'raw + normalized'} "
            f"state from {args.proprio_npy}"
        )
    else:
        proprio_array = np.zeros(8, dtype=np.float32)
        proprio = torch.from_numpy(proprio_array).unsqueeze(0).to(device=device, dtype=dtype)

    # ---- Run the image-editing forward pass ----
    print("[run] predicting future frame...")
    final_tokens = None

    def capture_final_tokens(step_idx, total_steps, tokens):
        nonlocal final_tokens
        if step_idx == total_steps - 1:
            final_tokens = tokens.detach().float().cpu().numpy()

    with torch.no_grad():
        out = model.infer_video_flux2(
            prompt=prompt,
            input_image=x,
            proprio=proprio,
            num_inference_steps=args.steps,
            seed=args.seed,
            step_callback=capture_final_tokens if args.latent_output else None,
        )
    pred = model_tensor_to_pil(out["image"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_image(pred, out_path)

    latent_path = None
    if args.latent_output:
        if final_tokens is None:
            raise RuntimeError("Image editor did not expose final denoising tokens.")
        latent_path = Path(args.latent_output)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(final_tokens, latent_path)
        print(f"[done] final editor latent saved to: {latent_path}")

    # Also save an input|prediction strip for easy comparison.
    in_resized = center_crop_resize(input_img.convert("RGB"), width=args.width, height=args.height)
    strip = Image.new("RGB", (args.width, args.height * 2))
    strip.paste(in_resized, (0, 0))
    strip.paste(pred, (0, args.height))
    strip_path = out_path.with_name(out_path.stem + "_compare.png")
    atomic_save_image(strip, strip_path)

    def file_provenance(value):
        if not value:
            return None
        path = Path(value).resolve()
        return {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        }

    metadata = {
        "schema_version": 1,
        "edit_run_id": uuid.uuid4().hex,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": {
            "from_dataset": args.from_dataset,
            "episode": args.episode if args.from_dataset else None,
            "frame": args.frame if args.from_dataset else None,
            "main_image": args.main_image,
            "wrist_image": args.wrist_image,
            "image": args.image,
        },
        "prompt": prompt,
        "editor_seed": args.seed,
        "inference_steps": args.steps,
        "height": args.height,
        "width": args.width,
        "device": device,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "checkpoint": file_provenance(args.ckpt),
        "flux2_model": file_provenance(resolve("FLUX2_MODEL_PATH")),
        "flux2_autoencoder": file_provenance(resolve("FLUX2_AE_MODEL_PATH")),
        "flux2_source": str(Path(FLUX2_SRC).resolve()),
        "qwen3_model_spec": resolve("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B"),
        "proprio_source": args.proprio_npy,
        "proprio_was_normalized": bool(args.proprio_normalized),
        "proprio_model_values": proprio_array.tolist(),
        "dataset_stats": file_provenance(args.dataset_stats),
        "output_image": str(out_path.resolve()),
        "output_image_sha256": sha256_file(out_path),
        "comparison_image": str(strip_path.resolve()),
        "comparison_image_sha256": sha256_file(strip_path),
        "final_latent": str(latent_path.resolve()) if latent_path is not None else None,
        "final_latent_sha256": (
            sha256_file(latent_path) if latent_path is not None else None
        ),
        "final_latent_shape": list(final_tokens.shape) if final_tokens is not None else None,
    }
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else out_path.with_name(out_path.stem + "_metadata.json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(metadata, metadata_path)

    print(f"[done] prediction saved to: {out_path}")
    print(f"[done] input|prediction comparison saved to: {strip_path}")
    print(f"[done] editor metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
