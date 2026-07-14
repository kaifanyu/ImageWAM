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
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------------------------------
# Paths / imports
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env_local(repo_root: Path) -> dict:
    """Parse .env.local (KEY=VALUE lines) the same way the shell wrappers do."""
    env = {}
    env_file = repo_root / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            env[key.strip()] = val.strip()
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
    args = ap.parse_args()

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
    proprio = torch.zeros(1, 8, device=device, dtype=dtype)  # placeholder robot state

    # ---- Run the image-editing forward pass ----
    print("[run] predicting future frame...")
    with torch.no_grad():
        out = model.infer_video_flux2(
            prompt=prompt,
            input_image=x,
            proprio=proprio,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
    pred = model_tensor_to_pil(out["image"])

    out_path = Path(args.output)
    pred.save(out_path)

    # Also save an input|prediction strip for easy comparison.
    in_resized = center_crop_resize(input_img.convert("RGB"), width=args.width, height=args.height)
    strip = Image.new("RGB", (args.width, args.height * 2))
    strip.paste(in_resized, (0, 0))
    strip.paste(pred, (0, args.height))
    strip_path = out_path.with_name(out_path.stem + "_compare.png")
    strip.save(strip_path)

    print(f"[done] prediction saved to: {out_path}")
    print(f"[done] input|prediction comparison saved to: {strip_path}")


if __name__ == "__main__":
    main()
