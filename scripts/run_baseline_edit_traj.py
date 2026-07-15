#!/usr/bin/env python
"""Base image-editor trajectory-visualization BASELINE (no ImageWAM fine-tune).

Question this answers: how well does an OFF-THE-SHELF image editor -- with no
robot fine-tuning and no ActionDiT -- predict future robot frames from the
current frame plus the task instruction?

Unlike experiments/libero/rollout_edit_traj.py (which loads a fine-tuned ImageWAM
checkpoint and drives the simulator with its ActionDiT), this script:

  * loads a RAW pretrained editor (base weights only, no checkpoint),
  * walks a REAL recorded LIBERO episode (no simulator, no action model),
  * at each step t, predicts frame (t + horizon) from frame t + instruction,
  * saves a filmstrip of rows  [ input_t | predicted_{t+H} | actual_{t+H} ].

Pick the editor with the IMAGE_EDITOR env var (or --editor):

  flux4b  FLUX.2-klein-base-4B    (base weights, fits ~11GB with CPU offload)
  flux9b  FLUX.2-klein-base-9B    (base weights, needs a bigger GPU)
  qwen    Qwen/Qwen-Image-Edit    (diffusers pipeline; ~20B, needs a big GPU
                                    and diffusers>=0.35 -- see docker/)

Everything else (weights paths, HF cache) comes from .env.local / the
environment, exactly like the other scripts.

Usage:
  IMAGE_EDITOR=flux9b python scripts/run_baseline_edit_traj.py \
      --suite libero_10 --episode 0 --horizon 16 --max-steps 8
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# .env.local resolution (same convention as run_image_edit.py)
# ---------------------------------------------------------------------------
def _load_env_local(repo_root: Path) -> dict:
    env = {}
    f = repo_root / ".env.local"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


_ENV = _load_env_local(REPO_ROOT)


def resolve(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _ENV.get(name) or default


# ---------------------------------------------------------------------------
# Backend-agnostic image + dataset helpers
# ---------------------------------------------------------------------------
def center_crop_resize(img: Image.Image, width: int, height: int) -> Image.Image:
    """Aspect-preserving center crop then resize to (width, height)."""
    img = img.convert("RGB")
    src_w, src_h = img.size
    tgt = width / height
    if src_w / src_h > tgt:  # too wide -> crop width
        new_w = int(round(src_h * tgt))
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:                    # too tall -> crop height
        new_h = int(round(src_w / tgt))
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    return img.resize((width, height), Image.BICUBIC)


def compose_two_views(main: Image.Image, wrist: Image.Image,
                      height: int, width: int) -> Image.Image:
    """Concatenate [ main | wrist ] side by side into a single HxW image."""
    half = width // 2
    m = center_crop_resize(main, half, height)
    w = center_crop_resize(wrist, width - half, height)
    canvas = Image.new("RGB", (width, height))
    canvas.paste(m, (0, 0))
    canvas.paste(w, (half, 0))
    return canvas


def load_episode_videos(ds_dir: Path, episode: int):
    """Decode the main + wrist videos once, return (main_frames, wrist_frames).

    Video containers here report an infinite length via improps, so we read the
    whole clip (a few hundred small frames -- cheap) and index into the arrays.
    """
    import imageio.v3 as iio
    chunk = "chunk-000"
    main_mp4 = ds_dir / "videos" / chunk / "observation.images.image" / f"episode_{episode:06d}.mp4"
    wrist_mp4 = ds_dir / "videos" / chunk / "observation.images.wrist_image" / f"episode_{episode:06d}.mp4"
    main = np.asarray(iio.imread(str(main_mp4)))    # (T, H, W, 3)
    wrist = np.asarray(iio.imread(str(wrist_mp4)))
    return main, wrist


def compose_from_arrays(main_frames, wrist_frames, frame_idx: int,
                        height: int, width: int) -> Image.Image:
    main = Image.fromarray(main_frames[frame_idx])
    wrist = Image.fromarray(wrist_frames[frame_idx])
    return compose_two_views(main, wrist, height, width)


def episode_instruction(ds_dir: Path, episode: int) -> str | None:
    ep_file = ds_dir / "meta" / "episodes.jsonl"
    if not ep_file.exists():
        return None
    for line in ep_file.read_text().splitlines():
        rec = json.loads(line)
        if int(rec.get("episode_index", -1)) == int(episode):
            tasks = rec.get("tasks") or []
            return tasks[0] if tasks else None
    return None




# ---------------------------------------------------------------------------
# Editor backends -- each exposes .predict(pil_image, instruction) -> pil_image
# ---------------------------------------------------------------------------
class Flux2BaseEditor:
    """Raw FLUX.2-klein base weights, NO ImageWAM checkpoint, NO ActionDiT.

    Uses ImageWAM.infer_video_flux2 (the pure image-editing forward pass) on a
    model built from base weights with proprio_dim=None. Verified to run without
    any fine-tuned checkpoint.
    """

    def __init__(self, variant: str, device: str, dtype, steps: int, seed: int):
        import torch
        self.torch = torch
        self.steps = steps
        self.seed = seed
        flux2_src = resolve("FLUX2_SRC", str(REPO_ROOT / "third_party" / "flux2"))
        for p in (str(REPO_ROOT / "src"), flux2_src, str(Path(flux2_src) / "src")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from imagewam.models.backbones.imagewam import ImageWAM

        # Prefer a variant-specific path (FLUX2_MODEL_PATH_9B / _4B) so both
        # variants can be configured at once and the switch just works; fall
        # back to the generic FLUX2_MODEL_PATH.
        suffix = variant.replace("flux", "").upper()   # "9B" / "4B"
        model_path = resolve(f"FLUX2_MODEL_PATH_{suffix}") or resolve("FLUX2_MODEL_PATH")
        ae_path = resolve("FLUX2_AE_MODEL_PATH")
        qwen3 = resolve("FLUX2_QWEN3_MODEL_SPEC",
                        "Qwen/Qwen3-8B" if variant.endswith("9b") else "Qwen/Qwen3-4B")
        if not model_path or not Path(model_path).exists():
            raise SystemExit(
                f"[flux] no base weights for {variant}: set FLUX2_MODEL_PATH_{suffix} "
                f"(or FLUX2_MODEL_PATH) to the klein-base-{suffix.lower()} .safetensors. "
                f"got {model_path!r}")
        print(f"[flux] building {variant} base (NO checkpoint), text-encoder={qwen3}", flush=True)
        self.model = ImageWAM.from_flux2_klein_pretrained(
            flux2_model_path=model_path,
            ae_model_path=ae_path,
            action_dit_config={"action_dim": 7, "hidden_dim": 1024, "mlp_ratio": 4.0,
                               "max_action_horizon": 64, "use_gradient_checkpointing": False},
            flux2_src_path=flux2_src,
            variant=f"klein-base-{variant.replace('flux', '')}",
            proprio_dim=None,          # base model: no proprio conditioning
            load_text_encoder=True,
            qwen3_model_spec=qwen3,
            qwen_context_len=512,
            pack_proprio_after_text=True,
            device=device, torch_dtype=dtype,
        )
        # deliberately NOT loading any checkpoint -- this is the base editor.
        self.model.eval()

    def predict(self, image: Image.Image, instruction: str) -> Image.Image:
        torch = self.torch
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * (2.0 / 255.0) - 1.0
        x = x.to(device=self.model.device, dtype=self.model.torch_dtype)
        with torch.no_grad():
            out = self.model.infer_video_flux2(
                prompt=instruction, input_image=x, proprio=None,
                num_inference_steps=self.steps, seed=self.seed)
        y = out["image"].detach().float().clamp(-1, 1)
        y = ((y + 1.0) * (255.0 / 2.0)).round().clamp(0, 255).byte()
        return Image.fromarray(y.permute(1, 2, 0).cpu().numpy())


class QwenImageEditor:
    """Off-the-shelf Qwen-Image-Edit via diffusers. No robot fine-tuning.

    Requires diffusers>=0.35 (QwenImageEditPipeline). ~20B params: realistically
    needs a large GPU (or aggressive offload). Model id defaults to
    Qwen/Qwen-Image-Edit, override with QWEN_IMAGE_EDIT_MODEL.
    """

    def __init__(self, device: str, dtype, steps: int, seed: int):
        import torch
        self.torch = torch
        self.steps = steps
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        model_id = resolve("QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit")
        try:
            from diffusers import QwenImageEditPipeline
        except ImportError as e:
            raise SystemExit(
                "[qwen] QwenImageEditPipeline not available. Needs diffusers>=0.35 "
                "-- rebuild the image with DIFFUSERS_VERSION set (see docker/). "
                f"(import error: {e})")
        print(f"[qwen] loading {model_id} (this is large) ...", flush=True)
        self.pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=dtype)
        # Offload rather than assume it fits; harmless on big GPUs, essential on small.
        if resolve("QWEN_CPU_OFFLOAD", "true").lower() in ("1", "true", "yes"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    def predict(self, image: Image.Image, instruction: str) -> Image.Image:
        out = self.pipe(image=image.convert("RGB"), prompt=instruction,
                        num_inference_steps=self.steps, generator=self.generator)
        return out.images[0].resize(image.size, Image.BICUBIC)


def build_editor(name: str, device: str, dtype_str: str, steps: int, seed: int):
    import torch
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    name = name.lower()
    if name in ("flux4b", "flux9b"):
        return Flux2BaseEditor(name, device, dtype, steps, seed)
    if name == "qwen":
        return QwenImageEditor(device, dtype, steps, seed)
    raise SystemExit(f"Unknown IMAGE_EDITOR={name!r}. Use flux4b | flux9b | qwen.")


# ---------------------------------------------------------------------------
# Filmstrip
# ---------------------------------------------------------------------------
def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, 14], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return out


def save_filmstrip(rows, out_path: Path, instruction: str):
    if not rows:
        raise SystemExit("no rows produced")
    h, w = rows[0][0].height, rows[0][0].width
    pad, header = 6, 20
    grid = Image.new("RGB", (w * 3 + pad * 2, header + len(rows) * (h + pad)), (30, 30, 30))
    d = ImageDraw.Draw(grid)
    d.text((4, 4), f"[input | base-editor prediction | actual]   {instruction}",
           fill=(255, 255, 255))
    for r, (inp, pred, actual) in enumerate(rows):
        y = header + r * (h + pad)
        grid.paste(_label(inp, f"t (row {r})"), (0, y))
        grid.paste(_label(pred, "predicted"), (w + pad, y))
        grid.paste(_label(actual, "actual"), (2 * (w + pad), y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editor", default=resolve("IMAGE_EDITOR", "flux9b"),
                    help="flux4b | flux9b | qwen (or set IMAGE_EDITOR).")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=16,
                    help="Predict this many frames into the future.")
    ap.add_argument("--stride", type=int, default=0,
                    help="Frames between consecutive rows (default: = horizon).")
    ap.add_argument("--max-steps", type=int, default=8,
                    help="Max rows (prediction points) along the episode.")
    ap.add_argument("--steps", type=int, default=int(resolve("EDIT_STEPS", "20") or 20),
                    help="Denoising steps per prediction.")
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-root", default=resolve("DATA_ROOT"))
    ap.add_argument("--out-dir", default=resolve("BASELINE_OUT_DIR",
                                                 str(REPO_ROOT / "baseline_edit_traj")))
    args = ap.parse_args()

    import torch
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    stride = args.stride or args.horizon

    if not args.data_root:
        raise SystemExit("DATA_ROOT is unset. Set it in .env.local or pass --data-root.")
    ds_dir = Path(args.data_root) / f"{args.suite}_no_noops_lerobot"
    if not ds_dir.exists():
        raise SystemExit(f"Dataset not found: {ds_dir}")

    instruction = episode_instruction(ds_dir, args.episode) or ""
    main_frames, wrist_frames = load_episode_videos(ds_dir, args.episode)
    n_frames = min(len(main_frames), len(wrist_frames))
    print(f"[data] {args.suite} ep{args.episode}: {n_frames} frames | "
          f"instruction={instruction!r}", flush=True)
    print(f"[cfg] editor={args.editor} horizon={args.horizon} stride={stride} "
          f"steps={args.steps} device={device}", flush=True)

    editor = build_editor(args.editor, device, args.dtype, args.steps, args.seed)

    rows = []
    t = 0
    while t + args.horizon < n_frames and len(rows) < args.max_steps:
        inp = compose_from_arrays(main_frames, wrist_frames, t, args.height, args.width)
        actual = compose_from_arrays(main_frames, wrist_frames, t + args.horizon,
                                     args.height, args.width)
        print(f"[predict] row {len(rows)}: frame {t} -> {t + args.horizon}", flush=True)
        pred = editor.predict(inp, instruction)
        rows.append((inp, pred, actual))
        t += stride

    out_path = Path(args.out_dir) / f"{args.editor}_{args.suite}_ep{args.episode}_h{args.horizon}.png"
    save_filmstrip(rows, out_path, instruction)
    print(f"[done] {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
