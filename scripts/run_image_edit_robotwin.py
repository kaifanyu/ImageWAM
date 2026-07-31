#!/usr/bin/env python
"""ImageWAM (FLUX.2 klein-base-4B) next-frame prediction on the RoboTwin checkpoint.

Same edit path as run_image_edit.py -- and it imports that module rather than
copying it -- but wired for RoboTwin instead of LIBERO. Three things differ, and
all three are load-bearing:

    LIBERO ckpt                        RoboTwin ckpt
    ---------------------------------  ---------------------------------
    224 (H) x 448 (W)                  288 (H) x 256 (W)
    [ agentview | wrist ]              head on top, [left | right] below
    proprio_dim=8, action_dim=7        proprio_dim=14, action_dim=14

Feeding a 224x448 LIBERO composite to this checkpoint (or vice versa) produces a
plausible-looking image that is entirely out of distribution. The defaults here
come from checkpoints/imagewam_release/robotwin/flux2_klein_4b/config.yaml.

Pair it with scripts/robotwin_start_obs.py, which produces start.png + proprio.npy
+ instruction.txt in the exact form this script wants:

    python scripts/run_image_edit_robotwin.py \
        --input-dir runs/robotwin_preview/place_dual_shoes \
        --output runs/robotwin_preview/place_dual_shoes/goal.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_image_edit as rie  # noqa: E402

from imagewam.models.backbones.imagewam import ImageWAM  # noqa: E402

# The prompt template the model was trained under. Raw instructions without this
# wrapper are a different conditioning distribution.
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)

PROPRIO_DIM = 14
ACTION_DIM = 14


def build_model(ckpt: str, device, dtype):
    action_dit_config = {
        "action_dim": ACTION_DIM,
        "hidden_dim": 1024,
        "mlp_ratio": 4.0,
        "max_action_horizon": 64,
        "use_gradient_checkpointing": False,
    }
    model = ImageWAM.from_flux2_klein_pretrained(
        flux2_model_path=rie.resolve("FLUX2_MODEL_PATH"),
        ae_model_path=rie.resolve("FLUX2_AE_MODEL_PATH"),
        action_dit_config=action_dit_config,
        flux2_src_path=rie.FLUX2_SRC,
        variant="klein-base-4b",
        proprio_dim=PROPRIO_DIM,
        load_text_encoder=True,
        qwen3_model_spec=rie.resolve("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B"),
        qwen_context_len=512,
        pack_proprio_after_text=True,
        device=device,
        torch_dtype=dtype,
    )
    model.load_checkpoint(ckpt)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_argument_group("input")
    src.add_argument("--input-dir",
                     help="A scripts/robotwin_start_obs.py output dir "
                          "(start.png + proprio.npy + instruction.txt).")
    src.add_argument("--image", help="A composed 256x288 image, if not using --input-dir.")
    src.add_argument("--prompt", help="Raw task instruction; overrides instruction.txt.")
    src.add_argument("--proprio-npy", help="Raw 14-dim state; overrides proprio.npy.")

    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument(
        "--ckpt",
        default=str(
            rie.REPO_ROOT / "checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt"
        ),
    )
    ap.add_argument(
        "--dataset-stats",
        default=str(
            rie.REPO_ROOT
            / "checkpoints/imagewam_release/robotwin/flux2_klein_4b/dataset_stats.json"
        ),
        help="Used to normalize raw proprio the way training did.",
    )
    ap.add_argument("--proprio-normalized", action="store_true",
                    help="Treat the proprio input as already normalized.")
    ap.add_argument("--no-proprio", action="store_true",
                    help="Pass zeros instead. Qualitatively fine, slightly less faithful.")
    ap.add_argument("--raw-prompt", action="store_true",
                    help="Skip the training prompt template and pass --prompt verbatim.")
    args = ap.parse_args()

    if args.height % 16 or args.width % 16:
        ap.error("--height and --width must be multiples of 16")

    image_path = None
    prompt = args.prompt
    proprio_path = args.proprio_npy

    if args.input_dir:
        d = Path(args.input_dir)
        image_path = d / "start.png"
        if prompt is None and (d / "instruction.txt").exists():
            prompt = (d / "instruction.txt").read_text(encoding="utf-8").strip()
        if proprio_path is None and (d / "proprio.npy").exists():
            proprio_path = d / "proprio.npy"
    elif args.image:
        image_path = Path(args.image)
    else:
        ap.error("Provide --input-dir or --image.")

    if not prompt:
        ap.error("No instruction found; pass --prompt.")

    full_prompt = prompt if args.raw_prompt else DEFAULT_PROMPT.format(task=prompt)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    input_img = Image.open(image_path)
    if input_img.size != (args.width, args.height):
        print(f"[warn] input is {input_img.size[0]}x{input_img.size[1]}, expected "
              f"{args.width}x{args.height}; it will be center-cropped/resized.")

    print(f"[input] {image_path}")
    print(f"[instruction] {prompt!r}")
    print(f"[prompt] {full_prompt!r}")
    print(f"[device] {device}  [dtype] {args.dtype}  [steps] {args.steps}  [seed] {args.seed}")
    print(f"[model] RoboTwin ImageWAM FLUX.2-klein-base-4B | {args.ckpt}")

    model = build_model(args.ckpt, device, dtype)
    x = rie.pil_to_model_tensor(
        input_img, height=args.height, width=args.width, device=device, dtype=dtype
    )

    if args.no_proprio or proprio_path is None:
        proprio_array = np.zeros(PROPRIO_DIM, dtype=np.float32)
        print("[proprio] zeros")
    else:
        proprio_array = np.asarray(np.load(proprio_path), dtype=np.float32).reshape(-1)
        if proprio_array.shape != (PROPRIO_DIM,):
            ap.error(f"proprio must have {PROPRIO_DIM} values, got {proprio_array.shape}")
        if not args.proprio_normalized:
            # Same global min/max scheme as LIBERO -- only the stats file differs.
            proprio_array = rie.normalize_libero_proprio(
                proprio_array, Path(args.dataset_stats)
            )
        print(f"[proprio] {proprio_path}")
    proprio = torch.from_numpy(proprio_array).unsqueeze(0).to(device=device, dtype=dtype)

    print("[run] predicting future frame...")
    with torch.no_grad():
        out = model.infer_video_flux2(
            prompt=full_prompt,
            input_image=x,
            proprio=proprio,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
    pred = rie.model_tensor_to_pil(out["image"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rie.atomic_save_image(pred, out_path)

    # Side-by-side rather than stacked: these frames are already tall (288x256),
    # so a vertical strip would be awkward to eyeball.
    in_resized = rie.center_crop_resize(
        input_img.convert("RGB"), width=args.width, height=args.height
    )
    strip = Image.new("RGB", (args.width * 2, args.height))
    strip.paste(in_resized, (0, 0))
    strip.paste(pred, (args.width, 0))
    strip_path = out_path.with_name(out_path.stem + "_compare.png")
    rie.atomic_save_image(strip, strip_path)

    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "ckpt": str(args.ckpt),
                "input_image": str(image_path),
                "instruction": prompt,
                "prompt": full_prompt,
                "steps": args.steps,
                "seed": args.seed,
                "height": args.height,
                "width": args.width,
                "proprio_source": "zeros" if (args.no_proprio or proprio_path is None)
                else str(proprio_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[done] prediction: {out_path}")
    print(f"[done] start|prediction: {strip_path}")


if __name__ == "__main__":
    main()
