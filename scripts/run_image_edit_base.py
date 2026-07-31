#!/usr/bin/env python
"""Same edit path as run_image_edit.py, but on the *base* FLUX.2 weights.

This is a diagnostic, not a production path. It builds the identical model
scaffolding as run_image_edit.py and then simply does NOT call
`load_checkpoint()`, so the DiT keeps its FLUX.2-klein-base-4B weights instead
of the ImageWAM LIBERO finetune. Everything else -- the 224x448 two-view
compose, the [-1,1] normalization, the ref/target time coordinates, the
denoise schedule -- is shared with run_image_edit.py by direct import.

Use it to answer one question: is a disappointing goal image the fault of the
ImageWAM checkpoint's fixed ~32-step training horizon, or of the edit path?

    base output looks better  -> horizon prior is the constraint; retrain with
                                 a larger global_sample_stride
    base output looks worse   -> the finetune is doing its job; look elsewhere

Known caveats -- read these before drawing conclusions
-----------------------------------------------------
* No proprio. The proprio encoder is a *trained* ImageWAM component; without
  the checkpoint it would be randomly initialized, so this script builds the
  model with proprio_dim=None and ignores any proprio input. The base run is
  therefore conditioned on strictly less information than the ImageWAM run.
* CFG is available but is NOT what the ImageWAM run uses. `infer_video_flux2`
  denoises without guidance, so `--guidance 1.0` is the apples-to-apples
  comparison and the 4.0 default is the "show the base model at its best" run.
  When comparing outputs, know which one you are looking at.
* Out-of-distribution input. Base FLUX.2 has never seen a 224x448 side-by-side
  [agentview | wrist] composite and has no reason to keep the two views
  mutually consistent.
* The latent is not comparable. `--latent-output` is accepted so the file
  layout matches, but score_endpoint_candidates.py compares editor latents
  against simulator terminals under the finetune's semantics. Scoring a base
  latent will produce numbers; they do not mean what the pipeline thinks.

Example
-------
python scripts/run_image_edit_base.py \
    --image runs/empty_arm_preview/start.png \
    --prompt "$PROMPT" \
    --output runs/empty_arm_preview/goal_edit_base.png
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_image_edit as rie  # noqa: E402

from imagewam.models.backbones.imagewam import ImageWAM  # noqa: E402


def build_base_model(args, device, dtype):
    """Identical to rie.build_model minus load_checkpoint() and proprio."""
    action_dit_config = {
        "action_dim": 7,
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
        proprio_dim=None,  # no trained proprio encoder without the checkpoint
        load_text_encoder=True,
        qwen3_model_spec=rie.resolve("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B"),
        qwen_context_len=512,
        pack_proprio_after_text=True,
        device=device,
        torch_dtype=dtype,
    )
    # NOTE: deliberately no model.load_checkpoint(...) -- that is the whole point.
    model.eval()
    return model


def denoise_with_cfg(
    model,
    prompt: str,
    negative_prompt: str,
    input_image: torch.Tensor,
    num_inference_steps: int,
    guidance: float,
    seed: int,
    step_callback=None,
):
    """`infer_video_flux2` with classifier-free guidance bolted on.

    Mirrors the denoise loop in ImageWAM.infer_video_flux2, but evaluates the
    velocity twice per step -- once on `prompt`, once on `negative_prompt` --
    and combines them with flux2.sampling.vanilla_guidance's formula:

        pred = pred_uncond + guidance * (pred_cond - pred_uncond)

    Two separate forward passes rather than one batched [uncond, cond] pass:
    2x the compute, but it avoids having to reason about batching the MoT
    attention mask, and at 224x448 the cost is not the bottleneck.

    This reaches into private ImageWAM methods on purpose -- there is no public
    CFG hook. It will need updating if the denoise loop upstream changes.
    """
    model.eval()
    if input_image.ndim == 3:
        input_image = input_image.unsqueeze(0)
    _, _, height, width = input_image.shape

    cond_hidden, cond_mask = model._prepare_flux2_infer_text(prompt, None, None)
    uncond_hidden, uncond_mask = model._prepare_flux2_infer_text(negative_prompt, None, None)

    input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
    ref_tokens, ref_img_ids = model._encode_flux2_image_tokens(input_image, time_value=10.0)
    batch_size = int(ref_tokens.shape[0])

    from imagewam.models.backbones.flux2_video_expert import Flux2VideoExpert

    generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        ref_tokens.shape, generator=generator, device="cpu", dtype=torch.float32
    ).to(device=model.device, dtype=model.torch_dtype)
    target_img_ids = Flux2VideoExpert.build_img_ids(
        batch_size=batch_size,
        token_height=int(height) // 16,
        token_width=int(width) // 16,
        time_value=0.0,
        device=model.device,
        dtype=model.torch_dtype,
    )

    timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=latents.dtype,
        shift_override=None,
    )

    def velocity(text_hidden, text_mask, step_t):
        timestep = step_t.expand(batch_size).to(dtype=latents.dtype, device=model.device)
        pre = model.video_expert.pre_dit(
            x=latents,
            timestep=model._scheduler_timestep_to_unit(timestep, model.infer_video_scheduler),
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_tokens,
            target_img_ids=target_img_ids,
            ref_img_ids=ref_img_ids,
        )
        attention_mask = model._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(pre["txt_len"]),
            target_len=int(pre["target_len"]),
            cond_len=int(pre["cond_len"]),
            action_len=0,
            device=latents.device,
            text_attention_mask=pre["text_mask"],
        )
        return model.video_expert.post_dit(
            model._forward_flux2_video_only(pre, attention_mask), pre
        )

    total_steps = len(timesteps)
    for step_idx, (step_t, step_delta) in enumerate(zip(timesteps, deltas)):
        pred_cond = velocity(cond_hidden, cond_mask, step_t)
        pred_uncond = velocity(uncond_hidden, uncond_mask, step_t)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        latents = model.infer_video_scheduler.step(pred, step_delta, latents)
        if step_callback is not None:
            step_callback(step_idx, total_steps, latents)

    image = model._decode_flux2_image_tokens(latents, height=height, width=width)
    return {"image": image[0].detach().cpu()}


def main():
    ap = rie.argparse.ArgumentParser(
        description=__doc__,
        formatter_class=rie.argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image", required=True,
                    help="A composed 224x448 two-view image (e.g. runs/<run>/start.png).")
    ap.add_argument("--prompt", required=True, help="Task instruction.")
    ap.add_argument("--output", required=True,
                    help="Where to save the result; an input|prediction strip is saved beside it.")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.0,
                    help="Classifier-free guidance scale; FLUX.2's own CLI defaults to 4.0. "
                         "1.0 disables CFG and costs one forward pass per step instead of two.")
    ap.add_argument("--negative-prompt", default="",
                    help="Unconditional branch prompt for CFG. Empty string is the usual choice.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--latent-output",
                    help="Optional .npy for the final denoising tokens. NOT comparable to "
                         "ImageWAM latents -- see the caveats in this file's docstring.")
    args = ap.parse_args()

    if args.steps <= 0:
        ap.error("--steps must be positive")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        ap.error("--height and --width must be positive multiples of 16")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    input_img = Image.open(args.image)
    print(f"[input] single image: {args.image}")
    print(f"[prompt] {args.prompt!r}")
    print(f"[device] {device}  [dtype] {args.dtype}  [steps] {args.steps}  [seed] {args.seed}")
    cfg_note = "no CFG" if args.guidance == 1.0 else f"CFG={args.guidance}"
    print(f"[model] BASE FLUX.2-klein-base-4B -- ImageWAM checkpoint NOT loaded, no proprio, {cfg_note}")

    model = build_base_model(args, device, dtype)
    x = rie.pil_to_model_tensor(
        input_img, height=args.height, width=args.width, device=device, dtype=dtype
    )

    final_tokens = None

    def capture_final_tokens(step_idx, total_steps, tokens):
        nonlocal final_tokens
        if step_idx == total_steps - 1:
            final_tokens = tokens.detach().float().cpu().numpy()

    print("[run] predicting future frame...")
    with torch.no_grad():
        if args.guidance == 1.0:
            out = model.infer_video_flux2(
                prompt=args.prompt,
                input_image=x,
                proprio=None,
                num_inference_steps=args.steps,
                seed=args.seed,
                step_callback=capture_final_tokens if args.latent_output else None,
            )
        else:
            out = denoise_with_cfg(
                model,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                input_image=x,
                num_inference_steps=args.steps,
                guidance=args.guidance,
                seed=args.seed,
                step_callback=capture_final_tokens if args.latent_output else None,
            )
    pred = rie.model_tensor_to_pil(out["image"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rie.atomic_save_image(pred, out_path)

    in_resized = rie.center_crop_resize(
        input_img.convert("RGB"), width=args.width, height=args.height
    )
    strip = Image.new("RGB", (args.width, args.height * 2))
    strip.paste(in_resized, (0, 0))
    strip.paste(pred, (0, args.height))
    strip_path = out_path.with_name(out_path.stem + "_compare.png")
    rie.atomic_save_image(strip, strip_path)

    if args.latent_output:
        if final_tokens is None:
            raise RuntimeError("Base editor did not expose final denoising tokens.")
        latent_path = Path(args.latent_output)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        rie.atomic_save_npy(np.asarray(final_tokens), latent_path)
        print(f"[done] final latent saved to: {latent_path} (NOT score-comparable)")

    print(f"[done] prediction saved to: {out_path}")
    print(f"[done] input|prediction comparison saved to: {strip_path}")


if __name__ == "__main__":
    main()
