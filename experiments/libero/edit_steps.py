#!/usr/bin/env python
"""Visualize ImageWAM's image-editing (next-frame prediction) step by step.

For each (task, env-condition) it:
  1. builds the LIBERO scene, optionally applying an object-perturbation spec,
  2. settles the scene and grabs the real observation (agentview + wrist),
  3. runs the FLUX.2 image-editing forward pass with the task instruction,
  4. decodes the latent at EVERY denoising step, so you can see noise -> predicted
     future frame, and saves the progression as a labelled grid.

It reuses the eval's own model, processor and preprocessing, so the model input is
identical to a real rollout (real proprio, same two-view 224x448 composition).

Driven by COMBOS below: each entry is (task_id, cond_slug, spec_path_or_None).
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _obs_to_model_input,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _resolve_model_cfg,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
)
from imagewam.datasets.lerobot.processors.imagewam_processor import ImageWAMProcessor
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from libero.libero import benchmark

from env_perturb import apply_perturbation, load_spec

WORK = Path("/home/kaifany/project-data/ImageWAM")

# (task_id, condition slug, perturbation spec or None)  -- None = original env
COMBOS = [
    (4, "orig", None),
    (4, "moved", WORK / "perturb_task4_all.json"),
    (5, "orig", None),
    (5, "small_perturb", WORK / "perturb_task5_small.json"),
    (6, "orig", None),
    (6, "reoriented", WORK / "perturb_task6_reoriented.json"),
]


def to_pil(x: torch.Tensor) -> Image.Image:
    """model image tensor [3,H,W] in [-1,1] (or [0,1]) -> PIL"""
    a = x.detach().float().cpu()
    if a.min() < -0.01:  # [-1,1] -> [0,1]
        a = (a + 1.0) / 2.0
    a = a.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((a * 255).round().astype(np.uint8))


def label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 16), (16, 16, 16))
    out.paste(img, (0, 16))
    ImageDraw.Draw(out).text((4, 3), text, fill=(235, 235, 235))
    return out


def grid(tiles, cols):
    if not tiles:
        return None
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (cols * w, rows * h), (16, 16, 16))
    for i, t in enumerate(tiles):
        g.paste(t, ((i % cols) * w, (i // cols) * h))
    return g


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_omnigen2")
def main(cfg: DictConfig):
    out_dir = Path(cfg.EVALUATION.get("edit_steps_out_dir", str(WORK / "edit_steps")))
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(cfg.EVALUATION.get("edit_steps", 20))
    seed = int(cfg.get("seed") or 0)
    trial = int(cfg.EVALUATION.get("edit_trial", 0))

    device = _resolve_eval_device(cfg)
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(_resolve_model_cfg(cfg), model_dtype=dtype, device=device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(device).eval()

    stats = load_dataset_stats_from_json(str(_resolve_dataset_stats_path(cfg)))
    processor: ImageWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 30))
    suite_name = str(cfg.EVALUATION.get("edit_suite", "libero_10"))
    task_suite = benchmark.get_benchmark_dict()[suite_name]()

    for task_id, cond, spec_path in COMBOS:
        task = task_suite.get_task(int(task_id))
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        init_states = task_suite.get_task_init_states(int(task_id))
        if spec_path is not None:
            init_states = apply_perturbation(env, init_states, load_spec(str(spec_path)))

        env.reset()
        obs = env.set_init_state(init_states[trial])
        for _ in range(num_steps_wait):  # let the scene settle, as the eval does
            obs, _, _, _ = env.step(get_libero_dummy_action())

        x, proprio, _ = _obs_to_model_input(
            obs, cfg=cfg, processor=processor, width=input_w, height=input_h,
            device=device, dtype=model.torch_dtype,
        )
        prompt = DEFAULT_PROMPT.format(task=task_description)
        logging.info("[edit] task %s / %s | %r", task_id, cond, task_description)

        frames: list[Image.Image] = []

        def cb(i, total, latents):
            img = model._decode_flux2_image_tokens(latents, height=input_h, width=input_w)
            frames.append(label(to_pil(img[0]), f"step {i + 1}/{total}"))

        with torch.no_grad():
            out = model.infer_video_flux2(
                prompt=prompt, input_image=x, proprio=proprio,
                num_inference_steps=steps, seed=seed, step_callback=cb,
            )

        tag = f"task{task_id}_{cond}"
        inp = label(to_pil(x[0]), "INPUT (obs: agentview | wrist)")
        pred = label(to_pil(out["image"]), "PREDICTED next frame")
        inp.save(out_dir / f"{tag}_input.png")
        pred.save(out_dir / f"{tag}_pred.png")
        grid([inp, pred], cols=1).save(out_dir / f"{tag}_input_vs_pred.png")
        g = grid(frames, cols=5)
        if g is not None:
            g.save(out_dir / f"{tag}_denoise_steps.png")
        logging.info("[edit] saved %s_* (%d step frames)", tag, len(frames))
        env.close()

    print(f"\n[done] wrote step-by-step edits to: {out_dir}")


if __name__ == "__main__":
    main()
