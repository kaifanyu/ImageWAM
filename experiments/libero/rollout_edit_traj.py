#!/usr/bin/env python
"""Full-trajectory ImageWAM image-edit check: is the predicted frame right at EVERY chunk?

Rolls out the policy for real, and at each replan (every `replan_steps`=12 env steps) it:
  * predicts the future frame with the image-editing model  (= obs at t + FUTURE_OFFSET),
  * predicts the action chunk and executes `replan_steps` of it,
  * later pairs that prediction with the frame that ACTUALLY occurred at t+FUTURE_OFFSET
    and scores it (PSNR).

The model was trained on pairs (frame_t, frame_{t+16}) -- data.train.num_frames=17,
action_video_freq_ratio=1 -- so a prediction made at t is a claim about t+16, while only
12 steps get executed before replanning. That offset is why we compare against t+16, not
against the next replan.

Outputs per (task, condition):
  chunk_XX.png    [ input obs_t | PREDICTED t+16 | ACTUAL t+16 | abs diff ]  (+ PSNR)
  filmstrip.png   all chunks stacked
  psnr.json       per-chunk PSNR + final success/failure
"""

import json
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
    _predict_action_chunk,
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

COMBOS = [
    (4, "orig", None),
    (4, "moved", WORK / "perturb_task4_all.json"),
    (5, "orig", None),
    (5, "small_perturb", WORK / "perturb_task5_small.json"),
    (6, "orig", None),
    (6, "reoriented", WORK / "perturb_task6_reoriented.json"),
]


def to_pil(x: torch.Tensor) -> Image.Image:
    a = x.detach().float().cpu()
    if a.min() < -0.01:
        a = (a + 1.0) / 2.0
    return Image.fromarray((a.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))


def diff_img(a: Image.Image, b: Image.Image) -> Image.Image:
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(d)


def label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 16), (16, 16, 16))
    out.paste(img, (0, 16))
    ImageDraw.Draw(out).text((4, 3), text, fill=(235, 235, 235))
    return out


def hstack(tiles):
    w = sum(t.width for t in tiles)
    h = max(t.height for t in tiles)
    out = Image.new("RGB", (w, h), (16, 16, 16))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width
    return out


def vstack(rows):
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows)
    out = Image.new("RGB", (w, h), (16, 16, 16))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_omnigen2")
def main(cfg: DictConfig):
    out_root = Path(cfg.EVALUATION.get("traj_out_dir", str(WORK / "edit_traj")))
    out_root.mkdir(parents=True, exist_ok=True)
    max_chunks = int(cfg.EVALUATION.get("traj_max_chunks", 30))
    trial = int(cfg.EVALUATION.get("traj_trial", 0))
    only = str(cfg.EVALUATION.get("traj_only", ""))  # e.g. "4" to run just task 4
    seed = int(cfg.get("seed") or 0)

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
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 12))
    action_horizon = int(cfg.EVALUATION.get("action_horizon", 16))
    steps = int(cfg.EVALUATION.get("num_inference_steps", 20) or 20)
    # trained on (frame_t, frame_{t+16}); offset = num_frames-1 scaled by freq ratio
    future_offset = (int(cfg.data.train.num_frames) - 1)
    suite_name = str(cfg.EVALUATION.get("traj_suite", "libero_10"))
    task_suite = benchmark.get_benchmark_dict()[suite_name]()

    combos = [c for c in COMBOS if not only or str(c[0]) == only]
    logging.info("[traj] future_offset=%d replan_steps=%d max_chunks=%d combos=%d",
                 future_offset, replan_steps, max_chunks, len(combos))

    force = bool(cfg.EVALUATION.get("traj_force", False))
    for task_id, cond, spec_path in combos:
        # resume: skip combos that already finished (psnr.json is written last)
        if not force and (out_root / f"task{task_id}_{cond}" / "psnr.json").exists():
            logging.info("[traj] skip task%s_%s (already done; traj_force=true to redo)", task_id, cond)
            continue
        task = task_suite.get_task(int(task_id))
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        init_states = task_suite.get_task_init_states(int(task_id))
        if spec_path is not None:
            init_states = apply_perturbation(env, init_states, load_spec(str(spec_path)))

        env.reset()
        obs = env.set_init_state(init_states[trial])
        done = False
        for _ in range(num_steps_wait):
            obs, _, done, _ = env.step(get_libero_dummy_action())

        prompt = DEFAULT_PROMPT.format(task=task_description)
        tag = f"task{task_id}_{cond}"
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info("[traj] === %s | %r ===", tag, task_description)

        frames: dict[int, Image.Image] = {}   # env step -> composed obs (model-input view)
        records = []
        t = 0

        def capture(o):
            x, proprio, _ = _obs_to_model_input(
                o, cfg=cfg, processor=processor, width=input_w, height=input_h,
                device=device, dtype=model.torch_dtype,
            )
            return x, proprio

        for chunk in range(max_chunks):
            if done:
                break
            x, proprio = capture(obs)
            frames[t] = to_pil(x[0])

            with torch.no_grad():
                pred = model.infer_video_flux2(
                    prompt=prompt, input_image=x, proprio=proprio,
                    num_inference_steps=steps, seed=seed,
                )
            pred_pil = to_pil(pred["image"])

            action_chunk, _, _ = _predict_action_chunk(
                obs, task_description, model, processor, cfg,
                action_horizon=action_horizon, input_w=input_w, input_h=input_h,
                model_device=device,
            )
            records.append({"chunk": chunk, "t": t, "pred": pred_pil})
            logging.info("[traj] %s chunk %d @t=%d", tag, chunk, t)

            for a in action_chunk[:replan_steps]:
                obs, _, done, _ = env.step(a.tolist())
                t += 1
                xx, _ = capture(obs)
                frames[t] = to_pil(xx[0])
                if done:
                    break

        # score each prediction against the frame that actually happened at t+offset
        rows, psnrs = [], []
        for r in records:
            t0 = r["t"]
            actual = frames.get(t0 + future_offset)
            inp = frames[t0]
            if actual is None:  # episode ended before t+offset
                continue
            p = psnr(r["pred"], actual)
            psnrs.append({"chunk": r["chunk"], "t": t0, "psnr": round(p, 2)})
            row = hstack([
                label(inp, f"chunk {r['chunk']}  input obs t={t0}"),
                label(r["pred"], f"PREDICTED t+{future_offset}"),
                label(actual, f"ACTUAL t+{future_offset}"),
                label(diff_img(r["pred"], actual), f"|diff|  PSNR={p:.1f}dB"),
            ])
            row.save(out_dir / f"chunk_{r['chunk']:02d}.png")
            rows.append(row)

        if rows:
            vstack(rows).save(out_dir / "filmstrip.png")
        summary = {
            "task_id": task_id, "condition": cond, "task": task_description,
            "success": bool(done), "chunks": len(records), "env_steps": t,
            "future_offset": future_offset, "replan_steps": replan_steps,
            "psnr_mean": round(float(np.mean([p["psnr"] for p in psnrs])), 2) if psnrs else None,
            "per_chunk": psnrs,
        }
        (out_dir / "psnr.json").write_text(json.dumps(summary, indent=2))
        logging.info("[traj] %s done: success=%s chunks=%d psnr_mean=%s",
                     tag, done, len(records), summary["psnr_mean"])
        env.close()

    print(f"\n[done] trajectories written to: {out_root}")


if __name__ == "__main__":
    main()
