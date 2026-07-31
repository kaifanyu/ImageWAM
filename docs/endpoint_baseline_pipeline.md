# Empty-table direct-endpoint baseline

This experiment tests one narrow question: can an edited goal image rank full
simulator rollouts that move the robot arm from image-left to image-right?

It skips learned or generated intermediate images. The simulator still executes
each action sequence, but only the terminal render is scored.

```text
fixed simulator state
        |
        +--> start.png --> image editor --> goal_edit.png
        |
        +--> K action trajectories --> K simulator terminal.png files
                                      |
goal_edit.png -------------------------+--> pixel / latent ranking
```

The three stages are serialized. This keeps every rollout tied to the exact same
MuJoCo state and prevents the simulator renderer and FLUX model from competing
for GPU memory.

## 0. Use the prepared environment

Run commands from the repository root. The wrapper loads `.env.local`, including
the released LIBERO checkpoint, dataset stats, FLUX source, and AE weights.

```bash
cd /home/kaifany/project-data/ImageWAM
```

Choose a fresh run directory:

```bash
export RUN_DIR="$(pwd)/runs/empty_arm_test_01"
```

Prefer a new directory for every experiment. If you intentionally rerun
simulation with `FORCE=true`, the sampler removes the previous candidates and
the default edit/score artifacts it owns so a stale goal cannot be scored
against a new start state.

## 1. Capture the start and sample simulator trajectories

This stage is CPU/headless. It creates an empty LIBERO table, stages the end
effector on image-left, freezes the complete MuJoCo state, then independently
rolls out 13 candidates from that state.

```bash
STAGE=simulate \
RUN_DIR="$RUN_DIR" \
NUM_TRAJECTORIES=13 \
SIM_SEED=0 \
TRAJECTORY_SEED=7 \
SAVE_VIDEOS=true \
bash scripts/flux2/run_endpoint_pipeline.sh
```

The first five candidates are controls:

- no-op;
- wrong direction;
- undershoot;
- physical oracle (the requested endpoint);
- overshoot.

The remaining eight candidates independently sample small endpoint errors and
curved Cartesian paths around the goal. The endpoint distribution is narrow on
purpose: several stochastic samples should reach the 3 cm success ball, while
the controls exercise obvious failures. This controlled set should work before
replacing it with a learned trajectory sampler.

Inspect these first:

```text
$RUN_DIR/start.png
$RUN_DIR/goal_oracle.png
$RUN_DIR/candidate_003/rollout.mp4
$RUN_DIR/manifest.json
```

`manifest.json` records the exact start state, target and terminal end-effector
poses, position/height/orientation/gripper errors, exact MuJoCo state-restoration
error, controller/gripper synchronization method, and the small OSMesa
render-noise floor for every candidate.
`reset_state_max_abs` should be zero; `reset_rgb_mae` should remain below 0.25
RGB levels.

LIBERO does not have a built-in predicate for Cartesian robot pose. The BDDL
therefore contains a transparent, always-false sentinel so its stock reward and
done signals never claim success. `physical_success` in the manifest is the
authoritative endpoint check.

### Simulator-only smoke test

Before loading an editor, use the physical oracle render as a temporary target.
This verifies state restoration, rollout, masking, and pixel ranking:

```bash
.venv/bin/python experiments/libero/score_endpoint_candidates.py \
  --run-dir "$RUN_DIR" \
  --goal "$RUN_DIR/goal_oracle.png"
```

The pixel metric should select the oracle candidate, and the no-op/wrong-way
controls should rank worse.

## 2. Edit one terminal goal image

Run this on a host where CUDA is visible. The editor consumes the same 224x448
`[agentview | wrist]` image used by the LIBERO checkpoint and the normalized
proprio vector captured with it.

```bash
STAGE=edit \
RUN_DIR="$RUN_DIR" \
EDITOR_SEED=0 \
EDIT_STEPS=20 \
bash scripts/flux2/run_endpoint_pipeline.sh
```

This writes:

```text
$RUN_DIR/goal_edit.png
$RUN_DIR/goal_edit_compare.png
$RUN_DIR/goal_editor_latent.npy
$RUN_DIR/goal_edit_metadata.json
```

`goal_editor_latent.npy` is the final FLUX denoising token tensor before decode.
Saving it lets the scorer compare simulator endpoints directly with the editor's
own terminal latent, in addition to comparing re-encoded images.
`goal_edit_metadata.json` records the prompt, seed, step count, checkpoint and
AE files, dtype/device, input proprio, output latent shape, and SHA-256 hashes
that bind the goal image and editor latent to the same completed edit run. The
scorer refuses an interrupted or mismatched pair.

Check `goal_edit_compare.png` before trusting any metric. The edit should move
the arm right, update the agent and wrist views consistently, and preserve the
table, fixed camera, lighting, and gripper. If it changes the background or
invents objects, change the prompt or editor seed before continuing.

## 3. Rank sampled terminal images

```bash
STAGE=score \
RUN_DIR="$RUN_DIR" \
LATENT_DEVICE=cpu \
bash scripts/flux2/run_endpoint_pipeline.sh
```

The scorer reports these spaces separately; it does not sum distances with
incompatible scales:

- full-frame RGB MAE, MSE, PSNR, and SSIM;
- change-mask RGB MAE, MSE, and PSNR;
- separate agentview and moving-wrist-view pixel diagnostics;
- deterministic FLUX AE latent RMS and cosine distance;
- final editor-denoising latent RMS and cosine distance;
- optional dynamics/Dyno latent RMS and cosine distance.

The change mask is derived from `abs(goal_edit - start)` and dilated. This is
important because a mostly static empty table can otherwise make the no-op image
look artificially strong.

The released editor expects the full 224x448 `[agentview | wrist]` composite, so
latent distances use that in-distribution shape. Pixel diagnostics also split
the fixed agentview and moving wrist camera: agentview is the clearest physical
sanity check, while wrist metrics test whether the edit updated that camera
consistently. The wrapper keeps FLUX AE scoring on CPU/fp32 by default so repeat
runs use a fixed device and dtype.

The two FLUX metrics use the same coordinate system, but different targets:
`flux_vae_*` targets the edited image after decode/re-encode, while
`editor_final_*` targets the editor's final pre-decode denoising tokens. Treat
them as two views of FLUX AE space, not as independent encoders.

Read the compact report and contact sheets:

```text
$RUN_DIR/summary.md
$RUN_DIR/metrics.csv
$RUN_DIR/metrics.json
$RUN_DIR/edit_mask.png
$RUN_DIR/edit_mask_agentview.png
$RUN_DIR/edit_mask_wrist.png
$RUN_DIR/contact_sheet_pixel.png
$RUN_DIR/contact_sheet_flux_vae.png
$RUN_DIR/contact_sheet_editor_final.png
```

For each ranking metric, `summary.md` shows two selections: one over the whole
diagnostic pool and one over stochastic samples only. Each includes the
image-space value, physical end-effector error, and success flag.
`metrics.json` also reports tie-aware rank correlations with physical error for
both pools. The oracle is a harness control and is never counted as a sampled
success.

## 4. Add the dynamics/Dyno representation

The strongest concrete match for "Dyno VAE" in this repository is the WAN2.2
dynamics VAE, not DINOv2 (DINO is not a VAE). Its encoder lives in
`src/imagewam/models/backbones/wan_video_vae.py` and maps one 224x448 frame to a
`[1, 48, 1, 14, 28]` latent. The expected WAN2.2 VAE weights are not installed,
and there is no local DINO model/checkpoint. The scorer therefore uses an
explicit feature-file contract instead of silently choosing or downloading an
encoder.

Encode the edited goal once:

```text
$RUN_DIR/goal_dynamics_latent.npy
```

Encode every simulator terminal with the same deterministic encoder and
preprocessing:

```text
$RUN_DIR/candidate_000/dynamics_latent.npy
$RUN_DIR/candidate_001/dynamics_latent.npy
...
```

Also save shared encoder provenance as
`$RUN_DIR/goal_dynamics_metadata.json`. The scorer requires this sidecar,
auto-discovers it, verifies every feature hash, and embeds it in `metrics.json`.
Its contract is:

```json
{
  "encoder_name": "WanVideoVAE38",
  "checkpoint": {"path": "...", "sha256": "..."},
  "preprocessing": {"range": "[-1,1]", "layout": "3,T,H,W"},
  "latent_shape": [1, 48, 1, 14, 28],
  "goal_latent_sha256": "...",
  "candidate_latent_sha256": {
    "candidate_000": "...",
    "candidate_001": "..."
  }
}
```

Include every candidate id in `candidate_latent_sha256`; additional fields such
as output layer and dtype are encouraged.

Then run:

```bash
STAGE=score \
RUN_DIR="$RUN_DIR" \
GOAL_DYNAMICS_LATENT="$RUN_DIR/goal_dynamics_latent.npy" \
DYNAMICS_METADATA="$RUN_DIR/goal_dynamics_metadata.json" \
bash scripts/flux2/run_endpoint_pipeline.sh
```

If "Dyno" means DINOv2 or a separate project-specific checkpoint, keep the same
file contract but use that deterministic feature extractor. Confirm the intended
model and checkpoint before comparing numbers.

## 5. Replace the controlled sampler only after the harness passes

The current ImageWAM ActionDiT samples actions from the start image, prompt, and
proprio; it does not condition directly on `goal_edit.png`. A learned-sampler
extension should therefore:

1. hold the simulator seed and start snapshot fixed;
2. call ActionDiT with distinct action seeds;
3. save `K` denormalized action chunks;
4. restore the same snapshot before every open-loop rollout;
5. use the independently edited image only to rank terminal simulator renders.

Keeping `editor_seed`, `action_seed`, and `sim_seed` separate is required for a
meaningful best-of-K experiment.

## Acceptance checks

Treat the baseline as working only when all of these hold:

- restored simulator states are exact (`reset_state_max_abs == 0`) and the
  renderer noise floor stays small (`reset_rgb_mae < 0.25` RGB levels);
- action arrays and terminal poses are not all duplicates;
- the oracle control is within 3 cm of the physical goal while preserving
  height, orientation, and gripper within their configured tolerances;
- no-op and wrong-direction controls rank below the oracle in the oracle smoke test;
- at least one sampled trajectory is physically successful;
- the edited goal is visually plausible;
- a sampled-only image/latent ranking selects a physically successful sampled
  candidate (do not use the oracle to satisfy this check);
- because the default samples are goal-biased, the selected physical error is
  below the sampled median and rank correlation is sensible; success alone is
  not strong evidence that the metric works;
- the result is stable across several simulator, editor, and trajectory seeds.
