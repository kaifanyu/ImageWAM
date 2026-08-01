# kai.md — quickstart

Minimal path: install → run the empty-table arm experiment → look at the
trajectories in 3D. Everything below runs from the repo root.

## 1. Install

```bash
uv sync --python 3.11 --extra shared
source .venv/bin/activate
cp .env.example .env.local        # local paths; scripts/ read this automatically
```

CUDA 11.8, Python 3.11, PyTorch 2.7.1.

The SVGD runs need the FLUX.2 autoencoder (`checkpoints/flux2/FLUX.2-dev/ae.safetensors`)
and `third_party/flux2` — see **Model Preparation** in [README.md](README.md).
Weights are not in the repo.

## 2. Docker (instead of step 1)

```bash
cp docker/.env.example docker/.env     # per-machine uid/gid
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm imagewam bash
```

The repo is bind-mounted, so `runs/`, `checkpoints/` and `data/` are the same
bytes as on the host — the ~50 GB of weights is *not* baked into the image and
must already be on the machine.

## 3. Run the empty-table arm experiment

**a. Build the scene once** — writes `runs/empty_arm_preview/` with
`manifest.json` and `goal_oracle.png`:

```bash
bash scripts/flux2/prepare_svgd_scene.sh empty
```

**b. Launch the SVGD sweep** — three trials (`rms`, `cosine`, `token_cosine`),
one per GPU:

```bash
SWEEP_PROFILE=metric_y040 PARTICLES=15 ITERATIONS=30 GPU_IDS="0 1 2" \
  bash scripts/flux2/run_empty_goal_start_cloud_3gpu_sweep.sh empty_start_metric_y040_15p_v1
```

It detaches immediately. Watch it:

```bash
tail -f runs/empty_arm_preview/empty_start_metric_y040_15p_v1/launcher.log
```

~8 hours for 15 particles × 30 iterations on 3 GPUs. Results land in
`runs/empty_arm_preview/<sweep-name>/trials/<trial>/`. The sweep name must not
already exist.

**Single trial on one GPU** (what the sweep runs internally):

```bash
python -u -B experiments/libero/svgd_endpoint.py \
  --run-dir runs/empty_arm_preview \
  --out-dir runs/empty_arm_preview/my_run/trials/token_cosine_y040 \
  --goal runs/empty_arm_preview/goal_oracle.png --goal-latent-source reencode \
  --editor-ae checkpoints/flux2/FLUX.2-dev/ae.safetensors --flux2-src third_party/flux2 \
  --device cuda:0 --particles 15 --iterations 30 \
  --init-mode start-cloud --init-radius 0.005 0.005 0.003 \
  --bounds -0.06 -0.04 -0.32 0.32 1.02 1.045 \
  --latent-views agentview --latent-distance token_cosine \
  --transport svgd --latent-weight 1.0 --repulsion-weight 0.01 \
  --fd-eps 0.01 0.04 0.01 --step-size 0.01 --temperature 0.10 \
  --max-update-norm 0.02 --bandwidth-scale 1.0 \
  --move-steps 40 --settle-steps 8 --controller-gain 15.0 \
  --seed 0 --save-rollout-traces --rollout-trace-mode base --save-all-particles
```

`--save-rollout-traces --save-all-particles` are what the 3D viewer reads —
without them the run is invisible to step 4. Keep `--rollout-trace-mode base`
too: on its own `--save-rollout-traces` means `all`, which also dumps every
finite-difference probe (~3200 rollouts instead of ~500) for no extra value in
the viewer.

## 4. View the trajectories in 3D

```bash
python experiments/libero/svgd_traj3d.py --runs-root runs
# -> http://127.0.0.1:8770/
```

Pick a run, scrub or play the iteration slider, and switch between all particles
/ the best particle / one particle.

Over SSH, forward the port from your own machine first:

```bash
ssh -N -L 8770:127.0.0.1:8770 <user>@<host>
```

then open `http://localhost:8770/`. In VS Code Remote-SSH the PORTS panel does
this for you.

**Committed runs.** `runs/` is gitignored, so runs are shared as small JSON
bundles in `viz_bundles/` (12 MB for all 22 trials, already committed):

```bash
python experiments/libero/svgd_traj3d.py --runs-root viz_bundles
```

That works on a fresh clone with no trace files present. Re-export after new
runs with `--runs-root runs --export-all viz_bundles`.

Details: [docs/svgd_trajectory_viewer.md](docs/svgd_trajectory_viewer.md).
