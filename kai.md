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

Every render this pipeline produces is a two-panel image. The left panel is
always `agentview`; the right one is a **pure side profile** of the table --
a camera on the `+y` axis at gripper height looking horizontally along `-y`, so
the table is edge on, `x` runs across the image and `z` runs up it. agentview
alone resolves the arm's `y` motion but confounds height with reach; the two
together span all three axes.

```bash
RIGHT_VIEW=wrist bash scripts/flux2/prepare_svgd_scene.sh empty   # LIBERO's stock pair
SIDE_CAMERA_ELEVATION_DEG=15 ...                                  # tilt down, see some tabletop
SIDE_CAMERA_MARGIN=0.7 ...                                        # closer, arm fills more frame
```

The framing is derived from the arena (table extent, robot mount) and written to
`manifest.json` as `side_camera`, so every optimiser reopens the scene with the
identical camera. To try framings against a run you already built, without
rebuilding it:

```bash
python -u -B experiments/libero/preview_side_camera.py \
  --run-dir runs/empty_arm_preview --elevation-deg 10
# -> runs/empty_arm_preview/side_preview/preview.png
```

Runs prepared before this camera existed have no `composed_right_view` in their
manifest and are reopened as `[agentview | wrist]`, so their goal images stay
comparable.

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

## 3b. Multi-object table: choose a goal, then run it

The empty-table scene has exactly one goal, `goal_oracle.png`. The multi-object
scene puts a red mug, a plate, a black bowl and a cream cheese box on the same
table and generates a catalogue of candidate goal poses to choose between.

**a. Build the scene**, same as step 3a but with a different scene name:

```bash
bash scripts/flux2/prepare_svgd_scene.sh multi-object
```

**b. Render the goal catalogue** — 16 poses, roughly a minute each:

```bash
python -u -B experiments/libero/prepare_multi_object_goals.py \
  --run-dir runs/multi_object_arm_preview
```

This hovers the arm over each object (straight down, wrist spun 90 degrees, and
rolled 30 degrees) plus four object-free table positions, and writes:

```text
runs/multi_object_arm_preview/goals/contact_sheet.png   all 16, labelled
runs/multi_object_arm_preview/goals/index.json          poses and errors
runs/multi_object_arm_preview/goals/<goal_id>/goal.png  one target image
```

Review the contact sheet. Each tile reports how far the arm actually settled
from the requested pose and how far the objects were nudged; tiles that blew a
tolerance are labelled in red. Re-render a subset after editing a spec with
`--only goal_a,goal_b`.

**c. Run a chosen goal.** `goal.png` is a drop-in for `--goal`, against the same
run directory:

```bash
python -u -B experiments/libero/svgd_endpoint.py \
  --run-dir runs/multi_object_arm_preview \
  --out-dir runs/multi_object_arm_preview/mug_v1/trials/token_cosine \
  --goal runs/multi_object_arm_preview/goals/red_coffee_mug_above/goal.png \
  --goal-latent-source reencode \
  --bounds -0.20 0.09 -0.32 0.32 0.98 1.16 \
  ... same flags as step 3
```

Note the wider `--bounds`: the empty-table sweep searches a narrow slab at
x in [-0.06, -0.04], which does not contain any of the objects.

**Orientation goals are probes, not targets.** `svgd_endpoint.py` optimizes a
3-D Cartesian endpoint and rolls out with the gripper orientation held at its
start value, so no pose it can reach has a rotated gripper. The `_yaw90` and
`_roll30` images are still useful — they ask whether the latent metric responds
to orientation at all — but the search cannot converge to one. `index.json`
records this per goal as `in_position_only_search_family`, and only goals where
that is true are quoted in the script's closing "run one with" hint.

## 3c. The two ImageSTL Visual-SVPIO experiments

[ImageSTL_Visual_SVPIO_Implementation_Spec.md](ImageSTL_Visual_SVPIO_Implementation_Spec.md)
asks for two arms that share a goal-latent objective and differ only in where
`dJ/dU` comes from. Both run over the same multi-object goal catalogue, so their
numbers are directly comparable.

**Experiment 1 — differentiable gradient** (`svgd_endpoint_differentiable.py`).
The spec's Experiment A wants MJX → nvdiffrast → FLUX → JAX VJP. This
environment has no `jax`, no `mujoco.mjx` and no `nvdiffrast`, and the LIBERO
scene is not ported to MJX, so the differentiable path here is the learned
surrogate `theta -> terminal FLUX feature`, differentiated with `torch.autograd`.
The gradient is exact *for the surrogate*, not for the world; MuJoCo is used only
for ground-truth scoring and online refinement.

```bash
bash scripts/flux2/run_multi_object_exp1_differentiable.sh SUITE_NAME
```

**Experiment 2 — estimated latent Jacobian** (`svgd_latent_jacobian.py`).
Spec section 10, unchanged in substance: the simulator is a black box from joint
configuration to image. `J_zq = dz/dq` is built by central differences over
image observations, maintained with rank-one Broyden updates, and contracted
with an autograd `dL/dz` to give `dL/dq = J_zq^T dL/dz`. Particles are short
joint-increment sequences rolled out through `z_{k+1} = z_k + J dq_k`; only the
first increment executes before re-observing.

```bash
bash scripts/flux2/run_multi_object_exp2_latent_jacobian.sh SUITE_NAME
```

`ITERATIONS` means different things in the two: 45 SVGD sweeps in experiment 1,
45 closed-loop control cycles in experiment 2. Experiment 1 is dominated by
MuJoCo rollouts (hours per goal); experiment 2 touches the simulator once per
cycle plus a Jacobian refresh (minutes per goal).

**Joint space buys orientation.** Experiment 2 optimizes the 7 arm joints, not a
3-D endpoint, so unlike the note above it *can* represent the `_roll30` and
`_yaw45` goals — a wrist roll moves the end-effector by zero and the image by a
lot.

**Check the Jacobian before trusting it** (spec section 18.9):

```bash
python -u -B experiments/libero/probe_latent_jacobian.py \
  --run-dir runs/multi_object_arm_preview \
  --editor-ae $FLUX2_AE_MODEL_PATH --flux2-src $FLUX2_SRC
```

On this scene it reports positive held-out cosine alignment at every step size
(peaking near `|dq| = 0.005` rad) but relative error near 1 — the direction is
usable, the magnitude is not. That is why the launcher caps the commanded step
an order of magnitude below the spec's hardware defaults.

## 3d. Whole-trajectory particles (joint setpoints)

`svgd_joint_traj.py` changes what a particle *is*. In `svgd_endpoint.py` a
particle is a 3-D terminal end-effector position and a fixed minimum-jerk helper
invents the path; here the particle is the trajectory itself:

```text
U_i in R^[300, 8]      u_t = [q1 ... q7, gripper]
```

`u_t[:7]` are absolute desired joint positions in radians at control step `t`.
The environment is built with robosuite's `JOINT_POSITION` controller, whose
Panda action vector is exactly 8 wide, so one particle row is one controller
command. The rollout converts each absolute setpoint into the delta the
controller wants: `a_t[:7] = clip((u_t[:7] - q_t) / 0.05, -1, 1)`.

The objective is unchanged — encode the terminal `[agentview | wrist]` render
with the FLUX.2 AE and measure `--latent-distance` against the goal image.

**The gradient is not finite differences.** A particle has 2400 coordinates;
central differences would cost 4800 rollouts per particle per iteration, and one
300-step rollout is ~36 s. Instead it is assembled through the Experiment-2
chain (`latent_jacobian.py`), which costs **no extra rollouts**:

```text
dE/du_t = (dq_s/du_t)^T  J_zq^T  dL/dz

  dL/dz     autograd through the distance                 exact, free
  J_zq      dz/dq by kinematic re-renders over 7 joints    14 renders, no physics
  dq_s/du_t a servo model, set by --credit-mode
```

`J_zq` is refreshed every `--jacobian-refresh-every` iterations and carried by
Broyden updates in between; `jacobian_relative_prediction_error` in
`history.json` is how far to trust it (~1.4 on this scene, matching
`probe_latent_jacobian.py`: direction usable, magnitude not — hence the trust
region `--max-joint-step-rad`).

**Credit assignment is the thing to think about.** The servo lands within
~1e-3 rad of its setpoint, so for a terminal objective the *literally* correct
sensitivity is `--credit-mode last-only` — and it is useless, because it
collapses a 2400-D search onto the last row. The default `uniform` treats `U` as
one trajectory-valued parameter and translates every free row along the same
joint direction; the start anchor and slew projection then reshape that into a
smooth ramp out of the real start pose. If you want the *interior* of the
trajectory to carry its own signal, use `--energy-mode eventually`, which scores
`--waypoints` sampled frames and softmins them (the STL "eventually" operator)
so the scored step moves along the horizon.

After every update each particle is projected back onto executable trajectories:
joint limits, slew rate `--max-setpoint-rate` (0.05 is where the controller
saturates), and the first `--anchor-start-steps` rows pinned to the measured
start configuration.

**Run both empty-table goals, one per GPU:**

```bash
bash scripts/flux2/run_joint_traj_two_goals.sh traj300_v1
```

GPU 0 takes `goal_half.png` (arm halfway across, y ~ 0.00), GPU 1 takes
`goal_oracle.png` (fully across, y ~ -0.22). Defaults are 300 steps, 10
particles, 100 iterations, checkpoints every 25. It detaches; watch it with

```bash
tail -f runs/empty_arm_preview/traj300_v1/launcher.log
tail -f runs/empty_arm_preview/traj300_v1/trials/half_across/backend.log
```

~10-12 h per trial (1000 rollouts x 300 steps), both in parallel.

**Checkpoints and resume.** `--checkpoint-every 25` writes
`checkpoints/iter_XXX.npz` — particles, RNG state, global best, per-particle
Jacobians, rollout counter — plus `checkpoints/latest.json`. `--resume auto` (the
default) picks up the newest one, truncates `history.json` back to it, and
continues, so **re-running the identical launcher command resumes**:

```bash
bash scripts/flux2/run_joint_traj_two_goals.sh traj300_v1          # resumes
ITERATIONS=200 bash scripts/flux2/run_joint_traj_two_goals.sh traj300_v1   # extends
RESUME=none  bash scripts/flux2/run_joint_traj_two_goals.sh traj300_v2     # fresh
```

A single trial by hand, or a rollback to a specific checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 imagewam_python -u -B experiments/libero/svgd_joint_traj.py \
  --run-dir runs/empty_arm_preview \
  --out-dir runs/empty_arm_preview/traj300_v1/trials/half_across \
  --goal runs/empty_arm_preview/goal_half.png \
  --editor-ae $FLUX2_AE_MODEL_PATH --flux2-src $FLUX2_SRC --device cuda:0 \
  --horizon 300 --particles 10 --max-iterations 100 --checkpoint-every 25 \
  --resume runs/empty_arm_preview/traj300_v1/trials/half_across/checkpoints/iter_050.npz \
  --save-all-particles --verbose-evaluations
```

**Smoke test first** (~3 minutes, both goals, one GPU):

```bash
FOREGROUND=true HORIZON=25 PARTICLES=3 ITERATIONS=4 CHECKPOINT_EVERY=2 \
  ANCHOR_START_STEPS=2 RESUME=none GPU_HALF=2 GPU_FULL=2 \
  bash scripts/flux2/run_joint_traj_two_goals.sh smoke
```

**Viewing works unchanged** — the run writes the same `history.json` and
`iter_XXX/traces/particle_NN_base.npz` schema step 4 reads, plus `joint_setpoints`,
`joint_path` and `actions` in each trace. The per-particle marker is where the
*final setpoint* puts the end effector, so the viewer's `track` column becomes
"how far the arm still was from its own last setpoint" — a real convergence
readout, not a controller artefact. Traces are strided by `--trace-stride 3`
(101 of 301 states); export bundles with `--stride 2 --arm-stride 8`, since a
300-step rollout is 6x the endpoint runs.

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
