# imageSTL: Visual-Latent STL-SVPIO Implementation Specification

## 0. Purpose

This document is an execution specification for a coding agent modifying the `imageSTL` / STL-SVPIO-derived codebase.

The goal is to add **goal-image-driven robot-arm trajectory optimization** while preserving the existing STL-SVPIO machinery:

1. Encode a **goal RGB image** with a frozen FLUX autoencoder.
2. Encode predicted/current RGB images with the same encoder.
3. Define a differentiable latent distance, primarily **spatial token/channel cosine distance**, optionally combined with latent L2.
4. Backpropagate the visual objective into robot controls.
5. Use the existing **SVGD particle update** to optimize multiple candidate robot trajectories.
6. Retain STL robustness terms for geometric/safety/temporal constraints.
7. Support two interchangeable gradient paths:
   - **Experiment A — Simulator exact-gradient path:** MJX dynamics -> differentiable renderer -> FLUX latent objective -> visual-state cotangent -> JAX VJP through MJX -> joint-control gradient.
   - **Experiment B — Real-camera latent-Jacobian path:** camera -> FLUX latent objective -> latent gradient -> online estimate of `d latent / d joint` -> joint gradient -> receding-horizon SVGD/local control.

The two experiments must share the same latent encoder, latent loss, optimizer interface, logging format, and safety objective definitions so their results are directly comparable.

---

# 1. Existing STL-SVPIO behavior that must be preserved

The current STL-SVPIO implementation optimizes a population of complete control trajectories.

For the Panda reference configuration:

- planning horizon: `H = 300`
- control dimension: `m = 8`
- one particle shape: `[300, 8]`
- number of particles: `N = 10`
- simulator timestep: `dt = 0.005 s`
- physical horizon: `1.5 s`
- SVGD/path-integral temperature: `lambda = 0.8`
- STL smooth approximation: log-sum-exp
- STL smoothing temperature/beta: approximately `100`
- SVGD step schedule: exponential, approximately `10.0 -> 0.01` over 300 optimization steps
- final particle selection: `best`
- Panda controls: joint-position actuator setpoints, not torque commands

The current optimization flow is conceptually:

```text
particle U_i [H,m]
      |
      v
MJX rollout
      |
      v
state/trace x_0:H
      |
      v
STL robustness rho(x)
      |
      v
J(U) = -rho
      |
      v
jax.grad(J)
      |
      v
dJ/dU
      |
      v
SVGD attraction + repulsion
      |
      v
updated particle population
```

The existing SVGD update must remain available and continue to support the old STL-only experiments unchanged.

Do **not** replace `_legacy_mppi.py` with a new optimizer. Extend it with an optional gradient-oracle interface while preserving the old JAX-native path.

---

# 2. Target architecture

Implement one shared visual-planning stack with pluggable gradient providers.

```text
                         Goal image
                            |
                            v
                    Frozen FLUX VAE
                            |
                       z_goal cache
                            |
              +-------------+-------------+
              |                           |
              |                           |
      Experiment A                  Experiment B
      SIMULATION                    REAL CAMERA
              |                           |
       SVGD particles                  camera RGB
        U_1 ... U_N                      |
              |                          v
              v                     FLUX encoder
         MJX rollout                      |
              |                        z_current
       body pose trace                    |
              |                          loss
              v                           |
   differentiable renderer                v
              |                        dL/dz
              v                           |
        predicted RGB                 J_zq estimate
              |                      = dz/dq
              v                           |
        FLUX encoder                     v
              |                        dL/dq
              v                           |
      visual latent loss                  |
              |                           |
       dL/d(body pose)                    |
              |                           |
       JAX VJP through MJX                |
              |                           |
              +-------------+-------------+
                            |
                         dJ/dU
                            |
                            v
                   existing SVGD update
                            |
                            v
                   safe best trajectory
                            |
                            v
                 execute/replan closed-loop
```

The shared components are:

- FLUX latent encoder
- latent preprocessing and tokenization
- goal latent caching
- visual distance function
- combined objective bookkeeping
- SVGD particle optimizer
- logging and result serialization
- receding-horizon execution wrapper

Only the method used to obtain `dJ/dU` differs between Experiment A and Experiment B.

---

# 3. Repository assumptions and file layout

The uploaded analysis describes an STL-SVPIO-derived tree containing approximately:

```text
src/stl_svpio/
  algorithms/stl_svpio.py
  _legacy_mppi.py
  specifications.py
  tasks/
  _paper_runners/
```

Assume `imageSTL` preserves this package or an equivalent renamed package.

If the package root has been renamed, apply the file structure below under the actual package root. Do not rename functioning existing modules just to match this document.

Add:

```text
src/stl_svpio/
  visual/
    __init__.py
    config.py
    flux_encoder.py
    latent_loss.py
    objective.py
    gradient_oracle.py
    dlpack_bridge.py
    sim_render_backend.py
    latent_jacobian.py
    local_latent_dynamics.py
    learned_latent_dynamics.py        # optional Phase 3
    diagnostics.py

  _paper_runners/
    run_panda_visual_goal.py
    run_panda_visual_goal_mpc.py
    run_panda_real_visual_servo.py    # only if hardware interface exists

tests/
  visual/
    test_flux_encoder.py
    test_latent_loss.py
    test_svgd_external_gradient.py
    test_sim_visual_vjp.py
    test_latent_jacobian.py
    test_visual_planner_smoke.py

configs/
  visual/
    panda_flux_sim.yaml
    panda_flux_sim_stl.yaml
    panda_flux_real_jacobian.yaml
```

If the project uses a different config system, preserve that system and create equivalent configuration entries rather than introducing another parser.

---

# 4. Core design rule: separate cost evaluation from gradient generation

The current implementation differentiates the cost internally with JAX:

```python
traj = rollout(x0, u_seq)
loss = cost_fn(traj)
grad = jax.grad(single_cost)(u_seq)
```

This works only when the full chain is JAX-differentiable.

FLUX is expected to run in PyTorch. The differentiable renderer is also expected to run in PyTorch. Therefore, do not try to force FLUX into the existing `jax.grad(single_cost)` closure.

Instead, add an optional external gradient oracle.

## 4.1 Required interface

In `_legacy_mppi.py`, define an interface equivalent to:

```python
class GradientOracle(Protocol):
    def value_and_grad_batch(
        self,
        x0,
        particles,      # [N,H,m], JAX array or accepted bridge type
    ) -> tuple[
        Array,          # costs [N]
        Array,          # gradients [N,H,m]
        dict,           # diagnostics
    ]:
        ...
```

Add to controller construction/configuration:

```python
gradient_oracle: Optional[GradientOracle] = None
```

In the existing SVGD step:

```python
if self.gradient_oracle is None:
    costs = ... existing cost evaluation ...
    grads = grad_cost(curr_particles)
else:
    costs, grads, oracle_info = self.gradient_oracle.value_and_grad_batch(
        x0, curr_particles
    )
```

Then preserve the existing score convention:

```python
score = -(1.0 / path_integral_temperature) * grads
```

where `grads = dJ/dU` and **lower cost is better**.

Do not reverse this sign elsewhere.

## 4.2 Backward compatibility requirement

When `gradient_oracle is None`, the old STL-SVPIO code path must produce the same outputs within normal numerical tolerance.

Add a regression test for this before enabling visual planning.

---

# 5. FLUX latent encoder

Create `visual/flux_encoder.py`.

## 5.1 Model loading

Use a frozen Diffusers-compatible FLUX autoencoder.

Configuration must specify either:

```yaml
flux:
  model_path: /absolute/or/repo/model/path
  subfolder: vae
```

or a local model identifier already available in the environment.

Do not hardcode a remote download as the only path.

Load the VAE with an API equivalent to:

```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(model_path, subfolder=subfolder)
vae.eval()
vae.requires_grad_(False)
```

Use `float32` for the first correctness implementation. Add fp16/bf16 only after gradient tests pass.

## 5.2 Image preprocessing

Define one function and use it for both goal and current/predicted images:

```python
preprocess_rgb(image) -> torch.Tensor[B,3,H,W]
```

Requirements:

1. RGB channel order.
2. Float tensor.
3. Values normalized exactly as required by the FLUX VAE; normally image data presented to `AutoencoderKL.encode` should be in `[-1,1]`.
4. Fixed configured resolution.
5. Height and width compatible with the VAE downsampling factor.
6. No random crops or augmentations during optimization.
7. The simulator renderer and physical camera path must use the same crop/resize convention.

Recommended initial resolution:

```yaml
image:
  width: 256
  height: 256
```

Use 256x256 first for tractability. Do not start at 1024x1024.

## 5.3 Deterministic latent

For control, do not sample from the posterior.

Use the posterior mean/mode:

```python
posterior = vae.encode(image).latent_dist
z_raw = posterior.mode()   # or mean if mode unavailable
```

Apply the VAE's configured scaling/shift convention using values read from `vae.config` rather than hardcoded constants.

Implement a helper:

```python
z = normalize_vae_latent(z_raw, vae.config)
```

The same transform must be applied to goal and current latents.

## 5.4 Gradient rule

The VAE parameters remain frozen, but the current/predicted image must **not** be wrapped in `torch.no_grad()`.

Correct:

```python
with torch.no_grad():
    z_goal = encoder(goal_image)

predicted_rgb.requires_grad_(True)
z_current = encoder(predicted_rgb)
loss = latent_loss(z_current, z_goal)
loss.backward()
```

Incorrect:

```python
with torch.no_grad():
    z_current = encoder(predicted_rgb)
```

That would destroy `dL/dimage`.

## 5.5 Goal cache

Encode the goal once per episode:

```python
GoalLatentCache:
    goal_rgb
    goal_latent
    goal_tokens
    metadata
```

Do not repeatedly encode the unchanged goal during every SVGD iteration.

---

# 6. Spatial latent/token representation

FLUX VAE latents are spatial feature maps, not language tokens.

Treat each spatial cell as a feature token.

Given:

```text
z: [B,C,h,w]
```

convert to:

```text
tokens: [B,P,C]
P = h*w
```

with:

```python
tokens = z.permute(0, 2, 3, 1).reshape(B, h*w, C)
```

Optionally pool before flattening to control dimensionality:

```yaml
latent:
  spatial_pool_h: 16
  spatial_pool_w: 16
```

Use `adaptive_avg_pool2d` before tokenization if native latent resolution is larger.

The same pooling must be used for current and goal images.

---

# 7. Visual latent loss

Create `visual/latent_loss.py`.

## 7.1 Per-frame cosine loss

For token `p`:

```math
c_p = \frac{z_p^T z^g_p}{||z_p||_2 ||z^g_p||_2 + \epsilon}
```

Define:

```math
L_cos = 1 - \frac{1}{P}\sum_p c_p
```

Use `eps = 1e-8`.

PyTorch form:

```python
z = F.normalize(tokens, dim=-1, eps=1e-8)
zg = F.normalize(goal_tokens, dim=-1, eps=1e-8)
loss_cos = 1.0 - (z * zg).sum(dim=-1).mean(dim=-1)
```

Output shape for a batch of frames should be `[B]`.

## 7.2 L2 term

Also compute raw latent mean-squared error:

```math
L_l2 = mean((z-z_g)^2)
```

Default combined frame loss:

```math
L_frame = w_cos L_cos + w_l2 L_l2
```

Initial defaults:

```yaml
latent_loss:
  cosine_weight: 1.0
  l2_weight: 0.10
  eps: 1.0e-8
```

These weights are starting values, not theoretical constants. Log both unweighted terms separately.

## 7.3 Optional spatial mask

Support a mask, but do not require it initially.

Interface:

```python
latent_distance(z, z_goal, spatial_mask=None)
```

If a mask exists, resize it to latent spatial resolution and normalize weights so their mean contribution is unchanged.

Do not silently use a simulator segmentation mask in experiments that are meant to compare with the real-camera path unless the real path has an equivalent mask.

## 7.4 Time aggregation: visual EVENTUALLY objective

For a candidate trajectory, evaluate images at selected planning timesteps:

```text
T_vis = [0, visual_stride, 2*visual_stride, ..., H-1]
```

Default:

```yaml
visual:
  stride: 10
```

With `H=300`, this yields approximately 30 visual evaluations per particle instead of 300.

For frame losses `d_t`, define a normalized soft minimum:

```math
J_visual = -tau_v * [logsumexp(-d_t/tau_v) - log(T)]
```

The `-log(T)` normalization prevents the cost from changing solely because the number of sampled timesteps changes.

Initial default:

```yaml
visual:
  time_softmin_temperature: 0.05
```

This behaves like STL `Eventually`: the gradient concentrates on the predicted timestep whose image is closest to the goal while remaining differentiable.

Also support:

```yaml
visual:
  temporal_mode: softmin   # default
  # alternatives: terminal, mean
```

For `terminal`, use only the final rendered frame.

---

# 8. Combined image + STL objective

Create `visual/objective.py`.

The primary objective is:

```math
J_total(U) =
  w_v J_visual(U)
  + w_task J_task_stl(U)
  + w_safe J_safety(U)
  + w_smooth J_smooth(U)
  + w_vel J_command_velocity(U)
```

All terms must obey the same convention: **smaller cost is better**.

## 8.1 Visual term

```math
J_visual = softmin_t D(E(I_t), E(I_goal))
```

## 8.2 Optional existing STL task term

If an existing STL task robustness is retained:

```math
J_task_stl = -rho_task
```

This preserves the old behavior.

If the goal is entirely visual, set `w_task = 0`.

## 8.3 Safety STL term

Safety should be represented by a dedicated STL robustness `rho_safe`, for example collision avoidance / forbidden regions / joint-safe predicates where available.

Use a soft hinge penalty:

```math
J_safety = tau_s * softplus((margin_safe-rho_safe)/tau_s)
```

Initial values:

```yaml
objective:
  safety_margin: 0.0
  safety_temperature: 0.02
```

The safety term should be zero-ish when comfortably safe and rise when robustness approaches or crosses the required margin.

For hard actuator bounds, continue using the existing projection/clipping after the SVGD update.

## 8.4 Smoothness term

The current Panda optimizer can produce discontinuous 5 ms position setpoints because the physics/PD controller is what smooths them.

Add an explicit smoothness regularizer:

```math
J_smooth = \frac{1}{H-1}\sum_{t=0}^{H-2} ||u_{t+1}-u_t||_2^2
```

## 8.5 Command acceleration term

Optional second-difference penalty:

```math
J_command_velocity = \frac{1}{H-2}\sum_t ||u_{t+2}-2u_{t+1}+u_t||_2^2
```

Use this only if needed.

## 8.6 Initial objective weights

Start with:

```yaml
objective:
  visual_weight: 1.0
  task_stl_weight: 0.0
  safety_weight: 0.25
  smooth_weight: 0.001
  command_accel_weight: 0.0001
```

For the mixed visual + original-task ablation:

```yaml
objective:
  visual_weight: 1.0
  task_stl_weight: 0.10
  safety_weight: 0.25
```

Log each raw loss and each weighted contribution separately.

---

# 9. Experiment A: MJX -> differentiable renderer -> FLUX -> JAX VJP

This experiment proves the complete differentiable visual-control chain in simulation.

## 9.1 Critical constraint

The standard MuJoCo/OpenGL renderer is not differentiable with respect to body poses.

Do not implement:

```text
MJX -> ordinary MuJoCo render -> FLUX -> backward
```

and claim gradients reach the robot.

They will not.

Use a differentiable rasterizer.

Recommended implementation: a PyTorch-compatible differentiable triangle rasterizer such as `nvdiffrast`.

The design below deliberately keeps MJX in JAX and the renderer/FLUX stack in PyTorch.

## 9.2 Cross-framework gradient strategy

Do not attempt to differentiate directly across JAX <-> PyTorch.

Use an explicit cotangent bridge:

```text
U --JAX--> render_state
              |
              | detach / DLPack
              v
         PyTorch renderer
              |
              v
          FLUX + loss
              |
              v
        dL/d(render_state)
              |
              | DLPack
              v
        JAX VJP pullback
              |
              v
             dL/dU
```

This is mathematically the chain rule:

```math
dL/dU = (d render_state/dU)^T * dL/d(render_state)
```

## 9.3 Define the JAX render state

Do not pass only `qpos` unless the PyTorch side also implements the robot kinematics.

Prefer to expose body/world transforms already computed by MJX.

Create a JAX trace extractor returning, for every visual timestep:

```python
RenderState = {
    "body_pos":  [T_vis, B_render, 3],
    "body_quat": [T_vis, B_render, 4],
}
```

where `B_render` includes all robot links and movable task objects that appear in the image.

If MJX exposes geometry transforms directly and their gradients are stable, geometry transforms may be used instead:

```python
geom_pos  [T_vis, G, 3]
geom_quat [T_vis, G, 4]
```

Pick one representation and use it consistently.

Do not pass contact impulses, constraints, or irrelevant state into the visual renderer.

## 9.4 Static scene cache

At runner initialization, parse/copy all non-dynamic rendering information from the MJCF:

- triangle mesh vertices
- triangle indices
- geom-to-body mapping
- geom local transform relative to body
- material/base color
- static table/floor geometry
- camera intrinsics
- camera extrinsics
- near/far plane

Create:

```python
DifferentiableSceneCache
```

This cache must not be reconstructed during each SVGD iteration.

If a MuJoCo geom primitive has no triangle mesh, convert it once at initialization to a sufficiently tessellated triangle mesh.

The differentiable renderer only needs visual equivalence sufficient for the FLUX latent objective; it does not participate in physics.

## 9.5 Camera calibration

Use a fixed camera for Experiment A.

Save exact camera parameters in config:

```yaml
camera:
  width: 256
  height: 256
  fx: ...
  fy: ...
  cx: ...
  cy: ...
  position: [...]
  quaternion_wxyz: [...]
  near: 0.01
  far: 10.0
```

The ordinary reference renderer used for videos and the differentiable renderer must be calibrated to approximately the same viewpoint.

Add a one-time validation image showing both renderers from the same state. They need not be pixel-identical, but the robot/object positions must align visually.

## 9.6 Torch rendering function

Create:

```python
render_rgb(render_state_torch, scene_cache, camera) -> rgb
```

Expected output:

```text
rgb: [B_frames,3,H,W]
range: [0,1]
```

The input body positions/quaternions must be leaf tensors with `requires_grad=True`.

The renderer must preserve gradients to those tensors.

After rendering, convert to FLUX input normalization in `flux_encoder.preprocess_rgb`.

## 9.7 Simulator visual-gradient oracle

Create `SimVisualGradientOracle` in `visual/gradient_oracle.py`.

Per particle `U_i`:

### Step A — JAX rollout plus VJP closure

Use `jax.vjp`:

```python
render_state_i, pullback_i = jax.vjp(
    lambda u: rollout_to_render_state(x0, u, visual_indices),
    U_i,
)
```

Do not keep a full Python list of unbounded VJP closures across all optimization iterations. They are only needed for the current batch/step.

If batching `jax.vjp` over particles is awkward, implement per-particle first for correctness, then vectorize.

### Step B — move render state to Torch

Use DLPack when devices/framework versions permit zero-copy:

```python
torch.utils.dlpack.from_dlpack(...)
```

Otherwise use an explicit device copy as a correctness fallback.

Create new Torch leaves:

```python
body_pos = body_pos.detach().requires_grad_(True)
body_quat = body_quat.detach().requires_grad_(True)
```

Normalize quaternions inside the renderer using a differentiable normalization.

### Step C — render all selected visual frames

Batch frames where possible.

### Step D — encode with frozen FLUX VAE

Compute current/predicted tokens.

Goal tokens are cached.

### Step E — visual loss

Compute frame latent distances and temporal softmin.

### Step F — Torch gradient

Use:

```python
grad_pos, grad_quat = torch.autograd.grad(
    visual_loss,
    [body_pos, body_quat],
    retain_graph=False,
    create_graph=False,
)
```

Do not call `.backward()` on global model state if `autograd.grad` is sufficient.

### Step G — convert cotangent to JAX

Convert `(grad_pos, grad_quat)` back to arrays matching the JAX `RenderState` pytree.

### Step H — JAX pullback

Apply:

```python
(grad_u_visual,) = pullback_i(render_state_cotangent)
```

Now:

```text
grad_u_visual shape = [H,m]
```

### Step I — combine with native JAX terms

Compute STL/safety/smoothness terms in JAX.

For example:

```python
cost_native, grad_native = jax.value_and_grad(native_cost_fn)(U_i)
```

Then:

```python
cost_total = (
    w_visual * cost_visual
    + cost_native
)

grad_total = (
    w_visual * grad_u_visual
    + grad_native
)
```

Return `[N]` costs and `[N,H,m]` gradients to the existing SVGD step.

## 9.8 Quaternion gradient handling

Quaternions are constrained.

On the PyTorch side, always normalize:

```python
q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
```

The JAX pullback receives the gradient with respect to the unnormalized MJX quaternion values through this normalization.

Do not manually project quaternion gradients unless gradient checks show a problem.

## 9.9 Visual frame stride

Start with:

```yaml
visual:
  stride: 10
```

For a 300-step horizon, render about 30 frames per particle.

With 10 particles, one SVGD optimization iteration therefore processes about 300 rendered images rather than 3000.

If still too expensive, test stride `20` before lowering the number of particles.

## 9.10 Closed-loop runner

The original paper runner executes the entire 300-step plan open-loop.

The visual experiment should support receding horizon.

Create `run_panda_visual_goal_mpc.py`.

Default execution policy:

```yaml
mpc:
  replan: true
  execute_steps: 5
  warm_start: true
```

Loop:

```python
while not done:
    observe current simulator state
    optimize particles for horizon H
    select best U_star
    execute first K=5 controls
    shift/warm-start particles by K steps
    repeat
```

For the first correctness run, also support:

```yaml
mpc:
  replan: false
```

so results can be compared with the original open-loop runner.

---

# 10. Experiment B: real-camera FLUX latent Jacobian

This experiment removes the need for a differentiable renderer and differentiable real-world physics.

The core identity is:

```math
dL/dq = (dz/dq)^T * dL/dz
```

where:

- `dL/dz` comes from frozen FLUX autograd
- `dz/dq` is estimated from physical robot motions and camera observations

## 10.1 Real-camera loop

At each control update:

```text
capture RGB
   |
FLUX encoder
   |
z_current
   |
latent distance to z_goal
   |
dL/dz
   |
online J_zq
   |
dL/dq = J_zq^T dL/dz
   |
SVGD/local trajectory update
   |
safe joint command
   |
robot
   |
repeat
```

This must run closed-loop.

Do not execute a long real-world sequence open-loop based only on a local Jacobian.

## 10.2 Descriptor dimensionality

Do not construct a Jacobian against the raw 256x256 RGB image.

Use the pooled FLUX latent descriptor.

Example:

```text
z [C,h,w]
 -> adaptive pool to [C,16,16]
 -> flatten to d-dimensional descriptor
```

If `C=16`, `d=4096`.
If `C=32`, `d=8192`.

A Jacobian of shape `[d,7]` is tractable.

The gripper may be excluded from the visual Jacobian initially and handled separately unless the task explicitly requires gripper opening/closing.

## 10.3 Initial finite-difference latent Jacobian

At safe configuration `q`:

```math
J[:,j] ~= [z(q+delta e_j)-z(q-delta e_j)]/(2 delta)
```

Initial default:

```yaml
jacobian:
  finite_difference_delta_rad: 0.005
```

That is approximately 0.29 degrees.

Before each perturbation:

- check joint limits
- check a configurable safety margin
- reject perturbation if unsafe
- if central difference is impossible near a limit, use a one-sided difference and log it

For 7 arm joints, central difference requires up to 14 image captures.

Return the robot to the nominal `q` after each perturbation, or execute the perturbations in an order that returns exactly to `q` before control begins.

## 10.4 Descriptor consistency

The goal/current/perturbation observations must use:

- same camera
- same crop
- same resize
- same exposure policy where possible
- same FLUX encoder
- same latent scaling
- same pooling

Freeze auto-exposure if the camera allows it. Large exposure changes can dominate reconstruction-oriented VAE features.

## 10.5 Online Broyden update

Do not repeat 14 finite-difference perturbations at every control cycle.

After an initial Jacobian estimate, update it online from normal motion.

Given:

```math
Delta q = q_{t+1}-q_t
Delta z = z_{t+1}-z_t
```

use the rank-one Broyden update:

```math
J_{t+1} = J_t + ((Delta z - J_t Delta q) Delta q^T)/(Delta q^T Delta q + eps)
```

Use:

```yaml
jacobian:
  broyden_eps: 1.0e-8
  max_condition_number: 1.0e6
  refresh_every_steps: 25
```

Every `refresh_every_steps`, or whenever Jacobian diagnostics fail, re-estimate selected columns with physical finite differences.

## 10.6 Jacobian confidence diagnostics

For each real transition compute prediction error:

```math
r_t = ||Delta z - J Delta q|| / (||Delta z|| + eps)
```

Log `r_t`.

If:

```yaml
jacobian:
  max_relative_prediction_error: 0.5
```

is exceeded for 3 consecutive steps, trigger a Jacobian refresh.

Do not continue large control updates with a clearly invalid local model.

## 10.7 Direct latent-Jacobian servo baseline

Implement a non-SVGD baseline first:

```python
g_z = dL_dz(z_current, z_goal)
g_q = J_zq.T @ g_z

dq = -eta * g_q
```

Normalize/clip:

```python
dq = clip_norm(dq, max_joint_step_norm)
dq = elementwise_clip(dq, -max_joint_step, +max_joint_step)
q_cmd = project_joint_limits(q + dq)
```

Initial conservative defaults:

```yaml
real_control:
  servo_rate_hz: 5.0
  gradient_step_size: 0.05
  max_joint_step_rad: 0.02
  max_joint_step_norm_rad: 0.04
  joint_limit_margin_rad: 0.05
```

These are software defaults only. The hardware driver must remain responsible for its own velocity, torque, collision, and emergency-stop limits.

## 10.8 SVGD with a local linear latent model

After the direct baseline works, use the same SVGD machinery.

Define short-horizon joint increments:

```text
DeltaQ_i [H_real, n_joints]
```

Initial default:

```yaml
real_planner:
  horizon: 10
  num_particles: 10
  execute_steps: 1
```

Use a local latent transition:

```math
z_{k+1} = z_k + J_zq Delta q_k
```

Optionally include a simple configuration update:

```math
q_{k+1} = q_k + Delta q_k
```

and update collision/joint-limit penalties from `q_k` if a kinematic safety model is available.

The goal cost is:

```math
J_visual = softmin_k D(z_k, z_goal)
```

Because this local model is differentiable, `dJ/dDeltaQ` can be computed entirely in PyTorch or JAX.

To reuse the existing optimizer, return the resulting cost and gradient through the same `GradientOracle` interface.

Execute only the first increment, capture a new image, update the Jacobian, shift/warm-start particles, and replan.

---

# 11. Optional Phase 3: learned latent dynamics model

This is not required for the first two experiments, but the architecture should not block it.

Create `visual/learned_latent_dynamics.py` with interface:

```python
predict_next(z_t, q_t, u_t) -> z_next
```

Recommended residual model:

```math
z_{t+1} = z_t + f_theta(z_t, q_t, u_t)
```

Train from tuples:

```text
(z_t, q_t, u_t, z_{t+1})
```

collected from either simulator rollouts or physical robot data.

Loss:

```math
L_dyn = MSE(z_pred, z_next) + beta * cosine_distance(z_pred, z_next)
```

The learned model can later replace the constant local Jacobian for longer-horizon real-world SVGD.

Do not start here. Establish Experiment A and the direct Jacobian baseline first.

---

# 12. DLPack bridge utilities

Create `visual/dlpack_bridge.py`.

Provide explicit functions:

```python
jax_to_torch(x, requires_grad=False)
torch_to_jax(x)
```

Requirements:

- preserve shape
- preserve dtype where possible
- preserve GPU placement when supported
- never silently convert a gradient tensor to CPU unless configured
- call `.contiguous()` where required
- document framework-version constraints

Add a unit test moving a random tensor JAX -> Torch -> JAX and verifying values.

Do not expect gradients to cross DLPack automatically. DLPack is only the data transport. Gradient propagation is accomplished manually through the Torch cotangent + JAX VJP procedure.

---

# 13. SVGD integration details

The existing Stein update remains:

```math
phi_i = 1/N * sum_j [k(U_j,U_i) s_j + alpha grad_{U_j} k(U_j,U_i)]
```

with:

```math
s_j = -(1/lambda) dJ(U_j)/dU_j
```

Do not alter this convention for visual planning.

The existing:

- RBF kernel
- dimension normalization
- median bandwidth heuristic
- repulsion term
- post-update control clipping
- particle selection modes
- annealed step size

should remain unchanged unless a separate ablation explicitly tests them.

## 13.1 Visual planner defaults

Start Experiment A with the same SVGD settings as Panda unless memory forces a reduction:

```yaml
svgd:
  num_particles: 10
  num_iterations: 300
  path_integral_temperature: 0.8
  step_size_initial: 10.0
  step_size_final: 0.01
  step_schedule: exp
  selection: best
```

For development/smoke testing use:

```yaml
svgd:
  num_particles: 2
  num_iterations: 3
```

Never evaluate research behavior from the smoke-test config.

## 13.2 NaN handling

The current code zeroes NaN/Inf gradients.

For visual experiments add diagnostics before sanitization:

```text
num_nan_grad_particles
num_inf_grad_particles
visual_grad_norm
native_grad_norm
```

Keep compatibility with existing behavior, but emit a warning if any particle gradient is sanitized.

---

# 14. Control parameterization

## 14.1 Simulation

Preserve the Panda semantics:

```text
U_t = absolute joint position actuator setpoints
```

Shape:

```text
[H,8]
```

Use the existing actuator `ctrl_range` and projection.

## 14.2 Physical latent-Jacobian path

Use arm joint increments for the local model:

```text
Delta q_t
```

Do not directly optimize torques.

Create a small adapter:

```python
class ActionParameterization:
    def optimizer_to_command(...)
    def project(...)
```

Implement:

- `AbsolutePositionAction` for simulator Panda
- `JointDeltaAction` for real latent-Jacobian control

The SVGD optimizer should remain agnostic to the physical interpretation of the action vector.

---

# 15. Goal image handling

The runner must support two goal-image sources.

## 15.1 Simulator-generated goal

For controlled evaluation:

1. Define a known target simulator configuration/state.
2. Render one RGB goal image from the same fixed camera.
3. Save the target state separately for evaluation only.
4. The optimizer receives **only the goal image**, not target joint values or target object coordinates.

The hidden simulator target state can be used after planning to calculate ground-truth metrics.

## 15.2 User-provided goal image

Load a PNG/JPEG specified by config:

```yaml
goal:
  image_path: path/to/goal.png
```

Validate:

- readable
- RGB conversion successful
- same crop/resize preprocessing is applied

For a real robot, the preferred initial evaluation is a goal image captured from the **same fixed physical camera** in the same workspace.

Do not begin with an arbitrary internet image or different camera viewpoint; that would confound control with viewpoint/domain mismatch.

---

# 16. Evaluation metrics

Every episode must report at least:

```json
{
  "success": true,
  "final_visual_cosine_loss": 0.0,
  "final_visual_l2_loss": 0.0,
  "best_visual_loss": 0.0,
  "initial_visual_loss": 0.0,
  "visual_loss_reduction_fraction": 0.0,
  "final_task_stl_robustness": 0.0,
  "final_safety_stl_robustness": 0.0,
  "true_stl_satisfied": true,
  "planning_runtime_seconds": 0.0,
  "num_replans": 0,
  "num_nan_gradient_events": 0
}
```

For simulator-generated goals also record:

- final joint distance to hidden target configuration
- final end-effector distance to hidden target pose if meaningful
- final object pose error if manipulation is involved

These hidden-state metrics are evaluation-only and must not enter the visual objective.

---

# 17. Logging required during each SVGD optimization

At every optimization iteration log:

- iteration index
- best total cost
- mean total cost
- best visual cost
- mean visual cost
- best task STL term
- best safety term
- best smoothness term
- minimum STL robustness across particles
- gradient norm per particle
- visual gradient norm per particle
- native JAX gradient norm per particle
- kernel bandwidth `h`
- mean attraction norm
- mean repulsion norm
- number of NaN/Inf gradients
- best particle index

Every configurable number of iterations save preview images for the current best particle:

```yaml
logging:
  save_preview_every: 25
```

Preview should include:

- goal image
- best predicted closest-to-goal frame
- initial frame
- optionally a horizontal strip of selected trajectory frames

Do not save every rendered frame every iteration; that will dominate I/O.

---

# 18. Tests that must pass before running long experiments

## 18.1 FLUX encoder test

`test_flux_encoder.py`

Assertions:

1. same image encoded twice produces same deterministic latent within tolerance
2. output finite
3. current-image tensor receives finite gradient through the encoder
4. VAE parameters receive no gradients / remain frozen

## 18.2 Latent loss test

`test_latent_loss.py`

Assertions:

```text
D(z,z) approximately 0 for cosine component
D(z,z) < D(z,random)
```

Gradient must be finite and nonzero for non-identical latents.

## 18.3 DLPack round trip

Verify values and shapes across JAX/Torch conversion.

## 18.4 SVGD external gradient regression

Construct a quadratic toy objective:

```math
J(U)=0.5 ||U-U_goal||^2
```

Implement it once using existing internal JAX differentiation and once using the new external gradient oracle.

With repulsion disabled and identical particles, one update must agree numerically.

This isolates the optimizer integration from vision.

## 18.5 Differentiable renderer gradient test

Place one simple rendered object in front of the camera.

Autograd gradient:

```math
d mean(image)/d x_object
```

must have nonzero finite value.

Compare a directional derivative against finite differences.

Target relative error for this local test:

```text
< 5%
```

where rasterization discontinuities are not crossed.

## 18.6 Full visual VJP gradient check

Use a very short simulation horizon:

```text
H=2 or H=3
N=1
```

Compute:

```text
g_autodiff = SimVisualGradientOracle dJ/dU
```

Then choose a random normalized direction `v` and finite-difference the scalar cost:

```math
g_fd = [J(U+eps v)-J(U-eps v)]/(2eps)
```

Compare with:

```math
g_proj = <g_autodiff, v>
```

Require:

- same sign
- relative error below 10% initially
- tighten to 5% if renderer behavior permits

This is the most important correctness test in Experiment A.

## 18.7 Visual planning smoke test

Simple scene, no contact:

- goal differs by reachable arm pose
- `N=2`
- 3-10 SVGD iterations

Assert best visual loss decreases.

## 18.8 STL-only regression

Run a small existing STL-only test before/after changes.

External visual code must not change the result when disabled.

## 18.9 Latent Jacobian finite-difference test in simulation

Before real hardware, test `LatentJacobianEstimator` using simulator images as if they came from a physical camera.

Estimate `J_zq` via image perturbations.

For a held-out small `Delta q`, compare:

```math
Delta z_pred = J Delta q
```

against observed:

```math
Delta z_true = z(q+Delta q)-z(q)
```

Log relative error and cosine alignment.

---

# 19. Experiment sequence

Do not jump directly to the full mixed objective.

Run experiments in this order.

## Stage 0 — baseline regression

Run existing STL-SVPIO unchanged.

Save reference metrics.

Pass/fail criterion:

- no regression caused by gradient-oracle refactor

## Stage 1 — FLUX latent sanity

No control optimization.

Generate a sequence of simulator configurations interpolating from start to a known goal.

Render and encode each image.

Plot/log latent distance to goal against interpolation fraction.

Expected result:

- distance generally decreases as pose approaches goal

If it does not, do not proceed to SVGD. Investigate crop, encoder choice, background dominance, or representation.

## Stage 2 — renderer + FLUX gradient check

Run the full directional derivative test.

Pass criterion:

- correct sign
- acceptable finite-difference agreement

## Stage 3 — one-particle visual gradient descent

Disable Stein repulsion and use `N=1`.

This isolates the visual gradient.

Success criterion:

- latent goal loss decreases reliably

## Stage 4 — visual SVGD

Restore `N=10` and repulsion.

Compare against `N=1` and 10 independent gradient-descent restarts.

Metrics:

- best visual loss
- success rate
- runtime
- particle diversity

This separates the benefit of exact visual gradients from the benefit of Stein coupling.

## Stage 5 — visual + STL safety

Enable safety robustness penalty.

Success criterion:

- visual progress remains
- safety violation rate decreases / remains zero

## Stage 6 — receding-horizon simulation

Execute `K=5`, replan, warm-start.

Compare with open-loop.

## Stage 7 — latent Jacobian in simulation

Pretend simulator camera is non-differentiable.

Use only finite-difference image observations to estimate `J_zq`.

Compare direct Jacobian servo and local-linear SVGD against Experiment A.

## Stage 8 — physical robot, direct Jacobian servo

Use small safe motions only.

Do not enable long-horizon SVGD until direct `-J^T grad_z` control reliably decreases the latent loss.

## Stage 9 — physical robot, short-horizon SVGD

Use local linear latent dynamics, horizon 10, execute one increment, replan.

## Stage 10 — optional learned latent dynamics

Only after sufficient transition data exists.

---

# 20. Ablations to implement

Every experimental result should support these ablations through config, not code edits.

## Representation

```text
FLUX cosine only
FLUX L2 only
FLUX cosine + L2
```

If DINO/SigLIP are later added, they should plug into the same encoder interface.

## Temporal aggregation

```text
terminal
mean
softmin
```

## Optimizer

```text
N=1 gradient descent
N=10 independent gradient descent
N=10 SVGD
MPPI baseline if compatible
```

## Objective

```text
visual only
STL only
visual + STL task
visual + STL safety
visual + STL task + STL safety
```

## Execution

```text
open-loop
receding horizon
```

## Gradient source

```text
exact sim visual VJP
finite-difference latent Jacobian
Broyden-updated latent Jacobian
learned latent dynamics, later
```

---

# 21. Suggested config: `panda_flux_sim.yaml`

```yaml
seed: 0

planner:
  horizon: 300
  dt: 0.005

svgd:
  num_particles: 10
  num_iterations: 300
  path_integral_temperature: 0.8
  stein_repulsion_coef: 1.0
  step_schedule: exp
  step_size_initial: 10.0
  step_size_final: 0.01
  selection: best

flux:
  model_path: /path/to/local/flux/model
  subfolder: vae
  dtype: float32
  deterministic_latent: true

image:
  width: 256
  height: 256

latent:
  spatial_pool_h: 16
  spatial_pool_w: 16

latent_loss:
  cosine_weight: 1.0
  l2_weight: 0.10
  eps: 1.0e-8

visual:
  temporal_mode: softmin
  stride: 10
  time_softmin_temperature: 0.05

objective:
  visual_weight: 1.0
  task_stl_weight: 0.0
  safety_weight: 0.0
  safety_margin: 0.0
  safety_temperature: 0.02
  smooth_weight: 0.001
  command_accel_weight: 0.0001

mpc:
  replan: false
  execute_steps: 5
  warm_start: true

logging:
  save_preview_every: 25
  save_particle_costs: true
  fail_on_nan: false
```

---

# 22. Suggested config: `panda_flux_sim_stl.yaml`

Same as above, except:

```yaml
objective:
  visual_weight: 1.0
  task_stl_weight: 0.0
  safety_weight: 0.25
  safety_margin: 0.0
  safety_temperature: 0.02
  smooth_weight: 0.001
  command_accel_weight: 0.0001

mpc:
  replan: true
  execute_steps: 5
  warm_start: true
```

Use the task's real safety predicates rather than fabricating image-space collision penalties.

---

# 23. Suggested config: `panda_flux_real_jacobian.yaml`

```yaml
seed: 0

flux:
  model_path: /path/to/local/flux/model
  subfolder: vae
  dtype: float32
  deterministic_latent: true

image:
  width: 256
  height: 256

latent:
  spatial_pool_h: 16
  spatial_pool_w: 16

latent_loss:
  cosine_weight: 1.0
  l2_weight: 0.10
  eps: 1.0e-8

jacobian:
  finite_difference_delta_rad: 0.005
  broyden_eps: 1.0e-8
  refresh_every_steps: 25
  max_relative_prediction_error: 0.5
  bad_prediction_patience: 3

real_control:
  servo_rate_hz: 5.0
  gradient_step_size: 0.05
  max_joint_step_rad: 0.02
  max_joint_step_norm_rad: 0.04
  joint_limit_margin_rad: 0.05

real_planner:
  mode: svgd_local_linear
  horizon: 10
  num_particles: 10
  num_iterations: 20
  execute_steps: 1
  path_integral_temperature: 0.8
  stein_repulsion_coef: 1.0
  warm_start: true

objective:
  visual_weight: 1.0
  safety_weight: 0.25
  smooth_weight: 0.001
```

Hardware-specific speed/torque/collision limits are not replaced by these planner settings.

---

# 24. Runner CLI requirements

`run_panda_visual_goal.py` should expose at minimum:

```text
--config
--goal-image
--seed
--num-particles
--svgd-iters
--visual-stride
--open-loop / --mpc
--execute-steps
--visual-only
--enable-safety-stl
--save-dir
```

The CLI should override config values without editing source.

Example:

```bash
python -m stl_svpio._paper_runners.run_panda_visual_goal \
  --config configs/visual/panda_flux_sim.yaml \
  --goal-image assets/goals/panda_goal_01.png \
  --save-dir results/visual/panda_goal_01
```

Smoke test:

```bash
python -m stl_svpio._paper_runners.run_panda_visual_goal \
  --config configs/visual/panda_flux_sim.yaml \
  --goal-image assets/goals/panda_goal_01.png \
  --num-particles 2 \
  --svgd-iters 3 \
  --visual-stride 50 \
  --save-dir results/visual/smoke
```

---

# 25. Output files

Each experiment directory should contain:

```text
config_resolved.yaml
metrics.json
iteration_metrics.jsonl
selected_controls.npy
particle_final_costs.npy
trajectory_state.npz
goal.png
initial.png
best_predicted.png
preview_iter_0000.png
preview_iter_0025.png
...
run.log
```

For MPC runs also save:

```text
replan_000/
replan_001/
...
```

or an equivalent structured representation.

For real-camera runs additionally save:

```text
jacobian_initial.npy
jacobian_latest.npy
jacobian_diagnostics.jsonl
camera_frames/
command_log.jsonl
```

---

# 26. Performance implementation notes

## 26.1 Cache everything static

Cache:

- FLUX model
- goal latent
- scene meshes
- camera matrices
- visual timestep indices
- static renderer buffers
- JAX-compiled rollout functions

## 26.2 Batch FLUX encoding

For simulation, combine visual frames across particles when memory permits:

```text
[N,T_vis,3,H,W] -> [N*T_vis,3,H,W]
```

Encode in chunks if necessary.

Config:

```yaml
flux:
  encode_batch_size: 32
```

Do not use a Python FLUX forward pass for each individual frame.

## 26.3 Start float32

Correct gradients are more important than throughput initially.

Only after all gradient checks pass:

- benchmark bf16
- benchmark fp16
- verify gradient direction remains consistent

## 26.4 Visual stride before particle reduction

If memory/runtime is too high, reduce in this order:

1. increase visual stride from 10 -> 20
2. reduce image resolution if needed
3. chunk FLUX batch
4. reduce SVGD iterations for debugging
5. reduce particles last for final research comparison

Particle diversity is part of the method being evaluated.

---

# 27. Important conceptual checks for the coding agent

## 27.1 FLUX is not the robot dynamics model

FLUX supplies:

```math
dL/dI
```

or equivalently the latent objective gradient.

It does not supply:

```math
dI/dq
```

Experiment A obtains the missing derivative from differentiable rendering + MJX VJP.

Experiment B obtains it from an estimated latent Jacobian.

Do not report that the camera itself is differentiable.

## 27.2 DLPack does not create a shared autograd graph

It only transports tensor memory/data.

The cross-framework derivative is explicitly:

1. Torch computes a cotangent wrt render state.
2. JAX VJP pulls that cotangent back to controls.

## 27.3 The visual objective is not guaranteed to be task-semantic

FLUX VAE features may respond to:

- texture
- lighting
- background
- shadows
- viewpoint

Therefore Stage 1 latent-distance sanity checks are mandatory.

If background dominates, add one of the following as a controlled ablation:

- workspace crop
- latent spatial mask
- foreground segmentation mask
- semantic encoder such as DINO/SigLIP through the same encoder interface

Do not silently change representation mid-experiment.

## 27.4 Goal images should initially share viewpoint

Start with fixed-camera same-viewpoint goals.

Cross-view goal matching is a separate research problem.

## 27.5 Contact gradients remain a weakness

The existing MJX STL-SVPIO analysis already notes that contact gradients can be spiky and NaN/Inf values may occur.

The visual objective does not remove this problem.

Log visual and MJX gradient norms separately so failures can be attributed correctly.

---

# 28. Acceptance criteria

The implementation is not complete until all of the following are true.

## Software correctness

- [ ] Existing STL-only runner still works.
- [ ] External gradient oracle can reproduce a toy internal gradient.
- [ ] FLUX current-image gradient is nonzero and finite.
- [ ] Goal latent is cached and deterministic.
- [ ] Spatial cosine + L2 loss has unit tests.
- [ ] Differentiable renderer produces nonzero pose gradients.
- [ ] Torch visual cotangent -> JAX VJP -> action gradient passes directional finite-difference check.
- [ ] SVGD accepts the external gradient without changing its sign convention.
- [ ] All objective terms are logged separately.

## Experiment A

- [ ] Interpolating toward a known goal generally decreases FLUX latent distance.
- [ ] N=1 gradient optimization reduces visual loss.
- [ ] N=10 SVGD reduces visual loss.
- [ ] Simulator hidden-state goal error decreases in visual-only evaluation.
- [ ] Visual + safety STL preserves safety better than visual-only when safety matters.
- [ ] Receding-horizon execution works and warm-starts the next solve.

## Experiment B

- [ ] Central-difference latent Jacobian is estimated safely.
- [ ] A held-out small joint perturbation is predicted with positive latent-delta cosine alignment.
- [ ] Direct `-J^T dL/dz` servoing reduces goal loss in simulation before hardware use.
- [ ] Direct servoing reduces goal loss on hardware using conservative bounded motions.
- [ ] Broyden update lowers or maintains Jacobian prediction error over normal motion.
- [ ] Short-horizon local-linear SVGD works closed-loop and executes only one/few steps before replanning.

---

# 29. Recommended first milestone pull request

The first PR should **not** attempt the entire robot experiment.

It should contain only:

1. `visual/flux_encoder.py`
2. `visual/latent_loss.py`
3. optional external `GradientOracle` support in `_legacy_mppi.py`
4. `visual/dlpack_bridge.py`
5. a minimal differentiable-renderer prototype for one simple body
6. tests for:
   - FLUX gradient
   - latent loss
   - external gradient oracle
   - DLPack round trip
   - renderer directional derivative
7. no change to existing STL behavior when the visual subsystem is disabled

Only after that passes should the full Panda visual runner be implemented.

---

# 30. Recommended second milestone pull request

Implement Experiment A end-to-end:

1. MJX render-state trace extraction
2. static MJCF -> differentiable render scene cache
3. `SimVisualGradientOracle`
4. goal-image cache
5. visual softmin-over-time objective
6. mixed visual + STL objective
7. `run_panda_visual_goal.py`
8. `run_panda_visual_goal_mpc.py`
9. full directional gradient test
10. structured experiment logging

Do not start real hardware before this PR produces stable decreasing visual loss in simulation.

---

# 31. Recommended third milestone pull request

Implement Experiment B:

1. `LatentJacobianEstimator`
2. central finite-difference initialization
3. Broyden updates
4. Jacobian prediction diagnostics
5. direct Jacobian-transpose servo baseline
6. local-linear latent rollout model
7. external gradient oracle for local-linear SVGD
8. closed-loop real/sim observation interface
9. hard joint-step projection
10. complete per-step camera/command logging

First run Experiment B against simulator observations with the differentiable gradient deliberately disabled. Only then connect a physical arm.

---

# 32. Minimal pseudocode: Experiment A

```python
# goal setup
z_goal = flux_encoder.encode_goal(goal_rgb)  # cached, no grad

for svgd_iter in range(num_svgd_iters):
    total_costs = []
    total_grads = []

    for U in particles:
        # JAX: exact simulated sensitivity
        render_state, pullback = jax.vjp(
            lambda u: rollout_to_render_state(x0, u, visual_indices),
            U,
        )

        # Bridge to Torch as differentiable leaves
        rs_torch = bridge.to_torch_leaves(render_state, requires_grad=True)

        # Torch: differentiable appearance + representation
        rgb = diff_renderer.render(rs_torch)
        z_pred = flux_encoder.encode(rgb)
        visual_cost = visual_objective(z_pred, z_goal)

        # Visual cotangent wrt simulated body poses
        cotangent_torch = torch.autograd.grad(
            visual_cost,
            rs_torch,
        )
        cotangent_jax = bridge.to_jax(cotangent_torch)

        # JAX: exact pullback through MJX dynamics
        (grad_visual_U,) = pullback(cotangent_jax)

        # Native JAX STL/smoothness terms
        native_cost, grad_native_U = jax.value_and_grad(native_cost_fn)(U)

        cost = w_visual * float(visual_cost) + native_cost
        grad = w_visual * grad_visual_U + grad_native_U

        total_costs.append(cost)
        total_grads.append(grad)

    particles = svgd_update(
        particles,
        costs=stack(total_costs),
        grads=stack(total_grads),
    )

U_star = select_best(particles)
```

---

# 33. Minimal pseudocode: Experiment B

```python
# one-time goal
z_goal = flux_encoder.encode_goal(goal_rgb)

# initial real observation
q = robot.get_joint_positions()
rgb = camera.capture()
z = flux_encoder.encode_descriptor(rgb)

# initial finite-difference visual Jacobian
J = jacobian_estimator.central_difference(robot, camera, flux_encoder, q)

while not done:
    rgb = camera.capture()
    q = robot.get_joint_positions()

    z = flux_encoder.encode_descriptor(rgb, requires_grad=True)
    loss = latent_loss(z, z_goal)
    g_z = torch.autograd.grad(loss, z)[0]

    if planner_mode == "direct":
        g_q = J.T @ g_z.flatten()
        dq = bounded_step(-eta * g_q)

    elif planner_mode == "svgd_local_linear":
        # particles are short future dq sequences
        costs, grads = local_linear_oracle.value_and_grad_batch(
            z0=z.detach(),
            q0=q,
            J_zq=J,
            particles=particles,
            z_goal=z_goal,
        )
        particles = svgd_update(particles, costs, grads)
        dq = select_best(particles)[0]

    q_cmd = safety_project(q + dq)
    robot.command_position(q_cmd)

    # after motion settles / next control sample
    rgb_next = camera.capture()
    q_next = robot.get_joint_positions()
    z_next = flux_encoder.encode_descriptor(rgb_next)

    J = broyden_update(
        J,
        delta_q=q_next-q,
        delta_z=z_next-z.detach(),
    )

    jacobian_monitor.update(...)
```

---

# 34. Final intended research comparison

The completed codebase should support the following direct comparison under one visual goal definition:

| Method | Gradient source | World model | Population | Closed loop |
|---|---|---|---|---|
| STL-SVPIO baseline | exact STL gradient | MJX | SVGD | optional |
| Visual GD | exact visual VJP | MJX + diff renderer | 1 | yes/no |
| Visual multi-start GD | exact visual VJP | MJX + diff renderer | 10 independent | yes/no |
| **Visual SVPIO** | exact visual VJP | MJX + diff renderer | SVGD | yes/no |
| Jacobian servo | estimated `dz/dq` | local real Jacobian | 1 | yes |
| **Jacobian Visual SVPIO** | estimated `dz/dq` | local linear latent model | SVGD | yes |
| Learned-latent SVPIO | learned dynamics gradient | learned latent model | SVGD | yes |

The central scientific question is then measurable:

> Can a frozen image representation provide a useful optimization landscape for robot control, and does SVGD's multimodal population improve success over single-trajectory visual gradient descent when the gradient is propagated through either exact simulator dynamics or an estimated real-world latent Jacobian?

---

# 35. Non-negotiable implementation constraints

1. Do not claim real camera pixels are directly differentiable with respect to joint positions.
2. Do not use a standard non-differentiable MuJoCo renderer in the exact-gradient path.
3. Do not put the FLUX current-image forward pass inside `torch.no_grad()`.
4. Do not sample stochastic VAE latents during control.
5. Do not silently change the existing SVGD gradient sign convention.
6. Do not remove the STL-only path.
7. Do not execute long real-world trajectories open-loop with a local Jacobian.
8. Do not use hidden simulator goal state inside the visual optimization objective.
9. Do not benchmark the method before the directional gradient test passes.
10. Do not interpret decreasing FLUX latent distance as task success without also reporting physical/hidden-state or STL success metrics.

---

# 36. Definition of done

The code is considered complete when a single codebase can run:

```text
A) Panda + MJX + differentiable render + FLUX + SVGD
B) Panda/simulated camera + finite-difference latent Jacobian + closed-loop SVGD
C) Physical camera/robot adapter + latent Jacobian + closed-loop bounded commands
```

using the same goal-latent objective and producing directly comparable metrics.

The strongest final implementation is not simply “FLUX controls the robot.” It is:

```text
Goal image
   -> frozen visual representation
   -> differentiable visual objective
   -> one of two explicit visuomotor derivative models
        A. exact simulated dynamics + differentiable rendering
        B. estimated real-world latent Jacobian
   -> STL-constrained combined cost
   -> SVGD multimodal trajectory optimization
   -> receding-horizon execution
```

That is the architecture the coding agent should implement.
