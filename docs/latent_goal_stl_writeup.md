# Goal-Latent Distances as a Planning Objective

**Baseline:** ImageWAM (image editing as the world-action backbone)
**Inspiration:** STL-SVPIO (temporal logic as a differentiable shaping signal, optimized with Stein variational particles)

**Core design rule:** the goal image enters the system *only* as a latent distance. It is never decoded, never compared in pixels, never handed to a policy, and never used to compute a physical target. Every physical measurement in this document is a held-out diagnostic that the optimizer cannot see.

---

## 1. Problem Statement

ImageWAM makes a strong claim: a robot policy does not need to imagine a whole video. It only needs to know *what should change* between now and the task-relevant future state. It gets that signal from an image-editing backbone, and it consumes it as key-value caches feeding a flow-matching action expert.

Two things about that design limit where it can be used.

**It needs training.** The action expert is learned from demonstrations. On LIBERO, that is 500 demos per suite; on RoboTwin, 27,500 trajectories. If you have a new scene, a new robot, or a task with no demonstrations, ImageWAM has nothing to condition on.

**It cannot express a constraint.** The action expert is a conditional sampler. You can tell it *where to end up* (through the instruction and the edited target state), but you have no way to say "and never tip the mug over," "reach A only after B is done," or "stay above the table for the whole approach." Those are properties of the entire trajectory, not of one endpoint. Nothing in the flow-matching objective can represent them.

STL-SVPIO solves exactly that second problem, but assumes something this setting does not have. It requires the task to be written as predicates over *known physical state* — obstacle centers, goal coordinates, torso pitch angle. It also requires differentiable physics (MuJoCo MJX under JAX) so that robustness gradients flow back to controls. Neither assumption holds when your specification of the task is an image and your simulator is not differentiable.

So there is a gap:

| | Where the task comes from | Handles trajectory constraints | Needs demos | Needs differentiable physics |
|---|---|---|---|---|
| ImageWAM | An image (editing model) | No | Yes | No |
| STL-SVPIO | Hand-written physical predicates | Yes | No | Yes |
| **This work** | **An image (latent distance only)** | **Yes (planned)** | **No** | **No** |

The question this project asks: **can a distance in the image-editor's latent space act as the objective that a particle optimizer descends — with no demonstrations, no differentiable simulator, and no access to physical state?**

If yes, then the goal image stops being a conditioning signal and becomes a *cost function*, and everything STL-SVPIO does with hand-written predicates can be redone with predicates written over latent distances.

---

## 2. Background

### 2.1 What we take from ImageWAM

The useful part is the premise, not the architecture. ImageWAM argues that an image-editing model, given a start image and an instruction, produces a target frame that captures the task-relevant change and discards irrelevant appearance detail. We accept that premise and use it in the simplest possible way: run the editor once, encode the result, and keep the latent.

We deliberately drop:

- the action expert (needs demos),
- the KV-cache conditioning (needs the action expert),
- decoding the edited image at any point in the loop.

What is left is one vector: `z_goal`.

### 2.2 What we take from STL-SVPIO

Three ideas transfer cleanly:

1. **A population beats a point.** Optimizing one trajectory with gradient descent gets stuck. A set of mutually repulsive particles explores multiple modes. This is the SVGD update:

   ```
   φ*(u_i) = (1/N) Σ_j [ K(u_j, u_i) ∇_{u_j} log p*(u_j)  +  ∇_{u_j} K(u_j, u_i) ]
                          └──── attraction ────┘              └──── repulsion ────┘
   ```

2. **The cost defines the posterior.** With a uniform prior over admissible controls, `∇_u log p*(u) = -(1/λ) ∇_u J(u)`. The whole method is then determined by what you put in `J`.

3. **Logic can be a smooth cost.** STL robustness `ρ` is a real number: positive means satisfied, and larger means satisfied by a wider margin. With smooth min/max it is differentiable, so `J = -ρ` is a usable objective.

What does *not* transfer is how STL-SVPIO gets its gradients. It differentiates through MJX. We cannot: our loop is `target → MuJoCo rollout → render → FLUX encoder → distance`, and MuJoCo/LIBERO is not differentiable. Gradients have to be estimated another way (Section 4.4).

---

## 3. Goal

Build a planner that takes a start state and a goal image, and returns a trajectory — with three properties:

1. **Image-only objective.** The optimizer sees exactly one number per rollout: the latent distance from the rendered terminal state to the goal latent. It never sees end-effector coordinates, object poses, or the physical goal.
2. **No demonstrations.** Nothing is trained on expert data. The prior over trajectories is uniform inside the workspace bounds.
3. **Constraint-expressive.** The objective can be extended from "reach the goal state" to a temporal specification — always avoid, eventually reach, reach A after B — with the predicates written over latent distances rather than physical coordinates.

Goals 1 and 2 are what has been tested so far. Goal 3 is the designed extension, and is not yet implemented.

**How success is measured.** Because the optimizer is blind to physics, we can validate it honestly. We record physical error separately and ask two questions:

- *Does the latent distance track physical error?* (Spearman correlation between the two across the particle population.)
- *Does descending the latent distance actually move the robot to the goal?* (Physical error of the selected particle.)

---

## 4. Method

### 4.1 Setup

Fixed LIBERO scene. The complete MuJoCo state is frozen at the start and restored before every single evaluation, so the objective is a deterministic function of the decision variable up to renderer noise. (Measured re-evaluation drift: exactly 0.0.)

A particle is a **decision variable `θ`**, not a full control sequence. Two parameterizations are in use:

| Experiment | `θ` | Dim |
|---|---|---|
| Endpoint reaching | terminal end-effector position `(x, y, z)` | 3 |
| Obstacle avoidance | path shape `(midpoint_x, arc_height)`, terminal point fixed | 2 |

A minimum-jerk controller turns `θ` into actions: 40 move steps + 8 settle steps, gain 15.0. Tracking error between the commanded and achieved terminal point stays under ~3 mm, so `θ` is a faithful stand-in for the trajectory it produces.

### 4.2 The goal image, and the only way it is used

```
start.png ──> FLUX.2 image editor (instruction prompt) ──> goal image
                                                              │
                                                        encode │  (never decoded)
                                                              ▼
                                                          z_goal
```

At evaluation time:

```
θ ──> restore frozen state ──> rollout ──> render terminal frame ──> encode ──> z(θ)
                                                                                  │
                                                            E(θ) = d(z(θ), z_goal) ◄┘
```

`E(θ)` is a scalar. It is the only thing the optimizer receives.

**Encoder.** FLUX.2 autoencoder. Renders are `[agentview | wrist]` side by side at 224×448, which the patch-16 encoder turns into a 14×28 token grid. DINOv3 (`vit_base_patch16_dinov3.lvd1689m`) is supported as a drop-in alternative and is used as a control.

**View restriction.** The token grid is split down the middle column, and by default only the agentview half (14×14 = 196 tokens) is scored. The wrist half barely changes until the final millimetres of approach, so including it adds magnitude to the distance without adding gradient.

**Goal latent source.** Two options exist: `editor` (use the editing model's own output latent directly) and `reencode` (decode-free re-encoding of the goal image through the same encoder used for candidates). All optimization runs so far use `reencode`, so candidate and goal latents come from the same encoder and the distance is not measuring an encoder mismatch.

### 4.3 The three latent distances

Let `z, z* ∈ R^{N×C}` be `N` latent tokens of `C` channels.

| Name | Definition | What it measures |
|---|---|---|
| `rms` | `sqrt( mean_{n,c} (z_{nc} - z*_{nc})² )` | Raw magnitude difference. Sensitive to overall brightness/contrast shifts. |
| `cosine` | `1 - cos(flatten(z), flatten(z*))` | Direction of the whole latent as one vector. Discards spatial correspondence. |
| `token_cosine` | `mean_n [ 1 - cos(z_n, z*_n) ]` | Per-token direction, averaged. **Keeps spatial correspondence:** each latent cell is compared to the cell at the same image location. |

`token_cosine` is the one we expect to work best, because a robot arm moving across a table is a *local* change: most tokens should match, a few should not, and averaging per-token disagreement isolates exactly that. The experiments in Section 5 test this rather than assume it.

### 4.4 Getting a gradient without differentiable physics

SVGD needs `∇_θ E`. Two mechanisms are implemented.

**(a) Finite differences (primary).** Six extra MuJoCo rollouts per particle per iteration — a central difference on each axis. The probe step is deliberately anisotropic: `(0.01, 0.04, 0.01)` m. The `y` axis is the task axis and the latent landscape is flat along it far from the goal, so a small probe there returns pure renderer noise.

**(b) Learned differentiable surrogate (alternative).** A network learns `θ → projected terminal feature`, which is then differentiated with autograd. MuJoCo is still run once per particle for ground-truth scoring and online refinement of the surrogate, but never for a gradient. This is the closest analogue to STL-SVPIO's differentiable physics.

**Transport.** SVGD with an RBF kernel and the median heuristic bandwidth, matching STL-SVPIO. Step size 0.01, temperature 0.10, repulsion weight 0.01, and a 0.02 m trust region on the applied update. A `particle_gd` mode (repulsion disabled) is available as an ablation.

### 4.5 The STL layer — designed, not yet built

This is the planned extension. STL-SVPIO writes predicates over physical state. We write them over latent distances, which keeps the image-only rule intact:

| STL-SVPIO predicate | Latent version |
|---|---|
| `µ_reach := ‖x_ee - c_g‖ ≤ r_g` | `µ_reach := δ_g - d(z_t, z_goal) > 0` |
| `µ_avoid := ‖x - c_obs‖ > r_obs` | `µ_avoid := d(z_t, z_violation) - δ_v > 0` |
| — (no analogue) | `µ_preserve := δ_p - d(mask ⊙ z_t, mask ⊙ z_start) > 0` |

`z_violation` is the encoding of a reference image showing a failure state — a tipped mug, an object knocked aside. `µ_preserve` uses a region mask over the token grid to say "this part of the scene must not change," which has no clean equivalent in coordinate-space STL.

The specification is then built with the standard operators and evaluated with smooth min/max:

```
φ = □[0,H] µ_avoid  ∧  ◇[0,H] µ_reach          (avoid always, reach eventually)
J(θ) = -ρ^φ(z_0:H)
```

and `J` drops straight into the SVGD update already in place.

**The cost this adds.** Terminal-only scoring needs one render+encode per rollout. A temporal specification needs the latent at every timestep it quantifies over. Planned mitigations, in order: subsample to keyframes, score with DINOv3 instead of the FLUX autoencoder, and use the learned surrogate to predict the latent sequence.

### 4.6 What the optimizer is not allowed to see

Recorded every iteration, used only for analysis:

- physical goal error (m) of every particle,
- centroid goal error and progress along the goal axis,
- tracked object displacement and rotation,
- controller tracking error.

None of these appear in `E(θ)`.

---

## 5. Progress and Experiments Run So Far

### 5.1 Ranking check — can the goal latent tell good rollouts from bad?

**Setup.** Empty table, arm staged image-left, goal image-right. 13 candidate rollouts from one frozen state: five controls (no-op, wrong direction, undershoot, physical oracle, overshoot) and eight sampled endpoint perturbations. Score terminal renders only.

**Result.** The latent distance orders the controls correctly. This was the gate for building an optimizer on top, and it passed.

### 5.2 Landscape probes — what the objective looks like along a path

**Setup.** Sample 11 points along different paths (oracle arc, straight, halfway goal, and a zoomed 0.9→1.0 segment near the goal), with repeats to separate signal from renderer noise.

**Result.** Reference scale: start→goal `rms` = 0.830, `cosine` = 0.410. The landscape descends monotonically near the goal but is close to flat far from it — a real, measured local minimum problem. This is what motivated the anisotropic finite-difference step, agentview-only scoring, and the particle population.

### 5.3 Transport ablation

Smoke runs isolating the two SVGD forces: `latent_only` (attraction), `repulsion_only`, and `full_svgd`. Confirms each term does what it should before spending GPU time on sweeps.

### 5.4 Metric comparison, particles spanning the goal (20 particles, 8 iterations)

**Setup.** 20 particles, bounds `y ∈ [-0.32, 0.32]` spanning the goal. One trial per metric. Traces saved for all particles.

**Result — this is the strongest evidence so far:**

| Trial | Latent↔physical Spearman | Objective mean | Centroid error (m) | Best particle physical error |
|---|---|---|---|---|
| `rms` | 0.982 | 0.803 → 0.730 | 0.245 → 0.207 | **4.9 mm** |
| `cosine` | 0.972 | 0.368 → 0.315 | 0.245 → 0.209 | **4.0 mm** |
| `token_cosine` | **0.991** | 0.288 → 0.242 | 0.245 → 0.203 | **4.0 mm** |

All three latent distances rank candidates in near-perfect agreement with physical goal error, which the optimizer never saw. `token_cosine` correlates best in every trial, confirming Section 4.3. Re-evaluation drift is exactly 0.0, so the objective is deterministic. Controller tracking error averages 2.4–2.8 mm.

### 5.5 Far start, short budget (15 particles, 30 iterations) — the failure case

**Setup.** Same scene, but all particles initialized in a tight cloud at the *start* pose, 0.44 m from the goal. This is the honest hard version: the population begins in the flat region.

**Result:**

| Trial | Objective change | Centroid error change | Spearman | Best physical error | Success @ 5 cm |
|---|---|---|---|---|---|
| `cosine` | −5.6% | **+2.3 cm (worse)** | −0.23 | 0.369 m | 0/15 |
| `rms` | −4.1% | −1.8 cm | −0.06 | 0.308 m | 0/15 |
| `token_cosine` | −10.2% | −5.9 cm | +0.28 | 0.255 m | 0/15 |

Every trial reduced its own objective while barely moving the robot. `cosine` moved it *away*. This is the flat-landscape problem measured directly: far from the goal, the latent distance is weakly informative, and correlation collapses from ~0.98 to ~0. `token_cosine` degrades most gracefully. Trust region and bounds were active on 59% and 65% of updates respectively, so the search was also fighting its own step limits.

### 5.6 DINOv3 control (15 particles, 60 iterations)

**Setup.** Identical protocol, FLUX autoencoder swapped for DINOv3 features. Tests whether the far-field weakness is a property of *this* encoder or of latent distances generally.

**Result:**

| Trial | Objective change | Centroid error | Spearman | Best physical error |
|---|---|---|---|---|
| `dino_rms` (ε=0.04) | −7.4% | −3.2 cm | 0.595 | 0.359 m |
| `dino_token_cosine` (ε=0.02) | −13.3% | −3.8 cm | 0.628 | 0.353 m |
| `dino_token_cosine` (ε=0.04) | −14.3% | −5.1 cm | **0.682** | 0.317 m |

DINOv3 keeps a far-field correlation of 0.60–0.68 where the FLUX autoencoder collapses to ~0. Reconstruction-trained latents and semantically-trained features behave differently at distance, and the trust region never activated for DINO (0% vs 59%), meaning its gradients were better conditioned.

### 5.7 Far start, full budget (15 particles, 100 iterations) — the result that works

**Setup.** Same hard far-start initialization as 5.5, run to 100 iterations instead of 30. FLUX `token_cosine` vs DINOv3 `token_cosine`. Both complete.

**Result:**

| Encoder | Final physical goal error | Final `token_cosine` |
|---|---|---|
| FLUX.2 AE | **12.7 mm** | 0.0089 |
| DINOv3 | **19.2 mm** | 0.0092 |

**Starting from 0.44 m away, with no demonstrations and no access to any physical coordinate, latent-distance SVGD drives the arm to within 1.3 cm of the goal.** The 30-iteration result in 5.5 was a budget problem, not a method problem — the population has to cross the flat region before the informative basin takes over. FLUX ends up more accurate than DINOv3 despite being the weaker far-field signal, consistent with 5.6: DINOv3 gets you across the plateau, FLUX resolves the final approach.

### 5.8 Learned differentiable surrogate (15 particles, 100 iterations)

**Setup.** Finite differences replaced entirely by the learned surrogate. Bootstrapped from the 5.5 history, then refined online: 1,616 ground-truth rollouts, **0 finite-difference rollouts**.

**Result.** Selected at iteration 97: energy 0.275, physical error **0.172 m**. Better than the 30-iteration finite-difference run (0.255 m), much worse than the 100-iteration one (0.013 m).

The surrogate works — it produces usable gradients with no probe rollouts — but its accuracy currently bounds the final solution. It matters because it is the path to affording temporal specifications: if the surrogate can predict a latent *sequence*, STL robustness becomes cheap.

### 5.9 Obstacle avoidance (20 particles, 10 iterations)

**Setup.** Living-room scene with an upright mug between start and goal. `θ` = `(midpoint_x, arc_height)`; terminal point fixed. Particles initialized as a *collision* cloud — every one starts by knocking the mug over. The optimizer sees only the latent distance to a reference image in which the mug remains upright. Ground-truth safe arc: 0.08 m. Tolerances: 2 mm displacement, 2°, 3 cm goal.

**Result:**

| Trial | Energy reduction | Success 0 → final | Mug displacement | Recovered arc height |
|---|---|---|---|---|
| `cosine` | 95.4% | 0.0 → **1.0** | 0.096 m → ~0 | 0.0798 m |
| `rms` | 80.5% | 0.0 → **1.0** | 0.096 m → ~0 | 0.0829 m |
| `token_cosine` | 94.9% | 0.0 → **1.0** | 0.096 m → ~0 | 0.0791 m |
| `particle_gd` (no repulsion) | 96.0% | 0.0 → **1.0** | 0.096 m → ~0 | 0.0789 m |

Every particle in every trial ends safe, and all four recover the 0.08 m arc to within 3 mm, with terminal goal error ~3.4 mm. **A latent distance to a single "nothing was disturbed" image is enough to express an obstacle constraint** — no collision geometry, no object pose, no penalty term.

One honest caveat: plain particle gradient descent matched SVGD here. This task has one safe mode, so repulsion had nothing to find. That is precisely the argument for the STL extension — multi-modality is where a particle population earns its cost, and terminal-distance objectives on unimodal tasks do not create it.

### 5.10 In flight

- **Single-object push, agentview vs wrist** (15 particles, 100 iterations, 3 trials — FLUX agentview, DINO agentview, DINO wrist). At ~67/101 iterations. Tests whether the agentview-only choice from 4.2 holds when the task is object motion rather than arm motion.
- **Living-room scene, larger sweep** (3 metric trials). Early, ~3 iterations in.

### 5.11 Supporting infrastructure built

- `svgd_endpoint.py` — main optimizer (finite differences, three metrics, two encoders, SVGD/particle-GD, trace saving).
- `svgd_endpoint_differentiable.py` — learned-surrogate variant.
- `svgd_obstacle_path.py` — path-shape parameterization for obstacle tasks.
- `probe_latent_landscape.py`, `probe_latent_path.py` — landscape characterization.
- `svgd_traj3d.py` — interactive 3D trajectory viewer with an iteration slider; runs ship as ~12 MB JSON bundles in `viz_bundles/` so they replay on a fresh clone.
- Containerized environment; multi-GPU sweep launchers (one trial per GPU).
- Off-the-shelf image-editor trajectory baseline (no ImageWAM fine-tune).

---

## 6. What the Results Say

1. **A goal image works as a cost function.** Latent distance tracks physical error at ρ = 0.97–0.99 when particles span the goal (5.4), and descending it reaches 1.3 cm from 0.44 m away with no demos and no physical state (5.7).
2. **Spatial correspondence matters.** `token_cosine` wins on correlation in every trial and degrades most gracefully far from the goal.
3. **The failure mode is a flat far field, and it is fixable.** Correlation collapses at distance (5.5). Two independent fixes work: more iterations (5.7), and a semantically-trained encoder (5.6). They are complementary — DINOv3 crosses the plateau, FLUX resolves the approach.
4. **Constraints are already expressible as images.** A single "undisturbed" reference image encodes an obstacle constraint well enough to take 20/20 colliding particles to safe (5.9).
5. **Gradients without differentiable physics are viable but currently limiting.** The surrogate reaches 0.172 m against finite differences' 0.013 m (5.8).

## 7. Next Steps

1. Finish 5.10 and settle the agentview/wrist question.
2. Implement latent-space STL predicates (4.5) and the smooth robustness objective; the SVGD machinery is already in place and only `J` changes.
3. Start with `□ µ_preserve ∧ ◇ µ_reach` on the mug scene — a spec whose two clauses are already validated separately (5.7, 5.9).
4. Build a multi-modal task where SVGD's repulsion is required, to justify particles over gradient descent (5.9 did not).
5. Push the surrogate to predict latent *sequences*, which is what makes temporal specifications affordable.
