# SVGD trajectory viewer (3D)

`experiments/libero/svgd_traj3d.py` serves a small local UI that shows **how the
rollout trajectories of an SVGD run evolve across iterations**: pick a run, scrub
or play the iteration slider, and watch the particle population — or the two or
three particles you picked, or the running best — move through 3D end-effector
space, inside the scene they actually moved through.

```bash
imagewam_python experiments/libero/svgd_traj3d.py --runs-root runs
# -> viewer at http://127.0.0.1:8770/
```

Then open the printed URL. Over SSH, forward the port first
(`ssh -L 8770:127.0.0.1:8770 <host>`); VS Code forwards it automatically when the
server is started from an integrated terminal.

## What it reads

Nothing is precomputed — the viewer reads the run artefacts that
`svgd_endpoint.py` and `svgd_obstacle_path.py` already write:

| Source | Used for |
|---|---|
| `history.json` -> `history[i].energies`, `latent_metrics`, `goal_errors_m`, `best_particle`, `global_best_*` | per-iteration statistics, colouring, the metric strip |
| `history.json` -> `particles_before_update` | the particle endpoint each iteration (endpoint search) |
| `iter_XXX/traces/particle_NN_base.npz` -> `eef_path (49, 3)` | the 3D rollout curve drawn per particle |
| the same trace -> `object_positions (49, K, 3)` | each tracked object's **track** per rollout, not just its endpoints |
| the same trace -> `arm_link_positions (49, L, 3)`, `arm_link_names` | the arm skeleton at the scrubbed rollout step |
| `iter_XXX/particle_NN.png`, `iter_XXX/best.png`, `goal_reference.png` | the terminal renders shown beside the goal, per particle per iteration |
| `scene.json` (at the trial or any run directory above it) | table top, robot base, link parents, object rest poses |
| `config.bounds`, `actual_start_eef`, `diagnostic_goal_eef`, the test `manifest.json` | search-bounds box and scene anchors |

A run therefore has to have been launched with `--save-rollout-traces`
(and `--save-all-particles` for the whole population, not just the best one);
runs without traces are skipped by the index.

Arm links and `scene.json` are written by `experiments/libero/scene_geometry.py`,
which the two optimizers call automatically. **Runs that finished before that
existed still load** — the arm and landmark layers are simply greyed out, with
the reason on hover. Landmarks (though not per-step arm poses) can be retrofitted
without re-running the optimizer, since they only depend on the start state:

```bash
imagewam_python experiments/libero/scene_geometry.py runs/living_room_mug_obstacle
# -> runs/living_room_mug_obstacle/scene.json, picked up by every trial underneath
```

Two optimizer families are normalised into the same bundle:

* **endpoint search** — particles *are* 3D endpoints, so the per-particle marker
  is the target endpoint and the trail shows the endpoint migrating;
* **path search** — every particle shares one fixed endpoint and differs in arc
  shape, so the marker is instead the **path apex** (the rollout point furthest
  from its own start→end chord), which is what the two path parameters move.

## The UI

* **Run** — every trial under `--runs-root` that has traces, grouped by scene.
* **Show** — all particles / best particle of each iteration / global best.
* **Picked particles** — click a path, a marker, or a table row to pick a
  particle; shift-click to add up to three. Picked particles get their own hue,
  a direct label, their own line on the strip below, and their own read-out
  (energy, goal error, target-tracking error, terminal position, and how far
  their rollout pushed each object). **Show only the picked particles** hides
  everything else, which is the "one particle, whole timeline" view.
* **Rollout layers** — paths, direction arrows and the time bead, *draw path only
  up to the current step* (progressive reveal), endpoint/apex markers, trails
  across the iterations so far, ghost paths of the previous five iterations, and
  the arm skeleton at the current step.
* **Scene layers** — start/goal/target anchors, the goal tolerance ring plus the
  dashed labelled residual from the best terminal to the goal, the table and
  robot base, tracked objects and their motion, and the search-bounds box.
* **Framing** — the extents are nested, because the scales are: particles alone
  (often millimetres), plus goal and objects, plus table/base/arm (tens of
  centimetres), plus the search bounds. Ticking the arm layer widens the framing
  automatically, otherwise the arm would reach out of view.
* **Proportions** — how the three axes are scaled against each other:
  * *True* (default) — 1 m of x is 1 m of y is 1 m of z. What the arm actually
    did, at the cost of a thin box when one axis barely moves.
  * *True, but thin axes floored* — keeps the metric ratio wherever it is
    legible and only lifts an axis that would otherwise vanish (to 22 % of the
    longest), renormalised to the same on-screen volume so it reads as flatter
    rather than smaller.
  * *Stretch every axis to a cube* — the most readable and the least honest.

  The line underneath always states the exaggeration actually applied
  (`x ×24, z ×26` on a typical y-sweep run in cube mode), so a distorted plot
  can never quietly be read as a shape.
* **Terminal renders** — the strip under the 3D scene shows, for the current
  iteration, the goal image beside the frame each picked particle actually
  produced (the composed `[agentview | wrist]` view the objective was computed
  on), captioned with its energy and goal error. With nothing picked it shows
  the iteration's best. The view switch crops to one camera; clicking a tile
  opens it full size. Pixels are fetched per view from `/api/image` rather than
  carried in the bundle — a trial holds several hundred renders — so **this
  panel is live-run only**: served from an exported `*.traj3d.json` the layer is
  greyed out, because a bundle carries the numbers and not the images.
* **Scene check** — the distances that decide whether the plot means what it
  looks like it means: start vs. where paths actually begin, each particle's
  target vs. where its rollout actually stopped, the goal vs. the best terminal,
  the population span, and which file the goal came from.
* **Transport** — two rows. *Iteration* plays the optimisation (space, ←/→);
  *Rollout* scrubs within one rollout (shift ←/→), moving the time bead and the
  arm skeleton. Clicking the strip jumps to an iteration.
* **Strip metric** — energy, goal error, tracking error, or particle spread.

Colour encodes two things and nothing else: the **objective energy** on a
single-hue blue ramp whose range is fixed across the whole run, and the
**identity of the picked particles** in orange / aqua / violet. Three is a hard
cap — that trio is the largest set that clears the all-pairs CVD and
normal-vision separation floors in both themes, and picking a fourth drops the
oldest. Rankings (best, global best) and anchors are shape plus a direct label in
neutral ink, never hue; landmarks are muted. Light and dark themes are separately
stepped; the toggle is the ◐ button.

## Checking a run's anchors without opening the UI

```bash
imagewam_python experiments/libero/svgd_traj3d.py --runs-root runs --verify
```

prints one row per trial: `start` (start anchor vs. the first waypoint of every
path), `track` (worst gap between a particle's target and its own terminal),
`goal` (best terminal to the goal marker), `span` (how far the population
spreads), and where the goal came from. `start` at 0.0000 and `track` at
controller scale mean the anchors are trustworthy; a large `goal` is expected in
runs where the goal is diagnostic only and never entered the objective.

## Sharing a run — what to commit

`runs/` is gitignored, and the raw read set is large: one 15-particle x
31-iteration trial needs its `history.json` (2.3 MB) plus 496 trace `.npz` files
(6.0 MB). Instead, export **bundles** — the normalised JSON the UI already
consumes, one file per trial:

```bash
# every trial that has traces -> viz_bundles/<scene>/<trial>.traj3d.json
imagewam_python experiments/libero/svgd_traj3d.py --runs-root runs \
    --export-all viz_bundles

# serve straight from the bundles; no traces, no history.json needed
imagewam_python experiments/libero/svgd_traj3d.py --runs-root viz_bundles
```

`--runs-root` indexes live trial directories and `*.traj3d.json` bundles
interchangeably, so a clone with only `viz_bundles/` committed gets the identical
UI. `--stride 4` roughly quarters a bundle if a run is bigger than you want in
git; arm skeletons are the bulkiest single layer and carry their own
`--arm-stride` (default 8, `0` to leave them out entirely).

Bundles exported before a feature existed keep working, but they carry only what
they were exported with — re-export to pick up object tracks, arm poses, the
scene landmarks, and the goal-from-manifest fallback. No bundle carries the
terminal renders; for those, serve the run directory itself.

Keep bundles **outside** `runs/` (that path is ignored) — `viz_bundles/` at the
repo root works.

## Other entry points

```bash
# list the trials that can be visualised
imagewam_python experiments/libero/svgd_traj3d.py --list

# dump one trial's normalised bundle (the exact JSON the UI consumes)
imagewam_python experiments/libero/svgd_traj3d.py \
    --export-bundle runs/empty_arm_preview/empty_start_metric_y040_15p_v1/trials/token_cosine_y040 \
    --out bundle.json
```

Useful flags: `--port`, `--host` (defaults to `127.0.0.1`; the server refuses any
path outside `--runs-root`), `--stride N` to keep every Nth rollout waypoint, and
`--arm-stride N` for the arm poses (a 15-particle x 31-iteration trial is ~0.7 MB
at full waypoint resolution before arm data).

`plotly.min.js` is served from the installed `plotly` package, so the viewer works
without network access; if plotly is not importable it falls back to the CDN.
