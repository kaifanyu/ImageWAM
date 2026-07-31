# SVGD trajectory viewer (3D)

`experiments/libero/svgd_traj3d.py` serves a small local UI that shows **how the
rollout trajectories of an SVGD run evolve across iterations**: pick a run, scrub
or play the iteration slider, and watch the particle population — or one particle,
or the running best — move through 3D end-effector space.

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
| `history.json` -> `history[i].energies`, `latent_metrics`, `goal_errors_m`, `best_particle`, `global_best_*` | per-iteration statistics, colouring, the energy strip |
| `history.json` -> `particles_before_update` | the particle endpoint each iteration (endpoint search) |
| `iter_XXX/traces/particle_NN_base.npz` -> `eef_path (49, 3)` | the 3D rollout curve drawn per particle |
| the same trace -> `object_positions` | tracked-object start/end markers (obstacle runs) |
| `config.bounds`, `actual_start_eef`, `diagnostic_goal_eef` | search-bounds box and scene anchors |

A run therefore has to have been launched with `--save-rollout-traces`
(and `--save-all-particles` for the whole population, not just the best one);
runs without traces are skipped by the index.

Two optimizer families are normalised into the same bundle:

* **endpoint search** — particles *are* 3D endpoints, so the per-particle marker
  is the target endpoint and the trail shows the endpoint migrating;
* **path search** — every particle shares one fixed endpoint and differs in arc
  shape, so the marker is instead the **path apex** (the rollout point furthest
  from its own start→end chord), which is what the two path parameters move.

## The UI

* **Run** — every trial under `--runs-root` that has traces, grouped by scene.
* **Show** — all particles / best particle of each iteration / global best /
  one specific particle. The focused particle is drawn in orange with the rest
  dimmed, and its own energy curve is added to the strip below.
* **Layers** — rollout paths, endpoint (or apex) markers, endpoint trails across
  the iterations so far, ghost paths of the previous five iterations, scene
  anchors, the search-bounds box, and equal-axis scaling.
* **Transport** — play/pause (space), step (←/→), speed, or click the energy
  strip to jump to an iteration.
* **Particles this iteration** — the same numbers as a sortable table; click a
  row to focus that particle.

Colour encodes one thing only: the **objective energy**, on a single-hue blue
ramp whose range is fixed across the whole run so frames are comparable. Orange
is the focused particle, aqua-green the best particle of the iteration (always
directly labelled), and neutral ink the static scene anchors. Light and dark
themes are separately stepped; the toggle is the ◐ button.

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
UI. All 22 trials of the current `runs/` tree come to 11.9 MB; `--stride 4`
roughly quarters that if a run is bigger than you want in git.

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
path outside `--runs-root`), and `--stride N` to keep every Nth rollout waypoint
when a run is large enough for the bundles to feel heavy (a 15-particle x
31-iteration trial is ~0.7 MB at full resolution).

`plotly.min.js` is served from the installed `plotly` package, so the viewer works
without network access; if plotly is not importable it falls back to the CDN.
