#!/usr/bin/env bash
# Optimize a list of multi-object-table goal images against a frozen scoring
# encoder (the FLUX.2 autoencoder, DINOv3, or both).
#
# Trials are dealt round-robin across GPUS and each GPU works its own queue
# strictly one trial at a time, so peak memory is one trial per GPU no matter how
# long the list is. GPUS="0" therefore runs everything sequentially; GPUS="0 1 2"
# with three trials runs one on each card concurrently.
#
#   bash scripts/flux2/run_multi_object_goals.sh [suite_name]
#
# A trial is roughly 16 min per iteration, so a 100-iteration trial is about a
# day. The script does not self-detach -- launch it so it outlives the shell:
#
#   setsid nohup bash scripts/flux2/run_multi_object_goals.sh NAME \
#     > /dev/null 2>&1 < /dev/null &
#
# then follow $BASE_RUN_DIR/$NAME/launcher.log.
#
# Environment overrides:
#   GOAL_IDS    space-separated goal ids under $BASE_RUN_DIR/goals
#   GPUS        space-separated physical GPU indices (default "0 1 2")
#   ITERATIONS  SVGD iterations per trial (default 100)
#   PARTICLES   particles per trial (default 15)
#   ENCODERS    subset of "flux_ae dinov3" (default "flux_ae")
#   FD_EPS      three central-difference half-steps in metres, X Y Z
#   DRY_RUN     true to print the plan without launching or writing anything
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/common.sh
imagewam_init .
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env FLUX2_SRC

export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONUNBUFFERED=1

BASE_RUN_DIR="${BASE_RUN_DIR:-runs/multi_object_arm_preview}"
GOALS_DIR="$BASE_RUN_DIR/goals"
SUITE_NAME="${1:-${SUITE_NAME:-multi_object_agentview_15p_100it_flux_3gpu_v1}}"
SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
TRIAL_ROOT="$SUITE_DIR/trials"

ITERATIONS="${ITERATIONS:-100}"
PARTICLES="${PARTICLES:-15}"
DINO_MODEL="${DINO_MODEL:-vit_base_patch16_dinov3.lvd1689m}"
read -r -a GPUS <<< "${GPUS:-0 1 2}"
read -r -a GOAL_IDS <<< "${GOAL_IDS:-red_coffee_mug_above cream_cheese_above table_center}"
read -r -a ENCODERS <<< "${ENCODERS:-flux_ae}"

# The empty-table sweep searches a narrow slab at x in [-0.06, -0.04]. None of
# the object-hover goals live there, so the prior has to cover the whole
# reachable box or start-cloud init rejects the start pose outright.
read -r -a BOUNDS <<< "${BOUNDS:--0.20 0.09 -0.32 0.32 0.98 1.16}"

# Central-difference half-steps, one per axis. The empty-table runs used an
# asymmetric 0.01 0.04 0.01 because that task moved almost purely in y; these
# goals move in all three axes, so the probe is symmetric.
#
# Every probe is clipped to BOUNDS before it is evaluated, so a half-step wider
# than an axis's range stops being a local gradient: it becomes the same
# box-corner secant for every particle, identical across the population, and the
# per-particle gradient diversity SVGD relies on disappears. The z range here is
# only 0.18 m wide, so keep FD_EPS well under that.
read -r -a FD_EPS <<< "${FD_EPS:-0.03 0.03 0.03}"

if [[ ! -f "$BASE_RUN_DIR/manifest.json" ]]; then
  echo "Missing scene manifest: $BASE_RUN_DIR/manifest.json" >&2
  echo "Build it with: bash scripts/flux2/prepare_svgd_scene.sh multi-object" >&2
  exit 2
fi

# Validate every goal before launching: discovering a typo eight hours into a
# ten-trial suite is the expensive way to find it.
for goal_id in "${GOAL_IDS[@]}"; do
  if [[ ! -f "$GOALS_DIR/$goal_id/goal.png" || ! -f "$GOALS_DIR/$goal_id/metadata.json" ]]; then
    echo "No rendered goal at $GOALS_DIR/$goal_id/" >&2
    echo "Render the catalogue with: $PYTHON_BIN -u -B experiments/libero/prepare_multi_object_goals.py --run-dir $BASE_RUN_DIR" >&2
    exit 2
  fi
done
for encoder in "${ENCODERS[@]}"; do
  if [[ "$encoder" != "flux_ae" && "$encoder" != "dinov3" ]]; then
    echo "Unknown encoder '$encoder'; use flux_ae and/or dinov3." >&2
    exit 2
  fi
done
if (( ${#GPUS[@]} == 0 )); then
  echo "GPUS must list at least one GPU index." >&2
  exit 2
fi
VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd' ')"
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPUS entries must be integers; got '$gpu'." >&2
    exit 2
  fi
  if [[ -n "$VISIBLE_GPUS" && " $VISIBLE_GPUS " != *" $gpu "* ]]; then
    echo "GPU $gpu is not present on this host (have: $VISIBLE_GPUS)." >&2
    exit 2
  fi
done
if (( ${#FD_EPS[@]} != 3 )); then
  echo "FD_EPS must contain exactly three values (X Y Z); got '${FD_EPS[*]}'." >&2
  exit 2
fi
# A half-step wider than its axis clips to the same box corners for every
# particle, which silently turns the finite difference into a constant.
"$PYTHON_BIN" - "${FD_EPS[@]}" "${BOUNDS[@]}" <<'PY' || exit 2
import sys
eps = [float(v) for v in sys.argv[1:4]]
bounds = [float(v) for v in sys.argv[4:10]]
axes = ("x", "y", "z")
bad = []
for index, axis in enumerate(axes):
    low, high = bounds[2 * index], bounds[2 * index + 1]
    span = high - low
    if eps[index] >= span / 2.0:
        bad.append(f"  {axis}: fd_eps={eps[index]:g} m vs bounds span {span:.3f} m [{low:g}, {high:g}]")
if bad:
    print("FD_EPS is too wide for the search bounds:", file=sys.stderr)
    print("\n".join(bad), file=sys.stderr)
    print(
        "Probes clip to the bounds, so the finite difference degenerates into the\n"
        "same box-corner secant for every particle. Use a smaller FD_EPS, or widen\n"
        "BOUNDS if the larger probe is deliberate.",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
if [[ -e "$SUITE_DIR" ]]; then
  echo "Refusing to overwrite existing suite: $SUITE_DIR" >&2
  exit 2
fi

# The goal image is the only thing the objective sees. The pose below is passed
# as --diagnostic-goal-eef, which the optimizer uses for plots and reported
# distances only -- it never enters the energy or the gradients.
goal_pose() {
  "$PYTHON_BIN" - "$GOALS_DIR/$1/metadata.json" <<'PY'
import json, sys
print(" ".join(f"{v:.6f}" for v in json.load(open(sys.argv[1]))["target_eef"]))
PY
}

COMMON_ARGS=(
  --run-dir "$BASE_RUN_DIR"
  --goal-latent-source reencode
  --device cuda:0
  --particles "$PARTICLES"
  --iterations "$ITERATIONS"
  --init-mode start-cloud
  --init-radius 0.005 0.005 0.003
  --bounds "${BOUNDS[@]}"
  --latent-views agentview
  --latent-distance token_cosine
  --transport svgd
  --latent-weight 1.0
  --repulsion-weight 0.01
  --fd-eps "${FD_EPS[@]}"
  --step-size 0.01
  --temperature 0.10
  --max-update-norm 0.02
  --bandwidth-scale 1.0
  --move-steps 40
  --settle-steps 8
  --controller-gain 15.0
  --repeatability-particles 1
  --rollout-trace-mode base
  --seed 0
  --save-rollout-traces
  --save-all-particles
  --verbose-evaluations
)

run_trial() {
  local gpu="$1"
  local goal_id="$2"
  local encoder="$3"
  local name="${goal_id}__${encoder}"
  local trial_dir="$TRIAL_ROOT/$name"
  local pose
  read -r -a pose <<< "$(goal_pose "$goal_id")"

  local args=(
    -u -B experiments/libero/svgd_endpoint.py
    --out-dir "$trial_dir"
    --goal "$GOALS_DIR/$goal_id/goal.png"
    --feature-encoder "$encoder"
    --diagnostic-goal-eef "${pose[@]}"
    "${COMMON_ARGS[@]}"
  )
  if [[ "$encoder" == "flux_ae" ]]; then
    args+=(--editor-ae "$FLUX2_AE_MODEL_PATH" --flux2-src "$FLUX2_SRC")
  else
    args+=(--dino-model "$DINO_MODEL")
  fi

  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf '[dry-run] gpu=%s trial=%s goal_eef=(%s)\n' "$gpu" "$name" "${pose[*]}"
    return 0
  fi

  mkdir -p "$trial_dir"
  {
    echo "physical_gpu=$gpu"
    echo "goal_image=$GOALS_DIR/$goal_id/goal.png"
    echo "diagnostic_goal_eef=${pose[*]}"
    echo "fd_eps=${FD_EPS[*]}"
    echo "bounds=${BOUNDS[*]}"
    printf 'command=imagewam_python'
    printf ' %q' "${args[@]}"
    echo
  } > "$trial_dir/command.txt"

  echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$gpu trial=$name"
  if (
    export CUDA_VISIBLE_DEVICES="$gpu"
    imagewam_python "${args[@]}"
    imagewam_python -u -B experiments/libero/plot_svgd_latent_pull.py \
      --history "$trial_dir/history.json" \
      --prefix diagnostics \
      --title "$name"
  ) > "$trial_dir/backend.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$gpu trial=$name"
    return 0
  fi
  local status="$?"
  echo "$status" > "$trial_dir/failed.status"
  echo "[failed] $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$gpu trial=$name status=$status (see $trial_dir/backend.log)" >&2
  return "$status"
}

# One worker per GPU, each draining its own slice of the trial list in order.
# Serialising within a worker is what keeps memory to one trial per card.
gpu_worker() {
  local gpu="$1"
  shift
  local failures=0
  local spec
  for spec in "$@"; do
    if ! run_trial "$gpu" "${spec%%:*}" "${spec##*:}"; then
      failures=$(( failures + 1 ))
      echo "[continue] gpu=$gpu still has queued trials; ${spec} is marked failed" >&2
    fi
  done
  return "$failures"
}

# Nothing above this line may write to disk: a dry run that creates $SUITE_DIR
# leaves behind exactly the directory whose absence the guard above checks for,
# so the real launch is refused.
if [[ "${DRY_RUN:-false}" != "true" ]]; then
  mkdir -p "$TRIAL_ROOT"
  exec > "$SUITE_DIR/launcher.log" 2>&1
  echo "$$" > "$SUITE_DIR/launcher.pid"
fi

# Goal outer, encoder inner: an interrupted suite then still leaves whole
# FLUX-vs-DINO pairs, which is the comparison a two-encoder suite exists to make.
TRIAL_SPECS=()
for goal_id in "${GOAL_IDS[@]}"; do
  for encoder in "${ENCODERS[@]}"; do
    TRIAL_SPECS+=("${goal_id}:${encoder}")
  done
done

echo "[plan] suite=$SUITE_DIR"
echo "[plan] gpus=${GPUS[*]} iterations=$ITERATIONS particles=$PARTICLES views=agentview"
echo "[plan] bounds=${BOUNDS[*]}"
echo "[plan] fd_eps=${FD_EPS[*]}"
echo "[plan] encoders=${ENCODERS[*]} dino_model=$DINO_MODEL"
echo "[plan] goals=${#GOAL_IDS[@]} trials=${#TRIAL_SPECS[@]} across ${#GPUS[@]} gpu(s)"

# Deal the trials round-robin so the cards finish together rather than one taking
# every long trial.
declare -A QUEUE=()
for index in "${!TRIAL_SPECS[@]}"; do
  slot=$(( index % ${#GPUS[@]} ))
  QUEUE[$slot]+="${TRIAL_SPECS[$index]} "
done
for slot in "${!GPUS[@]}"; do
  echo "[plan] gpu=${GPUS[$slot]} queue=${QUEUE[$slot]:-<empty>}"
done

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  for slot in "${!GPUS[@]}"; do
    read -r -a queued <<< "${QUEUE[$slot]:-}"
    for spec in "${queued[@]:-}"; do
      [[ -n "$spec" ]] && run_trial "${GPUS[$slot]}" "${spec%%:*}" "${spec##*:}"
    done
  done
  exit 0
fi

WORKER_PIDS=()
for slot in "${!GPUS[@]}"; do
  read -r -a queued <<< "${QUEUE[$slot]:-}"
  if (( ${#queued[@]} == 0 )); then
    continue
  fi
  gpu_worker "${GPUS[$slot]}" "${queued[@]}" &
  WORKER_PIDS+=("$!")
done

failed=0
for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid" || failed=1
done

if (( failed )); then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/failed.status"
  echo "[done] suite finished with at least one failed trial"
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/complete.status"
echo "[done] all ${#TRIAL_SPECS[@]} trials finished"
