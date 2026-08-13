#!/usr/bin/env bash
# Experiment 1 -- differentiable-gradient arm of the ImageSTL Visual-SVPIO spec.
#
# The spec's Experiment A propagates an exact gradient from the visual objective
# back to the controls.  Its literal recipe (MJX -> nvdiffrast -> FLUX -> JAX
# VJP) needs jax, mujoco.mjx and nvdiffrast, none of which this environment has,
# and the LIBERO scene is not ported to MJX.  The differentiable path that does
# exist here is svgd_endpoint_differentiable.py: a learned surrogate
#
#     theta -> terminal FLUX feature -> distance to the goal feature
#
# differentiated with torch.autograd.  MuJoCo is still used for ground-truth
# scoring, best-particle selection, and online surrogate refinement -- never to
# estimate a gradient.  So the gradient is exact for the surrogate, not for the
# world, and that is the honest label for this arm.
#
#   bash scripts/flux2/run_multi_object_exp1_differentiable.sh [suite_name]
#
# Strictly sequential on one GPU: the next goal starts only after the previous
# one exits, so peak memory is one trial's worth.  It does not self-detach --
# launch it so it outlives the shell:
#
#   setsid nohup bash scripts/flux2/run_multi_object_exp1_differentiable.sh NAME \
#     > /dev/null 2>&1 < /dev/null &
#
# then follow $BASE_RUN_DIR/$NAME/launcher.log.
#
# Environment overrides:
#   GOAL_IDS    space-separated goal ids under $BASE_RUN_DIR/goals
#   GPU         physical GPU index (default 1)
#   ITERATIONS  SVGD iterations per trial (default 45)
#   PARTICLES   particles per trial (default 15)
#   DRY_RUN     true to print the plan without launching anything
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
SUITE_NAME="${1:-${SUITE_NAME:-multi_object_exp1_differentiable_15p_45it_flux_v1}}"
SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
TRIAL_ROOT="$SUITE_DIR/trials"

GPU="${GPU:-1}"
ITERATIONS="${ITERATIONS:-45}"
PARTICLES="${PARTICLES:-15}"
read -r -a GOAL_IDS <<< "${GOAL_IDS:-akita_black_bowl_above akita_black_bowl_tilt_roll30 plate_tilt_roll30 cream_cheese_above table_far_side_yaw45}"

# The empty-table sweep searches a narrow slab at x in [-0.06, -0.04]. None of
# the object-hover goals live there, so the prior has to cover the whole
# reachable box or start-cloud init rejects the start pose outright.
read -r -a BOUNDS <<< "${BOUNDS:--0.20 0.09 -0.32 0.32 0.98 1.16}"

if [[ ! -f "$BASE_RUN_DIR/manifest.json" ]]; then
  echo "Missing scene manifest: $BASE_RUN_DIR/manifest.json" >&2
  echo "Build it with: bash scripts/flux2/prepare_svgd_scene.sh multi-object" >&2
  exit 2
fi

# Validate every goal before launching: discovering a typo eight hours into a
# five-trial suite is the expensive way to find it.
for goal_id in "${GOAL_IDS[@]}"; do
  if [[ ! -f "$GOALS_DIR/$goal_id/goal.png" || ! -f "$GOALS_DIR/$goal_id/metadata.json" ]]; then
    echo "No rendered goal at $GOALS_DIR/$goal_id/" >&2
    echo "Render the catalogue with: $PYTHON_BIN -u -B experiments/libero/prepare_multi_object_goals.py --run-dir $BASE_RUN_DIR" >&2
    exit 2
  fi
done
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
  # No --goal-latent-source here: unlike svgd_endpoint.py this runner has no such
  # flag and always re-encodes the goal image, which is the setting we want.
  --feature-encoder flux_ae
  --editor-ae "$FLUX2_AE_MODEL_PATH"
  --flux2-src "$FLUX2_SRC"
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
  --step-size 0.01
  --temperature 0.10
  --max-update-norm 0.02
  --bandwidth-scale 1.0
  --move-steps 40
  --settle-steps 8
  --controller-gain 15.0
  --repeatability-particles 1
  --seed 0
  --save-all-particles
  --verbose-evaluations
  # Section 18.6 in spirit: audit the surrogate's gradient against MuJoCo finite
  # differences on held-out samples before spending the iteration budget on it.
  --gradient-audit-samples 16
  --gradient-audit-fd-eps 0.01 0.04 0.01
)

run_trial() {
  local goal_id="$1"
  local trial_dir="$TRIAL_ROOT/$goal_id"
  local pose
  read -r -a pose <<< "$(goal_pose "$goal_id")"

  local args=(
    -u -B experiments/libero/svgd_endpoint_differentiable.py
    --out-dir "$trial_dir"
    --goal "$GOALS_DIR/$goal_id/goal.png"
    --diagnostic-goal-eef "${pose[@]}"
    "${COMMON_ARGS[@]}"
  )

  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf '[dry-run] gpu=%s trial=%s goal_eef=(%s)\n' "$GPU" "$goal_id" "${pose[*]}"
    return 0
  fi

  # Sibling file, not inside $trial_dir: the runner refuses to start into a
  # non-empty output directory, and it writes its own command.txt there.
  {
    echo "experiment=1_differentiable_surrogate"
    echo "physical_gpu=$GPU"
    echo "goal_image=$GOALS_DIR/$goal_id/goal.png"
    echo "diagnostic_goal_eef=${pose[*]}"
    printf 'command=imagewam_python'
    printf ' %q' "${args[@]}"
    echo
  } > "$TRIAL_ROOT/$goal_id.launch.txt"

  echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) trial=$goal_id"
  # Capture the status explicitly: after an `if cmd; then ... fi` whose
  # condition failed, `$?` is the status of the `if` itself, which is 0.
  local status=0
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    imagewam_python "${args[@]}"
    imagewam_python -u -B experiments/libero/plot_svgd_latent_pull.py \
      --history "$trial_dir/history.json" \
      --prefix diagnostics \
      --title "exp1_$goal_id"
  ) > "$TRIAL_ROOT/$goal_id.backend.log" 2>&1 || status=$?
  if (( status == 0 )); then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] $(date -u +%Y-%m-%dT%H:%M:%SZ) trial=$goal_id"
    return 0
  fi
  echo "$status" > "$TRIAL_ROOT/$goal_id.failed.status"
  echo "[failed] $(date -u +%Y-%m-%dT%H:%M:%SZ) trial=$goal_id status=$status (see $TRIAL_ROOT/$goal_id.backend.log)" >&2
  return "$status"
}

# Nothing above this line may write to disk: a dry run that creates $SUITE_DIR
# leaves behind exactly the directory whose absence the guard above checks for,
# so the real launch is refused.
if [[ "${DRY_RUN:-false}" != "true" ]]; then
  mkdir -p "$TRIAL_ROOT"
  exec > "$SUITE_DIR/launcher.log" 2>&1
  echo "$$" > "$SUITE_DIR/launcher.pid"
fi

echo "[plan] experiment=1 gradient_source=learned_differentiable_surrogate"
echo "[plan] suite=$SUITE_DIR"
echo "[plan] gpu=$GPU iterations=$ITERATIONS particles=$PARTICLES views=agentview encoder=flux_ae"
echo "[plan] bounds=${BOUNDS[*]}"
echo "[plan] goals=${#GOAL_IDS[@]} trials=${#GOAL_IDS[@]} (strictly sequential)"

failed=0
for goal_id in "${GOAL_IDS[@]}"; do
  if ! run_trial "$goal_id"; then
    failed=1
    echo "[continue] remaining trials still run; $goal_id is marked failed" >&2
  fi
done

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

if (( failed )); then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/failed.status"
  echo "[done] suite finished with at least one failed trial"
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/complete.status"
echo "[done] all ${#GOAL_IDS[@]} trials finished"
