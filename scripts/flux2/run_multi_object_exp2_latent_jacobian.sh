#!/usr/bin/env bash
# Experiment 2 -- estimated-latent-Jacobian arm of the ImageSTL Visual-SVPIO spec.
#
# Section 10: the simulator is treated as a black box that turns a joint
# configuration into an image.  Nothing differentiates MuJoCo or the renderer.
#
#     dL/dq = J_zq^T dL/dz
#
# with dL/dz from autograd through the frozen FLUX.2 AE and J_zq = dz/dq
# estimated by central differences over image observations, then maintained
# online with rank-one Broyden updates.  Particles are short joint-increment
# sequences rolled out through the local linear model z_{k+1} = z_k + J dq_k and
# transported by the same SVGD update the endpoint runs use; only the first
# increment is executed before re-observing.
#
#   bash scripts/flux2/run_multi_object_exp2_latent_jacobian.sh [suite_name]
#
# Strictly sequential on one GPU.  It does not self-detach -- launch it so it
# outlives the shell:
#
#   setsid nohup bash scripts/flux2/run_multi_object_exp2_latent_jacobian.sh NAME \
#     > /dev/null 2>&1 < /dev/null &
#
# then follow $BASE_RUN_DIR/$NAME/launcher.log.
#
# ITERATIONS counts closed-loop control cycles here, not optimizer sweeps: each
# cycle captures one image, runs SVGD_ITERS_PER_STEP updates on the local linear
# model (no rollouts), and commands one bounded joint step.
#
# Environment overrides:
#   GOAL_IDS    space-separated goal ids under $BASE_RUN_DIR/goals
#   GPU         physical GPU index (default 2)
#   ITERATIONS  closed-loop control cycles per trial (default 45)
#   PARTICLES   particles per trial (default 15)
#   PLANNER     svgd_local_linear (default) or direct
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
SUITE_NAME="${1:-${SUITE_NAME:-multi_object_exp2_latent_jacobian_15p_45it_flux_v1}}"
SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
TRIAL_ROOT="$SUITE_DIR/trials"

GPU="${GPU:-2}"
ITERATIONS="${ITERATIONS:-45}"
PARTICLES="${PARTICLES:-15}"
PLANNER="${PLANNER:-svgd_local_linear}"
SVGD_ITERS_PER_STEP="${SVGD_ITERS_PER_STEP:-20}"
read -r -a GOAL_IDS <<< "${GOAL_IDS:-akita_black_bowl_above akita_black_bowl_tilt_roll30 plate_tilt_roll30 cream_cheese_above table_far_side_yaw45}"

if [[ ! -f "$BASE_RUN_DIR/manifest.json" ]]; then
  echo "Missing scene manifest: $BASE_RUN_DIR/manifest.json" >&2
  echo "Build it with: bash scripts/flux2/prepare_svgd_scene.sh multi-object" >&2
  exit 2
fi

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
# as --diagnostic-goal-eef, which is used for plots and reported distances only.
goal_pose() {
  "$PYTHON_BIN" - "$GOALS_DIR/$1/metadata.json" <<'PY'
import json, sys
print(" ".join(f"{v:.6f}" for v in json.load(open(sys.argv[1]))["target_eef"]))
PY
}

# Step bounds come from probe_latent_jacobian.py on this scene, not from taste.
# Held-out cosine alignment between J dq and the true dz peaks near |dq| = 0.005
# rad (+0.46) and decays with step size (+0.28 at 0.02, +0.18 at 0.05), so the
# commanded step is capped an order of magnitude below the spec's hardware
# defaults to keep the linear model inside the range where it still points the
# right way.
COMMON_ARGS=(
  --run-dir "$BASE_RUN_DIR"
  --goal-latent-source reencode
  --feature-encoder flux_ae
  --editor-ae "$FLUX2_AE_MODEL_PATH"
  --flux2-src "$FLUX2_SRC"
  --device cuda:0
  --latent-views agentview
  --iterations "$ITERATIONS"
  --planner "$PLANNER"
  --particles "$PARTICLES"
  --horizon 10
  --svgd-iters-per-step "$SVGD_ITERS_PER_STEP"
  --cosine-weight 1.0
  --l2-weight 0.0
  --temporal-mode softmin
  --time-softmin-temperature 0.05
  --fd-delta-rad 0.005
  --refresh-every-steps 25
  --max-relative-prediction-error 0.5
  --bad-prediction-patience 3
  --temperature 0.10
  --repulsion-weight 0.01
  --latent-weight 1.0
  --bandwidth-scale 1.0
  --step-size 0.01
  --max-update-norm 0.02
  --init-scale-rad 0.002
  --safety-weight 0.25
  --safety-temperature 0.02
  --smooth-weight 0.001
  --gradient-step-size 0.05
  --max-joint-step-rad 0.01
  --max-joint-step-norm-rad 0.02
  --joint-limit-margin-rad 0.05
  --seed 0
  --save-camera-frames
  --verbose
)

run_trial() {
  local goal_id="$1"
  local trial_dir="$TRIAL_ROOT/$goal_id"
  local pose
  read -r -a pose <<< "$(goal_pose "$goal_id")"

  local args=(
    -u -B experiments/libero/svgd_latent_jacobian.py
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
    echo "experiment=2_estimated_latent_jacobian"
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

echo "[plan] experiment=2 gradient_source=finite_difference_latent_jacobian_with_broyden"
echo "[plan] suite=$SUITE_DIR"
echo "[plan] gpu=$GPU control_cycles=$ITERATIONS particles=$PARTICLES planner=$PLANNER encoder=flux_ae"
echo "[plan] svgd_iters_per_cycle=$SVGD_ITERS_PER_STEP horizon=10 views=agentview"
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
