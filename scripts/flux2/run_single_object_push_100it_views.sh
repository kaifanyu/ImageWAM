#!/usr/bin/env bash
# Prepare and run the one-object goal-image push comparison.
#
# Default invocation only prepares/validates the scene.  Launch the long run
# explicitly with STAGE=run (or use STAGE=all to prepare and launch).
set -euo pipefail

cd "$(dirname "$0")/../.."
source scripts/common.sh
imagewam_init .

STAGE="${STAGE:-prepare}"
BASE_RUN_DIR="${BASE_RUN_DIR:-runs/single_object_push_preview}"
SUITE_NAME="${SUITE_NAME:-push_goal_15p_100it_agent_vs_wrist_v1}"
SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
GOAL_IMAGE="${GOAL_IMAGE:-$BASE_RUN_DIR/goal_oracle.png}"
PARTICLES="${PARTICLES:-15}"
ITERATIONS="${ITERATIONS:-100}"
REPULSION_WEIGHT="${REPULSION_WEIGHT:-0.01}"
DINO_MODEL="${DINO_MODEL:-vit_base_patch16_dinov3.lvd1689m}"
SURROGATE_SAMPLES="${SURROGATE_SAMPLES:-384}"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2}"

prepare_scene() {
  local args=(
    -u -B experiments/libero/prepare_single_object_push.py
    --run-dir "$BASE_RUN_DIR"
  )
  [[ "${FORCE:-false}" == "true" ]] && args+=(--force)
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
  imagewam_python "${args[@]}"
}

validate_launch() {
  imagewam_require_env FLUX2_AE_MODEL_PATH
  imagewam_require_env FLUX2_SRC
  if (( ${#GPU_LIST[@]} < 3 )); then
    echo "GPU_IDS must contain at least three physical GPU indices." >&2
    exit 2
  fi
  if [[ ! -f "$BASE_RUN_DIR/manifest.json" || ! -f "$GOAL_IMAGE" ]]; then
    echo "Prepare the scene first: STAGE=prepare bash $0" >&2
    exit 2
  fi
  if [[ -e "$SUITE_DIR" ]]; then
    echo "Refusing to overwrite existing suite: $SUITE_DIR" >&2
    exit 2
  fi
}

launch_suite() {
  validate_launch
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exec bash "$0" --worker
  fi
  mkdir -p "$SUITE_DIR"
  nohup bash "$0" --worker > "$SUITE_DIR/launcher.log" 2>&1 < /dev/null &
  echo "$!" > "$SUITE_DIR/launcher.pid"
  echo "Started the six-trial one-object push comparison."
  echo "Suite: $SUITE_DIR"
  echo "Watch: tail -f $SUITE_DIR/launcher.log"
}

if [[ "${1:-}" != "--worker" ]]; then
  case "$STAGE" in
    prepare)
      prepare_scene
      ;;
    run)
      launch_suite
      ;;
    all)
      prepare_scene
      launch_suite
      ;;
    *)
      echo "Unknown STAGE=$STAGE. Use prepare | run | all." >&2
      exit 2
      ;;
  esac
  exit 0
fi

echo "[plan] suite=$SUITE_DIR"
echo "[plan] goal_image=$GOAL_IMAGE"
echo "[plan] particles=$PARTICLES iterations=$ITERATIONS repulsion=$REPULSION_WEIGHT"
echo "[plan] gpu=${GPU_LIST[0]}: flux_ae agentview -> flux_ae wrist"
echo "[plan] gpu=${GPU_LIST[1]}: DINO agentview -> differentiable agentview"
echo "[plan] gpu=${GPU_LIST[2]}: DINO wrist -> differentiable wrist"
echo "[plan] differentiable bootstrap=$SURROGATE_SAMPLES fresh Latin-hypercube rollouts"
echo "[plan] differentiable guard=16-point gradient audit, minimum mean cosine=0.50"
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] no directories or GPU processes created"
  exit 0
fi

export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONUNBUFFERED=1
TRIAL_ROOT="$SUITE_DIR/trials"
mkdir -p "$TRIAL_ROOT"

COMMON_ARGS=(
  --run-dir "$BASE_RUN_DIR"
  --goal "$GOAL_IMAGE"
  --device cuda:0
  --particles "$PARTICLES"
  --iterations "$ITERATIONS"
  --init-mode start-cloud
  --init-radius 0.004 0.004 0.003
  --bounds -0.095 -0.055 -0.14 0.14 0.915 0.94
  --latent-distance token_cosine
  --transport svgd
  --latent-weight 1.0
  --repulsion-weight "$REPULSION_WEIGHT"
  --step-size 0.005
  --temperature 0.10
  --max-update-norm 0.01
  --bandwidth-scale 1.0
  --move-steps 48
  --settle-steps 8
  --controller-gain 12.0
  --fixed-arc-height 0.0
  --fixed-midpoint-x 0.0
  --repeatability-particles 1
  --seed 0
  --save-all-particles
  --verbose-evaluations
)

run_fd_trial() {
  local name="$1" gpu="$2" encoder="$3" view="$4"
  local trial_dir="$TRIAL_ROOT/$name"
  local args=(
    -u -B experiments/libero/svgd_endpoint.py
    --out-dir "$trial_dir"
    --feature-encoder "$encoder"
    --goal-latent-source reencode
    --latent-views "$view"
    --fd-eps 0.005 0.02 0.005
    --save-rollout-traces
    --rollout-trace-mode base
    "${COMMON_ARGS[@]}"
  )
  if [[ "$encoder" == "flux_ae" ]]; then
    args+=(--editor-ae "$FLUX2_AE_MODEL_PATH" --flux2-src "$FLUX2_SRC")
  else
    args+=(--dino-model "$DINO_MODEL")
  fi
  mkdir -p "$trial_dir"
  {
    echo "physical_gpu=$gpu"
    printf 'command=imagewam_python'
    printf ' %q' "${args[@]}"
    echo
  } > "$trial_dir/command.txt"
  echo "[start] trial=$name gpu=$gpu encoder=$encoder view=$view"
  if CUDA_VISIBLE_DEVICES="$gpu" imagewam_python "${args[@]}" \
      > "$trial_dir/backend.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] trial=$name"
  else
    local status="$?"
    echo "$status" > "$trial_dir/failed.status"
    echo "[failed] trial=$name status=$status" >&2
    return "$status"
  fi
}

run_differentiable_trial() {
  local name="$1" gpu="$2" view="$3"
  local trial_dir="$TRIAL_ROOT/$name"
  local backend_log="$TRIAL_ROOT/${name}.backend.log"
  local args=(
    -u -B experiments/libero/svgd_endpoint_differentiable.py
    --out-dir "$trial_dir"
    --feature-encoder flux_ae
    --editor-ae "$FLUX2_AE_MODEL_PATH"
    --flux2-src "$FLUX2_SRC"
    --latent-views "$view"
    --surrogate-samples "$SURROGATE_SAMPLES"
    --validation-fraction 0.20
    --projection-dim 512
    --surrogate-hidden-dim 512
    --surrogate-train-steps 5000
    --online-train-steps 150
    --gradient-audit-samples 16
    --gradient-audit-fd-eps 0.005 0.02 0.005
    --minimum-gradient-audit-cosine 0.50
    "${COMMON_ARGS[@]}"
  )
  mkdir -p "$trial_dir"
  echo "[start] trial=$name gpu=$gpu encoder=flux_ae+differentiable view=$view"
  if CUDA_VISIBLE_DEVICES="$gpu" imagewam_python "${args[@]}" \
      > "$backend_log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] trial=$name"
  else
    local status="$?"
    echo "$status" > "$trial_dir/failed.status"
    echo "[failed] trial=$name status=$status" >&2
    return "$status"
  fi
}

worker_flux() {
  run_fd_trial flux_ae_agentview "${GPU_LIST[0]}" flux_ae agentview
  run_fd_trial flux_ae_wrist "${GPU_LIST[0]}" flux_ae wrist
}

worker_agent() {
  run_fd_trial dino_agentview "${GPU_LIST[1]}" dinov3 agentview
  run_differentiable_trial differentiable_agentview "${GPU_LIST[1]}" agentview
}

worker_wrist() {
  run_fd_trial dino_wrist "${GPU_LIST[2]}" dinov3 wrist
  run_differentiable_trial differentiable_wrist "${GPU_LIST[2]}" wrist
}

worker_flux & pid_flux="$!"
worker_agent & pid_agent="$!"
worker_wrist & pid_wrist="$!"
failed=0
wait "$pid_flux" || failed=1
wait "$pid_agent" || failed=1
wait "$pid_wrist" || failed=1

if (( failed )); then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/failed.status"
  exit 1
fi

imagewam_python -u -B experiments/libero/summarize_single_object_push.py \
  --suite-dir "$SUITE_DIR"
date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/complete.status"
echo "[complete] all six trials and held-out object diagnostics finished"
