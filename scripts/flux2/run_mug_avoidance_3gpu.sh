#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE_RUN_DIR="${BASE_RUN_DIR:-runs/living_room_mug_obstacle}"
TEST_DIR="${TEST_DIR:-$BASE_RUN_DIR/mug_avoidance_test}"
PARTICLES="${PARTICLES:-20}"
ITERATIONS="${ITERATIONS:-10}"
TRACE_MODE="${TRACE_MODE:-base}"
INIT_MODE="${INIT_MODE:-collision-cloud}"
SAVE_ALL_PARTICLES="${SAVE_ALL_PARTICLES:-false}"
VERBOSE_EVALUATIONS="${VERBOSE_EVALUATIONS:-false}"

if [[ "${1:-}" != "--worker" ]]; then
  SUITE_NAME="${1:-svgd_path_$(date -u +%Y%m%dT%H%M%SZ)}"
  SUITE_DIR="$TEST_DIR/$SUITE_NAME"
  if [[ ! -f "$TEST_DIR/manifest.json" ]]; then
    echo "Missing avoidance test: $TEST_DIR/manifest.json" >&2
    echo "Run experiments/libero/prepare_obstacle_avoidance_test.py first." >&2
    exit 2
  fi
  if [[ -e "$SUITE_DIR" && "${RESUME:-false}" != "true" ]]; then
    echo "Suite already exists: $SUITE_DIR" >&2
    exit 2
  fi
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exec bash "$0" --worker "$SUITE_NAME"
  fi
  mkdir -p "$SUITE_DIR"
  nohup bash "$0" --worker "$SUITE_NAME" \
    > "$SUITE_DIR/launcher.log" 2>&1 < /dev/null &
  echo "$!" > "$SUITE_DIR/launcher.pid"
  echo "Started mug-avoidance path trials."
  echo "Suite: $SUITE_DIR"
  echo "Watch: tail -f $SUITE_DIR/launcher.log"
  exit 0
fi

SUITE_NAME="${2:?Worker mode requires a suite name}"
SUITE_DIR="$TEST_DIR/$SUITE_NAME"
TRIAL_ROOT="$SUITE_DIR/trials"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2}"
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU." >&2
  exit 2
fi

# name|objective|transport
TRIALS=(
  "metric_rms|rms|svgd"
  "metric_cosine|cosine|svgd"
  "metric_token|token_cosine|svgd"
  "particle_gd_token|token_cosine|particle_gd"
)

echo "[plan] suite=$SUITE_DIR particles=$PARTICLES iterations=$ITERATIONS"
echo "[plan] init=$INIT_MODE trace=$TRACE_MODE gpus=${GPU_LIST[*]}"
for ((index=0; index<${#TRIALS[@]}; index++)); do
  IFS='|' read -r name objective transport <<< "${TRIALS[$index]}"
  echo "[plan] gpu=${GPU_LIST[$((index % ${#GPU_LIST[@]}))]} trial=$name objective=$objective transport=$transport"
done
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] no suite directory or GPU process was created"
  exit 0
fi

mkdir -p "$TRIAL_ROOT"
source scripts/common.sh
imagewam_init .
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env FLUX2_SRC
export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONUNBUFFERED=1

run_trial() {
  local gpu="$1"
  local spec="$2"
  local name objective transport
  IFS='|' read -r name objective transport <<< "$spec"
  local trial_dir="$TRIAL_ROOT/$name"
  mkdir -p "$trial_dir"
  if [[ "${RESUME:-false}" == "true" && -f "$trial_dir/best_metadata.json" ]]; then
    echo "[skip] $name"
    return 0
  fi

  local args=(
    -u -B experiments/libero/svgd_obstacle_path.py
    --run-dir "$BASE_RUN_DIR"
    --test-dir "$TEST_DIR"
    --out-dir "$trial_dir"
    --editor-ae "$FLUX2_AE_MODEL_PATH"
    --flux2-src "$FLUX2_SRC"
    --device cuda:0
    --particles "$PARTICLES"
    --iterations "$ITERATIONS"
    --init-mode "$INIT_MODE"
    --latent-distance "$objective"
    --latent-views agentview
    --transport "$transport"
    --repulsion-weight 0.0
    --fd-eps 0.01 0.01
    --step-size 0.01
    --temperature 0.10
    --max-update-norm 0.02
    --trace-mode "$TRACE_MODE"
    --seed 0
  )
  [[ "$SAVE_ALL_PARTICLES" == "true" ]] && args+=(--save-all-particles)
  [[ "$VERBOSE_EVALUATIONS" == "true" ]] && args+=(--verbose-evaluations)
  {
    echo "physical_gpu=$gpu"
    printf "command=imagewam_python"
    printf " %q" "${args[@]}"
    echo
  } > "$trial_dir/command.txt"

  echo "[start] trial=$name gpu=$gpu"
  if (
    export CUDA_VISIBLE_DEVICES="$gpu"
    imagewam_python "${args[@]}"
  ) > "$trial_dir/backend.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] trial=$name gpu=$gpu"
  else
    local status="$?"
    echo "$status" > "$trial_dir/failed.status"
    echo "[failed] trial=$name gpu=$gpu status=$status" >&2
    return 1
  fi
}

run_queue() {
  local slot="$1"
  local failed=0
  local index
  for ((index=slot; index<${#TRIALS[@]}; index+=${#GPU_LIST[@]})); do
    run_trial "${GPU_LIST[$slot]}" "${TRIALS[$index]}" || failed=1
  done
  return "$failed"
}

PIDS=()
for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
  run_queue "$slot" &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done

if compgen -G "$TRIAL_ROOT/*/history.json" > /dev/null; then
  imagewam_python -u -B experiments/libero/summarize_obstacle_path_trials.py \
    --suite-dir "$SUITE_DIR"
fi
if (( FAILED != 0 )); then
  echo "[done-with-failures] $SUITE_DIR" >&2
  exit 1
fi
echo "[done] $SUITE_DIR"
