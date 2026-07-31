#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE_RUN_DIR="${BASE_RUN_DIR:-runs/living_room_mug_obstacle}"
TEST_DIR="${TEST_DIR:-$BASE_RUN_DIR/mug_avoidance_test}"
GOAL_IMAGE="${GOAL_IMAGE:-$TEST_DIR/goal_avoidance.png}"
SWEEP_PROFILE="${SWEEP_PROFILE:-repulsion}"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2}"

if (( ${#GPU_LIST[@]} < 3 )); then
  echo "GPU_IDS must contain at least three GPU indices." >&2
  exit 2
fi

case "$SWEEP_PROFILE" in
  repulsion)
    # name|repulsion|step|temperature|bandwidth
    TRIALS=(
      "repulsion_000|0.0|0.01|0.10|1.0"
      "repulsion_001|0.01|0.01|0.10|1.0"
      "repulsion_005|0.05|0.01|0.10|1.0"
    )
    ;;
  step)
    TRIALS=(
      "step_005|0.01|0.005|0.10|1.0"
      "step_010|0.01|0.010|0.10|1.0"
      "step_020|0.01|0.020|0.10|1.0"
    )
    ;;
  temperature)
    TRIALS=(
      "temperature_005|0.01|0.01|0.05|1.0"
      "temperature_010|0.01|0.01|0.10|1.0"
      "temperature_020|0.01|0.01|0.20|1.0"
    )
    ;;
  bandwidth)
    TRIALS=(
      "bandwidth_050|0.01|0.01|0.10|0.5"
      "bandwidth_100|0.01|0.01|0.10|1.0"
      "bandwidth_200|0.01|0.01|0.10|2.0"
    )
    ;;
  *)
    echo "Unknown SWEEP_PROFILE=$SWEEP_PROFILE" >&2
    echo "Use repulsion | step | temperature | bandwidth." >&2
    exit 2
    ;;
esac

if [[ "${1:-}" != "--worker" ]]; then
  SWEEP_NAME="${1:-goal_image_${SWEEP_PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)}"
  SWEEP_DIR="$TEST_DIR/$SWEEP_NAME"
  if [[ ! -f "$GOAL_IMAGE" ]]; then
    echo "Missing goal image: $GOAL_IMAGE" >&2
    exit 2
  fi
  if [[ -e "$SWEEP_DIR" ]]; then
    echo "Refusing to reuse existing sweep: $SWEEP_DIR" >&2
    exit 2
  fi
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exec bash "$0" --worker "$SWEEP_NAME"
  fi
  mkdir -p "$SWEEP_DIR"
  nohup bash "$0" --worker "$SWEEP_NAME" \
    > "$SWEEP_DIR/launcher.log" 2>&1 < /dev/null &
  echo "$!" > "$SWEEP_DIR/launcher.pid"
  echo "Started three-GPU goal-image hyperparameter sweep."
  echo "Profile: $SWEEP_PROFILE"
  echo "Sweep: $SWEEP_DIR"
  echo "Watch: tail -f $SWEEP_DIR/launcher.log"
  exit 0
fi

SWEEP_NAME="${2:?Worker mode requires a sweep name}"
SWEEP_DIR="$TEST_DIR/$SWEEP_NAME"
TRIAL_ROOT="$SWEEP_DIR/trials"

echo "[plan] sweep=$SWEEP_DIR profile=$SWEEP_PROFILE"
for index in 0 1 2; do
  IFS='|' read -r name repulsion step temperature bandwidth \
    <<< "${TRIALS[$index]}"
  echo "[plan] gpu=${GPU_LIST[$index]} trial=$name repulsion=$repulsion step=$step temperature=$temperature bandwidth=$bandwidth"
done
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] no sweep directory or GPU process was created"
  exit 0
fi

mkdir -p "$TRIAL_ROOT"
source scripts/common.sh
imagewam_init .
export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONUNBUFFERED=1

run_trial() {
  local index="$1"
  local gpu="${GPU_LIST[$index]}"
  local name repulsion step temperature bandwidth
  IFS='|' read -r name repulsion step temperature bandwidth \
    <<< "${TRIALS[$index]}"
  local trial_dir="$TRIAL_ROOT/$name"
  mkdir -p "$trial_dir"
  echo "[start] trial=$name gpu=$gpu"
  if (
    BASE_RUN_DIR="$BASE_RUN_DIR" \
    TEST_DIR="$TEST_DIR" \
    OUTPUT_ROOT="$TRIAL_ROOT" \
    GOAL_IMAGE="$GOAL_IMAGE" \
    GPU_ID="$gpu" \
    ENDPOINT_PARTICLES="${ENDPOINT_PARTICLES:-20}" \
    ENDPOINT_ITERATIONS="${ENDPOINT_ITERATIONS:-15}" \
    PATH_PARTICLES="${PATH_PARTICLES:-20}" \
    PATH_ITERATIONS="${PATH_ITERATIONS:-10}" \
    TRACE_MODE="${TRACE_MODE:-base}" \
    LATENT_DISTANCE="${LATENT_DISTANCE:-token_cosine}" \
    SEED="${SEED:-0}" \
    ENDPOINT_REPULSION_WEIGHT="$repulsion" \
    PATH_REPULSION_WEIGHT="$repulsion" \
    ENDPOINT_STEP_SIZE="$step" \
    PATH_STEP_SIZE="$step" \
    ENDPOINT_TEMPERATURE="$temperature" \
    PATH_TEMPERATURE="$temperature" \
    ENDPOINT_BANDWIDTH_SCALE="$bandwidth" \
    PATH_BANDWIDTH_SCALE="$bandwidth" \
    bash scripts/flux2/run_mug_goal_image_pipeline.sh --worker "$name"
  ) > "$trial_dir/launcher.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] trial=$name gpu=$gpu"
  else
    local status="$?"
    echo "$status" > "$trial_dir/failed.status"
    echo "[failed] trial=$name gpu=$gpu status=$status" >&2
    return 1
  fi
}

PIDS=()
for index in 0 1 2; do
  run_trial "$index" &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done

imagewam_python -u -B experiments/libero/summarize_goal_image_sweep.py \
  --sweep-dir "$SWEEP_DIR" || FAILED=1
if (( FAILED != 0 )); then
  echo "[done-with-failures] $SWEEP_DIR" >&2
  exit 1
fi
echo "[done] $SWEEP_DIR"
