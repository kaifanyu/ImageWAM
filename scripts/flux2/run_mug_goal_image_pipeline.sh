#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE_RUN_DIR="${BASE_RUN_DIR:-runs/living_room_mug_obstacle}"
TEST_DIR="${TEST_DIR:-$BASE_RUN_DIR/mug_avoidance_test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$TEST_DIR}"
GOAL_IMAGE="${GOAL_IMAGE:-$TEST_DIR/goal_avoidance.png}"
GPU_ID="${GPU_ID:-0}"
ENDPOINT_PARTICLES="${ENDPOINT_PARTICLES:-20}"
ENDPOINT_ITERATIONS="${ENDPOINT_ITERATIONS:-15}"
PATH_PARTICLES="${PATH_PARTICLES:-20}"
PATH_ITERATIONS="${PATH_ITERATIONS:-10}"
TRACE_MODE="${TRACE_MODE:-base}"
LATENT_DISTANCE="${LATENT_DISTANCE:-token_cosine}"
SEED="${SEED:-0}"
ENDPOINT_REPULSION_WEIGHT="${ENDPOINT_REPULSION_WEIGHT:-0.0}"
PATH_REPULSION_WEIGHT="${PATH_REPULSION_WEIGHT:-0.0}"
ENDPOINT_FD_EPS="${ENDPOINT_FD_EPS:-0.01}"
PATH_FD_EPS="${PATH_FD_EPS:-0.01}"
ENDPOINT_STEP_SIZE="${ENDPOINT_STEP_SIZE:-0.01}"
PATH_STEP_SIZE="${PATH_STEP_SIZE:-0.01}"
ENDPOINT_TEMPERATURE="${ENDPOINT_TEMPERATURE:-0.10}"
PATH_TEMPERATURE="${PATH_TEMPERATURE:-0.10}"
ENDPOINT_BANDWIDTH_SCALE="${ENDPOINT_BANDWIDTH_SCALE:-1.0}"
PATH_BANDWIDTH_SCALE="${PATH_BANDWIDTH_SCALE:-1.0}"
ENDPOINT_MAX_UPDATE_NORM="${ENDPOINT_MAX_UPDATE_NORM:-0.02}"
PATH_MAX_UPDATE_NORM="${PATH_MAX_UPDATE_NORM:-0.02}"

if [[ "${1:-}" != "--worker" ]]; then
  SUITE_NAME="${1:-goal_image_path_$(date -u +%Y%m%dT%H%M%SZ)}"
  SUITE_DIR="$OUTPUT_ROOT/$SUITE_NAME"
  if [[ ! -f "$GOAL_IMAGE" ]]; then
    echo "Missing goal image: $GOAL_IMAGE" >&2
    exit 2
  fi
  if [[ -e "$SUITE_DIR" ]]; then
    echo "Refusing to reuse existing suite: $SUITE_DIR" >&2
    exit 2
  fi
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exec bash "$0" --worker "$SUITE_NAME"
  fi
  mkdir -p "$SUITE_DIR"
  nohup bash "$0" --worker "$SUITE_NAME" \
    > "$SUITE_DIR/launcher.log" 2>&1 < /dev/null &
  echo "$!" > "$SUITE_DIR/launcher.pid"
  echo "Started goal-image endpoint and mug-avoidance pipeline."
  echo "Suite: $SUITE_DIR"
  echo "Watch: tail -f $SUITE_DIR/launcher.log"
  exit 0
fi

SUITE_NAME="${2:?Worker mode requires a suite name}"
SUITE_DIR="$OUTPUT_ROOT/$SUITE_NAME"
ENDPOINT_DIR="$SUITE_DIR/endpoint_search"
PATH_DIR="$SUITE_DIR/path_search"

echo "[plan] goal_image=$GOAL_IMAGE"
echo "[plan] endpoint=[$ENDPOINT_PARTICLES particles x $ENDPOINT_ITERATIONS updates]"
echo "[plan] path=[$PATH_PARTICLES particles x $PATH_ITERATIONS updates]"
echo "[plan] physical_gpu=$GPU_ID trace_mode=$TRACE_MODE"
echo "[plan] metric=$LATENT_DISTANCE seed=$SEED"
echo "[plan] endpoint repulsion=$ENDPOINT_REPULSION_WEIGHT step=$ENDPOINT_STEP_SIZE temperature=$ENDPOINT_TEMPERATURE bandwidth=$ENDPOINT_BANDWIDTH_SCALE"
echo "[plan] path repulsion=$PATH_REPULSION_WEIGHT step=$PATH_STEP_SIZE temperature=$PATH_TEMPERATURE bandwidth=$PATH_BANDWIDTH_SCALE"
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] no suite directory or GPU process was created"
  exit 0
fi

source scripts/common.sh
imagewam_init .
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env FLUX2_SRC
export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONUNBUFFERED=1

mkdir -p "$ENDPOINT_DIR" "$PATH_DIR"
{
  echo "goal_image=$GOAL_IMAGE"
  echo "physical_gpu=$GPU_ID"
  echo "latent_distance=$LATENT_DISTANCE"
  echo "seed=$SEED"
  echo "endpoint_particles=$ENDPOINT_PARTICLES"
  echo "endpoint_iterations=$ENDPOINT_ITERATIONS"
  echo "endpoint_repulsion_weight=$ENDPOINT_REPULSION_WEIGHT"
  echo "endpoint_fd_eps=$ENDPOINT_FD_EPS"
  echo "endpoint_step_size=$ENDPOINT_STEP_SIZE"
  echo "endpoint_temperature=$ENDPOINT_TEMPERATURE"
  echo "endpoint_bandwidth_scale=$ENDPOINT_BANDWIDTH_SCALE"
  echo "endpoint_max_update_norm=$ENDPOINT_MAX_UPDATE_NORM"
  echo "path_particles=$PATH_PARTICLES"
  echo "path_iterations=$PATH_ITERATIONS"
  echo "path_repulsion_weight=$PATH_REPULSION_WEIGHT"
  echo "path_fd_eps=$PATH_FD_EPS"
  echo "path_step_size=$PATH_STEP_SIZE"
  echo "path_temperature=$PATH_TEMPERATURE"
  echo "path_bandwidth_scale=$PATH_BANDWIDTH_SCALE"
  echo "path_max_update_norm=$PATH_MAX_UPDATE_NORM"
  echo "trace_mode=$TRACE_MODE"
} > "$SUITE_DIR/pipeline_config.txt"

echo "[stage 1] infer terminal EEF from the goal image"
imagewam_python -u -B experiments/libero/svgd_endpoint.py \
  --run-dir "$BASE_RUN_DIR" \
  --out-dir "$ENDPOINT_DIR" \
  --goal "$GOAL_IMAGE" \
  --goal-latent-source reencode \
  --editor-ae "$FLUX2_AE_MODEL_PATH" \
  --flux2-src "$FLUX2_SRC" \
  --device cuda:0 \
  --particles "$ENDPOINT_PARTICLES" \
  --iterations "$ENDPOINT_ITERATIONS" \
  --init-mode uniform \
  --bounds -0.12 0.04 -0.28 0.28 0.52 0.60 \
  --latent-views agentview \
  --latent-distance "$LATENT_DISTANCE" \
  --transport svgd \
  --repulsion-weight "$ENDPOINT_REPULSION_WEIGHT" \
  --fd-eps "$ENDPOINT_FD_EPS" \
  --bandwidth-scale "$ENDPOINT_BANDWIDTH_SCALE" \
  --step-size "$ENDPOINT_STEP_SIZE" \
  --temperature "$ENDPOINT_TEMPERATURE" \
  --max-update-norm "$ENDPOINT_MAX_UPDATE_NORM" \
  --fixed-arc-height 0.12 \
  --fixed-midpoint-x 0.0 \
  --move-steps 40 \
  --settle-steps 8 \
  --controller-gain 15.0 \
  --rollout-trace-mode "$TRACE_MODE" \
  --seed "$SEED" \
  --save-all-particles \
  --verbose-evaluations

read -r TARGET_X TARGET_Y TARGET_Z < <(
  "$PYTHON_BIN" -B -c '
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(*payload["selection"]["target_eef"])
' "$ENDPOINT_DIR/best_metadata.json"
)
echo "[stage 1] inferred_target=[$TARGET_X, $TARGET_Y, $TARGET_Z]"

echo "[stage 2] optimize a mug-preserving path to the inferred EEF"
imagewam_python -u -B experiments/libero/svgd_obstacle_path.py \
  --run-dir "$BASE_RUN_DIR" \
  --test-dir "$TEST_DIR" \
  --target-eef "$TARGET_X" "$TARGET_Y" "$TARGET_Z" \
  --out-dir "$PATH_DIR" \
  --editor-ae "$FLUX2_AE_MODEL_PATH" \
  --flux2-src "$FLUX2_SRC" \
  --device cuda:0 \
  --particles "$PATH_PARTICLES" \
  --iterations "$PATH_ITERATIONS" \
  --init-mode collision-cloud \
  --latent-distance "$LATENT_DISTANCE" \
  --latent-views agentview \
  --transport svgd \
  --repulsion-weight "$PATH_REPULSION_WEIGHT" \
  --fd-eps "$PATH_FD_EPS" "$PATH_FD_EPS" \
  --bandwidth-scale "$PATH_BANDWIDTH_SCALE" \
  --step-size "$PATH_STEP_SIZE" \
  --temperature "$PATH_TEMPERATURE" \
  --max-update-norm "$PATH_MAX_UPDATE_NORM" \
  --trace-mode "$TRACE_MODE" \
  --seed "$SEED" \
  --save-all-particles \
  --verbose-evaluations

echo "[done] inferred endpoint: $ENDPOINT_DIR/best_metadata.json"
echo "[done] selected trajectory: $PATH_DIR/best_metadata.json"
echo "[done] rollout: $PATH_DIR/best_rollout.mp4"
