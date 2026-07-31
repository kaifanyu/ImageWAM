#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ "${1:-}" != "--worker" ]]; then
  SUITE_NAME="${1:-svgd_metrics_20p_$(date -u +%Y%m%dT%H%M%SZ)}"
  SUITE_DIR="runs/empty_arm_preview/$SUITE_NAME"
  if [[ -e "$SUITE_DIR" ]]; then
    echo "Refusing to reuse existing suite directory: $SUITE_DIR" >&2
    exit 1
  fi
  mkdir -p "$SUITE_DIR"
  nohup bash "$0" --worker "$SUITE_NAME" \
    > "$SUITE_DIR/launcher.log" 2>&1 < /dev/null &
  echo "$!" > "$SUITE_DIR/launcher.pid"
  echo "Started three-GPU SVGD metric suite."
  echo "Suite: $SUITE_DIR"
  echo "Launcher PID: $(<"$SUITE_DIR/launcher.pid")"
  echo "Watch: tail -f $SUITE_DIR/launcher.log"
  exit 0
fi

SUITE_NAME="${2:?Worker mode requires a suite name}"
SUITE_DIR="runs/empty_arm_preview/$SUITE_NAME"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2}"
if (( ${#GPU_LIST[@]} < 3 )); then
  echo "GPU_IDS must contain at least three GPU indices, for example: GPU_IDS='0 1 2'" >&2
  exit 1
fi

source scripts/common.sh
imagewam_init .

export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONUNBUFFERED=1

PARTICLES="${PARTICLES:-20}"
ITERATIONS="${ITERATIONS:-10}"
MOVE_STEPS="${MOVE_STEPS:-40}"
SETTLE_STEPS="${SETTLE_STEPS:-8}"
CONTROLLER_GAIN="${CONTROLLER_GAIN:-15.0}"
FD_EPS="${FD_EPS:-0.01}"
STEP_SIZE="${STEP_SIZE:-0.01}"
TEMPERATURE="${TEMPERATURE:-0.10}"
MAX_UPDATE_NORM="${MAX_UPDATE_NORM:-0.01}"
REPEATABILITY_PARTICLES="${REPEATABILITY_PARTICLES:-2}"
SEED="${SEED:-0}"

OBJECTIVES=(rms cosine token_cosine)
PIDS=()

for INDEX in 0 1 2; do
  OBJECTIVE="${OBJECTIVES[$INDEX]}"
  GPU="${GPU_LIST[$INDEX]}"
  TRIAL_DIR="$SUITE_DIR/$OBJECTIVE"
  mkdir -p "$TRIAL_DIR"

  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "[trial] objective=$OBJECTIVE physical_gpu=$GPU visible_device=cuda:0"
    echo "[trial] particles=$PARTICLES iterations=$ITERATIONS move=$MOVE_STEPS settle=$SETTLE_STEPS"
    imagewam_python -u -B experiments/libero/svgd_endpoint.py \
      --run-dir runs/empty_arm_preview \
      --out-dir "$TRIAL_DIR" \
      --goal runs/empty_arm_preview/goal_oracle.png \
      --goal-latent-source reencode \
      --editor-ae "$FLUX2_AE_MODEL_PATH" \
      --flux2-src "$FLUX2_SRC" \
      --device cuda:0 \
      --init-mode uniform \
      --particles "$PARTICLES" \
      --iterations "$ITERATIONS" \
      --bounds -0.06 -0.04 -0.32 0.32 1.02 1.045 \
      --latent-views agentview \
      --latent-distance "$OBJECTIVE" \
      --fd-eps "$FD_EPS" \
      --step-size "$STEP_SIZE" \
      --temperature "$TEMPERATURE" \
      --max-update-norm "$MAX_UPDATE_NORM" \
      --latent-weight 1.0 \
      --repulsion-weight 0.0 \
      --move-steps "$MOVE_STEPS" \
      --settle-steps "$SETTLE_STEPS" \
      --controller-gain "$CONTROLLER_GAIN" \
      --repeatability-particles "$REPEATABILITY_PARTICLES" \
      --seed "$SEED" \
      --save-all-particles \
      --save-rollout-traces \
      --verbose-evaluations

    imagewam_python -u -B experiments/libero/plot_svgd_latent_pull.py \
      --history "$TRIAL_DIR/history.json" \
      --prefix diagnostics \
      --title "SVGD 20-particle trial: $OBJECTIVE"
  ) > "$TRIAL_DIR/backend.log" 2>&1 &

  PID="$!"
  PIDS+=("$PID")
  echo "$PID" > "$TRIAL_DIR/backend.pid"
  echo "[launch] objective=$OBJECTIVE gpu=$GPU pid=$PID log=$TRIAL_DIR/backend.log"
done

FAILED=0
for INDEX in 0 1 2; do
  if wait "${PIDS[$INDEX]}"; then
    echo "[complete] ${OBJECTIVES[$INDEX]}"
  else
    echo "[failed] ${OBJECTIVES[$INDEX]} (see its backend.log)" >&2
    FAILED=1
  fi
done

if find "$SUITE_DIR" -mindepth 2 -maxdepth 2 -name history.json -print -quit | grep -q .; then
  imagewam_python -u -B experiments/libero/compare_svgd_metric_trials.py \
    --suite-dir "$SUITE_DIR"
fi

if (( FAILED != 0 )); then
  exit 1
fi
echo "[done] suite=$SUITE_DIR"
