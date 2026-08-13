#!/usr/bin/env bash
# Two whole-trajectory SVGD runs on the empty table, one per GPU:
#
#   GPU_HALF  -> goal_half.png    arm halfway across the table  (y ~  0.00)
#   GPU_FULL  -> goal_oracle.png  arm fully across the table    (y ~ -0.22)
#
# Both optimize a [--horizon, 8] joint-setpoint trajectory against the FLUX AE
# token-cosine distance of the terminal render.  Re-running the identical
# command resumes each trial from its newest checkpoint.
#
#   bash scripts/flux2/run_joint_traj_two_goals.sh SUITE_NAME
#   RESUME=none bash scripts/flux2/run_joint_traj_two_goals.sh SUITE_NAME   # restart
#   FOREGROUND=true ... ITERATIONS=2 HORIZON=30                             # smoke test
set -euo pipefail

cd "$(dirname "$0")/../.."
source scripts/common.sh
imagewam_init .

BASE_RUN_DIR="${BASE_RUN_DIR:-runs/empty_arm_preview}"
GOAL_HALF="${GOAL_HALF:-$BASE_RUN_DIR/goal_half.png}"
GOAL_FULL="${GOAL_FULL:-$BASE_RUN_DIR/goal_oracle.png}"
GPU_HALF="${GPU_HALF:-0}"
GPU_FULL="${GPU_FULL:-1}"

HORIZON="${HORIZON:-300}"
PARTICLES="${PARTICLES:-10}"
ITERATIONS="${ITERATIONS:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
RESUME="${RESUME:-auto}"
SEED="${SEED:-0}"

OBJECTIVE="${OBJECTIVE:-token_cosine}"
VIEWS="${VIEWS:-agentview}"
ENERGY_MODE="${ENERGY_MODE:-terminal}"
WAYPOINTS="${WAYPOINTS:-6}"
CREDIT_MODE="${CREDIT_MODE:-uniform}"
JACOBIAN_MODE="${JACOBIAN_MODE:-per-particle}"
JACOBIAN_REFRESH="${JACOBIAN_REFRESH:-5}"
JACOBIAN_DELTA="${JACOBIAN_DELTA:-0.005}"

INIT_MODE="${INIT_MODE:-ramp-cloud}"
INIT_JOINT_RADIUS="${INIT_JOINT_RADIUS:-0.02}"
ANCHOR_START_STEPS="${ANCHOR_START_STEPS:-10}"
MAX_SETPOINT_RATE="${MAX_SETPOINT_RATE:-0.04}"

TRANSPORT="${TRANSPORT:-svgd}"
KERNEL_SPACE="${KERNEL_SPACE:-full}"
REPULSION="${REPULSION:-0.01}"
STEP_SIZE="${STEP_SIZE:-0.01}"
TEMPERATURE="${TEMPERATURE:-0.10}"
MAX_JOINT_STEP="${MAX_JOINT_STEP:-0.01}"
TRACE_STRIDE="${TRACE_STRIDE:-3}"

# Diagnostic only -- plotted and reported, never part of the energy.  The
# halfway pose is the t=0.5 point of the oracle path that produced goal_half.png.
DIAG_GOAL_HALF="${DIAG_GOAL_HALF:--0.0492 0.0011 1.0309}"
DIAG_GOAL_FULL="${DIAG_GOAL_FULL:--0.05 -0.22 1.03}"
# Drawn by the 3D viewer as the search box; the joint search is not bounded by it.
BOUNDS="${BOUNDS:--0.30 0.10 -0.32 0.32 0.95 1.15}"

imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env FLUX2_SRC
export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONUNBUFFERED=1

if [[ "${1:-}" != "--worker" ]]; then
  SUITE_NAME="${1:-joint_traj_$(date -u +%Y%m%dT%H%M%SZ)}"
  SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
  for required in "$BASE_RUN_DIR/manifest.json" "$GOAL_HALF" "$GOAL_FULL"; do
    [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
  done
  if [[ -e "$SUITE_DIR" && "$RESUME" == "none" ]]; then
    echo "Refusing to overwrite $SUITE_DIR with RESUME=none." >&2
    exit 2
  fi
  mkdir -p "$SUITE_DIR"
  if [[ "${FOREGROUND:-false}" == "true" ]]; then
    exec bash "$0" --worker "$SUITE_NAME"
  fi
  nohup bash "$0" --worker "$SUITE_NAME" \
    > "$SUITE_DIR/launcher.log" 2>&1 < /dev/null &
  echo "$!" > "$SUITE_DIR/launcher.pid"
  echo "Started joint-trajectory SVGD suite."
  echo "Suite:  $SUITE_DIR"
  echo "Half:   gpu $GPU_HALF -> $SUITE_DIR/trials/half_across"
  echo "Full:   gpu $GPU_FULL -> $SUITE_DIR/trials/full_across"
  echo "Watch:  tail -f $SUITE_DIR/launcher.log"
  echo "Resume: re-run this exact command (RESUME=auto is the default)"
  exit 0
fi

SUITE_NAME="${2:?Worker mode requires a suite name}"
SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
TRIAL_ROOT="$SUITE_DIR/trials"
mkdir -p "$TRIAL_ROOT"

read -r -a BOUNDS_ARGS <<< "$BOUNDS"

run_trial() {
  local name="$1" gpu="$2" goal="$3" diag="$4"
  local trial_dir="$TRIAL_ROOT/$name"
  mkdir -p "$trial_dir"
  local diag_args
  read -r -a diag_args <<< "$diag"

  local args=(
    -u -B experiments/libero/svgd_joint_traj.py
    --run-dir "$BASE_RUN_DIR"
    --out-dir "$trial_dir"
    --goal "$goal"
    --goal-latent-source reencode
    --editor-ae "$FLUX2_AE_MODEL_PATH"
    --flux2-src "$FLUX2_SRC"
    --device cuda:0
    --latent-views "$VIEWS"
    --latent-distance "$OBJECTIVE"
    --horizon "$HORIZON"
    --particles "$PARTICLES"
    --max-iterations "$ITERATIONS"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --resume "$RESUME"
    --energy-mode "$ENERGY_MODE"
    --waypoints "$WAYPOINTS"
    --credit-mode "$CREDIT_MODE"
    --jacobian-mode "$JACOBIAN_MODE"
    --jacobian-refresh-every "$JACOBIAN_REFRESH"
    --jacobian-delta "$JACOBIAN_DELTA"
    --init-mode "$INIT_MODE"
    --init-joint-radius "$INIT_JOINT_RADIUS"
    --anchor-start-steps "$ANCHOR_START_STEPS"
    --max-setpoint-rate "$MAX_SETPOINT_RATE"
    --transport "$TRANSPORT"
    --kernel-space "$KERNEL_SPACE"
    --repulsion-weight "$REPULSION"
    --step-size "$STEP_SIZE"
    --temperature "$TEMPERATURE"
    --max-joint-step-rad "$MAX_JOINT_STEP"
    --bounds "${BOUNDS_ARGS[@]}"
    --diagnostic-goal-eef "${diag_args[@]}"
    --trace-stride "$TRACE_STRIDE"
    --seed "$SEED"
    --save-all-particles
    --verbose-evaluations
  )

  {
    echo "physical_gpu=$gpu"
    echo "goal=$goal"
    printf "command=%s" "$PYTHON_BIN"
    printf " %q" "${args[@]}"
    echo
  } > "$trial_dir/command.txt"

  echo "[start] trial=$name gpu=$gpu goal=$goal"
  if (
    export CUDA_VISIBLE_DEVICES="$gpu"
    imagewam_python "${args[@]}"
  ) >> "$trial_dir/backend.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] trial=$name gpu=$gpu"
  else
    local status="$?"
    echo "exit=$status at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$trial_dir/failed.status"
    echo "[failed] trial=$name gpu=$gpu exit=$status (see $trial_dir/backend.log)"
  fi
}

echo "[plan] suite=$SUITE_DIR"
echo "[plan] horizon=$HORIZON particles=$PARTICLES iterations=$ITERATIONS"
echo "[plan] checkpoint_every=$CHECKPOINT_EVERY resume=$RESUME"
echo "[plan] objective=$OBJECTIVE views=$VIEWS energy_mode=$ENERGY_MODE credit=$CREDIT_MODE"
echo "[plan] jacobian=$JACOBIAN_MODE/$JACOBIAN_REFRESH transport=$TRANSPORT kernel=$KERNEL_SPACE"
{
  echo "scene_run_dir=$BASE_RUN_DIR"
  echo "goal_half=$GOAL_HALF"
  echo "goal_full=$GOAL_FULL"
  echo "gpu_half=$GPU_HALF"
  echo "gpu_full=$GPU_FULL"
  echo "horizon=$HORIZON"
  echo "particles=$PARTICLES"
  echo "iterations=$ITERATIONS"
  echo "checkpoint_every=$CHECKPOINT_EVERY"
} > "$SUITE_DIR/suite_config.txt"

run_trial half_across "$GPU_HALF" "$GOAL_HALF" "$DIAG_GOAL_HALF" &
HALF_PID="$!"
run_trial full_across "$GPU_FULL" "$GOAL_FULL" "$DIAG_GOAL_FULL" &
FULL_PID="$!"
wait "$HALF_PID" "$FULL_PID"

echo
echo "Suite finished: $SUITE_DIR"
echo "Visualise: imagewam_python experiments/libero/svgd_traj3d.py --runs-root runs"
