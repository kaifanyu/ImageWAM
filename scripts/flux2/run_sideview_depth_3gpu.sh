#!/usr/bin/env bash
# Three side-camera trials, one per GPU, all scoring the FULL two-view render
# ([agentview | side profile]) rather than agentview alone.
#
# The question all three ask is the same: does a second, orthogonal camera give
# the objective enough information to place the gripper at the right *depth*?
# agentview looks down the +x axis, so it resolves the arm's y motion but
# confounds height with reach; the side profile looks down -y, so x and z read
# straight off the image.  Only together do they span all three axes.
#
#   gpu 0  endpoint       cream_cheese_above   theta = terminal (x, y, z), min-jerk path
#   gpu 1  joint traj     cream_cheese_above   theta = the whole [H, 8] setpoint trajectory
#   gpu 2  joint traj     plate_tilt_roll30    same, on a goal that needs wrist roll
#
# gpu 0 vs gpu 1 is a controlled comparison: identical goal, identical objective,
# different parameterisation.  gpu 2 is on a goal the endpoint parameterisation
# cannot represent at all -- it holds gripper orientation at its start value,
# whereas a joint-space particle can roll the wrist.
#
#   bash scripts/flux2/run_sideview_depth_3gpu.sh SUITE_NAME
#   RESUME=none bash scripts/flux2/run_sideview_depth_3gpu.sh SUITE_NAME   # restart
#   DRY_RUN=true bash scripts/flux2/run_sideview_depth_3gpu.sh SUITE_NAME  # print the plan
#
# Re-running the identical command resumes the two joint-trajectory trials from
# their newest checkpoint.  The endpoint trial has no checkpointing and restarts.
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

# The scene must have been prepared with the side camera; a run dir built before
# it carries no composed_right_view and reopens as [agentview | wrist], which
# would silently score the wrong pair of cameras.
BASE_RUN_DIR="${BASE_RUN_DIR:-runs/multi_object_sideview_preview}"
GOALS_DIR="$BASE_RUN_DIR/goals"

ENDPOINT_GOAL="${ENDPOINT_GOAL:-cream_cheese_above}"
TRAJ_GOAL="${TRAJ_GOAL:-cream_cheese_above}"
ROLL_GOAL="${ROLL_GOAL:-plate_tilt_roll30}"
GPU_ENDPOINT="${GPU_ENDPOINT:-0}"
GPU_TRAJ="${GPU_TRAJ:-1}"
GPU_ROLL="${GPU_ROLL:-2}"

ITERATIONS="${ITERATIONS:-100}"
VIEWS="${VIEWS:-both}"
OBJECTIVE="${OBJECTIVE:-token_cosine}"
SEED="${SEED:-0}"

# Endpoint search (gpu 0) -- the multi-object box, not the empty table's slab.
ENDPOINT_PARTICLES="${ENDPOINT_PARTICLES:-15}"
BOUNDS="${BOUNDS:--0.20 0.09 -0.32 0.32 0.98 1.16}"
# Symmetric because these goals move in all three axes.  Every probe is clipped
# to BOUNDS first, so a half-step past span/2 degenerates into the same
# box-corner secant for every particle and the population loses its diversity.
FD_EPS="${FD_EPS:-0.03 0.03 0.03}"

# Whole-trajectory search (gpu 1, gpu 2) -- matches empty_arm_preview/traj300_v1.
HORIZON="${HORIZON:-300}"
TRAJ_PARTICLES="${TRAJ_PARTICLES:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
RESUME="${RESUME:-auto}"
ENERGY_MODE="${ENERGY_MODE:-terminal}"
CREDIT_MODE="${CREDIT_MODE:-uniform}"
JACOBIAN_MODE="${JACOBIAN_MODE:-per-particle}"
JACOBIAN_REFRESH="${JACOBIAN_REFRESH:-5}"
JACOBIAN_DELTA="${JACOBIAN_DELTA:-0.005}"
INIT_MODE="${INIT_MODE:-ramp-cloud}"
INIT_JOINT_RADIUS="${INIT_JOINT_RADIUS:-0.02}"
ANCHOR_START_STEPS="${ANCHOR_START_STEPS:-10}"
MAX_SETPOINT_RATE="${MAX_SETPOINT_RATE:-0.04}"
KERNEL_SPACE="${KERNEL_SPACE:-full}"
MAX_JOINT_STEP="${MAX_JOINT_STEP:-0.01}"
TRACE_STRIDE="${TRACE_STRIDE:-3}"

TRANSPORT="${TRANSPORT:-svgd}"
REPULSION="${REPULSION:-0.01}"
STEP_SIZE="${STEP_SIZE:-0.01}"
TEMPERATURE="${TEMPERATURE:-0.10}"

read -r -a BOUNDS_ARGS <<< "$BOUNDS"
read -r -a FD_EPS_ARGS <<< "$FD_EPS"

# --------------------------------------------------------------------------- #
# validation -- every one of these costs seconds now and a wasted day later
# --------------------------------------------------------------------------- #

if [[ ! -f "$BASE_RUN_DIR/manifest.json" ]]; then
  echo "Missing scene manifest: $BASE_RUN_DIR/manifest.json" >&2
  echo "Build it with: RUN_DIR=$BASE_RUN_DIR bash scripts/flux2/prepare_svgd_scene.sh multi-object" >&2
  exit 2
fi

SCENE_RIGHT_VIEW="$("$PYTHON_BIN" - "$BASE_RUN_DIR/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("composed_right_view") or "wrist")
PY
)"
if [[ "$SCENE_RIGHT_VIEW" != "sideview" ]]; then
  echo "Scene $BASE_RUN_DIR composes [agentview | $SCENE_RIGHT_VIEW], not the side profile." >&2
  echo "Rebuild it with: RUN_DIR=$BASE_RUN_DIR bash scripts/flux2/prepare_svgd_scene.sh multi-object" >&2
  exit 2
fi

for goal_id in "$ENDPOINT_GOAL" "$TRAJ_GOAL" "$ROLL_GOAL"; do
  if [[ ! -f "$GOALS_DIR/$goal_id/goal.png" || ! -f "$GOALS_DIR/$goal_id/metadata.json" ]]; then
    echo "No rendered goal at $GOALS_DIR/$goal_id/" >&2
    echo "Render it with: $PYTHON_BIN -u -B experiments/libero/prepare_multi_object_goals.py --run-dir $BASE_RUN_DIR" >&2
    exit 2
  fi
done

VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd' ')"
for gpu in "$GPU_ENDPOINT" "$GPU_TRAJ" "$GPU_ROLL"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU indices must be integers; got '$gpu'." >&2
    exit 2
  fi
  if [[ -n "$VISIBLE_GPUS" && " $VISIBLE_GPUS " != *" $gpu "* ]]; then
    echo "GPU $gpu is not present on this host (have: $VISIBLE_GPUS)." >&2
    exit 2
  fi
done

"$PYTHON_BIN" - "${FD_EPS_ARGS[@]}" "${BOUNDS_ARGS[@]}" <<'PY' || exit 2
import sys
eps = [float(v) for v in sys.argv[1:4]]
bounds = [float(v) for v in sys.argv[4:10]]
bad = []
for index, axis in enumerate(("x", "y", "z")):
    low, high = bounds[2 * index], bounds[2 * index + 1]
    span = high - low
    if eps[index] >= span / 2.0:
        bad.append(f"  {axis}: fd_eps={eps[index]:g} m vs bounds span {span:.3f} m [{low:g}, {high:g}]")
if bad:
    print("FD_EPS is too wide for the search bounds:", file=sys.stderr)
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit(2)
PY

# The goal image is the only thing the objective sees.  This pose is reported and
# plotted as held-out truth; it never enters the energy or the gradients.
goal_pose() {
  "$PYTHON_BIN" - "$GOALS_DIR/$1/metadata.json" <<'PY'
import json, sys
print(" ".join(f"{v:.6f}" for v in json.load(open(sys.argv[1]))["target_eef"]))
PY
}

# --------------------------------------------------------------------------- #
# launch
# --------------------------------------------------------------------------- #

if [[ "${1:-}" != "--worker" ]]; then
  SUITE_NAME="${1:-${SUITE_NAME:-sideview_depth_$(date -u +%Y%m%dT%H%M%SZ)}}"
  SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
  if [[ -e "$SUITE_DIR" && "$RESUME" == "none" ]]; then
    echo "Refusing to overwrite $SUITE_DIR with RESUME=none." >&2
    exit 2
  fi
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "[dry-run] suite=$SUITE_DIR"
    echo "[dry-run] scene=$BASE_RUN_DIR composes [agentview | $SCENE_RIGHT_VIEW]"
    echo "[dry-run] gpu $GPU_ENDPOINT  endpoint    $ENDPOINT_GOAL   eef=($(goal_pose "$ENDPOINT_GOAL"))"
    echo "[dry-run] gpu $GPU_TRAJ  joint_traj  $TRAJ_GOAL   eef=($(goal_pose "$TRAJ_GOAL"))"
    echo "[dry-run] gpu $GPU_ROLL  joint_traj  $ROLL_GOAL   eef=($(goal_pose "$ROLL_GOAL"))"
    echo "[dry-run] views=$VIEWS objective=$OBJECTIVE iterations=$ITERATIONS"
    exit 0
  fi
  mkdir -p "$SUITE_DIR"
  if [[ "${FOREGROUND:-false}" == "true" ]]; then
    exec bash "$0" --worker "$SUITE_NAME"
  fi
  setsid nohup bash "$0" --worker "$SUITE_NAME" \
    > "$SUITE_DIR/launcher.log" 2>&1 < /dev/null &
  echo "$!" > "$SUITE_DIR/launcher.pid"
  echo "Started side-camera depth suite (all trials score [agentview | side profile])."
  echo "Suite:    $SUITE_DIR"
  echo "gpu $GPU_ENDPOINT:    endpoint    $ENDPOINT_GOAL  -> $SUITE_DIR/trials/${ENDPOINT_GOAL}__endpoint_xyz"
  echo "gpu $GPU_TRAJ:    joint traj  $TRAJ_GOAL  -> $SUITE_DIR/trials/${TRAJ_GOAL}__traj${HORIZON}"
  echo "gpu $GPU_ROLL:    joint traj  $ROLL_GOAL  -> $SUITE_DIR/trials/${ROLL_GOAL}__traj${HORIZON}"
  echo "Watch:    tail -f $SUITE_DIR/launcher.log"
  echo "Resume:   re-run this exact command (RESUME=auto is the default)"
  exit 0
fi

SUITE_NAME="${2:?Worker mode requires a suite name}"
SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
TRIAL_ROOT="$SUITE_DIR/trials"
mkdir -p "$TRIAL_ROOT"

record_and_run() {
  local name="$1" gpu="$2" goal_id="$3"
  shift 3
  local trial_dir="$TRIAL_ROOT/$name"
  mkdir -p "$trial_dir"
  {
    echo "physical_gpu=$gpu"
    echo "goal_id=$goal_id"
    echo "goal_image=$GOALS_DIR/$goal_id/goal.png"
    echo "scene_run_dir=$BASE_RUN_DIR"
    echo "composed_views=[agentview | $SCENE_RIGHT_VIEW]"
    echo "latent_views=$VIEWS"
    printf "command=%s" "$PYTHON_BIN"
    printf " %q" "$@"
    echo
  } > "$trial_dir/command.txt"

  echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$gpu trial=$name goal=$goal_id"
  if (
    export CUDA_VISIBLE_DEVICES="$gpu"
    imagewam_python "$@"
  ) >> "$trial_dir/backend.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$gpu trial=$name"
    return 0
  fi
  local status="$?"
  echo "exit=$status at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$trial_dir/failed.status"
  echo "[failed] $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$gpu trial=$name exit=$status (see $trial_dir/backend.log)" >&2
  return "$status"
}

run_endpoint_trial() {
  local goal_id="$1" gpu="$2"
  local name="${goal_id}__endpoint_xyz"
  local trial_dir="$TRIAL_ROOT/$name"
  local pose
  read -r -a pose <<< "$(goal_pose "$goal_id")"
  record_and_run "$name" "$gpu" "$goal_id" \
    -u -B experiments/libero/svgd_endpoint.py \
    --run-dir "$BASE_RUN_DIR" \
    --out-dir "$trial_dir" \
    --goal "$GOALS_DIR/$goal_id/goal.png" \
    --goal-latent-source reencode \
    --feature-encoder flux_ae \
    --editor-ae "$FLUX2_AE_MODEL_PATH" \
    --flux2-src "$FLUX2_SRC" \
    --device cuda:0 \
    --diagnostic-goal-eef "${pose[@]}" \
    --particles "$ENDPOINT_PARTICLES" \
    --iterations "$ITERATIONS" \
    --init-mode start-cloud \
    --init-radius 0.005 0.005 0.003 \
    --bounds "${BOUNDS_ARGS[@]}" \
    --latent-views "$VIEWS" \
    --latent-distance "$OBJECTIVE" \
    --transport "$TRANSPORT" \
    --latent-weight 1.0 \
    --repulsion-weight "$REPULSION" \
    --fd-eps "${FD_EPS_ARGS[@]}" \
    --step-size "$STEP_SIZE" \
    --temperature "$TEMPERATURE" \
    --max-update-norm 0.02 \
    --bandwidth-scale 1.0 \
    --move-steps 40 \
    --settle-steps 8 \
    --controller-gain 15.0 \
    --repeatability-particles 1 \
    --rollout-trace-mode base \
    --seed "$SEED" \
    --save-rollout-traces \
    --save-all-particles \
    --verbose-evaluations
}

run_traj_trial() {
  local goal_id="$1" gpu="$2"
  local name="${goal_id}__traj${HORIZON}"
  local trial_dir="$TRIAL_ROOT/$name"
  local pose
  read -r -a pose <<< "$(goal_pose "$goal_id")"
  record_and_run "$name" "$gpu" "$goal_id" \
    -u -B experiments/libero/svgd_joint_traj.py \
    --run-dir "$BASE_RUN_DIR" \
    --out-dir "$trial_dir" \
    --goal "$GOALS_DIR/$goal_id/goal.png" \
    --goal-latent-source reencode \
    --feature-encoder flux_ae \
    --editor-ae "$FLUX2_AE_MODEL_PATH" \
    --flux2-src "$FLUX2_SRC" \
    --device cuda:0 \
    --diagnostic-goal-eef "${pose[@]}" \
    --latent-views "$VIEWS" \
    --latent-distance "$OBJECTIVE" \
    --horizon "$HORIZON" \
    --particles "$TRAJ_PARTICLES" \
    --max-iterations "$ITERATIONS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --resume "$RESUME" \
    --energy-mode "$ENERGY_MODE" \
    --credit-mode "$CREDIT_MODE" \
    --jacobian-mode "$JACOBIAN_MODE" \
    --jacobian-refresh-every "$JACOBIAN_REFRESH" \
    --jacobian-delta "$JACOBIAN_DELTA" \
    --init-mode "$INIT_MODE" \
    --init-joint-radius "$INIT_JOINT_RADIUS" \
    --anchor-start-steps "$ANCHOR_START_STEPS" \
    --max-setpoint-rate "$MAX_SETPOINT_RATE" \
    --transport "$TRANSPORT" \
    --kernel-space "$KERNEL_SPACE" \
    --repulsion-weight "$REPULSION" \
    --step-size "$STEP_SIZE" \
    --temperature "$TEMPERATURE" \
    --max-joint-step-rad "$MAX_JOINT_STEP" \
    --bounds "${BOUNDS_ARGS[@]}" \
    --trace-stride "$TRACE_STRIDE" \
    --seed "$SEED" \
    --save-all-particles \
    --verbose-evaluations
}

echo "[plan] suite=$SUITE_DIR"
echo "[plan] scene=$BASE_RUN_DIR composes [agentview | $SCENE_RIGHT_VIEW]"
echo "[plan] latent_views=$VIEWS objective=$OBJECTIVE iterations=$ITERATIONS seed=$SEED"
echo "[plan] gpu $GPU_ENDPOINT endpoint  goal=$ENDPOINT_GOAL particles=$ENDPOINT_PARTICLES bounds=${BOUNDS_ARGS[*]} fd_eps=${FD_EPS_ARGS[*]}"
echo "[plan] gpu $GPU_TRAJ joint_traj goal=$TRAJ_GOAL horizon=$HORIZON particles=$TRAJ_PARTICLES resume=$RESUME"
echo "[plan] gpu $GPU_ROLL joint_traj goal=$ROLL_GOAL horizon=$HORIZON particles=$TRAJ_PARTICLES resume=$RESUME"
{
  echo "scene_run_dir=$BASE_RUN_DIR"
  echo "composed_right_view=$SCENE_RIGHT_VIEW"
  echo "latent_views=$VIEWS"
  echo "objective=$OBJECTIVE"
  echo "iterations=$ITERATIONS"
  echo "endpoint_goal=$ENDPOINT_GOAL gpu=$GPU_ENDPOINT particles=$ENDPOINT_PARTICLES"
  echo "traj_goal=$TRAJ_GOAL gpu=$GPU_TRAJ horizon=$HORIZON particles=$TRAJ_PARTICLES"
  echo "roll_goal=$ROLL_GOAL gpu=$GPU_ROLL horizon=$HORIZON particles=$TRAJ_PARTICLES"
} > "$SUITE_DIR/suite_config.txt"

run_endpoint_trial "$ENDPOINT_GOAL" "$GPU_ENDPOINT" &
ENDPOINT_PID="$!"
run_traj_trial "$TRAJ_GOAL" "$GPU_TRAJ" &
TRAJ_PID="$!"
run_traj_trial "$ROLL_GOAL" "$GPU_ROLL" &
ROLL_PID="$!"

failed=0
for pid in "$ENDPOINT_PID" "$TRAJ_PID" "$ROLL_PID"; do
  wait "$pid" || failed=1
done

if (( failed )); then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/failed.status"
  echo "[done] suite finished with at least one failed trial"
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/complete.status"
echo "[done] all three trials finished"
echo "Visualise: imagewam_python experiments/libero/svgd_traj3d.py --runs-root runs"
