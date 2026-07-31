#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source scripts/common.sh
imagewam_init .

SCENE="${1:-clutter}"
DEFAULT_START_EEF="-0.05 0.22 1.03"
DEFAULT_GOAL_EEF="-0.05 -0.22 1.03"
case "$SCENE" in
  empty)
    DEFAULT_RUN_DIR="runs/empty_arm_preview"
    DEFAULT_BDDL="experiments/libero/empty_table_move_arm.bddl"
    DEFAULT_PROMPT="Move the robot arm from one side of the empty table to the other while keeping its height, orientation, and gripper unchanged."
    ;;
  clutter)
    DEFAULT_RUN_DIR="runs/clutter_arm_preview"
    DEFAULT_BDDL="experiments/libero/clutter_table_move_arm.bddl"
    DEFAULT_PROMPT="Move the robot arm from one side of the cluttered living room table to the other while keeping its height, orientation, and gripper unchanged."
    ;;
  living-room)
    DEFAULT_RUN_DIR="runs/living_room_arm_preview"
    DEFAULT_BDDL="experiments/libero/living_room_table_move_arm.bddl"
    DEFAULT_PROMPT="Move the robot arm from one side of the living room table to the other while keeping its height, orientation, and gripper unchanged."
    ;;
  mug-obstacle)
    DEFAULT_RUN_DIR="runs/living_room_mug_obstacle"
    DEFAULT_BDDL="experiments/libero/mug_obstacle_table_move_arm.bddl"
    DEFAULT_PROMPT="Move the robot arm from the left side to the right side of the living room table. A red mug is sitting at the midpoint of the path. Keep the gripper open and preserve the mug's motion consistently."
    # The living-room table surface is z=0.41 versus z=0.90 for the empty
    # tabletop. These endpoints preserve the proven 0.13 m table clearance.
    DEFAULT_START_EEF="-0.05 0.22 0.54"
    DEFAULT_GOAL_EEF="-0.05 -0.22 0.54"
    ;;
  living-interaction)
    DEFAULT_RUN_DIR="runs/living_room_interaction_arm_preview"
    DEFAULT_BDDL="third_party/LIBERO/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate.bddl"
    DEFAULT_PROMPT="Move the robot arm across the populated living room table. Preserve the scene when possible and keep the gripper open."
    ;;
  custom)
    DEFAULT_RUN_DIR="${RUN_DIR:?SCENE=custom requires RUN_DIR}"
    DEFAULT_BDDL="${BDDL:?SCENE=custom requires BDDL}"
    DEFAULT_PROMPT="${PROMPT:-Move the robot arm between the requested Cartesian endpoints.}"
    ;;
  *)
    echo "Unknown scene '$SCENE'." >&2
    echo "Use empty | living-room | mug-obstacle | clutter | living-interaction | custom." >&2
    exit 2
    ;;
esac

RUN_DIR="${2:-${RUN_DIR:-$DEFAULT_RUN_DIR}}"
BDDL="${BDDL:-$DEFAULT_BDDL}"
PROMPT="${PROMPT:-$DEFAULT_PROMPT}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-13}"
SIM_SEED="${SIM_SEED:-0}"
TRAJECTORY_SEED="${TRAJECTORY_SEED:-7}"
START_EEF="${START_EEF:-$DEFAULT_START_EEF}"
GOAL_EEF="${GOAL_EEF:-$DEFAULT_GOAL_EEF}"

if [[ ! "$NUM_TRAJECTORIES" =~ ^[0-9]+$ ]] || (( NUM_TRAJECTORIES < 5 )); then
  echo "NUM_TRAJECTORIES must be an integer of at least 5." >&2
  exit 2
fi

read -r -a START_EEF_ARGS <<< "$START_EEF"
read -r -a GOAL_EEF_ARGS <<< "$GOAL_EEF"
if (( ${#START_EEF_ARGS[@]} != 3 || ${#GOAL_EEF_ARGS[@]} != 3 )); then
  echo "START_EEF and GOAL_EEF must each contain exactly three numbers." >&2
  exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
ARGS=(
  --run-dir "$RUN_DIR"
  --bddl "$BDDL"
  --num-trajectories "$NUM_TRAJECTORIES"
  --sim-seed "$SIM_SEED"
  --trajectory-seed "$TRAJECTORY_SEED"
  --start-eef "${START_EEF_ARGS[@]}"
  --goal-eef "${GOAL_EEF_ARGS[@]}"
  --prompt "$PROMPT"
)
if [[ -n "${DATASET_STATS_PATH:-}" ]]; then
  ARGS+=(--dataset-stats "$DATASET_STATS_PATH")
fi
if [[ "${SAVE_VIDEOS:-false}" == "true" ]]; then
  ARGS+=(--save-videos)
fi
if [[ "${FORCE:-false}" == "true" ]]; then
  ARGS+=(--force)
fi

echo "[scene] name=$SCENE"
echo "[scene] run_dir=$RUN_DIR"
echo "[scene] bddl=$BDDL"
echo "[scene] start_eef=$START_EEF"
echo "[scene] goal_eef=$GOAL_EEF"
imagewam_python -u -B experiments/libero/sample_endpoint_trajectories.py "${ARGS[@]}"

if [[ "$SCENE" == "mug-obstacle" ]]; then
  imagewam_python -u -B experiments/libero/plot_obstacle_trajectory.py \
    --run-dir "$RUN_DIR" \
    --candidate oracle
fi

echo
echo "Scene ready."
echo "Start: $RUN_DIR/start.png"
echo "Oracle goal: $RUN_DIR/goal_oracle.png"
