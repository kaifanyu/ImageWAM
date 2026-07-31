#!/usr/bin/env bash
# Direct-endpoint baseline:
#   simulate -> edit one goal image -> score K simulator terminal images.
#
# Stages are separate on purpose so MuJoCo and the FLUX model never need to
# occupy GPU memory at the same time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

STAGE="${STAGE:-simulate}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/empty_arm_endpoint}"
# preview only needs a start frame, so drop to the 5 diagnostic controls the
# simulator always requires (no_op, wrong_direction, undershoot, oracle, overshoot).
if [ "${STAGE}" = "preview" ]; then
  NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-5}"
else
  NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-13}"
fi
SIM_SEED="${SIM_SEED:-0}"
TRAJECTORY_SEED="${TRAJECTORY_SEED:-7}"
EDITOR_SEED="${EDITOR_SEED:-0}"
EDIT_STEPS="${EDIT_STEPS:-20}"
LATENT_DEVICE="${LATENT_DEVICE:-cpu}"
DEFAULT_PROMPT="A video recorded from a robot's point of view executing the following instruction: "
DEFAULT_PROMPT+="move the robot arm from the left side of the empty table to the right side. "
DEFAULT_PROMPT+="Keep the end-effector height, orientation, and gripper unchanged. "
DEFAULT_PROMPT+="Update the agent and wrist views consistently with the motion. "
DEFAULT_PROMPT+="Keep the fixed camera, table, lighting, and background unchanged. "
DEFAULT_PROMPT+="Do not add or remove anything."
PROMPT="${PROMPT:-${DEFAULT_PROMPT}}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

run_simulate() {
  imagewam_require_env DATASET_STATS_PATH
  local args=(
    --run-dir "${RUN_DIR}"
    --num-trajectories "${NUM_TRAJECTORIES}"
    --sim-seed "${SIM_SEED}"
    --trajectory-seed "${TRAJECTORY_SEED}"
    --prompt "${PROMPT}"
    --dataset-stats "${DATASET_STATS_PATH}"
  )
  [ "${SAVE_VIDEOS:-false}" = "true" ] && args+=(--save-videos)
  [ "${FORCE:-false}" = "true" ] && args+=(--force)
  imagewam_run imagewam_python experiments/libero/sample_endpoint_trajectories.py "${args[@]}"
}

run_edit() {
  imagewam_require_env CKPT_PATH
  imagewam_require_env FLUX2_SRC
  imagewam_require_env FLUX2_MODEL_PATH
  imagewam_require_env FLUX2_AE_MODEL_PATH
  imagewam_run imagewam_python scripts/run_image_edit.py \
    --ckpt "${CKPT_PATH}" \
    --image "${RUN_DIR}/start.png" \
    --prompt "${PROMPT}" \
    --proprio-npy "${RUN_DIR}/start_proprio_normalized.npy" \
    --proprio-normalized \
    --output "${RUN_DIR}/goal_edit.png" \
    --latent-output "${RUN_DIR}/goal_editor_latent.npy" \
    --metadata-output "${RUN_DIR}/goal_edit_metadata.json" \
    --steps "${EDIT_STEPS}" \
    --seed "${EDITOR_SEED}"
  echo "start : ${RUN_DIR}/start.png"
  echo "end   : ${RUN_DIR}/goal_edit.png"
  echo "strip : ${RUN_DIR}/goal_edit_compare.png"
}

run_score() {
  imagewam_require_env FLUX2_SRC
  imagewam_require_env FLUX2_AE_MODEL_PATH
  local args=(
    --run-dir "${RUN_DIR}"
    --goal "${RUN_DIR}/goal_edit.png"
    --editor-ae "${FLUX2_AE_MODEL_PATH}"
    --flux2-src "${FLUX2_SRC}"
    --latent-device "${LATENT_DEVICE}"
  )
  [ -f "${RUN_DIR}/goal_editor_latent.npy" ] && \
    args+=(--goal-editor-latent "${RUN_DIR}/goal_editor_latent.npy")
  [ -n "${GOAL_DYNAMICS_LATENT:-}" ] && \
    args+=(--goal-dynamics-latent "${GOAL_DYNAMICS_LATENT}")
  [ -n "${DYNAMICS_METADATA:-}" ] && \
    args+=(--dynamics-metadata "${DYNAMICS_METADATA}")
  imagewam_run imagewam_python experiments/libero/score_endpoint_candidates.py "${args[@]}"
}

imagewam_print_config \
  STAGE RUN_DIR NUM_TRAJECTORIES SIM_SEED TRAJECTORY_SEED \
  EDITOR_SEED EDIT_STEPS LATENT_DEVICE CKPT_PATH

case "${STAGE}" in
  simulate)
    run_simulate
    ;;
  edit)
    run_edit
    ;;
  score)
    run_score
    ;;
  preview)
    # Beginning + end only: one simulator snapshot, one FLUX.2 edit, no scoring.
    run_simulate
    run_edit
    ;;
  all)
    run_simulate
    run_edit
    run_score
    ;;
  *)
    echo "Unknown STAGE='${STAGE}'. Use simulate | edit | score | preview | all." >&2
    exit 2
    ;;
esac
