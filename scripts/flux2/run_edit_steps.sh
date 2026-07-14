#!/usr/bin/env bash
# Dump ImageWAM's image-editing (next-frame prediction) step by step, for
# tasks 4/5/6 in both the original and the perturbed environment.
# Same model/checkpoint wiring as run_eval_flux2_libero.sh, but no rollout.
#
# Usage:  bash scripts/flux2/run_edit_steps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

CONFIG_NAME="sim_libero_omnigen2"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}"
TASK="${TASK:-libero_flux2_klein_${FLUX2_VARIANT}_base_imagewam}"

imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env FLUX2_MODEL_PATH
imagewam_ckpt_from_exp
imagewam_require_env CKPT_PATH
imagewam_require_env DATASET_STATS_PATH

FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/experiments/libero:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
imagewam_prepare_eval_ckpt

ARGS=(
  --config-name "${CONFIG_NAME}"
  task="${TASK}"
  ckpt="${CKPT_PATH}"
  EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}"
  model.flux2_src_path="${FLUX2_SRC}"
  model.flux2_model_path="${FLUX2_MODEL_PATH}"
  model.ae_model_path="${FLUX2_AE_MODEL_PATH}"
  model.variant="klein-base-${FLUX2_VARIANT}"
  model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}"
  model.load_text_encoder=true
  model.pack_proprio_after_text=true
  model.proprio_dim="${PROPRIO_DIM:-8}"
  data.train.qwen_context_len="${QWEN_CONTEXT_LEN:-512}"
  data.train.qwen_text_cache_format=qwen3_flux2
  seed="${SEED:-0}"
  "+EVALUATION.edit_steps=${EDIT_STEPS:-20}"
  "+EVALUATION.edit_trial=${EDIT_TRIAL:-0}"
  "+EVALUATION.edit_suite=${EDIT_SUITE:-libero_10}"
  "+EVALUATION.edit_steps_out_dir=${EDIT_OUT_DIR:-/home/kaifany/project-data/ImageWAM/edit_steps}"
  hydra.run.dir=/tmp/imagewam_edit_steps
)

imagewam_print_config TASK CKPT_PATH DATASET_STATS_PATH FLUX2_MODEL_PATH
imagewam_run imagewam_python experiments/libero/edit_steps.py "${ARGS[@]}" "$@"
