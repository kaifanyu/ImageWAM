#!/usr/bin/env bash
# Base image-editor trajectory-viz BASELINE (no ImageWAM fine-tune, no ActionDiT).
# Predicts future frames of a real LIBERO episode with an OFF-THE-SHELF editor and
# compares them to the actual frames.
#
# Choose the editor with IMAGE_EDITOR (default flux9b):
#   IMAGE_EDITOR=flux4b   FLUX.2-klein-base-4B  (fits ~11GB with CPU offload)
#   IMAGE_EDITOR=flux9b   FLUX.2-klein-base-9B  (needs a bigger GPU)
#   IMAGE_EDITOR=qwen     Qwen/Qwen-Image-Edit  (~20B; needs diffusers>=0.35 + big GPU)
#
# Other knobs:  SUITE=libero_10  EPISODE=0  HORIZON=16  MAX_STEPS=8  EDIT_STEPS=20
#
# Usage:
#   IMAGE_EDITOR=flux9b bash scripts/flux2/run_baseline_edit_traj.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

IMAGE_EDITOR="${IMAGE_EDITOR:-flux9b}"

case "${IMAGE_EDITOR}" in
  flux4b|flux9b)
    imagewam_require_env FLUX2_SRC
    imagewam_require_env FLUX2_AE_MODEL_PATH
    # The base weights path is resolved in Python: it prefers the variant-specific
    # FLUX2_MODEL_PATH_9B / FLUX2_MODEL_PATH_4B and falls back to FLUX2_MODEL_PATH,
    # emitting a precise error if none is set. So no hard require here.
    ;;
  qwen)
    : "${QWEN_IMAGE_EDIT_MODEL:=Qwen/Qwen-Image-Edit}"
    export QWEN_IMAGE_EDIT_MODEL
    ;;
  *)
    echo "Unknown IMAGE_EDITOR='${IMAGE_EDITOR}'. Use flux4b | flux9b | qwen." >&2
    exit 2
    ;;
esac

export IMAGE_EDITOR
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

ARGS=(
  --editor "${IMAGE_EDITOR}"
  --suite "${SUITE:-libero_10}"
  --episode "${EPISODE:-0}"
  --horizon "${HORIZON:-16}"
  --max-steps "${MAX_STEPS:-8}"
  --steps "${EDIT_STEPS:-20}"
  --seed "${SEED:-0}"
  --out-dir "${BASELINE_OUT_DIR:-${REPO_ROOT}/baseline_edit_traj}"
)
[ -n "${STRIDE:-}" ] && ARGS+=(--stride "${STRIDE}")

imagewam_print_config IMAGE_EDITOR SUITE EPISODE HORIZON
imagewam_run imagewam_python scripts/run_baseline_edit_traj.py "${ARGS[@]}" "$@"
