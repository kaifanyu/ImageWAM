#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/common.sh
imagewam_init .

export SWEEP_PROFILE="dino_xyz"
export PARTICLES="${PARTICLES:-15}"
export ITERATIONS="${ITERATIONS:-60}"
export GPU_IDS="${GPU_IDS:-0 1 2}"
export TRACE_MODE="${TRACE_MODE:-base}"
export SAVE_ALL_PARTICLES="${SAVE_ALL_PARTICLES:-true}"
export VERBOSE_EVALUATIONS="${VERBOSE_EVALUATIONS:-true}"
export DINO_MODEL="${DINO_MODEL:-vit_base_patch16_dinov3.lvd1689m}"

RUN_NAME="${1:-dino_xyz_15p_60it_v1}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [[ "${DRY_RUN:-false}" != "true" && "${DINO_PREFETCH:-true}" == "true" ]]; then
  echo "[preflight] loading pretrained $DINO_MODEL into the Hugging Face cache"
  if ! "$PYTHON_BIN" -c \
      'import sys, timm; model = timm.create_model(sys.argv[1], pretrained=True, num_classes=0); print(f"[preflight] cached {sys.argv[1]} ({model.num_features} features)")' \
      "$DINO_MODEL"; then
    echo "DINOv3 weights could not be loaded." >&2
    echo "Accept the model license at:" >&2
    echo "  https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m" >&2
    echo "Then authenticate with: .venv/bin/hf auth login" >&2
    exit 1
  fi
fi

exec bash scripts/flux2/run_empty_goal_start_cloud_3gpu_sweep.sh "$RUN_NAME"
