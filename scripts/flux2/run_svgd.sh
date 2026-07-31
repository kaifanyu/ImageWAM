#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="runs/empty_arm_preview/svgd_uniform_corridor_newparams_seed0"
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/backend.log"
PID_FILE="$OUT_DIR/backend.pid"

nohup bash -lc '
set -euo pipefail

source scripts/common.sh
imagewam_init .

export PYTHON_BIN=.venv/bin/python
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

imagewam_python -B experiments/libero/svgd_endpoint.py \
  --run-dir runs/empty_arm_preview \
  --out-dir runs/empty_arm_preview/svgd_uniform_corridor_newparams_seed0 \
  --goal runs/empty_arm_preview/goal_oracle.png \
  --goal-latent-source reencode \
  --editor-ae "$FLUX2_AE_MODEL_PATH" \
  --flux2-src "$FLUX2_SRC" \
  --device cuda:0 \
  --init-mode uniform \
  --particles 8 \
  --iterations 10 \
  --bounds -0.06 -0.04 -0.32 0.32 1.02 1.045 \
  --fd-eps 0.01 \
  --step-size 0.002 \
  --temperature 0.10 \
  --max-update-norm 0.01 \
  --latent-weight 1.0 \
  --repulsion-weight 0.0 \
  --seed 0 \
  --save-all-particles
' > "$LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"

echo "Started backend job."
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG"
echo "Watch with: tail -f $LOG"