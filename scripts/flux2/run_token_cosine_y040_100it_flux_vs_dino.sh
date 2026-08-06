#!/usr/bin/env bash
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

BASE_RUN_DIR="${BASE_RUN_DIR:-runs/empty_arm_preview}"
GOAL_IMAGE="${GOAL_IMAGE:-$BASE_RUN_DIR/goal_oracle.png}"
SUITE_NAME="${SUITE_NAME:-empty_start_metric_y040_15p_100it_flux_vs_dino_v1}"
SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
TRIAL_ROOT="$SUITE_DIR/trials"

# The original FLUX trial used physical GPU 2. Keep that default and place the
# DINO comparison on a separate GPU so both trials can run concurrently.
FLUX_GPU="${FLUX_GPU:-2}"
DINO_GPU="${DINO_GPU:-1}"
DINO_MODEL="${DINO_MODEL:-vit_base_patch16_dinov3.lvd1689m}"

if [[ ! -f "$BASE_RUN_DIR/manifest.json" || ! -f "$GOAL_IMAGE" ]]; then
  echo "Missing scene manifest or goal image under $BASE_RUN_DIR." >&2
  exit 2
fi
if [[ -e "$SUITE_DIR" ]]; then
  echo "Refusing to overwrite existing suite: $SUITE_DIR" >&2
  exit 2
fi

mkdir -p "$TRIAL_ROOT"
exec > "$SUITE_DIR/launcher.log" 2>&1
echo "$$" > "$SUITE_DIR/launcher.pid"

COMMON_ARGS=(
  --run-dir "$BASE_RUN_DIR"
  --goal "$GOAL_IMAGE"
  --goal-latent-source reencode
  --device cuda:0
  --particles 15
  --iterations 100
  --init-mode start-cloud
  --init-radius 0.005 0.005 0.003
  --bounds -0.06 -0.04 -0.32 0.32 1.02 1.045
  --latent-views agentview
  --latent-distance token_cosine
  --transport svgd
  --latent-weight 1.0
  --repulsion-weight 0.01
  --fd-eps 0.01 0.04 0.01
  --step-size 0.01
  --temperature 0.10
  --max-update-norm 0.02
  --bandwidth-scale 1.0
  --move-steps 40
  --settle-steps 8
  --controller-gain 15.0
  --repeatability-particles 1
  --rollout-trace-mode base
  --seed 0
  --save-rollout-traces
  --save-all-particles
  --verbose-evaluations
)

run_trial() {
  local name="$1"
  local gpu="$2"
  local feature_encoder="$3"
  local trial_dir="$TRIAL_ROOT/$name"
  local args=(
    -u -B experiments/libero/svgd_endpoint.py
    --out-dir "$trial_dir"
    --feature-encoder "$feature_encoder"
    "${COMMON_ARGS[@]}"
  )

  if [[ "$feature_encoder" == "flux_ae" ]]; then
    args+=(--editor-ae "$FLUX2_AE_MODEL_PATH" --flux2-src "$FLUX2_SRC")
  else
    args+=(--dino-model "$DINO_MODEL")
  fi

  mkdir -p "$trial_dir"
  {
    echo "physical_gpu=$gpu"
    printf 'command=imagewam_python'
    printf ' %q' "${args[@]}"
    echo
  } > "$trial_dir/command.txt"

  echo "[start] trial=$name encoder=$feature_encoder gpu=$gpu"
  if (
    export CUDA_VISIBLE_DEVICES="$gpu"
    imagewam_python "${args[@]}"
    imagewam_python -u -B experiments/libero/plot_svgd_latent_pull.py \
      --history "$trial_dir/history.json" \
      --prefix diagnostics \
      --title "$name"
  ) > "$trial_dir/backend.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] trial=$name encoder=$feature_encoder gpu=$gpu"
  else
    local status="$?"
    echo "$status" > "$trial_dir/failed.status"
    echo "[failed] trial=$name encoder=$feature_encoder gpu=$gpu status=$status" >&2
    return "$status"
  fi
}

echo "[plan] suite=$SUITE_DIR"
echo "[plan] flux_gpu=$FLUX_GPU dino_gpu=$DINO_GPU iterations=100 particles=15"
echo "[plan] DINO model=$DINO_MODEL"

run_trial token_cosine_y040 "$FLUX_GPU" flux_ae &
flux_pid="$!"
run_trial dino_token_cosine_y040 "$DINO_GPU" dinov3 &
dino_pid="$!"
echo "$flux_pid" > "$TRIAL_ROOT/token_cosine_y040/backend.pid"
echo "$dino_pid" > "$TRIAL_ROOT/dino_token_cosine_y040/backend.pid"

failed=0
if ! wait "$flux_pid"; then
  failed=1
fi
if ! wait "$dino_pid"; then
  failed=1
fi

if (( failed )); then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/failed.status"
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$SUITE_DIR/complete.status"
echo "[complete] both 100-iteration trials finished"
