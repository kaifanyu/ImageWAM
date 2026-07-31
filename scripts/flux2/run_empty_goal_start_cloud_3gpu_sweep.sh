#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE_RUN_DIR="${BASE_RUN_DIR:-runs/empty_arm_preview}"
GOAL_IMAGE="${GOAL_IMAGE:-$BASE_RUN_DIR/goal_oracle.png}"
SWEEP_PROFILE="${SWEEP_PROFILE:-signal}"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2}"

if (( ${#GPU_LIST[@]} < 3 )); then
  echo "GPU_IDS must contain at least three GPU indices." >&2
  exit 2
fi

PARTICLES="${PARTICLES:-12}"
ITERATIONS="${ITERATIONS:-20}"
INIT_RADIUS_FROM_ENV="${INIT_RADIUS:-}"
BOUNDS_FROM_ENV="${BOUNDS:-}"
INIT_RADIUS="${INIT_RADIUS:-0.005 0.005 0.003}"
BOUNDS="${BOUNDS:--0.06 -0.04 -0.32 0.32 1.02 1.045}"
MOVE_STEPS="${MOVE_STEPS:-40}"
SETTLE_STEPS="${SETTLE_STEPS:-8}"
CONTROLLER_GAIN="${CONTROLLER_GAIN:-15.0}"
REPEATABILITY_PARTICLES="${REPEATABILITY_PARTICLES:-1}"
TRACE_MODE="${TRACE_MODE:-base}"
SAVE_ALL_PARTICLES="${SAVE_ALL_PARTICLES:-true}"
VERBOSE_EVALUATIONS="${VERBOSE_EVALUATIONS:-true}"
TUNED_OBJECTIVE="${TUNED_OBJECTIVE:-token_cosine}"
TUNED_VIEWS="${TUNED_VIEWS:-agentview}"
TUNED_FD_Y="${TUNED_FD_Y:-0.04}"
TUNED_TRANSPORT="${TUNED_TRANSPORT:-svgd}"
TUNED_REPULSION="${TUNED_REPULSION:-0.01}"
FEATURE_ENCODER="${FEATURE_ENCODER:-flux_ae}"
DINO_MODEL="${DINO_MODEL:-vit_base_patch16_dinov3.lvd1689m}"
if [[ "$TUNED_TRANSPORT" == "particle_gd" ]]; then
  TUNED_REPULSION=0.0
fi

# Fields:
# name|objective|views|transport|init|repulsion|fd_x|fd_y|fd_z|step|temperature|max_update|bandwidth|seed
case "$SWEEP_PROFILE" in
  metric_y040)
    TRIALS=(
      "rms_y040|rms|agentview|svgd|start-cloud|0.01|0.01|0.04|0.01|0.01|0.10|0.02|1.0|0"
      "cosine_y040|cosine|agentview|svgd|start-cloud|0.01|0.01|0.04|0.01|0.01|0.10|0.02|1.0|0"
      "token_cosine_y040|token_cosine|agentview|svgd|start-cloud|0.01|0.01|0.04|0.01|0.01|0.10|0.02|1.0|0"
    )
    ;;
  token_tuning|token_xyz_probes)
    if [[ -z "$INIT_RADIUS_FROM_ENV" ]]; then
      INIT_RADIUS="0.005 0.005 0.005"
    fi
    if [[ -z "$BOUNDS_FROM_ENV" ]]; then
      BOUNDS="-0.18 0.18 -0.32 0.32 0.98 1.10"
    fi
    TRIALS=(
      "token_probe_xyz020|token_cosine|agentview|svgd|start-cloud|0.01|0.02|0.02|0.02|0.01|0.10|0.02|1.0|0"
      "token_probe_xyz040|token_cosine|agentview|svgd|start-cloud|0.01|0.04|0.04|0.04|0.01|0.10|0.02|1.0|0"
      "token_probe_xyz080|token_cosine|agentview|svgd|start-cloud|0.01|0.08|0.08|0.08|0.01|0.10|0.02|1.0|0"
    )
    ;;
  dino_xyz)
    FEATURE_ENCODER="dinov3"
    if [[ -z "$INIT_RADIUS_FROM_ENV" ]]; then
      INIT_RADIUS="0.005 0.005 0.005"
    fi
    if [[ -z "$BOUNDS_FROM_ENV" ]]; then
      BOUNDS="-0.18 0.18 -0.32 0.32 0.98 1.10"
    fi
    TRIALS=(
      "dino_token_xyz040|token_cosine|agentview|svgd|start-cloud|0.01|0.04|0.04|0.04|0.01|0.10|0.02|1.0|0"
      "dino_rms_xyz040|rms|agentview|svgd|start-cloud|0.01|0.04|0.04|0.04|0.01|0.10|0.02|1.0|0"
      "dino_token_xyz020|token_cosine|agentview|svgd|start-cloud|0.01|0.02|0.02|0.02|0.01|0.10|0.02|1.0|0"
    )
    ;;
  signal)
    TRIALS=(
      "rms_both|rms|both|svgd|start-cloud|0.0|0.01|0.01|0.01|0.01|0.10|0.01|1.0|0"
      "rms_agentview|rms|agentview|svgd|start-cloud|0.0|0.01|0.01|0.01|0.01|0.10|0.01|1.0|0"
      "token_agentview|token_cosine|agentview|svgd|start-cloud|0.0|0.01|0.01|0.01|0.01|0.10|0.01|1.0|0"
    )
    ;;
  gradient)
    TRIALS=(
      "fd_iso_010|$TUNED_OBJECTIVE|$TUNED_VIEWS|svgd|start-cloud|0.0|0.01|0.01|0.01|0.01|0.10|0.01|1.0|0"
      "fd_y_040|$TUNED_OBJECTIVE|$TUNED_VIEWS|svgd|start-cloud|0.0|0.01|0.04|0.01|0.01|0.10|0.01|1.0|0"
      "fd_y_080|$TUNED_OBJECTIVE|$TUNED_VIEWS|svgd|start-cloud|0.0|0.01|0.08|0.01|0.01|0.10|0.01|1.0|0"
    )
    ;;
  transport)
    TRIALS=(
      "particle_gd|$TUNED_OBJECTIVE|$TUNED_VIEWS|particle_gd|start-cloud|0.0|0.01|$TUNED_FD_Y|0.01|0.01|0.10|0.01|1.0|0"
      "svgd_no_repulsion|$TUNED_OBJECTIVE|$TUNED_VIEWS|svgd|start-cloud|0.0|0.01|$TUNED_FD_Y|0.01|0.01|0.10|0.01|1.0|0"
      "svgd_repulsion_001|$TUNED_OBJECTIVE|$TUNED_VIEWS|svgd|start-cloud|0.01|0.01|$TUNED_FD_Y|0.01|0.01|0.10|0.01|1.0|0"
    )
    ;;
  seed)
    TRIALS=(
      "seed_0|$TUNED_OBJECTIVE|$TUNED_VIEWS|$TUNED_TRANSPORT|random-start-cloud|$TUNED_REPULSION|0.01|$TUNED_FD_Y|0.01|0.01|0.10|0.01|1.0|0"
      "seed_1|$TUNED_OBJECTIVE|$TUNED_VIEWS|$TUNED_TRANSPORT|random-start-cloud|$TUNED_REPULSION|0.01|$TUNED_FD_Y|0.01|0.01|0.10|0.01|1.0|1"
      "seed_2|$TUNED_OBJECTIVE|$TUNED_VIEWS|$TUNED_TRANSPORT|random-start-cloud|$TUNED_REPULSION|0.01|$TUNED_FD_Y|0.01|0.01|0.10|0.01|1.0|2"
    )
    ;;
  *)
    echo "Unknown SWEEP_PROFILE=$SWEEP_PROFILE" >&2
    echo "Use metric_y040 | token_xyz_probes | dino_xyz | signal | gradient | transport | seed." >&2
    exit 2
    ;;
esac

read -r -a INIT_RADIUS_ARGS <<< "$INIT_RADIUS"
read -r -a BOUNDS_ARGS <<< "$BOUNDS"
if (( ${#INIT_RADIUS_ARGS[@]} != 3 || ${#BOUNDS_ARGS[@]} != 6 )); then
  echo "INIT_RADIUS needs three values and BOUNDS needs six values." >&2
  exit 2
fi

if [[ "${1:-}" != "--worker" ]]; then
  SWEEP_NAME="${1:-empty_start_${SWEEP_PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)}"
  SWEEP_DIR="$BASE_RUN_DIR/$SWEEP_NAME"
  if [[ ! -f "$BASE_RUN_DIR/manifest.json" || ! -f "$GOAL_IMAGE" ]]; then
    echo "Missing scene manifest or goal image under $BASE_RUN_DIR." >&2
    exit 2
  fi
  if [[ -e "$SWEEP_DIR" && "${DRY_RUN:-false}" != "true" ]]; then
    echo "Refusing to reuse existing sweep: $SWEEP_DIR" >&2
    exit 2
  fi
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exec bash "$0" --worker "$SWEEP_NAME"
  fi
  mkdir -p "$SWEEP_DIR"
  nohup bash "$0" --worker "$SWEEP_NAME" \
    > "$SWEEP_DIR/launcher.log" 2>&1 < /dev/null &
  echo "$!" > "$SWEEP_DIR/launcher.pid"
  echo "Started empty-table goal-image start-cloud sweep."
  echo "Profile: $SWEEP_PROFILE"
  echo "Sweep: $SWEEP_DIR"
  echo "Watch: tail -f $SWEEP_DIR/launcher.log"
  exit 0
fi

SWEEP_NAME="${2:?Worker mode requires a sweep name}"
SWEEP_DIR="$BASE_RUN_DIR/$SWEEP_NAME"
TRIAL_ROOT="$SWEEP_DIR/trials"

echo "[plan] sweep=$SWEEP_DIR profile=$SWEEP_PROFILE"
echo "[plan] goal_image=$GOAL_IMAGE"
echo "[plan] feature_encoder=$FEATURE_ENCODER dino_model=$DINO_MODEL"
echo "[plan] particles=$PARTICLES iterations=$ITERATIONS init_radius=$INIT_RADIUS"
echo "[plan] bounds=$BOUNDS trace_mode=$TRACE_MODE"
for index in 0 1 2; do
  IFS='|' read -r name objective views transport init_mode repulsion fd_x fd_y fd_z \
    step temperature max_update bandwidth seed <<< "${TRIALS[$index]}"
  echo "[plan] gpu=${GPU_LIST[$index]} trial=$name objective=$objective views=$views transport=$transport init=$init_mode repulsion=$repulsion fd=[$fd_x,$fd_y,$fd_z] step=$step temperature=$temperature max_update=$max_update bandwidth=$bandwidth seed=$seed"
done
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] no sweep directory or GPU process was created"
  exit 0
fi

mkdir -p "$TRIAL_ROOT"
{
  echo "scene_run_dir=$BASE_RUN_DIR"
  echo "goal_image=$GOAL_IMAGE"
  echo "profile=$SWEEP_PROFILE"
  echo "feature_encoder=$FEATURE_ENCODER"
  echo "dino_model=$DINO_MODEL"
  echo "particles=$PARTICLES"
  echo "iterations=$ITERATIONS"
  echo "init_mode=start-cloud (random-start-cloud for the seed profile)"
  echo "init_radius=$INIT_RADIUS"
  echo "bounds=$BOUNDS"
  echo "gpu_ids=${GPU_LIST[*]}"
  echo "trace_mode=$TRACE_MODE"
  for index in 0 1 2; do
    echo "trial_$index=${TRIALS[$index]}"
  done
} > "$SWEEP_DIR/sweep_config.txt"

source scripts/common.sh
imagewam_init .
if [[ "$FEATURE_ENCODER" == "flux_ae" ]]; then
  imagewam_require_env FLUX2_AE_MODEL_PATH
  imagewam_require_env FLUX2_SRC
fi
export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONUNBUFFERED=1

run_trial() {
  local index="$1"
  local gpu="${GPU_LIST[$index]}"
  local name objective views transport init_mode repulsion fd_x fd_y fd_z
  local step temperature max_update bandwidth seed
  IFS='|' read -r name objective views transport init_mode repulsion fd_x fd_y fd_z \
    step temperature max_update bandwidth seed <<< "${TRIALS[$index]}"
  local trial_dir="$TRIAL_ROOT/$name"
  mkdir -p "$trial_dir"
  local args=(
    -u -B experiments/libero/svgd_endpoint.py
    --run-dir "$BASE_RUN_DIR"
    --out-dir "$trial_dir"
    --goal "$GOAL_IMAGE"
    --goal-latent-source reencode
    --feature-encoder "$FEATURE_ENCODER"
    --device cuda:0
    --particles "$PARTICLES"
    --iterations "$ITERATIONS"
    --init-mode "$init_mode"
    --init-radius "${INIT_RADIUS_ARGS[@]}"
    --bounds "${BOUNDS_ARGS[@]}"
    --latent-views "$views"
    --latent-distance "$objective"
    --transport "$transport"
    --latent-weight 1.0
    --repulsion-weight "$repulsion"
    --fd-eps "$fd_x" "$fd_y" "$fd_z"
    --step-size "$step"
    --temperature "$temperature"
    --max-update-norm "$max_update"
    --bandwidth-scale "$bandwidth"
    --move-steps "$MOVE_STEPS"
    --settle-steps "$SETTLE_STEPS"
    --controller-gain "$CONTROLLER_GAIN"
    --repeatability-particles "$REPEATABILITY_PARTICLES"
    --rollout-trace-mode "$TRACE_MODE"
    --seed "$seed"
    --save-rollout-traces
  )
  if [[ "$FEATURE_ENCODER" == "flux_ae" ]]; then
    args+=(--editor-ae "$FLUX2_AE_MODEL_PATH" --flux2-src "$FLUX2_SRC")
  else
    args+=(--dino-model "$DINO_MODEL")
  fi
  [[ "$SAVE_ALL_PARTICLES" == "true" ]] && args+=(--save-all-particles)
  [[ "$VERBOSE_EVALUATIONS" == "true" ]] && args+=(--verbose-evaluations)

  {
    echo "physical_gpu=$gpu"
    printf "command=imagewam_python"
    printf " %q" "${args[@]}"
    echo
  } > "$trial_dir/command.txt"

  echo "[start] trial=$name gpu=$gpu"
  if (
    export CUDA_VISIBLE_DEVICES="$gpu"
    imagewam_python "${args[@]}"
    imagewam_python -u -B experiments/libero/plot_svgd_latent_pull.py \
      --history "$trial_dir/history.json" \
      --prefix diagnostics \
      --title "$name"
  ) > "$trial_dir/backend.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] trial=$name gpu=$gpu"
  else
    local status="$?"
    echo "$status" > "$trial_dir/failed.status"
    echo "[failed] trial=$name gpu=$gpu status=$status" >&2
    return 1
  fi
}

PIDS=()
for index in 0 1 2; do
  run_trial "$index" &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done

if compgen -G "$TRIAL_ROOT/*/history.json" > /dev/null; then
  imagewam_python -u -B experiments/libero/summarize_svgd_matrix.py \
    --suite-dir "$SWEEP_DIR"
  imagewam_python -u -B experiments/libero/plot_svgd_best_trajectories.py \
    --suite-dir "$SWEEP_DIR" \
    --output "$SWEEP_DIR/best_rollout_trajectories_3d.png"
fi
if (( FAILED != 0 )); then
  echo "[done-with-failures] $SWEEP_DIR" >&2
  exit 1
fi
echo "[done] $SWEEP_DIR"
