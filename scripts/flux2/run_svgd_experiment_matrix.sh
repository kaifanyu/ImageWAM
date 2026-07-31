#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE_RUN_DIR="${BASE_RUN_DIR:-runs/empty_arm_preview}"
PROFILE="${PROFILE:-standard}"

if [[ "${1:-}" != "--worker" ]]; then
  SUITE_NAME="${1:-svgd_${PROFILE}_matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
  SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
  if [[ -e "$SUITE_DIR" && "${RESUME:-false}" != "true" \
      && "${DRY_RUN:-false}" != "true" ]]; then
    echo "Suite already exists: $SUITE_DIR" >&2
    echo "Use RESUME=true with the same suite name to skip completed trials." >&2
    exit 1
  fi
  if [[ ! -f "$BASE_RUN_DIR/manifest.json" ]]; then
    echo "Missing scene manifest: $BASE_RUN_DIR/manifest.json" >&2
    echo "Prepare the scene with scripts/flux2/prepare_svgd_scene.sh first." >&2
    exit 1
  fi
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exec bash "$0" --worker "$SUITE_NAME"
  fi
  mkdir -p "$SUITE_DIR"
  nohup bash "$0" --worker "$SUITE_NAME" \
    > "$SUITE_DIR/launcher.log" 2>&1 < /dev/null &
  echo "$!" > "$SUITE_DIR/launcher.pid"
  echo "Started SVGD experiment matrix."
  echo "Scene: $BASE_RUN_DIR"
  echo "Profile: $PROFILE"
  echo "Suite: $SUITE_DIR"
  echo "Launcher PID: $(<"$SUITE_DIR/launcher.pid")"
  echo "Watch: tail -f $SUITE_DIR/launcher.log"
  exit 0
fi

SUITE_NAME="${2:?Worker mode requires a suite name}"
SUITE_DIR="$BASE_RUN_DIR/$SUITE_NAME"
TRIAL_ROOT="$SUITE_DIR/trials"

read -r -a GPU_LIST <<< "${GPU_IDS:-0 1 2}"
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU index." >&2
  exit 2
fi

case "$PROFILE" in
  smoke)
    DEFAULT_PARTICLES=4
    DEFAULT_ITERATIONS=2
    ;;
  standard)
    DEFAULT_PARTICLES=12
    DEFAULT_ITERATIONS=8
    ;;
  large)
    DEFAULT_PARTICLES=20
    DEFAULT_ITERATIONS=15
    ;;
  *)
    echo "Unknown PROFILE='$PROFILE'. Use smoke | standard | large." >&2
    exit 2
    ;;
esac

PARTICLES="${PARTICLES:-$DEFAULT_PARTICLES}"
ITERATIONS="${ITERATIONS:-$DEFAULT_ITERATIONS}"
GOAL_PATH="${GOAL_PATH:-$BASE_RUN_DIR/goal_oracle.png}"
if [[ -z "${BOUNDS:-}" ]]; then
  BOUNDS="$(
    .venv/bin/python -c '
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
start = manifest["actual_start_eef"]
goal = manifest["physical_goal_eef"]
padding = (0.01, 0.10, 0.01)
values = []
for axis in range(3):
    values.extend((
        min(start[axis], goal[axis]) - padding[axis],
        max(start[axis], goal[axis]) + padding[axis],
    ))
print(" ".join(f"{value:.9g}" for value in values))
' "$BASE_RUN_DIR/manifest.json"
  )"
fi
INIT_RADIUS="${INIT_RADIUS:-0.005 0.02 0.003}"
CONTROLLER_GAIN="${CONTROLLER_GAIN:-15.0}"
REPEATABILITY_PARTICLES="${REPEATABILITY_PARTICLES:-1}"
TRACE_MODE="${TRACE_MODE:-base}"
PLOT_TRIALS="${PLOT_TRIALS:-false}"
SAVE_ALL_PARTICLES="${SAVE_ALL_PARTICLES:-false}"
VERBOSE_EVALUATIONS="${VERBOSE_EVALUATIONS:-false}"

read -r -a BOUNDS_ARGS <<< "$BOUNDS"
read -r -a INIT_RADIUS_ARGS <<< "$INIT_RADIUS"
if (( ${#BOUNDS_ARGS[@]} != 6 || ${#INIT_RADIUS_ARGS[@]} != 3 )); then
  echo "BOUNDS needs six numbers and INIT_RADIUS needs three." >&2
  exit 2
fi
if [[ ! -f "$GOAL_PATH" ]]; then
  echo "Goal image not found: $GOAL_PATH" >&2
  exit 2
fi

# Fields:
# name|objective|transport|init|repulsion|fd_eps|move|settle|seed|views|step|temperature|max_update|bandwidth
EXPERIMENTS=()
add_experiment() {
  EXPERIMENTS+=("$1")
}

add_experiment "metric_rms|rms|svgd|uniform|0.0|0.01|40|8|0|agentview|0.01|0.10|0.01|1.0"
add_experiment "metric_cosine|cosine|svgd|uniform|0.0|0.01|40|8|0|agentview|0.01|0.10|0.01|1.0"
add_experiment "metric_token|token_cosine|svgd|uniform|0.0|0.01|40|8|0|agentview|0.01|0.10|0.01|1.0"
add_experiment "optimizer_particle_gd|token_cosine|particle_gd|uniform|0.0|0.01|40|8|0|agentview|0.01|0.10|0.01|1.0"

if [[ "$PROFILE" != "smoke" ]]; then
  add_experiment "repulsion_025|token_cosine|svgd|uniform|0.25|0.01|40|8|0|agentview|0.01|0.10|0.01|1.0"
  add_experiment "repulsion_100|token_cosine|svgd|uniform|1.0|0.01|40|8|0|agentview|0.01|0.10|0.01|1.0"
  add_experiment "init_start_cloud|token_cosine|svgd|start-cloud|0.0|0.01|40|8|0|agentview|0.01|0.10|0.01|1.0"
  add_experiment "init_random_cloud|token_cosine|svgd|random-start-cloud|0.0|0.01|40|8|0|agentview|0.01|0.10|0.01|1.0"
  add_experiment "uniform_seed1|token_cosine|svgd|uniform|0.0|0.01|40|8|1|agentview|0.01|0.10|0.01|1.0"
  add_experiment "fd_005|token_cosine|svgd|uniform|0.0|0.005|40|8|0|agentview|0.01|0.10|0.01|1.0"
  add_experiment "fd_020|token_cosine|svgd|uniform|0.0|0.02|40|8|0|agentview|0.01|0.10|0.01|1.0"
  add_experiment "horizon_short|token_cosine|svgd|uniform|0.0|0.01|20|8|0|agentview|0.01|0.10|0.01|1.0"
  add_experiment "horizon_long|token_cosine|svgd|uniform|0.0|0.01|80|16|0|agentview|0.01|0.10|0.01|1.0"
  add_experiment "views_both|token_cosine|svgd|uniform|0.0|0.01|40|8|0|both|0.01|0.10|0.01|1.0"
  add_experiment "step_half|token_cosine|svgd|uniform|0.0|0.01|40|8|0|agentview|0.005|0.10|0.005|1.0"
fi

if [[ "$PROFILE" == "large" ]]; then
  add_experiment "uniform_seed2|token_cosine|svgd|uniform|0.0|0.01|40|8|2|agentview|0.01|0.10|0.01|1.0"
  add_experiment "temperature_005|token_cosine|svgd|uniform|0.0|0.01|40|8|0|agentview|0.01|0.05|0.01|1.0"
  add_experiment "temperature_020|token_cosine|svgd|uniform|0.0|0.01|40|8|0|agentview|0.01|0.20|0.01|1.0"
  add_experiment "bandwidth_050|token_cosine|svgd|uniform|0.0|0.01|40|8|0|agentview|0.01|0.10|0.01|0.5"
  add_experiment "bandwidth_200|token_cosine|svgd|uniform|0.0|0.01|40|8|0|agentview|0.01|0.10|0.01|2.0"
fi

echo "[matrix] scene=$BASE_RUN_DIR profile=$PROFILE trials=${#EXPERIMENTS[@]}"
echo "[matrix] particles=$PARTICLES iterations=$ITERATIONS bounds=$BOUNDS"
echo "[matrix] trace_mode=$TRACE_MODE gpu_ids=${GPU_LIST[*]}"
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  printf "%-24s %-14s %-12s %-18s %s\n" \
    "TRIAL" "OBJECTIVE" "TRANSPORT" "INIT" "GPU"
  for ((index=0; index<${#EXPERIMENTS[@]}; index++)); do
    IFS='|' read -r name objective transport init_mode _ \
      <<< "${EXPERIMENTS[$index]}"
    gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
    printf "%-24s %-14s %-12s %-18s %s\n" \
      "$name" "$objective" "$transport" "$init_mode" "$gpu"
  done
  echo "[dry-run] no suite directory or GPU process was created"
  exit 0
fi

mkdir -p "$TRIAL_ROOT"
{
  echo "scene_run_dir=$BASE_RUN_DIR"
  echo "goal_path=$GOAL_PATH"
  echo "profile=$PROFILE"
  echo "particles=$PARTICLES"
  echo "iterations=$ITERATIONS"
  echo "gpu_ids=${GPU_LIST[*]}"
  echo "bounds=$BOUNDS"
  echo "trace_mode=$TRACE_MODE"
  echo "experiments=${#EXPERIMENTS[@]}"
} > "$SUITE_DIR/suite_config.txt"

source scripts/common.sh
imagewam_init .
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env FLUX2_SRC

export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONUNBUFFERED=1

run_trial() {
  local gpu="$1"
  local spec="$2"
  local name objective transport init_mode repulsion fd_eps move_steps
  local settle_steps seed views step_size temperature max_update bandwidth
  IFS='|' read -r name objective transport init_mode repulsion fd_eps \
    move_steps settle_steps seed views step_size temperature max_update bandwidth <<< "$spec"
  local trial_dir="$TRIAL_ROOT/$name"
  mkdir -p "$trial_dir"

  if [[ "${RESUME:-false}" == "true" && -f "$trial_dir/best_metadata.json" ]]; then
    echo "[skip] $name already complete"
    return 0
  fi

  local args=(
    -u -B experiments/libero/svgd_endpoint.py
    --run-dir "$BASE_RUN_DIR"
    --out-dir "$trial_dir"
    --goal "$GOAL_PATH"
    --goal-latent-source reencode
    --editor-ae "$FLUX2_AE_MODEL_PATH"
    --flux2-src "$FLUX2_SRC"
    --device cuda:0
    --particles "$PARTICLES"
    --iterations "$ITERATIONS"
    --init-mode "$init_mode"
    --init-radius "${INIT_RADIUS_ARGS[@]}"
    --bounds "${BOUNDS_ARGS[@]}"
    --latent-views "$views"
    --latent-distance "$objective"
    --transport "$transport"
    --fd-eps "$fd_eps"
    --bandwidth-scale "$bandwidth"
    --step-size "$step_size"
    --temperature "$temperature"
    --max-update-norm "$max_update"
    --latent-weight 1.0
    --repulsion-weight "$repulsion"
    --move-steps "$move_steps"
    --settle-steps "$settle_steps"
    --controller-gain "$CONTROLLER_GAIN"
    --repeatability-particles "$REPEATABILITY_PARTICLES"
    --rollout-trace-mode "$TRACE_MODE"
    --seed "$seed"
  )
  [[ "$SAVE_ALL_PARTICLES" == "true" ]] && args+=(--save-all-particles)
  [[ "$VERBOSE_EVALUATIONS" == "true" ]] && args+=(--verbose-evaluations)

  {
    echo "physical_gpu=$gpu"
    printf "command=imagewam_python"
    printf " %q" "${args[@]}"
    echo
  } > "$trial_dir/command.txt"

  echo "[start] trial=$name gpu=$gpu objective=$objective transport=$transport"
  if (
    export CUDA_VISIBLE_DEVICES="$gpu"
    imagewam_python "${args[@]}"
    if [[ "$PLOT_TRIALS" == "true" ]]; then
      imagewam_python -u -B experiments/libero/plot_svgd_latent_pull.py \
        --history "$trial_dir/history.json" \
        --prefix diagnostics \
        --title "$name"
    fi
  ) > "$trial_dir/backend.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$trial_dir/complete.status"
    echo "[complete] trial=$name gpu=$gpu"
    return 0
  else
    local status="$?"
    echo "$status" > "$trial_dir/failed.status"
    echo "[failed] trial=$name gpu=$gpu status=$status" >&2
    return 1
  fi
}

run_gpu_queue() {
  local slot="$1"
  local gpu="${GPU_LIST[$slot]}"
  local queue_failed=0
  local index
  for ((index=slot; index<${#EXPERIMENTS[@]}; index+=${#GPU_LIST[@]})); do
    run_trial "$gpu" "${EXPERIMENTS[$index]}" || queue_failed=1
  done
  return "$queue_failed"
}

PIDS=()
for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
  run_gpu_queue "$slot" &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done

if compgen -G "$TRIAL_ROOT/*/history.json" > /dev/null; then
  imagewam_python -u -B experiments/libero/summarize_svgd_matrix.py \
    --suite-dir "$SUITE_DIR"
  if [[ "$TRACE_MODE" != "none" ]]; then
    imagewam_python -u -B experiments/libero/plot_svgd_best_trajectories.py \
      --suite-dir "$SUITE_DIR" \
      --output "$SUITE_DIR/best_rollout_trajectories_3d.png"
  fi
fi

if (( FAILED != 0 )); then
  echo "[done-with-failures] suite=$SUITE_DIR" >&2
  exit 1
fi
echo "[done] suite=$SUITE_DIR"
