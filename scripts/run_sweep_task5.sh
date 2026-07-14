#!/usr/bin/env bash
# libero_10 / task 5: "pick up the book and place it in the back compartment of the caddy".
# 3 prompts x 2 env conditions (original vs a SMALL perturbation: few-cm shifts + slight yaw).
# The desk caddy is a fixture (not movable), so only the book and mug are perturbed.
#
# Optional env overrides:  NUM_TRIALS=3  NUM_GPUS=1  FLUX2_VARIANT=4b
#   LAYOUT_SPEC=/path/to/perturb_task5_small.json
#
# Usage:  bash scripts/run_sweep_task5.sh
set -euo pipefail

REPO=/data3/kaifany/ImageWAM
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

WORK=/home/kaifany/project-data/ImageWAM
TASK_FILE="$WORK/one_task5.txt"
echo "libero_10,5" > "$TASK_FILE"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="$WORK/sweep_task5_$STAMP"
PROMPT_DIR="$OUT_BASE/prompts"
mkdir -p "$PROMPT_DIR"

NUM_TRIALS="${NUM_TRIALS:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}"
LAYOUT_SPEC="${LAYOUT_SPEC:-$WORK/perturb_task5_small.json}"

# Env conditions: "slug|spec_path"   (empty spec_path = original / unchanged env)
ENV_CONDS=(
  "orig|"
  "small_perturb|$LAYOUT_SPEC"
)

# 3 prompts: normal, fast, and a variational task DIRECTION (back -> front compartment).
PROMPTS=(
  "0_normal|pick up the book and place it in the back compartment of the caddy"
  "1_fast|pick up the book and place it in the back compartment of the caddy as fast as possible"
  "2_dir_front|pick up the book and place it in the front compartment of the caddy"
)

echo "Output base : $OUT_BASE"
echo "Task        : libero_10 / task 5   trials=$NUM_TRIALS   gpus=$NUM_GPUS   variant=$FLUX2_VARIANT"
echo "Env conds   : orig (unchanged), small_perturb ($LAYOUT_SPEC)"
echo "Prompts     : ${#PROMPTS[@]}  ->  ${#PROMPTS[@]} x 2 = $(( ${#PROMPTS[@]} * 2 )) runs"
echo

for entry in "${PROMPTS[@]}"; do
  slug="${entry%%|*}"
  text="${entry#*|}"
  pfile="$PROMPT_DIR/$slug.txt"
  printf '%s' "$text" > "$pfile"

  for cond in "${ENV_CONDS[@]}"; do
    env_slug="${cond%%|*}"
    spec="${cond#*|}"
    outdir="$OUT_BASE/$slug/$env_slug"

    echo "==================================================================="
    echo "[$slug / $env_slug] $text"
    echo "-> $outdir"
    echo "==================================================================="

    args=(
      "MULTIRUN.task_file=$TASK_FILE"
      "EVALUATION.output_dir=$outdir"
      "EVALUATION.prompt_override_file=$pfile"
    )
    [ -n "$spec" ] && args+=( "EVALUATION.object_overrides_file=$spec" )

    NUM_GPUS="$NUM_GPUS" FLUX2_VARIANT="$FLUX2_VARIANT" NUM_TRIALS="$NUM_TRIALS" \
      bash scripts/flux2/run_eval_flux2_libero.sh "${args[@]}"

    echo "[done] $slug / $env_slug"
    echo
  done
done

echo "==================================================================="
echo "All $(( ${#PROMPTS[@]} * 2 )) runs complete."
echo "Results : $OUT_BASE/<slug>/{orig,small_perturb}/"
echo "Videos  : $OUT_BASE/<slug>/{orig,small_perturb}/**/libero_10/videos/"
echo "==================================================================="
