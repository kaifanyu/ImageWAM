#!/usr/bin/env bash
# libero_10 / task 6: "put the white mug on the plate and put the chocolate
# pudding to the right of the plate". Runs 3 prompts x 2 env conditions
# (original layout vs a repositioned+reoriented layout), in order.
#
# Optional env overrides:  NUM_TRIALS=3  NUM_GPUS=1  FLUX2_VARIANT=4b
#   LAYOUT_SPEC=/path/to/perturb_task6_reoriented.json
#
# Usage:  bash scripts/run_sweep_task6.sh
set -euo pipefail

REPO="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# Already inside an activated env (e.g. the container venv)? Leave it alone.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

WORK="${WORK:-$REPO}"
TASK_FILE="$WORK/one_task6.txt"
echo "libero_10,6" > "$TASK_FILE"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="$WORK/sweep_task6_$STAMP"
PROMPT_DIR="$OUT_BASE/prompts"
mkdir -p "$PROMPT_DIR"

NUM_TRIALS="${NUM_TRIALS:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}"
LAYOUT_SPEC="${LAYOUT_SPEC:-$WORK/perturb_task6_reoriented.json}"

# Env conditions: "slug|spec_path"   (empty spec_path = original / unchanged env)
ENV_CONDS=(
  "orig|"
  "reoriented|$LAYOUT_SPEC"
)

# 3 prompts: normal, fast, and a variational task DIRECTION (right -> left).
PROMPTS=(
  "0_normal|put the white mug on the plate and put the chocolate pudding to the right of the plate"
  "1_fast|put the white mug on the plate and put the chocolate pudding to the right of the plate as fast as possible"
  "2_dir_left|put the white mug on the plate and put the chocolate pudding to the left of the plate"
)

echo "Output base : $OUT_BASE"
echo "Task        : libero_10 / task 6   trials=$NUM_TRIALS   gpus=$NUM_GPUS   variant=$FLUX2_VARIANT"
echo "Env conds   : orig (unchanged), reoriented ($LAYOUT_SPEC)"
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
echo "Results : $OUT_BASE/<slug>/{orig,reoriented}/"
echo "Videos  : $OUT_BASE/<slug>/{orig,reoriented}/**/libero_10/videos/"
echo "==================================================================="
