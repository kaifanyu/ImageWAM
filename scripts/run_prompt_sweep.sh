#!/usr/bin/env bash
# Run a list of prompt variations on libero_10 / task 4 (mugs & plates), in order,
# each under BOTH env conditions: the unchanged layout and a moved-object layout.
# The instruction is passed via a text file so commas/spaces don't trip Hydra.
#
# Optional env overrides:
#   NUM_TRIALS=3  NUM_GPUS=1  FLUX2_VARIANT=4b
#   LAYOUT_SPEC=/path/to/perturb_task4_all.json   # the "moved" layout spec
#
# Usage:
#   bash scripts/run_prompt_sweep.sh
set -euo pipefail

REPO="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# Already inside an activated env (e.g. the container venv)? Leave it alone.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

WORK="${WORK:-$REPO}"
TASK_FILE="$WORK/one_task.txt"
echo "libero_10,4" > "$TASK_FILE"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="$WORK/prompt_sweep_$STAMP"
PROMPT_DIR="$OUT_BASE/prompts"
mkdir -p "$PROMPT_DIR"

NUM_TRIALS="${NUM_TRIALS:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}"
LAYOUT_SPEC="${LAYOUT_SPEC:-$WORK/perturb_task4_all.json}"

# Env conditions: "slug|spec_path"   (empty spec_path = unchanged env)
ENV_CONDS=(
  "orig|"
  "moved|$LAYOUT_SPEC"
)

# "slug|instruction text"   (commas/spaces are fine — text is written to a file)
PROMPTS=(
  "00_baseline|put the white mug on the left plate and put the yellow and white mug on the right plate"
  "01_quick|put the white mug on the left plate and put the yellow and white mug on the right plate, quickly"
  "02_fast|put the white mug on the left plate and put the yellow and white mug on the right plate as fast as possible"
  "03_slow|move slowly and carefully put the white mug on the left plate and put the yellow and white mug on the right plate"
  "04_sequential|first put the white mug on the left plate, then put the yellow and white mug on the right plate"
  "05_pause|put the white mug on the left plate, pause, then put the yellow and white mug on the right plate"
  "06_stack_plates|stack the left plate on top of the right plate"
  "07_put_plate|pick up the left plate and place it on the right plate"
  "08_stack_both_mug|stack both plates and put the white mug on top"
  "09_mug_on_both|put the white mug on top of both plates"
)

echo "Output base : $OUT_BASE"
echo "Task        : libero_10 / task 4   trials=$NUM_TRIALS   gpus=$NUM_GPUS   variant=$FLUX2_VARIANT"
echo "Env conds   : orig (unchanged), moved ($LAYOUT_SPEC)"
echo "Prompts     : ${#PROMPTS[@]}   ->  ${#PROMPTS[@]} x 2 = $(( ${#PROMPTS[@]} * 2 )) runs"
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
echo "Results : $OUT_BASE/<slug>/{orig,moved}/"
echo "Videos  : $OUT_BASE/<slug>/{orig,moved}/**/libero_10/videos/"
echo "Prompts : $PROMPT_DIR/<slug>.txt"
echo "==================================================================="
