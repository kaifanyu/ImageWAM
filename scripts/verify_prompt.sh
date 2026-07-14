#!/usr/bin/env bash
# Prompt-sensitivity check: same task, same (original) layout, SAME fixed seed,
# only the prompt changes. If the videos differ across these prompts, the model
# is responding to language; if even drastic prompts give identical video, the
# policy is (nearly) language-agnostic. Either way the plumbing is already proven
# correct -- this measures the MODEL's sensitivity.
#
# Optional: NUM_TRIALS=1  NUM_GPUS=1  FLUX2_VARIANT=4b  SEED=0
# Usage: bash scripts/verify_prompt.sh
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
OUT_BASE="$WORK/verify_prompt_$STAMP"
PROMPT_DIR="$OUT_BASE/prompts"
mkdir -p "$PROMPT_DIR"

NUM_TRIALS="${NUM_TRIALS:-1}"   # 1 trial is enough with a fixed seed
NUM_GPUS="${NUM_GPUS:-1}"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}"
SEED="${SEED:-0}"               # FIXED seed -> prompt is the only variable

# Deliberately spans subtle -> drastic so you can see where sensitivity kicks in.
PROMPTS=(
  "0_original|put the white mug on the plate and put the chocolate pudding to the right of the plate"
  "1_dir_left|put the white mug on the plate and put the chocolate pudding to the left of the plate"
  "2_wrong_obj|pick up the red mug and lift it into the air"
  "3_other_task|push all the objects to the edge of the table"
  "4_swap|put the chocolate pudding on the plate"
)

echo "Output base : $OUT_BASE"
echo "Task        : libero_10 / task 6 (original layout)   trials=$NUM_TRIALS   SEED=$SEED (FIXED)"
echo "Prompts     : ${#PROMPTS[@]} (fixed seed => only the prompt varies)"
echo

for entry in "${PROMPTS[@]}"; do
  slug="${entry%%|*}"
  text="${entry#*|}"
  pfile="$PROMPT_DIR/$slug.txt"
  printf '%s' "$text" > "$pfile"
  outdir="$OUT_BASE/$slug"

  echo "==================================================================="
  echo "[$slug] '${text}'"
  echo "==================================================================="

  NUM_GPUS="$NUM_GPUS" FLUX2_VARIANT="$FLUX2_VARIANT" NUM_TRIALS="$NUM_TRIALS" \
    bash scripts/flux2/run_eval_flux2_libero.sh \
      "MULTIRUN.task_file=$TASK_FILE" \
      "EVALUATION.output_dir=$outdir" \
      "EVALUATION.prompt_override_file=$pfile" \
      "seed=$SEED"

  echo "[done] $slug"
  echo
done

echo "==================================================================="
echo "Done. Compare videos across prompts (same seed => any difference is the prompt):"
echo "  $OUT_BASE/<slug>/**/libero_10/videos/"
echo "If 2_wrong_obj / 3_other_task look the same as 0_original, the policy is"
echo "language-insensitive (a real finding), NOT a plumbing bug."
echo "==================================================================="
