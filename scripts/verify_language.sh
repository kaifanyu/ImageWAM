#!/usr/bin/env bash
# DECISIVE language-grounding test (in-distribution).
# LIBERO-goal: all tasks share ONE kitchen scene; only the language goal differs,
# and every goal below is a TRAINED behavior with the required objects present.
# We hold the scene + init + SEED fixed on task 8 ("put the bowl on the plate")
# and swap the prompt among trained goals.
#
#   - If the robot goes to DIFFERENT targets per prompt  -> language IS influential
#     and used to select behavior (your "not influential" claim is refuted; the
#     real issue is OOD generalization).
#   - If it does the SAME thing regardless of prompt     -> language-agnostic even
#     in-distribution (your claim holds strongly).
#
# NOTE: the success flag is only meaningful for 0_control (its goal matches the
# checker). For the others, JUDGE BY VIDEO (which object/target it moves toward).
#
# Optional: NUM_TRIALS=1  NUM_GPUS=1  FLUX2_VARIANT=4b  SEED=0
# Usage: bash scripts/verify_language.sh
set -euo pipefail

REPO="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# Already inside an activated env (e.g. the container venv)? Leave it alone.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

WORK="${WORK:-$REPO}"
TASK_FILE="$WORK/one_goal8.txt"
echo "libero_goal,8" > "$TASK_FILE"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="$WORK/verify_language_$STAMP"
PROMPT_DIR="$OUT_BASE/prompts"
mkdir -p "$PROMPT_DIR"

NUM_TRIALS="${NUM_TRIALS:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}"
SEED="${SEED:-0}"

# All are trained LIBERO-goal instructions valid in this shared scene.
PROMPTS=(
  "0_control|put the bowl on the plate"
  "1_bowl_stove|put the bowl on the stove"
  "2_cheese_bowl|put the cream cheese in the bowl"
  "3_turn_stove|turn on the stove"
  "4_open_drawer|open the middle drawer of the cabinet"
)

echo "Output base : $OUT_BASE"
echo "Scene       : libero_goal / task 8 (shared kitchen scene)   trials=$NUM_TRIALS   SEED=$SEED (FIXED)"
echo "Prompts     : ${#PROMPTS[@]} trained goals, scene+seed held constant"
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
echo "Done. Same scene + same seed => any behavior change is the LANGUAGE."
echo "Watch: $OUT_BASE/<slug>/**/libero_goal/videos/"
echo "Does the arm go to plate vs stove vs cabinet vs cream cheese per prompt?"
echo "==================================================================="
