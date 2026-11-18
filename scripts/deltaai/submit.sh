#!/usr/bin/env bash
# scripts/deltaai/submit.sh: the ONE way to submit control-run jobs (run on a DeltaAI login node).
#
# Usage:  scripts/deltaai/submit.sh smoke            [--dry-run]
#         scripts/deltaai/submit.sh pilot [RUN_ID]   [--dry-run]   (default RUN_ID pilot_v1)
#         scripts/deltaai/submit.sh full  [RUN_ID]   [--dry-run]   (default RUN_ID control_v1)
#         scripts/deltaai/submit.sh build-slime      [--dry-run]   (slime training image, build_slime.sbatch)
#         scripts/deltaai/submit.sh slime-setup      [--dry-run]   (weights -> Megatron + round-trip check)
#         scripts/deltaai/submit.sh train            [--dry-run]   (GRPO run; name/steps from configs/train_reinit.toml)
#         scripts/deltaai/submit.sh setup-train      [--dry-run]   (slime-setup, then train, in ONE job: one queue wait)
#         scripts/deltaai/submit.sh eval-trained [STEP|best] [--dry-run]   (trained checkpoint, control evaluation)
#
# It reads scripts/deltaai/env.sh (account, partition, time limits), prints exactly what will
# be requested (1 GPU, time limit, the maximum GPU-hours that limit allows), asks for
# confirmation (showing the account's remaining balance from `accounts`), then calls sbatch with --account/--partition/--time added on the command
# line (those can't be hard-coded in the .sbatch files because they differ per user).
# --dry-run prints the sbatch command without submitting.
# Re-submitting the same pilot/full command after a time-limit kill RESUMES the run.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ ! -f scripts/deltaai/env.sh ]; then
  echo "scripts/deltaai/env.sh not found: cp scripts/deltaai/env.example.sh scripts/deltaai/env.sh and fill it in"; exit 1
fi
source scripts/deltaai/env.sh

DRY=0; ARGS=()
for a in "$@"; do if [ "$a" = "--dry-run" ]; then DRY=1; else ARGS+=("$a"); fi; done
set -- "${ARGS[@]+"${ARGS[@]}"}"
KIND="${1:-}"

case "$KIND" in
  smoke) TIME="${SMOKE_TIME:-00:30:00}"; SCRIPT=scripts/deltaai/smoke_test.sbatch; JOB_ARGS=() ;;
  pilot) TIME="${PILOT_TIME:-02:00:00}"; SCRIPT=scripts/deltaai/run_control.sbatch; JOB_ARGS=(--run-id "${2:-pilot_v1}" --pilot) ;;
  full)  TIME="${FULL_TIME:-24:00:00}";  SCRIPT=scripts/deltaai/run_control.sbatch; JOB_ARGS=(--run-id "${2:-control_v1}") ;;
  build-slime) TIME="${BUILD_SLIME_TIME:-03:30:00}"; SCRIPT=scripts/deltaai/build_slime.sbatch; JOB_ARGS=() ;;
  slime-setup) TIME="${SLIME_SETUP_TIME:-00:45:00}"; SCRIPT=scripts/deltaai/slime_setup.sbatch; JOB_ARGS=() ;;
  train)       TIME="${TRAIN_TIME:-04:00:00}";       SCRIPT=scripts/deltaai/train_reinit.sbatch; JOB_ARGS=() ;;
  setup-train) TIME="${SETUP_TRAIN_TIME:-20:00:00}"; SCRIPT=scripts/deltaai/setup_and_train.sbatch; JOB_ARGS=() ;;
  eval-trained) TIME="${EVAL_TRAINED_TIME:-02:30:00}"; SCRIPT=scripts/deltaai/eval_trained.sbatch; JOB_ARGS=("${2:-best}") ;;
  *) echo "usage: $0 smoke|pilot [RUN_ID]|full [RUN_ID]|build-slime|slime-setup|train|setup-train|eval-trained [STEP] [--dry-run]"; exit 2 ;;
esac
for v in DELTAAI_ACCOUNT DELTAAI_PARTITION; do
  case "${!v:-}" in ""|"<"*) echo "$v is not set in scripts/deltaai/env.sh"; exit 1 ;; esac
done

# Maximum GPU-hours = 1 GPU x time limit (Slurm charges the time actually used, usually less).
IFS=: read -r hh mm ss <<< "$TIME"
MAX_GPUH=$(awk -v h="$hh" -v m="$mm" -v s="$ss" 'BEGIN { printf "%.2f", (h*3600 + m*60 + s) / 3600 }')
mkdir -p runs/_slurm   # Slurm does not create the --output directory itself

CMD=(sbatch --account="$DELTAAI_ACCOUNT" --partition="$DELTAAI_PARTITION" --time="$TIME" "$SCRIPT" "${JOB_ARGS[@]+"${JOB_ARGS[@]}"}")
case "$KIND" in
  smoke) DESC="smoke test: 2 questions x 1 sample x 3 modes" ;;
  build-slime) DESC="build the slime training image (compiles on 64 CPUs; holds 1 GPU, the billing unit)" ;;
  slime-setup) DESC="convert the model to Megatron format + round-trip weight check" ;;
  train)       DESC="GRPO training (configs/train_reinit.toml: run name and number of steps)" ;;
  setup-train) DESC="setup (skipped if already passed), then GRPO training, in one job" ;;
  eval-trained) DESC="convert + check checkpoint '${JOB_ARGS[0]}', then the control evaluation (198 x 4 x 3 modes)" ;;
  *) DESC="${JOB_ARGS[*]}" ;;
esac
echo "Job:       $KIND ($DESC)"
echo "Request:   1 node, 1 GPU, time limit $TIME on partition $DELTAAI_PARTITION, account $DELTAAI_ACCOUNT"
echo "Cost:      at most $MAX_GPUH GPU-hours (charged for time actually used)"
# Remaining balance of the account (DeltaAI's `accounts` prints: Account  Balance(Hours)  Deposited(Hours) ...).
# Display only: the allocation is shared, so every submission shows what is left.
BAL=$(accounts 2>/dev/null | awk -v a="$DELTAAI_ACCOUNT" '$1 == a { print $2 " of " $3 " hours left" }' || true)
echo "Balance:   ${BAL:-unavailable (run \`accounts\` to check)} on $DELTAAI_ACCOUNT"
echo "Command:   ${CMD[*]}"
if [ "$DRY" = 1 ]; then echo "(dry run: not submitted)"; exit 0; fi
read -r -p "Submit this job? [y/N] " ans
if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then echo "Not submitted."; exit 0; fi
"${CMD[@]}"
echo "Follow it with:  squeue -u \$USER   and   tail -f runs/_slurm/<job-name>_<job-id>.out"
