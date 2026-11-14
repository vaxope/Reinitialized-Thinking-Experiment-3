#!/usr/bin/env bash
# scripts/deltaai/submit.sh: the ONE way to submit control-run jobs (run on a DeltaAI login node).
#
# Usage:  scripts/deltaai/submit.sh smoke            [--dry-run]
#         scripts/deltaai/submit.sh pilot [RUN_ID]   [--dry-run]   (default RUN_ID pilot_v1)
#         scripts/deltaai/submit.sh full  [RUN_ID]   [--dry-run]   (default RUN_ID control_v1)
#
# It reads scripts/deltaai/env.sh (account, partition, time limits), prints exactly what will
# be requested (1 GPU, time limit, the maximum GPU-hours that limit allows), asks for
# confirmation, then calls sbatch with --account/--partition/--time added on the command
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
  *) echo "usage: $0 smoke|pilot [RUN_ID]|full [RUN_ID] [--dry-run]"; exit 2 ;;
esac
for v in DELTAAI_ACCOUNT DELTAAI_PARTITION; do
  case "${!v:-}" in ""|"<"*) echo "$v is not set in scripts/deltaai/env.sh"; exit 1 ;; esac
done

# Maximum GPU-hours = 1 GPU x time limit (Slurm charges the time actually used, usually less).
IFS=: read -r hh mm ss <<< "$TIME"
MAX_GPUH=$(awk -v h="$hh" -v m="$mm" -v s="$ss" 'BEGIN { printf "%.2f", (h*3600 + m*60 + s) / 3600 }')
mkdir -p runs/_slurm   # Slurm does not create the --output directory itself

CMD=(sbatch --account="$DELTAAI_ACCOUNT" --partition="$DELTAAI_PARTITION" --time="$TIME" "$SCRIPT" "${JOB_ARGS[@]+"${JOB_ARGS[@]}"}")
echo "Job:       $KIND (${JOB_ARGS[*]:-smoke test: 2 questions x 1 sample x 2 modes})"
echo "Request:   1 node, 1 GPU, time limit $TIME on partition $DELTAAI_PARTITION, account $DELTAAI_ACCOUNT"
echo "Cost:      at most $MAX_GPUH GPU-hours (charged for time actually used)"
echo "Command:   ${CMD[*]}"
if [ "$DRY" = 1 ]; then echo "(dry run: not submitted)"; exit 0; fi
read -r -p "Submit this job? [y/N] " ans
if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then echo "Not submitted."; exit 0; fi
"${CMD[@]}"
echo "Follow it with:  squeue -u \$USER   and   tail -f runs/_slurm/<job-name>_<job-id>.out"
