#!/bin/bash
# scripts/deltaai/smoke_test.sbatch: short end-to-end test of the whole DeltaAI pipeline.
#
# Submit through submit.sh:   scripts/deltaai/submit.sh smoke
# What it does, on 1 GPU, in a few minutes:
#   1. starts SGLang and runs src/generate.py on 2 questions x 1 sample x 2 modes, with the
#      output capped at SMOKE_MAX_NEW_TOKENS (default 4096) so it finishes fast. The cap is a
#      test-only override; the run id starts with "smoke_" so it can never be mistaken for, or
#      mixed into, a real run (generate.py refuses overrides for control_*/pilot_* ids).
#   2. src/check_run.py: mechanical checks of every record (prompts rebuilt and compared,
#      token ids, counts, finish reasons, think-token counting); FAIL -> job fails
#   3. src/score.py: scores the 4 generations and prints the summary
# Truncated think outputs are expected with the small cap and are only warnings.
#SBATCH --job-name=gar-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/_slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f scripts/deltaai/env.sh ]; then source scripts/deltaai/env.sh; fi
source scripts/deltaai/common.sh

RUN_ID="smoke_${SLURM_JOB_ID:-local$(date +%Y%m%d%H%M%S)}"
echo "[smoke] $(date -u +%FT%TZ) run id $RUN_ID"

set +e
bash scripts/deltaai/serve_and_generate.sh --run-id "$RUN_ID" --limit-questions 2 --samples 1 \
     --max-new-tokens "${SMOKE_MAX_NEW_TOKENS:-4096}"
gen_status=$?
set -e
echo "[smoke] generation exit status: $gen_status"
if [ ! -f "runs/$RUN_ID/generations.jsonl" ]; then
  echo "[smoke] FAIL: no generations were written (see the server log in runs/_server_logs/)"; exit 1
fi

echo "[smoke] ---- check_run ----"
set +e
HF_HUB_OFFLINE=1 run "$PYTHON" -m src.check_run "runs/$RUN_ID" 2>&1 | grep -v "PyTorch was not found"
check_status=${PIPESTATUS[0]}
set -e

echo "[smoke] ---- score ----"
run "$PYTHON" -m src.score "runs/$RUN_ID/generations.jsonl"

if [ "$gen_status" -eq 0 ] && [ "$check_status" -eq 0 ]; then
  echo "[smoke] PASS: pipeline works end to end. Read runs/$RUN_ID in the viewer before the pilot."
else
  echo "[smoke] FAIL (generate=$gen_status, check_run=$check_status)"; exit 1
fi
