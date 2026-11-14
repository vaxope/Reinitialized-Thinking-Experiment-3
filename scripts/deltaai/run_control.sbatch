#!/bin/bash
# scripts/deltaai/run_control.sbatch: Slurm batch job for the pilot and the full control run.
#
# Submit through submit.sh (it adds --account, --partition and --time and asks for
# confirmation):   scripts/deltaai/submit.sh pilot     /     scripts/deltaai/submit.sh full
# Arguments after the script name go to src/generate.py (submit.sh passes --run-id and --pilot).
#
# Resources: 1 node, 1 GPU (one GH200/H100 holds the 2B model plus a large KV cache), CPU
# cores for tokenization and the HTTP client, RAM for the server process. The job's time
# limit is set by submit.sh; if it is hit, everything written so far is kept, and submitting
# the same command again resumes where it stopped.
#SBATCH --job-name=gar-control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=runs/_slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # Slurm starts the job in the directory sbatch was run from
if [ -f scripts/deltaai/env.sh ]; then source scripts/deltaai/env.sh; fi

echo "[job] $(date -u +%FT%TZ) run_control: args: $*"
bash scripts/deltaai/serve_and_generate.sh "$@"
