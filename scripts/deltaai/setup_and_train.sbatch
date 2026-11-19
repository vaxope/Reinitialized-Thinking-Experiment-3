#!/bin/bash
# scripts/deltaai/setup_and_train.sbatch: the setup job and the training job in ONE Slurm job,
# so there is one queue wait instead of two (user, 2026-11-17). 1 GH200.
#
# Submit through submit.sh:   scripts/deltaai/submit.sh setup-train
# Steps:
#   0. refuse uncommitted changes in src/, configs/, scripts/ (checked here first, so a dirty tree
#      fails in seconds instead of after the 20-30 min setup);
#   1. setup = slime_setup.sbatch unchanged (image hash, HF -> Megatron, round-trip check, strict
#      flag check), run as a plain script inside this allocation. Skipped when its marker
#      <Megatron copy>.setup_pass exists (a resubmission after a time limit doesn't redo it);
#      stops if the Megatron copy exists WITHOUT the marker (an earlier setup did not finish);
#   1b. the smoke run (scripts/deltaai/smoke_train.sh: 2 real training steps, convert, serve),
#      only if its marker <Megatron copy>.smoke_pass is missing (slime_setup.sbatch runs it too);
#   2. training = train_reinit.sbatch unchanged (it re-checks the tree, image and both markers).
# Each step stops the job on failure (set -e), so training never starts after a failed setup.
# Resources are the training job's (the larger of the two); #SBATCH lines inside the two
# scripts are plain comments when they run this way.
#SBATCH --job-name=gar-setup-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "[setup-train] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job ${SLURM_JOB_ID:-local}"

# ---- 0
if [ -n "$(git status --porcelain -- src configs scripts)" ]; then
  echo "[setup-train] uncommitted changes in src/, configs/ or scripts/: commit first so the run is traceable"; exit 1
fi
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
slime_paths

# ---- 1
if [ -f "$REF_DIR.setup_pass" ]; then
  echo "[setup-train] setup already passed ($(cat "$REF_DIR.setup_pass")): skipping it"
elif [ -e "$REF_DIR" ]; then
  echo "[setup-train] $REF_DIR exists without $REF_DIR.setup_pass: an earlier setup did not finish."
  echo "[setup-train] Delete that folder yourself and resubmit."; exit 1
else
  T0=$SECONDS
  bash scripts/deltaai/slime_setup.sbatch
  echo "[setup-train] setup took $((SECONDS - T0)) s"
fi

# ---- 1b
if [ -f "$REF_DIR.smoke_pass" ]; then
  echo "[setup-train] smoke run already passed ($(cat "$REF_DIR.smoke_pass")): skipping it"
else
  bash scripts/deltaai/smoke_train.sh
fi

# ---- 2
bash scripts/deltaai/train_reinit.sbatch
echo "[setup-train] $(date -u +%FT%TZ) done"
