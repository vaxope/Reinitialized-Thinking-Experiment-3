#!/bin/bash
# scripts/deltaai/train_reinit.sbatch: the GRPO training job (pilot or main run), 1 GH200.
#
# Submit through submit.sh:   scripts/deltaai/submit.sh train
# What it runs: slime's train.py inside the slime image, with the flags src/train_args.py builds
# from configs/train_reinit.toml (the confirmed parameter sheet; the run name and step count come
# from there too). Outputs, all under runs/<run name>/ unless noted:
#   slime_args_<job>.txt   the exact flags of this job
#   rollouts/              every training and validation answer with its reward (src/reward.py)
#   <SLIME_CKPT_DIR>/<run name>/   Megatron checkpoints every [checkpoint] save_interval steps
# Resume: the checkpoint folder is also --load, so resubmitting the same command after a time
# limit continues from the last saved step (steps after it are redone).
# Requires: the image (submit.sh build-slime, hash recorded), the Megatron copy of the model
# (submit.sh slime-setup) and the slime input files (python3 -m src.train_prep).
#SBATCH --job-name=gar-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
# CPUs for SGLang, Ray and data handling; RAM: with --colocate the model, its fp32 master weights
# and Adam state (about 30 GB) are moved to CPU memory while SGLang generates.
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
JOB="${SLURM_JOB_ID:-local}"
echo "[train] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job $JOB"
if [ -n "$(git status --porcelain -- src configs scripts)" ]; then
  echo "[train] uncommitted changes in src/, configs/ or scripts/: commit first so the run is traceable"; exit 1
fi

check_slime_sif
slime_paths
RUN_DIR="runs/$RUN_NAME"
SAVE_DIR="$SLIME_CKPT_DIR/$RUN_NAME"
if [ ! -d "$REF_DIR" ]; then echo "[train] $REF_DIR missing: run submit.sh slime-setup first"; exit 1; fi
mkdir -p "$RUN_DIR/rollouts" "$SAVE_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

FLAGS_FILE="$RUN_DIR/slime_args_$JOB.txt"
S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --ref-load "$REF_DIR" --save "$SAVE_DIR" \
  --out "$FLAGS_FILE" > /dev/null
echo "[train] run $RUN_NAME | flags $FLAGS_FILE | checkpoints $SAVE_DIR"

RAY_TMP="/tmp/ray_gar_$JOB"   # short path: Ray's socket paths must stay under ~100 characters
S env GAR_TRAIN_CONFIG="$REPO_DIR/configs/train_reinit.toml" \
      GAR_ROLLOUT_LOG_DIR="$REPO_DIR/$RUN_DIR/rollouts" REPO_DIR="$REPO_DIR" \
  bash scripts/deltaai/train_inside.sh "$FLAGS_FILE" "$RAY_TMP" "$(free_port)" 1 "${SLURM_CPUS_PER_TASK:-32}"
echo "[train] $(date -u +%FT%TZ) train.py exited normally"
