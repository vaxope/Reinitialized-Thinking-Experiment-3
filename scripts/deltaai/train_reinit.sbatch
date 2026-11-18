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
# Time budget + evaluation ([job] in configs/train_reinit.toml, user 2026-11-17): with
# eval_after_training = true, training stops when [job] eval_reserve_min minutes are left in the
# Slurm job, any GPU processes left behind are stopped, and the best saved checkpoint is then
# evaluated in this same job (scripts/deltaai/eval_trained.sbatch best), so the evaluation does
# not wait in the queue again. A crash (not the budget) stops the job without evaluating.
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
if [ ! -f "$REF_DIR.setup_pass" ]; then
  echo "[train] $REF_DIR has no setup_pass marker: the setup job did not finish its checks"; exit 1
fi
mkdir -p "$RUN_DIR/rollouts" "$SAVE_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

FLAGS_FILE="$RUN_DIR/slime_args_$JOB.txt"
S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --ref-load "$REF_DIR" --save "$SAVE_DIR" \
  --out "$FLAGS_FILE" > /dev/null
echo "[train] run $RUN_NAME | flags $FLAGS_FILE | checkpoints $SAVE_DIR"

# ---- time budget (seconds of training allowed before the evaluation reserve)
EVAL_AFTER="$(cfg_value configs/train_reinit.toml job eval_after_training)"   # True / False
RESERVE_MIN="$(cfg_value configs/train_reinit.toml job eval_reserve_min)"
MIN_TRAIN_MIN="$(cfg_value configs/train_reinit.toml job min_train_min)"
BUDGET_S=0
if [ "$EVAL_AFTER" = "True" ]; then
  END="$(squeue -h -j "${SLURM_JOB_ID:-none}" -o %e 2>/dev/null || true)"   # job end time, e.g. 2026-11-19T09:44:10
  if END_S="$(date -d "$END" +%s 2>/dev/null)"; then
    BUDGET_S=$(( END_S - $(date +%s) - RESERVE_MIN * 60 ))
    echo "[train] job ends $END; training budget $((BUDGET_S / 60)) min, then $RESERVE_MIN min for the evaluation"
    if [ "$BUDGET_S" -lt $(( MIN_TRAIN_MIN * 60 )) ]; then
      echo "[train] less than $MIN_TRAIN_MIN min of budget: not training; submit a longer job"; exit 1
    fi
  else
    echo "[train] WARNING: could not read the job's end time ('$END'): training without a budget, no evaluation in this job"
    EVAL_AFTER="False"
  fi
fi

RAY_TMP="/tmp/ray_gar_$JOB"   # short path: Ray's socket paths must stay under ~100 characters
set +e
S env GAR_TRAIN_CONFIG="$REPO_DIR/configs/train_reinit.toml" \
      GAR_ROLLOUT_LOG_DIR="$REPO_DIR/$RUN_DIR/rollouts" REPO_DIR="$REPO_DIR" \
  bash scripts/deltaai/train_inside.sh "$FLAGS_FILE" "$RAY_TMP" "$(free_port)" 1 "${SLURM_CPUS_PER_TASK:-32}" "$BUDGET_S"
STATUS=$?
set -e
case "$STATUS" in
  0)       echo "[train] $(date -u +%FT%TZ) train.py exited normally (all steps done)" ;;
  124|137) echo "[train] $(date -u +%FT%TZ) time budget reached: training stopped (saved checkpoints kept)" ;;
  *)       echo "[train] $(date -u +%FT%TZ) train.py FAILED with status $STATUS: not evaluating"; exit "$STATUS" ;;
esac
if [ "$EVAL_AFTER" != "True" ]; then exit 0; fi

# ---- free the GPU: SGLang/Ray processes that outlived train.py would block the evaluation server
for _ in $(seq 1 36); do
  LEFT="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
  [ -z "$LEFT" ] && break
  sleep 5
done
if [ -n "$LEFT" ]; then
  echo "[train] stopping processes still on the GPU after 3 min: $(echo $LEFT)"
  for pid in $LEFT; do kill -9 "$pid" 2>/dev/null || true; done   # this job's GPU only (--gpus-per-node=1)
  sleep 10
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# ---- evaluate the best saved checkpoint in this job
echo "[train] $(date -u +%FT%TZ) starting the evaluation (eval_trained.sbatch best)"
bash scripts/deltaai/eval_trained.sbatch best
