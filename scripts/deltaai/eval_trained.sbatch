#!/bin/bash
# scripts/deltaai/eval_trained.sbatch: evaluate a trained checkpoint exactly like control_v1, 1 GH200.
#
# Submit through submit.sh:   scripts/deltaai/submit.sh eval-trained [STEP|best]   (default best)
# Steps:
#   0. refuse uncommitted changes in src/, configs/, scripts/ (the run must be traceable);
#   1. pick the checkpoint (src/pick_checkpoint.py): the saved step with the highest validation
#      v3 accuracy on the held-out SuperGPQA slice (ties -> later), or the STEP given. GPQA is
#      never used for this choice;
#   2. convert it to Hugging Face format with slime's tools/convert_torch_dist_to_hf.py (the tool
#      slime_setup.sbatch's round-trip check uses), copying the untouched vision/MTP weights and the
#      tokenizer/config files from the pinned base snapshot; then src/check_trained.py: same tensors
#      and files as the base, only language-model weights changed (at least one), and a
#      fingerprint (gar_checkpoint.json). Output: $SLIME_CKPT_DIR/<run>_hf/step<N> (about 5 GB).
#      Skipped if that folder already passed (a resubmission resumes); stops if it exists unchecked;
#   3. the control evaluation, unchanged except for the weights: serve_and_generate.sh with the
#      SGLang image of control_v1 (configs/control.toml [container]) and src/generate.py
#      --checkpoint: GPQA Diamond, 198 x 4 samples x 3 modes, the control sampling, 81,920 cap.
#      Run id eval_<run>_step<N>; resubmitting continues where a time limit stopped it;
#   4. score (src/score.py: rule v3 + lenient) and the mechanical checks (src/check_run.py).
#      Comparison with control_v1: python3 -m src.compare_runs (locally or here).
#SBATCH --job-name=gar-eval-trained
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
# RAM: the converter loads the whole Megatron checkpoint (weights + optimizer state, ~30 GB).
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
WHICH="${1:-best}"
JOB="${SLURM_JOB_ID:-local}"
echo "[eval] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job $JOB | checkpoint: $WHICH"

# ---- 0
if [ -n "$(git status --porcelain -- src configs scripts)" ]; then
  echo "[eval] uncommitted changes in src/, configs/ or scripts/: commit first so the run is traceable"; exit 1
fi
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
check_slime_sif
slime_paths
SAVE_DIR="$SLIME_CKPT_DIR/$RUN_NAME"

# ---- 1
PICK=(python3 -m src.pick_checkpoint --rollouts "runs/$RUN_NAME/rollouts" --ckpt-dir "$SAVE_DIR")
if [ "$WHICH" != "best" ]; then PICK+=(--step "$WHICH"); fi
S "${PICK[@]}"
eval "$(S "${PICK[@]}" --shell)"   # sets STEP and ITER_DIR
EVAL_DIR="$SLIME_CKPT_DIR/${RUN_NAME}_hf/step$STEP"
RUN_ID="eval_${RUN_NAME}_step$STEP"
echo "[eval] step $STEP: $ITER_DIR -> $EVAL_DIR, run id $RUN_ID"

# ---- 2
if [ -f "$EVAL_DIR/gar_checkpoint.json" ]; then
  echo "[eval] $EVAL_DIR already converted and checked: skipping conversion"
elif [ -e "$EVAL_DIR" ]; then
  echo "[eval] $EVAL_DIR exists without gar_checkpoint.json; delete it yourself and resubmit"; exit 1
else
  PART="$EVAL_DIR.partial_$JOB"   # made by this job only; renamed once the check passes
  mkdir -p "$(dirname "$EVAL_DIR")"
  T0=$SECONDS
  S env PYTHONPATH=/opt/Megatron-LM:/opt/slime \
    python3 /opt/slime/tools/convert_torch_dist_to_hf.py --input-dir "$ITER_DIR" --output-dir "$PART" \
    --origin-hf-dir "$HF_SNAPSHOT" --add-missing-from-origin-hf --vocab-size "$VOCAB_SIZE"
  SOURCE="{\"run\": \"$RUN_NAME\", \"step\": $STEP, \"iter_dir\": \"$ITER_DIR\", \"job\": \"$JOB\", \"git\": \"$(git rev-parse HEAD)\"}"
  set +e
  S python3 -m src.check_trained "$HF_SNAPSHOT" "$PART" --source "$SOURCE"
  CHECK=$?
  set -e
  if [ "$CHECK" -ne 0 ]; then
    rm -rf "$PART"   # this job's own temporary copy
    echo "[eval] FAIL: the converted checkpoint did not pass src/check_trained.py"; exit 1
  fi
  mv "$PART" "$EVAL_DIR"
  echo "[eval] converted and checked in $((SECONDS - T0)) s"
fi

# ---- 3
export EVAL_MODEL_DIR="$EVAL_DIR"   # serve_and_generate.sh serves this folder; common.sh binds it
set +e
bash scripts/deltaai/serve_and_generate.sh --run-id "$RUN_ID" --checkpoint "$EVAL_DIR"
GEN=$?
set -e
echo "[eval] generation exited with status $GEN"

# ---- 4 (also after errors: errors are records and are counted; a time limit never reaches here)
source scripts/deltaai/common.sh
run python3 -m src.score "runs/$RUN_ID/generations.jsonl" --overwrite
run python3 -m src.check_run "runs/$RUN_ID" || echo "[eval] check_run reported problems (see above)"
echo "[eval] $(date -u +%FT%TZ) done: runs/$RUN_ID/summary.md"
exit "$GEN"
