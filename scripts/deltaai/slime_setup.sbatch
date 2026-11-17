#!/bin/bash
# scripts/deltaai/slime_setup.sbatch: one-time setup before training (about 20-30 min, 1 GPU).
#
# Submit through submit.sh:   scripts/deltaai/submit.sh slime-setup
# Steps:
#   1. check the slime image is the recorded one (SHA-256);
#   2. convert the pinned Qwen3.5-2B Hugging Face weights into Megatron format with slime's
#      tools/convert_hf_to_torch_dist.py, using EXACTLY the architecture flags training uses
#      (src/train_args.py --model-args, checked against the model's config.json). This copy is
#      both the starting point of training and the frozen KL reference (--ref-load);
#   3. convert it back to Hugging Face format (tools/convert_torch_dist_to_hf.py) and require every
#      language-model weight to match the original (src/check_roundtrip.py); then delete that
#      temporary copy (it was created by this job and is not needed again);
#   4. build the training flags once (src/train_args.py) to catch any config problem now.
# Output: $SLIME_CKPT_DIR/<model>_<rev>_torch_dist (about 5 GB). Never overwritten: if it exists,
# the job stops (delete it yourself to redo the conversion).
#SBATCH --job-name=gar-slime-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
echo "[setup] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job ${SLURM_JOB_ID:-local}"

# ---- 1
check_slime_sif
slime_paths
echo "[setup] model $HF_SNAPSHOT"
echo "[setup] Megatron copy -> $REF_DIR"
if [ -e "$REF_DIR" ]; then echo "[setup] $REF_DIR already exists; not overwriting (delete it to redo)."; exit 1; fi

# ---- 2
mapfile -t MODEL_ARGS < <(S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --model-args)
echo "[setup] architecture flags: ${MODEL_ARGS[*]}"
T0=$SECONDS
S env MASTER_ADDR=127.0.0.1 MASTER_PORT="$(free_port)" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  PYTHONPATH=/opt/Megatron-LM:/opt/slime \
  python3 /opt/slime/tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_SNAPSHOT" --save "$REF_DIR"
echo "[setup] conversion done in $((SECONDS - T0)) s: $(du -sh "$REF_DIR" | cut -f1)"

# ---- 3
CKPT_SUBDIR="$(dirname "$(find "$REF_DIR" -name common.pt | head -n 1)")"
RT_DIR="$SLIME_CKPT_DIR/_roundtrip_check_${SLURM_JOB_ID:-local}"
S python3 /opt/slime/tools/convert_torch_dist_to_hf.py --input-dir "$CKPT_SUBDIR" \
  --output-dir "$RT_DIR" --origin-hf-dir "$HF_SNAPSHOT" --vocab-size "$VOCAB_SIZE"
set +e
S python3 -m src.check_roundtrip "$HF_SNAPSHOT" "$RT_DIR"
RT_STATUS=$?
set -e
rm -rf "$RT_DIR"   # temporary copy made by this job
if [ "$RT_STATUS" -ne 0 ]; then
  echo "[setup] FAIL: the round trip changed weights; do not train on $REF_DIR"; exit 1
fi

# ---- 4
S python3 -m src.train_args --hf-checkpoint "$HF_SNAPSHOT" --ref-load "$REF_DIR" \
  --save "$SLIME_CKPT_DIR/$RUN_NAME" --out "runs/$RUN_NAME/slime_args_setup.txt" > /dev/null
echo "[setup] training flags OK ($(wc -l < "runs/$RUN_NAME/slime_args_setup.txt") lines, runs/$RUN_NAME/slime_args_setup.txt)"
echo "[setup] $(date -u +%FT%TZ) PASS"
