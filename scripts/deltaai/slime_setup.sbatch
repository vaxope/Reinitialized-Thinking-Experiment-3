#!/bin/bash
# scripts/deltaai/slime_setup.sbatch: one-time setup before training (about 1-1.5 h with the smoke run, 1 GPU).
#
# Submit through submit.sh:   scripts/deltaai/submit.sh slime-setup
# Steps:
#   1. check the slime image is the recorded one (SHA-256), then run its GPU check
#      (scripts/deltaai/check_slime_gpu.py, ~1-2 min; the build job runs it too, but build job
#      3620006's run had a broken reference in the check itself, so it is repeated here);
#   2. convert the pinned Qwen3.5-2B Hugging Face weights into Megatron format with slime's
#      tools/convert_hf_to_torch_dist.py, using EXACTLY the architecture flags training uses
#      (src/train_args.py --model-args, checked against the model's config.json). This copy is
#      both the starting point of training and the frozen KL reference (--ref-load);
#   3. convert it back to Hugging Face format (scripts/deltaai/convert_to_hf.py = slime's
#      tools/convert_torch_dist_to_hf.py, reading the checkpoint format Megatron ae76c60e writes) and require every
#      language-model weight to match the original (src/check_roundtrip.py); then delete that
#      temporary copy (it was created by this job and is not needed again);
#   4. build the training flags once (src/train_args.py) and run them through slime's REAL parser
#      (scripts/deltaai/check_slime_args.py, seconds): slime drops unknown Megatron flags
#      silently, so a flag it doesn't know, or a value it changes, would otherwise only show up
#      as a wrong setting in a paid training run. Needs the converted copy from step 2 (--ref-load).
#   5. the smoke run (scripts/deltaai/smoke_train.sh, ~40-60 min): 2 real training steps, checkpoint
#      saved, converted, checked and served with the control image; marker <folder>.smoke_pass.
#      Training refuses to start without that marker (user, 2026-11-19, after job 3622008).
# Output: $SLIME_CKPT_DIR/<model>_<rev>_torch_dist (about 5 GB). Never overwritten: if it exists,
# the job stops (delete it yourself to redo the conversion). On success it also writes the marker
# <that folder>.setup_pass (job, git, date): training refuses a Megatron copy without it, so a
# half-finished conversion is never trained on (setup_and_train.sbatch skips setup when it exists).
#SBATCH --job-name=gar-slime-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
# The training job's resources, because step 5 runs real training (with --colocate the optimizer
# state is moved to CPU memory while SGLang generates).
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/deltaai/env.sh
source scripts/deltaai/slime_common.sh
echo "[setup] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | job ${SLURM_JOB_ID:-local}"

# ---- 1
check_slime_sif
S python3 scripts/deltaai/check_slime_gpu.py
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
# The checkpoint folder is the one with torch_dist's .metadata file (the converter saves it as
# <REF_DIR>/release). Megatron ae76c60e writes no common.pt any more (setup job 3631056).
mapfile -t META < <(find "$REF_DIR" -name .metadata)
if [ "${#META[@]}" -ne 1 ]; then echo "[setup] expected one checkpoint (.metadata) in $REF_DIR, found ${#META[@]}"; exit 1; fi
CKPT_SUBDIR="$(dirname "${META[0]}")"
echo "[setup] Megatron checkpoint folder: $CKPT_SUBDIR"
RT_DIR="$SLIME_CKPT_DIR/_roundtrip_check_${SLURM_JOB_ID:-local}"
S env PYTHONPATH=/opt/Megatron-LM:/opt/slime python3 scripts/deltaai/convert_to_hf.py --input-dir "$CKPT_SUBDIR" \
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
echo "[setup] training flags built ($(wc -l < "runs/$RUN_NAME/slime_args_setup.txt") lines, runs/$RUN_NAME/slime_args_setup.txt)"
S env GAR_TRAIN_CONFIG="$REPO_DIR/configs/train_reinit.toml" PYTHONPATH="/opt/Megatron-LM" \
  python3 scripts/deltaai/check_slime_args.py "runs/$RUN_NAME/slime_args_setup.txt"
echo "job ${SLURM_JOB_ID:-local} | git $(git rev-parse --short HEAD) | $(date -u +%FT%TZ)" > "$REF_DIR.setup_pass"
echo "[setup] $(date -u +%FT%TZ) PASS (marker $REF_DIR.setup_pass)"

# ---- 5
bash scripts/deltaai/smoke_train.sh
