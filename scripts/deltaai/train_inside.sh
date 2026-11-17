#!/bin/bash
# scripts/deltaai/train_inside.sh: runs INSIDE the slime image; started by train_reinit.sbatch.
#
# Starts a private Ray instance for this job and runs slime's train.py with the flags file that
# src/train_args.py wrote. Private = its own temp folder and free ports: a 1-GPU job shares its
# node with other jobs, and slime connects to "whatever Ray cluster it finds" (it never calls
# ray.init itself), so a default-port Ray could belong to someone else.
# Args: <flags file> <ray temp dir> <ray port> <num gpus> <num cpus>
# Env (set by train_reinit.sbatch): GAR_TRAIN_CONFIG, GAR_ROLLOUT_LOG_DIR, REPO_DIR.
set -euo pipefail
FLAGS_FILE="$1"; RAY_TMP="$2"; RAY_PORT="$3"; NUM_GPUS="$4"; NUM_CPUS="$5"
mapfile -t FLAGS < "$FLAGS_FILE"

# Environment the Ray workers inherit (they are started by this Ray instance):
#   our plug-ins (src.reward, src.slime_hooks) and Megatron's source tree must be importable;
#   CUDA_DEVICE_MAX_CONNECTIONS=1 and NCCL_NVLS_ENABLE=0 as in slime's run scripts (one GPU, no NVLink).
export PYTHONPATH="$REPO_DIR:/opt/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0 PYTHONUNBUFFERED=1
: "${GAR_TRAIN_CONFIG:?}" "${GAR_ROLLOUT_LOG_DIR:?}"

stop_ray() { ray stop --force > /dev/null 2>&1 || true; }
trap stop_ray EXIT
ray start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" --num-gpus "$NUM_GPUS" --num-cpus "$NUM_CPUS" \
  --temp-dir "$RAY_TMP" --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"

echo "[train] slime train.py with ${#FLAGS[@]} flag tokens from $FLAGS_FILE"
cd /opt/slime
python3 train.py "${FLAGS[@]}"
