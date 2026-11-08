#!/bin/bash
# Bring-up step 1: confirm what a DeltaAI GPU node provides. Read-only; writes one report file.
# Run INSIDE a short interactive allocation (ask before allocating), e.g.:
#   salloc --account=<ACCOUNT> --partition=<GPU_INTERACTIVE_PARTITION> --nodes=1 --gpus-per-node=4 --time=00:30:00
#   srun --pty bash
#   bash scripts/deltaai/01_check_env.sh
# Look up <ACCOUNT> and the partition names in DeltaAI's documentation; they are not filled in here.
set -u
OUT="runs/env_report_$(hostname)_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p runs
{
  echo "== host / arch";        hostname; uname -m; cat /etc/os-release 2>/dev/null | head -2
  echo "== GPUs";               nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
  echo "== GPU topology";       nvidia-smi topo -m 2>/dev/null | head -20
  echo "== CUDA";               (nvcc --version 2>/dev/null | tail -2) || echo "nvcc not on PATH"
  echo "== modules";            (module avail 2>&1 | grep -iE "cuda|apptainer|singularity|python|nccl" | head -30) || true
  echo "== container runtime";  (command -v apptainer && apptainer --version) || echo "apptainer not on PATH"
  echo "== python";             (command -v python3 && python3 --version) || true
  echo "== disk (home/project/scratch)"; df -h ~ 2>/dev/null | tail -1
  echo "== slurm job";          env | grep -E "^SLURM_(JOB_ID|JOB_PARTITION|GPUS|NODELIST)" || true
} | tee "$OUT"
echo "Report written to $OUT. Expected: aarch64, 4 GPUs, compute_cap 9.0 (Hopper)."
