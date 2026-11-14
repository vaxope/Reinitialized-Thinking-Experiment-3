#!/usr/bin/env bash
# scripts/deltaai/setup.sh: one-time preparation on a DeltaAI LOGIN node (no GPU, no job).
#
# Run once after cloning the repo on DeltaAI and filling in scripts/deltaai/env.sh:
#     bash scripts/deltaai/setup.sh
# Steps (each prints what it checks; any failure stops the script):
#   1. env.sh values set, the SGLang .sif image exists, HF cache directory created
#   2. package versions inside the container (python, sglang, torch, transformers)
#   3. download the model + tokenizer at the PINNED revision into $HF_HOME (about 5 GB), so jobs
#      can run with HF_HUB_OFFLINE=1 and never fetch anything mid-run
#   4. data/gpqa_diamond.jsonl present with 198 questions (copy it from your Mac with
#      scp/rsync: data/ is gitignored and GPQA must not go through GitHub)
#   5. the unit tests, inside the container, on the CPU (includes template parity and the
#      generate.py tests against the fake server)
# It does not submit jobs or use a GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ ! -f scripts/deltaai/env.sh ]; then
  echo "scripts/deltaai/env.sh not found: cp scripts/deltaai/env.example.sh scripts/deltaai/env.sh and fill it in"; exit 1
fi
source scripts/deltaai/env.sh
source scripts/deltaai/common.sh

echo "== 1. settings"
for v in DELTAAI_ACCOUNT DELTAAI_PARTITION REPO_DIR SGLANG_SIF HF_HOME; do
  case "${!v:-}" in ""|*"<"*) echo "  $v is not filled in (env.sh)"; exit 1 ;; *) echo "  $v=${!v}" ;; esac
done
[ "$(cd "$REPO_DIR" && pwd)" = "$(pwd)" ] || { echo "  REPO_DIR does not point at this repository"; exit 1; }
if [ "${#CONTAINER[@]}" -gt 0 ] && [ ! -f "$SGLANG_SIF" ]; then echo "  image not found: $SGLANG_SIF"; exit 1; fi
mkdir -p "$HF_HOME" runs/_slurm runs/_server_logs

echo "== 2. container"
env_report

echo "== 3. model + tokenizer at the pinned revision -> $HF_HOME"
HF_HUB_OFFLINE=0 run "$PYTHON" -c '
import tomllib
from huggingface_hub import snapshot_download
c = tomllib.load(open("configs/control.toml", "rb"))["model"]
p = snapshot_download(c["name"], revision=c["revision"])
print("  downloaded", c["name"], "@", c["revision"], "->", p)
'

echo "== 4. data"
if [ ! -f data/gpqa_diamond.jsonl ]; then
  echo "  data/gpqa_diamond.jsonl missing. From your Mac:"
  echo "    rsync -av data/gpqa_diamond.jsonl <user>@<deltaai-login>:$REPO_DIR/data/"
  exit 1
fi
n=$(grep -c . data/gpqa_diamond.jsonl); echo "  data/gpqa_diamond.jsonl: $n questions"
[ "$n" = 198 ] || { echo "  expected 198"; exit 1; }

echo "== 5. unit tests (CPU, inside the container)"
HF_HUB_OFFLINE=1 run "$PYTHON" tests/run_all.py 2>&1 | grep -v "PyTorch was not found" | tail -n 15

echo "Setup complete. Next: scripts/deltaai/submit.sh smoke"
