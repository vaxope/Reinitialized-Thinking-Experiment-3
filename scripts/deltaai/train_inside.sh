#!/bin/bash
# scripts/deltaai/train_inside.sh: runs INSIDE the slime image; started by train_reinit.sbatch.
#
# Starts a private Ray instance for this job and runs slime's train.py with the flags file that
# src/train_args.py wrote. Private = its own temp folder and free ports: a 1-GPU job shares its
# node with other jobs, and slime connects to "whatever Ray cluster it finds" (it never calls
# ray.init itself), so a default-port Ray could belong to someone else.
# Args: <flags file> <ray temp dir> <ray port> <num gpus> <num cpus> [<time budget s, 0 = none>]
# Time budget: train.py gets SIGTERM after that many seconds (coreutils `timeout`, sent to its
# whole process group), SIGKILL 120 s later if still alive; then Ray is stopped as usual. Exit
# code 124 (or 137) then means "budget reached", not a crash. Checkpoints already saved are kept.
# Env (set by train_reinit.sbatch): GAR_TRAIN_CONFIG, GAR_ROLLOUT_LOG_DIR, REPO_DIR.
# Loopback only (smoke job 3633088): on Linux, `ray start` replaces --node-ip-address 127.0.0.1 with
# the node's network address (ray/scripts/scripts.py: resolve_ip_for_localhost -> services.
# get_node_ip_address -> node_ip_address_from_perspective, because ray_constants.ENABLE_RAY_CLUSTER
# defaults to True on Linux). On the DeltaAI node, connections to that address (141.142.249.84)
# timed out for 13 min and the GCS never answered. RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0 turns
# ENABLE_RAY_CLUSTER off, so get_node_ip_address() returns get_localhost_ip() and everything stays
# on 127.0.0.1 inside this one node (all a 1-GPU job needs). It must be set for `ray start` AND
# train.py (the driver resolves RAY_ADDRESS=127.0.0.1:<port> the same way), hence exported here.
# On failure, Ray's own logs (in the node's /tmp, wiped when the job ends) are copied next to the
# rollout log: <run dir>/ray_logs_<pid>/.
set -euo pipefail
FLAGS_FILE="$1"; RAY_TMP="$2"; RAY_PORT="$3"; NUM_GPUS="$4"; NUM_CPUS="$5"; BUDGET_S="${6:-0}"
mapfile -t FLAGS < "$FLAGS_FILE"

# Environment the Ray workers inherit (they are started by this Ray instance):
#   our plug-ins (src.reward, src.slime_hooks) and Megatron's source tree must be importable;
#   CUDA_DEVICE_MAX_CONNECTIONS=1 and NCCL_NVLS_ENABLE=0 as in slime's run scripts (one GPU, no NVLink).
export PYTHONPATH="$REPO_DIR:/opt/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0 PYTHONUNBUFFERED=1
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0   # keep Ray on 127.0.0.1 (see the header)
: "${GAR_TRAIN_CONFIG:?}" "${GAR_ROLLOUT_LOG_DIR:?}"

# Ray's own logs are in the node's /tmp, which is wiped when the job ends: on failure, keep the
# .out/.err files (last 2,000 lines each) next to the rollout log.
keep_ray_logs() {
  local src="$RAY_TMP/session_latest/logs" dst
  dst="$(dirname "$GAR_ROLLOUT_LOG_DIR")/ray_logs_$$"
  [ -d "$src" ] || return 0
  mkdir -p "$dst"
  for f in "$src"/*.out "$src"/*.err; do [ -f "$f" ] && tail -n 2000 "$f" > "$dst/$(basename "$f")"; done
  echo "[train] Ray logs copied to $dst"
}
on_exit() {
  local status=$?
  if [ "$status" -ne 0 ]; then keep_ray_logs || true; fi
  ray stop --force > /dev/null 2>&1 || true
}
trap on_exit EXIT
# `ray start` retries the GCS for a long time when it can't reach it (13 min in job 3633088);
# 300 s is far more than a healthy start needs (seconds), so a broken start fails fast.
RAY_START_TIMEOUT_S=300
timeout "$RAY_START_TIMEOUT_S" ray start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" \
  --num-gpus "$NUM_GPUS" --num-cpus "$NUM_CPUS" --temp-dir "$RAY_TMP" --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
# proof the cluster is up, and on which address (information only: never stops the job)
ray status --address "$RAY_ADDRESS" 2>&1 | head -n 20 || true

echo "[train] slime train.py with ${#FLAGS[@]} flag tokens from $FLAGS_FILE"
cd /opt/slime
if [ "$BUDGET_S" -gt 0 ]; then
  echo "[train] time budget: $BUDGET_S s ($((BUDGET_S / 60)) min), then training stops"
  timeout --signal=TERM --kill-after=120 "$BUDGET_S" python3 train.py "${FLAGS[@]}"
else
  python3 train.py "${FLAGS[@]}"
fi
