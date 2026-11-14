#!/usr/bin/env bash
# scripts/deltaai/serve_and_generate.sh: start SGLang, run src/generate.py against it, stop SGLang.
#
# Runs ON the GPU compute node, inside a Slurm job (called by smoke_test.sbatch and
# run_control.sbatch; never run it on a login node). All arguments are passed to
# src/generate.py, e.g.  --run-id pilot_v1 --pilot
#
# Steps:
#   1. print the environment (node, GPU, container, package versions) into the job log
#   2. start the SGLang server in the background, inside the container, with the model,
#      revision, dtype, host, port and seed from configs/control.toml; its log goes to
#      runs/_server_logs/
#   3. wait until GET /health answers (or fail fast if the server process dies)
#   4. run src/generate.py (it re-checks health, template hash and parity, then generates)
#   5. on ANY exit (success, error, Slurm time-limit SIGTERM): stop the server
# Exit status = generate.py's (0 ok, 3 = some generations logged errors, other = failure).
#
# Local test mode: CONTAINER_EXEC="" PYTHON=.venv/bin/python SGLANG_SERVER_MODULE=tests.fake_sglang
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_DIR"

SERVER_MODULE="${SGLANG_SERVER_MODULE:-sglang.launch_server}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # never download mid-job (setup.sh pre-fetches)

env_report
load_cfg

mkdir -p runs/_server_logs
SERVER_LOG="runs/_server_logs/sglang_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"

# The configured port is only the preferred one: another job on the same node may already use
# it, and a health check would then be answered by THEIR server. Take the next free port.
for _ in $(seq 1 100); do port_free "$PORT" && break; echo "[serve] port $PORT busy, trying $((PORT + 1))"; PORT=$((PORT + 1)); done
port_free "$PORT" || { echo "[serve] ERROR: no free port found"; exit 1; }

echo "[serve] starting $SERVER_MODULE: $MODEL @ $REVISION ($DTYPE) on $HOST:$PORT, log $SERVER_LOG"
# SERVER_EXTRA_ARGS: optional extra SGLang flags (e.g. --mem-fraction-static 0.85); unquoted
# on purpose so it splits into separate arguments. run_exec makes $! the server's own pid.
# shellcheck disable=SC2086
run_exec "$PYTHON" -m "$SERVER_MODULE" --model-path "$MODEL" --revision "$REVISION" --dtype "$DTYPE" \
    --host "$HOST" --port "$PORT" --random-seed "$SEED" ${SERVER_EXTRA_ARGS:-} \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

stop_server() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] stopping server (pid $SERVER_PID and its child processes)"
    kill_tree TERM "$SERVER_PID"
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill_tree KILL "$SERVER_PID"   # anything still alive after 30 s is forced
  fi
}
# EXIT covers normal and error exits; TERM is what Slurm sends at the time limit. Once cleanup
# starts, further signals are ignored so a second SIGTERM can't cut it short.
cleanup() { trap '' TERM INT; stop_server; }
trap cleanup EXIT
trap 'echo "[serve] received SIGTERM/SIGINT (time limit or cancel)"; exit 143' TERM INT

echo "[serve] waiting up to ${WAIT_S}s for http://$HOST:$PORT/health"
t_wait=$SECONDS
deadline=$((SECONDS + WAIT_S))
until curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] ERROR: server exited during startup. Last 40 log lines:"; tail -n 40 "$SERVER_LOG"; exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "[serve] ERROR: server not healthy after ${WAIT_S}s. Last 40 log lines:"; tail -n 40 "$SERVER_LOG"; exit 1
  fi
  sleep 5
done
kill -0 "$SERVER_PID" 2>/dev/null || { echo "[serve] ERROR: health answered but our server process is gone"; tail -n 40 "$SERVER_LOG"; exit 1; }
echo "[serve] server (pid $SERVER_PID) healthy after $((SECONDS - t_wait))s"

set +e
run "$PYTHON" -m src.generate --server-url "http://$HOST:$PORT" "$@"
status=$?
set -e
echo "[serve] generate.py exited with status $status"
exit "$status"
