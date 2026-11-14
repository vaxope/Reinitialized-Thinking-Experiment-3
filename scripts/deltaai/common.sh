# scripts/deltaai/common.sh: shared helpers, sourced (not run) by the other DeltaAI scripts.
#
# Provides:
#   run <cmd...>   run a command inside the SGLang Apptainer container (or directly, for local tests)
#   PYTHON         the python inside the container ("python3")
#   load_cfg       sets MODEL, REVISION, DTYPE, HOST, PORT, SEED, WAIT_S from configs/control.toml
#   env_report     prints node, GPU, container and package versions into the job log
#
# Local testing (no Apptainer/GPU): set CONTAINER_EXEC="" so commands run directly, and point
# PYTHON at the .venv python (example in the header of serve_and_generate.sh).

: "${REPO_DIR:?REPO_DIR is not set; create scripts/deltaai/env.sh from env.example.sh}"
PYTHON="${PYTHON:-python3}"

# Container command as an array. CONTAINER_EXEC unset -> Apptainer with the GPU (--nv) and the
# repo + HF cache bound inside; CONTAINER_EXEC="" -> no container (local tests).
if [ -z "${CONTAINER_EXEC+x}" ]; then
  : "${SGLANG_SIF:?SGLANG_SIF is not set (path to the SGLang .sif image)}"
  : "${HF_HOME:?HF_HOME is not set (Hugging Face cache directory)}"
  CONTAINER=(apptainer exec --nv --bind "$REPO_DIR" --bind "$HF_HOME" "$SGLANG_SIF")
else
  read -r -a CONTAINER <<< "$CONTAINER_EXEC"
fi

run() {
  # An empty array can't be expanded under `set -u` in old bash, hence the branch.
  if [ "${#CONTAINER[@]}" -gt 0 ]; then "${CONTAINER[@]}" "$@"; else "$@"; fi
}

run_exec() {
  # Like run, but REPLACES the current (sub)shell with the command. Used for the background
  # server, so that $! is the server process itself and not a wrapper shell around it.
  if [ "${#CONTAINER[@]}" -gt 0 ]; then exec "${CONTAINER[@]}" "$@"; else exec "$@"; fi
}

kill_tree() {
  # Signal a process and all its descendants (SGLang starts worker processes of its own).
  # Children are collected first, then everyone is signalled, so no orphan escapes.
  local sig="$1" pid="$2" kids
  kids=$(pgrep -P "$pid" 2>/dev/null || true)
  for k in $kids; do kill_tree "$sig" "$k"; done
  kill "-$sig" "$pid" 2>/dev/null || true
}

port_free() {
  # True if nothing answers on HOST:$1 (a 1-GPU job shares its node with other jobs).
  ! (exec 3<>"/dev/tcp/$HOST/$1") 2>/dev/null
}

load_cfg() {
  # One python call reads every value the shell needs from the config (single source of truth).
  eval "$(run "$PYTHON" -c '
import shlex, tomllib
c = tomllib.load(open("configs/control.toml", "rb"))
for k, v in {"MODEL": c["model"]["name"], "REVISION": c["model"]["revision"],
             "DTYPE": c["model"]["dtype"], "HOST": c["server"]["host"],
             "PORT": c["server"]["port"], "SEED": c["server"]["random_seed"],
             "WAIT_S": c["generation"]["server_wait_timeout_s"]}.items():
    print(f"{k}={shlex.quote(str(v))}")
')"
}

env_report() {
  echo "===== environment ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ====="
  echo "host: $(hostname)   arch: $(uname -m)   slurm job: ${SLURM_JOB_ID:-none}   git: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi: not available"
  if [ "${#CONTAINER[@]}" -gt 0 ]; then
    echo "apptainer: $(apptainer --version 2>/dev/null || echo 'not found')   image: $SGLANG_SIF"
  else
    echo "container: none (local test mode)"
  fi
  run "$PYTHON" -c '
import importlib, sys
print("python", sys.version.split()[0])
for m in ("sglang", "torch", "transformers"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "NOT AVAILABLE:", type(e).__name__)
try:
    import torch; print("cuda available:", torch.cuda.is_available(), "| cuda", torch.version.cuda)
except Exception:
    pass
' 2>&1 | grep -v "PyTorch was not found"
  echo "============================================"
}
