# scripts/deltaai/slime_common.sh: shared helpers for the slime jobs (sourced, not run).
#
# Used by slime_setup.sbatch and train_reinit.sbatch. Provides:
#   S <cmd...>          run a command inside the slime image (GPU, repo, HF cache, checkpoints bound)
#   slime_paths         sets SLIME_SIF, HF_SNAPSHOT (pinned model revision), REF_DIR (Megatron copy
#                       of the start model), VOCAB_SIZE, RUN_NAME, from the configs
#   check_slime_sif     refuses an image whose SHA-256 differs from configs/slime_container.toml
#   free_port           prints an unused TCP port (1-GPU jobs share nodes with other jobs)
# Every value comes from configs/train_reinit.toml or configs/slime_container.toml, read with the
# image's own Python (it has tomllib), so nothing is duplicated here.

for v in REPO_DIR HF_HOME SLIME_SIF_DIR SLIME_CKPT_DIR; do
  if [ -z "${!v:-}" ]; then echo "[slime] $v is not set in scripts/deltaai/env.sh"; exit 1; fi
done
SLIME_SIF="$SLIME_SIF_DIR/$(apptainer exec --bind "$REPO_DIR" "$SGLANG_SIF" python3 -m src.slime_container --sif-name)"

S() {
  # --nv: the GPU; CC/CXX: DeltaAI's Cray settings would break kernel builds (smoke job 3611226);
  # HF_HUB_OFFLINE: nothing is downloaded during a job (the model is in the HF cache already).
  apptainer exec --nv --env "CC=gcc,CXX=g++,HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1" \
    --bind "$REPO_DIR" --bind "$HF_HOME" --bind "$SLIME_CKPT_DIR" "$SLIME_SIF" "$@"
}

cfg_value() {  # cfg_value <toml file> <section> <key>
  apptainer exec --bind "$REPO_DIR" "$SLIME_SIF" python3 -c \
    "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))[sys.argv[2]][sys.argv[3]])" "$@"
}

slime_paths() {
  local repo rev
  repo="$(cfg_value configs/train_reinit.toml model hf_repo)"
  rev="$(cfg_value configs/train_reinit.toml model revision)"
  HF_SNAPSHOT="$HF_HOME/hub/models--${repo//\//--}/snapshots/$rev"
  REF_DIR="$SLIME_CKPT_DIR/$(basename "$repo")_${rev:0:8}_torch_dist"
  VOCAB_SIZE="$(cfg_value configs/train_reinit.toml megatron_model vocab_size)"
  RUN_NAME="$(cfg_value configs/train_reinit.toml run name)"
  if [ ! -f "$HF_SNAPSHOT/config.json" ]; then
    echo "[slime] model not in the HF cache: $HF_SNAPSHOT (setup.sh downloads it)"; exit 1
  fi
}

check_slime_sif() {
  local want got
  if [ ! -f "$SLIME_SIF" ]; then echo "[slime] image not found: $SLIME_SIF (submit.sh build-slime)"; exit 1; fi
  want="$(apptainer exec --bind "$REPO_DIR" "$SGLANG_SIF" python3 -m src.slime_container --sif-sha256)"
  if [ -z "$want" ]; then
    echo "[slime] configs/slime_container.toml [output] sif_sha256 is empty: record the build's hash first"; exit 1
  fi
  got="$(sha256sum "$SLIME_SIF" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then echo "[slime] image hash $got != recorded $want"; exit 1; fi
  echo "[slime] image $SLIME_SIF sha256 = recorded"
}

free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}
