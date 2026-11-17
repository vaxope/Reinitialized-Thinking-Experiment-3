#!/bin/bash
# scripts/deltaai/build_slime.sbatch: build the slime training image on a DeltaAI compute node.
#
# Submit through submit.sh (it adds --account/--partition/--time and asks first):
#     scripts/deltaai/submit.sh build-slime
# Steps (each stops the job on failure, so a problem costs minutes, not the full build):
#   1. read every pin from configs/slime_container.toml (via src/slime_container.py, run with the
#      SGLang control image's Python, which has tomllib);
#   2. a seconds-long test build (configs/slime_container.toml [build] fakeroot_test_image) to
#      check DeltaAI allows unprivileged builds, BEFORE the 15 GB base image is downloaded;
#   3. the real build: containers/slime_gh200.def -> <name>.sif.partial, renamed only on success;
#   4. SHA-256 of the image (compared with [output] sif_sha256 once recorded);
#   5. GPU check inside the new image (scripts/deltaai/check_slime_gpu.py, about 1-2 minutes).
# Why a compute node: login nodes throttle CPU (the SGLang image pull stalled there for hours);
# DeltaAI has only GPU nodes, so the job holds 1 GPU (the billing unit) while it compiles.
# Never overwrites an existing image: delete it yourself first if a rebuild is wanted.
#SBATCH --job-name=gar-build-slime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
# 64 CPUs for parallel compiles (MAX_JOBS = 32 compile processes, configs/slime_container.toml),
# 240 GB RAM: CUDA compiles take several GB each (the SGLang image build peaked at 67.5 GB).
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=runs/_slurm/%x_%j.out
#SBATCH --constraint=projects
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/deltaai/env.sh
for v in REPO_DIR SGLANG_SIF SLIME_SIF_DIR APPTAINER_CACHEDIR APPTAINER_TMPDIR; do
  if [ -z "${!v:-}" ]; then echo "[build] $v is not set in scripts/deltaai/env.sh"; exit 1; fi
done
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
mkdir -p "$SLIME_SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

cfg() { apptainer exec --bind "$REPO_DIR" "$SGLANG_SIF" python3 -m src.slime_container "$@"; }
SIF="$SLIME_SIF_DIR/$(cfg --sif-name)"
RECORDED_SHA="$(cfg --sif-sha256)"
TEST_IMAGE="$(cfg --fakeroot-test-image)"
mapfile -t KV < <(cfg --build-args)
BUILD_ARGS=()
for kv in "${KV[@]}"; do BUILD_ARGS+=(--build-arg "$kv"); done

echo "[build] $(date -u +%FT%TZ) host $(hostname) | git $(git rev-parse --short HEAD) | $(apptainer --version) | cpus $(nproc)"
echo "[build] image: $SIF"
printf '[build] arg %s\n' "${KV[@]}"
if [ -e "$SIF" ]; then echo "[build] $SIF already exists; not overwriting. Delete it first to rebuild."; exit 1; fi

# ---- step 2: can we build at all? (seconds)
T0=$SECONDS
TEST_DIR="$(mktemp -d "$APPTAINER_TMPDIR/fakeroot_test.XXXX")"
printf 'Bootstrap: docker\nFrom: %s\n%%post\n    touch /fakeroot_ok\n' "$TEST_IMAGE" > "$TEST_DIR/test.def"
if ! apptainer build --fakeroot "$TEST_DIR/test.sif" "$TEST_DIR/test.def"; then
  echo "[build] FAIL: unprivileged (fakeroot) builds are not available on this node. Stopping before"
  echo "        the large download. Fallback (needs a new plan): pinned base image + install folder."
  rm -rf "$TEST_DIR"; exit 3
fi
rm -rf "$TEST_DIR"
echo "[build] fakeroot test build OK ($((SECONDS - T0)) s)"

# ---- step 3: the image (about 1-1.5 h, mostly TransformerEngine)
T0=$SECONDS
apptainer build --fakeroot "${BUILD_ARGS[@]}" "$SIF.partial" containers/slime_gh200.def
mv "$SIF.partial" "$SIF"
echo "[build] image built in $(( (SECONDS - T0) / 60 )) min: $(du -h "$SIF" | cut -f1)"

# ---- step 4: hash
SHA="$(sha256sum "$SIF" | cut -d' ' -f1)"
if [ -z "$RECORDED_SHA" ]; then
  echo "[build] sif sha256: $SHA  (not recorded yet: copy it to configs/slime_container.toml [output] sif_sha256)"
elif [ "$SHA" = "$RECORDED_SHA" ]; then
  echo "[build] sif sha256 $SHA = recorded"
else
  echo "[build] sif sha256 $SHA != recorded $RECORDED_SHA (a rebuild is never byte-identical; record the new one)"
fi
apptainer exec "$SIF" cat /opt/gar/build_info.json

# ---- step 5: GPU check in the new image
apptainer exec --nv --env CC=gcc,CXX=g++ --bind "$REPO_DIR" "$SIF" python3 scripts/deltaai/check_slime_gpu.py
echo "[build] $(date -u +%FT%TZ) done"
