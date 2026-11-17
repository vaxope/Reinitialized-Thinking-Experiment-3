# scripts/deltaai/env.example.sh: per-user DeltaAI settings (TEMPLATE).
#
# Copy to scripts/deltaai/env.sh (gitignored) on DeltaAI and fill in the values:
#     cp scripts/deltaai/env.example.sh scripts/deltaai/env.sh && nano scripts/deltaai/env.sh
# Sourced by submit.sh, setup.sh and the .sbatch job scripts. Nothing experimental lives
# here (that is configs/control.toml); only where things are and who pays.

# Slurm account that GPU-hours are charged to. On DeltaAI, `accounts` lists yours.
export DELTAAI_ACCOUNT="<your-account>"
# GPU partitions (`sinfo -s`, checked 2026-11-14). ghx4: batch, up to 48 h, charge factor 1.0.
# ghx4-interactive: up to 2 h, charge factor 2.0 (twice the price): not used by submit.sh.
export DELTAAI_PARTITION="ghx4"
export DELTAAI_PARTITION_INTERACTIVE="ghx4-interactive"

# Absolute path of this repository on DeltaAI (project storage, not $HOME: it has more space).
export REPO_DIR="/projects/<your-project>/<you>/Reinitialized-Thinking-Experiment-3"

# The SGLang Apptainer image (.sif), built for ARM (aarch64) + CUDA. The image is pinned in
# configs/control.toml [container] (lmsysorg/sglang:v0.5.23-runtime); build the .sif with
#     apptainer pull "$SGLANG_SIF" docker://lmsysorg/sglang:v0.5.23-runtime
# (inside tmux: it takes hours). setup.sh checks the SGLang version inside it.
export SGLANG_SIF="/projects/<your-project>/<you>/containers/sglang-v0.5.23-runtime.sif"

# Folder for the slime training image built by `submit.sh build-slime` (about 20 GB). The file
# name and every version inside it come from configs/slime_container.toml.
export SLIME_SIF_DIR="/projects/<your-project>/<you>/containers"
# Megatron checkpoints: the converted start model (~5 GB) and each training run's checkpoints
# (~30 GB per saved step with optimizer state). Project storage, not $HOME.
export SLIME_CKPT_DIR="/projects/<your-project>/<you>/slime_checkpoints"

# Apptainer's download cache and its unpacking scratch space. Both default to places that
# are small on a login node ($HOME/.apptainer and /tmp); a 14 GB image needs project storage.
export APPTAINER_CACHEDIR="/projects/<your-project>/<you>/apptainer_cache"
export APPTAINER_TMPDIR="/projects/<your-project>/<you>/apptainer_tmp"

# Hugging Face cache for the model weights + tokenizer (about 5 GB). setup.sh downloads the
# pinned revision here once; jobs then run with HF_HUB_OFFLINE=1 so nothing is fetched mid-run.
export HF_HOME="/projects/<your-project>/<you>/hf_cache"

# Slurm time limits (HH:MM:SS) used by submit.sh. A limit is a cap, not a charge: you pay for
# time used. Tight caps limit the damage of a run that misbehaves. Set FULL_TIME from the pilot.
export SMOKE_TIME="00:20:00"
export PILOT_TIME="01:30:00"
export FULL_TIME="06:00:00"
# The slime image build. Job 3616837 (failed at step 7) took 1 h 38 min to get there; the
# rest (steps 7-12, writing the ~20-30 GB image, hash, GPU check) adds roughly 30-50 min, so
# 2.5 h is too tight: a time-out while the image file is written loses the whole build.
export BUILD_SLIME_TIME="03:30:00"
# One-time weight conversion + check: about 20-30 min.
export SLIME_SETUP_TIME="00:45:00"
# Training: set from the pilot's measured step time (the 25-step pilot: about 2-3 h projected).
export TRAIN_TIME="04:00:00"
