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
