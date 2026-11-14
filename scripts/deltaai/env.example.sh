# scripts/deltaai/env.example.sh: per-user DeltaAI settings (TEMPLATE).
#
# Copy to scripts/deltaai/env.sh (gitignored) on DeltaAI and fill in the values:
#     cp scripts/deltaai/env.example.sh scripts/deltaai/env.sh && nano scripts/deltaai/env.sh
# Sourced by submit.sh, setup.sh and the .sbatch job scripts. Nothing experimental lives
# here (that is configs/control.toml); only where things are and who pays.

# Slurm account that GPU-hours are charged to. On DeltaAI, `accounts` lists yours.
export DELTAAI_ACCOUNT="<your-account>"
# GPU partition for batch jobs, and the interactive one for short tests. Check the names
# with `sinfo -s` and the DeltaAI user guide (the GH200 partitions are expected to be
# named like "ghx4" / "ghx4-interactive"; verify before use).
export DELTAAI_PARTITION="<gpu-partition>"
export DELTAAI_PARTITION_INTERACTIVE="<interactive-gpu-partition>"

# Absolute path of this repository on DeltaAI (project storage, not $HOME: it has more space).
export REPO_DIR="/projects/<your-project>/<you>/Reinitialized-Thinking-Experiment-3"

# The SGLang Apptainer image (.sif), built for ARM (aarch64) + CUDA. Which image is still to
# be decided (advisor group's environment, an official SGLang ARM image, or SGLang installed
# into an NGC PyTorch ARM container). Record its digest in the vault when chosen.
export SGLANG_SIF="/projects/<your-project>/containers/<sglang-image>.sif"

# Hugging Face cache for the model weights + tokenizer (about 5 GB). setup.sh downloads the
# pinned revision here once; jobs then run with HF_HUB_OFFLINE=1 so nothing is fetched mid-run.
export HF_HOME="/projects/<your-project>/<you>/hf_cache"

# Optional: Slurm time limits (HH:MM:SS) used by submit.sh. Set FULL_TIME from the pilot.
export SMOKE_TIME="00:30:00"
export PILOT_TIME="02:00:00"
export FULL_TIME="24:00:00"
