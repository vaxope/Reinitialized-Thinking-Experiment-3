# Infrastructure: DeltaAI + slime

## Hardware (DeltaAI)
- Node: 4 × NVIDIA GH200 (H100 GPU, 96 GB HBM each) with **Grace ARM (aarch64) CPUs**.
- Login nodes have no GPUs; all GPU work goes through Slurm.
- Containers: DeltaAI runs Apptainer, not Docker. Check the DeltaAI container docs for the exact module and commands.

## The main blocker: slime's images are x86_64 only
slime @79aa41a, `docker/NOTES_GB10.md`: *"slime's published Docker images (`slimerl/slime:*`) are x86_64-only."*
There is an ARM port for NVIDIA DGX Spark (`docker/Dockerfile.gb10`), but it targets a Blackwell GPU (sm_121), not GH200's Hopper GPU (sm_90).

**Options, in order of preference:**
1. **Ask Ruida's group** whether they already have a working slime (or any RL stack) environment on DeltaAI. GAR was trained on Delta/DeltaAI.
2. **Adapt `Dockerfile.gb10` for Hopper**: same ARM base (`nvcr.io/nvidia/vllm`), but build kernels for sm_90 instead of sm_121. Its 15 documented blockers are a map of what to expect. Build it as an Apptainer image.
3. **Fallback framework** if (1–2) take more than a few days: verl or TRL's GRPOTrainer on an NGC PyTorch ARM container. Less scalable, but runs on GH200 more easily. The rest of this project (modes, reward, logging, gates) is framework-independent.

## Software to pin (record in every run's config)
- Container image + digest
- slime commit (`79aa41a` at setup time), SGLang version, Megatron-LM commit
- Qwen3.5-2B revision (Hugging Face commit hash), tokenizer/chat-template hash
- This repo's git hash

## Checkpoint flow
```
Qwen/Qwen3.5-2B (HF) --tools/convert_hf_to_torch_dist.py--> Megatron torch_dist  (--ref-load, initial --load)
training saves  --tools/convert_torch_dist_to_hf.py-->  HF checkpoint --> evaluation in SGLang
```
Gate 4: after converting the base model HF → torch_dist → HF, compare logits on ~5 prompts with the original. They must match closely. This verifies `configs/qwen3.5-2B.sh`.

## GPU layout (one node)
- `--colocate`: training and SGLang share all 4 GPUs, taking turns. Simplest on one node.
- 2B model: tensor parallel 1; one SGLang engine per GPU (`--rollout-num-gpus-per-engine 1`).
- Start with `--sglang-mem-fraction-static 0.6`; lower it on out-of-memory errors.

## Bring-up order (each step needs approval; see CLAUDE.md)
1. `scripts/deltaai/01_check_env.sh`, in a short interactive job: GPU, CUDA and Python visible; the container starts.
2. Inside the container: `python -c "import slime, sglang, megatron.core"`.
3. SGLang alone: load Qwen3.5-2B, generate one answer per mode with `src/modes.py` inputs (gate 3).
4. Checkpoint round trip (gate 4).
5. Zero-step baseline evaluation (gate 5).
6. Smoke training run: ~32 problems, 5 rollouts, group size 8 (gate 6). Record seconds per rollout.
7. Full Phase A run.
