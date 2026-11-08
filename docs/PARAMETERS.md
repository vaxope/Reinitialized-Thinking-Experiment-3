# Training parameters: GAR official vs this project

Source of GAR values: `RickySkywalker/GAR-Official` @ `aa04bed`. Files: `main.py`, `GAR_trainer.py` (`GAR_TrainerMetaConfig`, `GAR_TrainerForwardConfig`, `GRPO_Training_hyperparams`), `RL_trainner.py` (`GRPO_loss`, `_calculate_advantage`, optimizer, scheduler).
In GAR, the prover is the model that solves problems, so its settings are the reference for our solver.

## Core parameters

| Parameter | What it does | GAR official | Phase A (replicate) | Phase B (candidate changes) |
|---|---|---|---|---|
| **Group size** | Answers sampled per problem; GRPO compares them | 16 proofs per theorem (`proofs_per_theorem`). `GrpoConfig.group_size = 8` exists but isn't what training uses | 16 | 8 if compute is tight |
| **Learning rate** ("step size") | How far each update moves the weights | 5e-6, AdamW, constant, no warmup | 1e-6, constant | sweep {5e-7, 1e-6, 2e-6} |
| **Batch size** | Samples per optimizer step | global 32 = micro 1 × grad-accum 8 × 4 GPUs | 256 (16 prompts × 16), 1 step per rollout | 64, 4 steps per rollout |
| **Updates per generation** | How stale the data is when used | GAR generates once per round, then runs **one epoch of many 32-sample updates** over it: increasingly off-policy, guarded only by clipping | 1 (fully on-policy) | 2–4 steps per rollout |
| **Training length** | Total amount of training | 5 GAR rounds in `main.py` (16 in the dataclass default) × 1,024 fused problems, filtered by pass rate | 200 rollouts × 16 prompts = 3,200 prompts | set from the held-out curve |
| **Training token length** | Max tokens per sample | generation cap 16,385 (prover); training sequence cap 16,384 | response 16,384; prompt 2,048 | shrink once the truncation rate is known |
| **Sampling** | Randomness of rollouts | prover temp 1.0, top-p 0.95 | temp 1.0, top-p 1.0 | top-p 0.95 with slime's top-p masking |
| **Clip ε** | Max change in each token's probability per update | 0.2 / 0.2 | 0.2 / 0.2 | 0.2 / 0.28 (Clip-Higher) |
| **KL penalty** | Pull toward the starting model | β = 0.01, k3 estimator | 0.01, k3 | 0 (DAPO-style) |
| **Advantage** | How rewards become per-answer credit | (r − mean) / std within each theorem's group (prover); one batch-wide mean/std for the fuser | GRPO with std | `--disable-grpo-std-normalization` (Dr. GRPO) |
| **Loss reduction** | How token losses are averaged | per-sequence mean, then batch mean (favours long wrong answers; see Dr. GRPO) | same | `--calculate-per-token-loss` |
| **Data filter** | Which groups train | keep 0 < pass rate ≤ 0.5 (prover); fuser ≤ 0.4 | keep non-zero-std groups (0 < p < 1) | add GAR's p ≤ 0.5 cut |
| **Data order** | Order of training samples | **sorted by output length, shortest first** (`use_data_sorting=True`) | shuffled | – |
| **Weight decay** | Shrinks weights each step | 1e-8 | 1e-8 | – |
| **Gradient clipping** | Caps the update size | none found | 1.0 (logged) | – |
| **Precision** | Number format | bf16 | bf16 | – |
| **Reward** | Score per answer | Lean pass/fail; statement modification = fail | strict "Answer: X" match, 1/0 | + soft overlong penalty (correct answers only) |

## GAR details worth knowing

- **Reinitialized thinking in GAR's code** is `"\n<think>\n\n</think>\n"` appended by hand (`utils/PromptWriter.py`, `skip_thinking=True`). Qwen3.5's template emits `<think>\n\n</think>\n\n`. Different whitespace means different tokens; we use the template's version.
- **`skip_thinking` has conflicting defaults**: `GAR_TrainerForwardConfig.NL_fuser_skip_thinking = False`, while `StatementFuser.py` and `GAR_NodeWorker.py` set `True`. Check which path a GAR run actually takes before quoting "GAR skips thinking".
- **The fuser** (Qwen3-8B) samples at temperature 0.8, top-p 0.95, max 16,384 (or 8,192 in some paths).
- **The KL code is labelled "DEBUG: try the reverse KL"**, but `exp(ref − θ) − (ref − θ) − 1` is the standard k3 estimator used by GRPO.

## Why Phase A first

Change **one thing at a time**. Phase A reproduces GAR's choices as closely as slime allows, so any result can be compared with GAR's. The intentional deviations (lr 1e-6, on-policy updates, top-p 1.0, shuffled data, gradient clipping) are each listed above with a reason. Phase B changes are ablations, one per run.

## Compute estimate (to confirm with the smoke test)
Per rollout: 256 samples. At a few thousand tokens per GAR-mode answer, that's roughly 0.5–1M generated tokens, plus one training step. On one DeltaAI node (4 × H100) the rough guess is a few minutes per rollout, so 200 rollouts ≈ 10–40 GPU-hours. **Measure the real time per rollout in the smoke test before requesting the full run.**
