# CLAUDE.md: GAR reinitialized-thinking rebuild (Qwen3.5-2B, slime, DeltaAI)

## Project overview

A from-scratch rebuild of the experiment. **No earlier runs are reused**; their results had clear errors.

**Goal:** train **Qwen3.5-2B** with RL, using GAR's **reinitialized thinking** (an empty native `<think>` block, then reasoning inside `<analysis>` with numbered steps), on **multiple-choice science data similar to GPQA**. Evaluate on **GPQA Diamond** against the untrained model in every reasoning mode. Everything must be transparent: exact inputs, every output token, every parameter, every error.

**References, pinned:**
- GAR official code: `RickySkywalker/GAR-Official` @ `aa04bed` (training-parameter reference; see `docs/PARAMETERS.md`)
- slime: `THUDM/slime` @ `79aa41a` (training framework)

---

## Working rules for Claude (read first)

### 1. Learning mode: the user must understand every component
- **Before writing code, explain the plan**: what the component does, how it fits the pipeline (`docs/PIPELINE.md`), its inputs and outputs, and what could go wrong. Wait for approval.
- **One component at a time.** Each ships with: comments on non-obvious lines, unit tests the user runs, and a vault note (what it does, inputs/outputs, failure modes).
- **No magic numbers.** Every parameter lives in `configs/`, commented with what it does, its value, its source (GAR / slime default / chosen) and why. The `PARAMETERS.md` table must stay in sync.
- On request, give a line-by-line walkthrough, or quiz the user on the component.

### 2. Consult before acting
**Always stop and ask before:**
- `git commit`, `git push`, opening PRs, merging, rebasing, or anything forced;
- deleting or overwriting data, checkpoints, rollouts, logs or results;
- **any DeltaAI job** (`sbatch`, `salloc`, `srun`) or long local run. State the estimated GPU-hours and allocation charge;
- installing packages, pulling or building containers, changing the environment;
- changing prompts, chat-template settings, sampling, answer extraction, rewards or metrics after a run has started.

When asking, give: **what**, **why**, **files affected**, **cost/risk**, **how to undo**. Reading files, unit tests, CPU-only checks and ≤ 20-question local pilots need no approval.

### 3. Transparency
- **Never let a server apply the chat template silently.** Build each mode's exact input text with `src/modes.py`, and log both the text and its token IDs.
- Log **every** generation (see Logging) and save slime rollouts with `--save-debug-rollout-data`.
- Errors and exceptions are recorded in their own field and never dropped. A crashed or truncated sample is counted, not skipped.
- Report only numbers from real logs. Mark anything unverified.

### 4. Data rules
- `data/` and `runs/` are gitignored. **GPQA is test-only**: never train on it, never publish its text (notes and shared pages use question IDs only).
- Training data is decontaminated against GPQA (embedding similarity + n-gram overlap); log what was removed.

---

## Reasoning modes (exact definitions)

Qwen3.5's chat template **turns thinking OFF by default**: thinking is on only if `enable_thinking` is explicitly `true`. Every run must set it explicitly. Exact strings are in `docs/MODES.md`; `src/modes.py` builds them; `tests/test_modes.py` checks them.

| Mode | Assistant turn starts with | Instruction | Model may write |
|---|---|---|---|
| `think` | `<think>\n` (open) | standard question prompt | hidden reasoning, `</think>`, answer |
| `nothink` | `<think>\n\n</think>\n\n` | standard question prompt | visible reasoning, then answer |
| `direct` | `<think>\n\n</think>\n\n` | "Respond with only the final answer line." + generation capped at 16 tokens | answer line only |
| `gar` | `<think>\n\n</think>\n\n` | standard prompt + the `<analysis>` steps | `<analysis>…</analysis>`, then answer |

All modes end with the same required final line: `Answer: X`.

---

## Pipeline (see `docs/PIPELINE.md`)
data prep → `prompts.jsonl` → slime (SGLang rollouts → custom reward `src/reward.py` → Megatron GRPO update → weight sync) → checkpoints → convert to Hugging Face format → **evaluation in the same engine (SGLang) with the same templates and extraction** → trajectory analysis.

## Verification gates (do not skip; record each result in the vault)
1. **Data:** print 20 formatted prompts exactly as the model sees them; all answers parse.
2. **Reward:** `tests/test_reward.py` passes (including truncated, malformed and decoy-letter cases).
3. **Template parity:** `src/modes.py` output == `tokenizer.apply_chat_template(...)` for every mode; token IDs identical between the training and evaluation code.
4. **Model config:** the HF → Megatron → HF round trip gives near-identical logits on a few prompts (needed because `configs/qwen3.5-2B.sh` was written by us).
5. **Zero-step baseline:** the untrained model evaluated through the new harness, all four modes, with outputs inspected by hand (lengths, finish reasons, answer lines).
6. **Overfit smoke test:** ~32 problems, a few rollouts: reward must rise.
7. **Training metrics:** reward, response length, truncation rate, entropy, KL, clip fraction. Investigate any collapse in length or entropy.

## Logging (one JSONL record per generation)
`run_id, mode, model, checkpoint, qid, sample_idx, seed, chat_template_hash, input_text, input_token_ids, output_text, output_token_ids, output_logprobs, finish_reason, n_input_tokens, n_output_tokens, n_think_tokens, extracted_answer, extraction_rule, gold, correct, sampling_params, engine_version, wall_time_s, error`

## Documentation (Obsidian vault)
Notes go in `VAULT/Research/GAR-rebuild/` (replace `VAULT` with the real path; add via `--add-dir`). Never edit outside that folder; never touch `.obsidian/`. Folders: `components/`, `runs/`, `decisions/`, `changes/`, plus `index.md`. Frontmatter: `type, date, status, mode, where (local|deltaai), job, git, related`. Body: Summary · What · Why · Results (real numbers only) · Risks & undo (`> [!warning]`) · Next steps.

## Code conventions
- New behavior goes behind a flag or a new mode name; existing modes never change meaning.
- Small, readable modules, with tests next to every function that touches prompts, rewards or metrics.
- Pin versions (container digest, slime commit, SGLang version) in every run's config dump.
