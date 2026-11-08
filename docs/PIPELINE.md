# Pipeline: components and how they connect

```
[1] data prep (src/data_prep.py, TODO)
    GPQA-like MC source -> decontaminate vs GPQA -> shuffle options (src/modes.shuffle_options)
    -> train_{mode}.jsonl / heldout_{mode}.jsonl  {messages, label, metadata}
            |
            v
[2] slime training (scripts/deltaai/run_train.sh, TODO)
    SGLang rollout: 16 answers per prompt, chat template with explicit enable_thinking
      -> [3] reward (src/reward.custom_rm): strict last "Answer: X" line
      -> dynamic filter: drop groups with identical rewards
      -> Megatron: GRPO advantages, clipped policy loss + KL, one update
      -> new weights synced to SGLang
    every rollout: trajectories saved (--save-debug-rollout-data)
    every 10 rollouts: held-out eval; every 20: checkpoint
            |
            v
[4] convert checkpoint torch_dist -> HF (slime tools)
            |
            v
[5] evaluation (src/evaluate.py, TODO)
    SGLang, same templates (src/modes.build_text), same extraction (src/reward.extract_answer)
    GPQA Diamond + held-out, all 4 modes, base and trained models
    -> runs/<run>/eval.jsonl (full logging schema in CLAUDE.md)
            |
            v
[6] trajectory analysis (src/analyze.py + local viewer, TODO)
    per question, side by side across modes: answer commits and switches, where the answer settles,
    tokens after settling, loops, finish reason, errors
```

## Component status
| # | Component | Status | Gate |
|---|---|---|---|
| – | `src/modes.py` | written, unit-tested; tokenizer check pending | 3 |
| 3 | `src/reward.py` | written, unit-tested | 2 |
| 1 | data prep | next | 1 |
| 5 | evaluation harness | after data prep | 3, 5 |
| 2 | slime launch script | after the environment works on DeltaAI | 4, 6 |
| 6 | analysis + viewer | after the first evaluation | – |
