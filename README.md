# GAR reinitialized-thinking rebuild: control run

The baseline for later experiments: untrained **Qwen3.5-2B** answering **GPQA Diamond** (198 multiple-choice questions) in three modes, with every input, output token and error logged.

| Mode | Prompt | Assistant turn starts with | Model writes |
|---|---|---|---|
| `think` | short task line + question | `<think>\n` | hidden reasoning, `</think>`, then the answer |
| `default` | same as `think` | `<think>\n\n</think>\n\n` | the answer directly |
| `reinit` | system persona + step-by-step analysis instructions + question | `<think>\n\n</think>\n\n<analysis>\n` | analysis, `</analysis>`, then the answer |

`reinit` ("reinitialized thinking") turns Qwen's built-in thinking off and has the model reason in a plain-text `<analysis>` block instead. All modes get the same question, the same shuffled options, the same sampling settings and the same 81,920-token output cap. Every prompt ends with Qwen's recommended instruction: ``Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.``

## Files

| File | Purpose |
|---|---|
| `configs/control.toml` | every run parameter, with its source |
| `src/data.py` | pinned GPQA CSV to `data/gpqa_diamond.jsonl` |
| `src/prompts.py` | exact model input per mode (all prompt text lives here); option shuffle |
| `src/extract.py` | strict answer extraction (last `"answer": "X"` after the reasoning block) |
| `src/generate.py` | sends prompts to SGLang, logs every generation (runs on DeltaAI) |
| `src/check_run.py` | pass/fail checks of a finished run |
| `src/score.py` | scoring and metrics; `src/schema.py` lists the log fields |
| `scripts/deltaai/` | setup, job submission and Slurm job scripts |
| `viewer/index.html` | offline dashboard for browsing every generation |
| `tests/` | `run_all.py` runs the tests; `fake_sglang.py` stands in for SGLang; `fake_run.py` makes a synthetic run |

## Usage

```bash
.venv/bin/python tests/run_all.py                 # tests (template parity needs transformers in .venv)
python3 -m src.data                               # build data/gpqa_diamond.jsonl
python3 -m src.data --show 5                      # print prompts exactly as sent
scripts/deltaai/submit.sh smoke|pilot|full        # on DeltaAI (after setup.sh)
python3 -m src.check_run runs/<run>               # pass/fail checks
python3 -m src.score runs/<run>/generations.jsonl # -> scored.jsonl, summary.md
open viewer/index.html                            # load scored.jsonl
```

## Status

- [x] Data prepared and gold letters verified
- [x] Unit tests pass
- [x] Prompts match the official chat template (text and token IDs)
- [x] Generation and job scripts written, verified locally against a fake server
- [ ] DeltaAI setup and smoke test, then pilot (20 questions)
- [ ] Full run and summary

**GPQA is test-only:** never train on it or commit its text. `data/` and `runs/` are gitignored.
