# GAR reinitialized-thinking rebuild

Train Qwen3.5-2B with RL using GAR-style reinitialized thinking on GPQA-like multiple-choice data (slime, NCSA DeltaAI), and evaluate on GPQA Diamond against the untrained model in four reasoning modes, with full trajectory transparency.

| Read first | |
|---|---|
| `CLAUDE.md` | rules for Claude Code: learning mode, consult-before-acting, gates, logging |
| `docs/MODES.md` | exact input for think / nothink / direct / gar |
| `docs/PARAMETERS.md` | every training parameter: GAR official vs ours, and why |
| `docs/PIPELINE.md` | components, data flow, status |
| `docs/INFRASTRUCTURE.md` | DeltaAI + slime setup, the ARM blocker, bring-up order |

Tests: `python -m pytest tests` (or run the test functions directly). `data/` and `runs/` are gitignored.
