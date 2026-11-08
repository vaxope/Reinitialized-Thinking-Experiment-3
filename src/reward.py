"""Answer extraction and reward for multiple-choice questions.

Rule (strict, documented): the answer is the LAST line matching `Answer: X`
outside any <think>/<analysis> block, where X is a valid option letter
(case-sensitive capital). Anything else -> no answer -> reward 0.

Why not slime's built-in `--rm-type gpqa`? Its extractor takes the FIRST match of
patterns like "answer is X" (case-insensitive, so "the answer is a bit…" yields A)
and falls back to any standalone capital letter. That can reward the wrong letter.
A strict rule makes failures visible instead of silently guessing.

Pipeline position: rollout response -> [extract_answer -> reward] -> GRPO advantages.
"""
import re
import string

ANSWER_LINE = re.compile(r"^\s*\**\s*Answer\s*:\s*\**\s*\(?([A-Z])\)?\s*\**\s*\.?\s*$", re.MULTILINE)
BLOCKS = re.compile(r"<(think|analysis)>.*?</\1>", re.DOTALL)


def visible_text(response: str) -> str:
    """Remove closed <think>/<analysis> blocks and anything before a final </think>."""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    return BLOCKS.sub("", response)


def extract_answer(response: str, n_options: int = 4) -> dict:
    """Return {'letter', 'rule', 'n_answer_lines'} so every decision is loggable."""
    if not response:
        return {"letter": None, "rule": "empty_response", "n_answer_lines": 0}
    if "<think>" in response and "</think>" not in response:
        return {"letter": None, "rule": "unclosed_think", "n_answer_lines": 0}
    valid = set(string.ascii_uppercase[:n_options])
    matches = ANSWER_LINE.findall(visible_text(response))
    if not matches:
        return {"letter": None, "rule": "no_answer_line", "n_answer_lines": 0}
    letter = matches[-1]
    if letter not in valid:
        return {"letter": None, "rule": "invalid_letter", "n_answer_lines": len(matches)}
    return {"letter": letter, "rule": "last_answer_line", "n_answer_lines": len(matches)}


def reward(response: str, gold: str, n_options: int = 4, truncated: bool = False) -> float:
    """1.0 if correct, else 0.0. Truncated responses score 0 here; length shaping
    (soft overlong penalty) is a separate, flagged component, not hidden in this rule."""
    if truncated:
        return 0.0
    return 1.0 if extract_answer(response, n_options)["letter"] == gold else 0.0


# ---- slime adapter: --custom-rm-path src.reward.custom_rm ----------------------
async def custom_rm(args, sample) -> float:
    """slime custom reward (signature from slime docs: async def custom_rm(args, sample) -> float).

    Expects sample.label = gold letter and sample.metadata['n_options'].
    """
    meta = getattr(sample, "metadata", None) or {}
    truncated = str(getattr(sample, "status", "")).endswith("TRUNCATED")
    return reward(sample.response, sample.label, int(meta.get("n_options", 4)), truncated)
