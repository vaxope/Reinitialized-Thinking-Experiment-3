"""Answer extraction: find the model's chosen letter in its output.

Purpose : turn an output string into (letter or None, rule that decided it). Every
          decision is named, so failures are counted by cause instead of guessed away.
Inputs  : output_text (what the model generated AFTER the prompt), the mode, n_options.
Output  : {"letter": "B" | None, "rule": <str>, "n_answer_fields": int, "n_strict_fields": int,
           "strict_letter": "B" | None (what strict rule v2 would pick), "n_near_misses": int,
           "think_closed": bool | None, "analysis_closed": bool | None}
Pipeline: runs/<run>/generations.jsonl -> src/score.py -> [extract_answer] -> scored.jsonl.
          viewer/index.html uses the same two patterns only to highlight text.

The prompt (src/prompts.MCQ_FORMAT_INSTRUCTION, Qwen's recommended MCQ format) asks:
    Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.

Rule "last_answer_field_v3" (2026-11-16, user's choice after pilot_v1). v3 = v2 with a wider
definition of an answer field (step 3); the block handling (steps 1-2b, 4) is unchanged.
Why: in pilot_v1 the model almost always ended with an answer line, but only 5/60 used the
exact `"answer": "X"` form (e.g. `answer: C`, `answer: "C"`, `answer": "C"`), so v2 measured
punctuation rather than knowledge, and as a GRPO reward it would mostly teach formatting.
  1. empty output                               -> None, "empty_output"
  2. think mode: split at the FIRST "</think>". The prompt already opened <think>, so the
     output begins inside the reasoning; the first </think> closes that block.
     No "</think>" at all                       -> None, "unclosed_think"
     (reasoning never finished; any answer in it is a draft, not an answer)
  2b. reinit mode: split at the FIRST "</analysis>". The prompt already opened <analysis>,
     so the output begins inside the analysis; the first </analysis> closes it.
     No "</analysis>" at all                    -> None, "unclosed_analysis"
  3. in the visible text (think: after </think>; reinit: after </analysis>; default: whole
     output), find every answer field. An answer field is EITHER
       a. the strict v2 field  "answer": "C"  anywhere (inside JSON, a code block, or prose), OR
       b. an answer LINE: the word `answer` (any case, not part of a longer word), optionally
          wrapped in quotes / backticks / asterisks, then `:`, then one CAPITAL letter,
          optionally wrapped in quotes / backticks / asterisks / parentheses, and then NOTHING
          else up to the end of the line, except an HTML-like tag (`</answer>`), `,`, `}` or `]`.
          Accepted: answer: C | Answer: C | answer: "C" | answer": "C" | `answer`: "C" |
                    **Answer:** C | Answer: **C** | answer": "C"</answer> | Final Answer: (C)
     none                                       -> None, "no_answer_field"
  4. take the LAST field by position (the model's final word after any self-correction)
     letter outside A..(n_options)              -> None, "invalid_letter"
     otherwise                                  -> letter, "last_answer_field"

Still NOT accepted: prose ("the answer is C", "Option C is correct"), a bare letter, "\\boxed{C}",
<answer>C</answer>, a lower-case letter, any other key ("choice": "C"), and an answer line with
more text after the letter ("Answer: C because ..."), since text after the letter may change
its meaning. `n_near_misses` still counts loose look-alikes that are not fields.
The strict v2 result is kept beside it (`strict_letter`), so v2 accuracy stays reportable.
"""
from __future__ import annotations

import re
import string

from src.prompts import ANALYSIS_CLOSE

RULE_NAME = "last_answer_field_v3"

# 3a. Strict (= the v2 field): "answer" : "X" with straight double quotes and one capital letter.
ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"([A-Z])"')
# 3b. Answer line (new in v3). Read left to right:
#   (?<![A-Za-z])           `answer` is not the end of a longer word ("ananswer")
#   [`"'*]*(?i:answer)[`"'*]*   the key, any case, optionally quoted / in backticks / bold
#   [ \t]*:                 a colon (spaces/tabs only, so the field stays on one line)
#   [`"'*( \t]*([A-Z])[`"'*)]*  ONE capital letter, optionally quoted / bold / in parentheses
#                           (spaces allowed among the wrappers: "**Answer:** B")
#   [ \t]*(?=$|<|[,}\]])     then the line must end (or a tag / JSON punctuation follows)
ANSWER_LINE = re.compile(
    r"""(?m)(?<![A-Za-z])[`"'*]*(?i:answer)[`"'*]*[ \t]*:[`"'*( \t]*([A-Z])[`"'*)]*[ \t]*(?=$|<|[,}\]])""")
# Diagnostic only (never used for the letter): any case, optional quotes/markdown, ':' '=' or 'is'.
NEAR_MISS = re.compile(r'(?i)\banswer\b["*`\']*\s*(?:is|[:=])\s*["*`\'(]*\s*([A-Z])\b')
THINK_CLOSE = "</think>"


def split_output(output_text: str, mode: str) -> dict:
    """Separate the reasoning block from the visible answer.

    think  : reasoning = text before the first </think>, visible = text after it.
    reinit : reasoning = text before the first </analysis>, visible = text after it.
    default: the prompt contains a closed empty think block, so everything is visible.
    """
    if mode in ("think", "reinit"):
        close = THINK_CLOSE if mode == "think" else ANALYSIS_CLOSE
        closed = close in output_text
        reasoning, visible = output_text.split(close, 1) if closed else (output_text, "")
        flag = "think_closed" if mode == "think" else "analysis_closed"
        other = "analysis_closed" if mode == "think" else "think_closed"
        return {"reasoning": reasoning, "visible": visible, flag: closed, other: None}
    if mode == "default":
        return {"reasoning": "", "visible": output_text, "think_closed": None, "analysis_closed": None}
    raise ValueError(f"unknown mode {mode!r}")


def find_fields(visible: str) -> list[tuple[int, str]]:
    """Every answer field in `visible` as (position of the letter, letter), in text order.
    A strict field that is also an answer line is counted once (same letter position)."""
    found = {m.start(1): m.group(1) for m in ANSWER_FIELD.finditer(visible)}
    found.update({m.start(1): m.group(1) for m in ANSWER_LINE.finditer(visible)})
    return sorted(found.items())


def _blank(text: str, *patterns: re.Pattern) -> str:
    """Replace every match of the patterns with spaces (keeps positions; for near-miss counting)."""
    for p in patterns:
        text = p.sub(lambda m: " " * len(m.group(0)), text)
    return text


def extract_answer(output_text: str | None, mode: str, n_options: int = 4) -> dict:
    """Apply rule last_answer_field_v3 (see module docstring)."""
    if mode not in ("think", "default", "reinit"):
        raise ValueError(f"unknown mode {mode!r}")
    empty = {"n_answer_fields": 0, "n_strict_fields": 0, "strict_letter": None, "n_near_misses": 0}
    if not output_text or not output_text.strip():
        return {"letter": None, "rule": "empty_output", **empty,
                "think_closed": None, "analysis_closed": None}
    parts = split_output(output_text, mode)
    flags = {"think_closed": parts["think_closed"], "analysis_closed": parts["analysis_closed"]}
    if parts["think_closed"] is False:
        return {"letter": None, "rule": "unclosed_think", **empty, **flags}
    if parts["analysis_closed"] is False:
        return {"letter": None, "rule": "unclosed_analysis", **empty, **flags}
    visible = parts["visible"]
    fields = find_fields(visible)
    strict = ANSWER_FIELD.findall(visible)
    valid = string.ascii_uppercase[:n_options]
    # What rule v2 would have decided (same blocks, strict fields only), for comparison.
    strict_letter = strict[-1] if strict and strict[-1] in valid else None
    near = NEAR_MISS.findall(_blank(visible, ANSWER_FIELD, ANSWER_LINE))
    base = {"n_answer_fields": len(fields), "n_strict_fields": len(strict),
            "strict_letter": strict_letter, "n_near_misses": len(near), **flags}
    if not fields:
        return {"letter": None, "rule": "no_answer_field", **base}
    letter = fields[-1][1]
    if letter not in valid:
        return {"letter": None, "rule": "invalid_letter", **base}
    return {"letter": letter, "rule": "last_answer_field", **base}
