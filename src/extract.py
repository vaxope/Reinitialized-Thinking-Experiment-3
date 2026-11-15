"""Answer extraction: find the model's chosen letter in its output.

Purpose : turn an output string into (letter or None, rule that decided it). Every
          decision is named, so failures are counted by cause instead of guessed away.
Inputs  : output_text (what the model generated AFTER the prompt), the mode, n_options.
Output  : {"letter": "B" | None, "rule": <str>, "n_answer_fields": int,
           "n_near_misses": int, "think_closed": bool | None, "analysis_closed": bool | None}
Pipeline: runs/<run>/generations.jsonl -> src/score.py -> [extract_answer] -> scored.jsonl.
          viewer/index.html uses the same pattern (ANSWER_FIELD) only to highlight text.

The prompt (src/prompts.MCQ_FORMAT_INSTRUCTION, Qwen's recommended MCQ format) asks:
    Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.

Rule "last_answer_field_v2" (strict, in this order). v2 = v1 + the reinit mode (step 2b);
think and default are decided exactly as in v1:
  1. empty output                               -> None, "empty_output"
  2. think mode: split at the FIRST "</think>". The prompt already opened <think>, so the
     output begins inside the reasoning; the first </think> closes that block.
     No "</think>" at all                       -> None, "unclosed_think"
     (reasoning never finished; any answer in it is a draft, not an answer)
  2b. reinit mode: split at the FIRST "</analysis>". The prompt already opened <analysis>,
     so the output begins inside the analysis; the first </analysis> closes it.
     No "</analysis>" at all                    -> None, "unclosed_analysis"
     (same reasoning as unclosed_think; the prompt asks for the answer after </analysis>)
  3. in the visible text (think: after </think>; reinit: after </analysis>; default: whole
     output), find every
     answer field:  "answer": "C"   (key exactly `answer` in double quotes, a colon,
     one capital letter in double quotes; spaces around the colon allowed; it may sit
     inside a JSON object or a ```json block, or stand alone)
     none                                       -> None, "no_answer_field"
  4. take the LAST field (the model's final word after any self-correction)
     letter outside A..(n_options)              -> None, "invalid_letter"
     otherwise                                  -> letter, "last_answer_field"

Deliberately NOT accepted: "Answer: C", "the answer is C", "\\boxed{C}", 'answer: "C"'
(unquoted key), "Answer": "C" (capitalised key), "c". Looser matching silently guesses.
Instead, `n_near_misses` counts loose look-alikes, so the summary and viewer can show
how many outputs answered in a different format (a format failure, not a knowledge one).
"""
from __future__ import annotations

import re
import string

from src.prompts import ANALYSIS_CLOSE

RULE_NAME = "last_answer_field_v2"

# Strict: "answer" : "X"  with straight double quotes and one capital letter.
ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"([A-Z])"')
# Diagnostic only (never used for the letter): any case, optional quotes/markdown, ':' or '='.
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


def extract_answer(output_text: str | None, mode: str, n_options: int = 4) -> dict:
    """Apply rule last_answer_field_v2 (see module docstring)."""
    if mode not in ("think", "default", "reinit"):
        raise ValueError(f"unknown mode {mode!r}")
    if not output_text or not output_text.strip():
        return {"letter": None, "rule": "empty_output", "n_answer_fields": 0,
                "n_near_misses": 0, "think_closed": None, "analysis_closed": None}
    parts = split_output(output_text, mode)
    flags = {"think_closed": parts["think_closed"], "analysis_closed": parts["analysis_closed"]}
    if parts["think_closed"] is False:
        return {"letter": None, "rule": "unclosed_think", "n_answer_fields": 0,
                "n_near_misses": 0, **flags}
    if parts["analysis_closed"] is False:
        return {"letter": None, "rule": "unclosed_analysis", "n_answer_fields": 0,
                "n_near_misses": 0, **flags}
    visible = parts["visible"]
    matches = ANSWER_FIELD.findall(visible)
    # Near misses = loose look-alikes that are not strict fields (strict ones removed first).
    near = NEAR_MISS.findall(ANSWER_FIELD.sub("", visible))
    base = {"n_answer_fields": len(matches), "n_near_misses": len(near), **flags}
    if not matches:
        return {"letter": None, "rule": "no_answer_field", **base}
    letter = matches[-1]
    if letter not in string.ascii_uppercase[:n_options]:
        return {"letter": None, "rule": "invalid_letter", **base}
    return {"letter": letter, "rule": "last_answer_field", **base}
