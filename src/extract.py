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


# ================================================================ lenient reading (beside v3)
# Rule "any_answer_lenient_v1" (2026-11-16, user's choice after control_v1): "if the model gives
# an answer, count it", even when it ignores the requested format. It is reported BESIDE v3,
# never instead of it: v3 stays the official accuracy and the training reward. The gap between
# the two is the cost of format failures (control_v1 reinit: 35% of outputs unreadable by v3).
# Differences from v3:
#   - No block rule: the WHOLE output is searched, so an answer written inside an <analysis>
#     that was closed with </think> (or never closed) still counts. Truncated outputs are
#     still wrong (src/score.py), because a cut-off output never reached a final answer.
#   - More answer shapes (all need ONE capital letter; the last one in the output wins):
#       field   : every v3 field (strict "answer": "X" and answer lines)
#       tag     : any tag whose name/attributes contain "answer" or "choice": <answer>X</answer>,
#                 <final_answer>X</final_answer>, <choice>X</choice>, <div class="answer">X</div>
#       quoted  : the key "answer" then the letter in quotes, colon optional: answer"A", answer": "B"\
#       key     : a choice/option line, e.g. choice: X, "choice": "X", Final choice: (X)
#       phrase  : "the answer is X", "correct option is X", "Option X is correct", "I will select X";
#                 IGNORED when the same line has "if", "whether", "suppose" or "assume" just
#                 before it ("What if the answer is C?" is a hypothesis, not an answer)
#       bare    : the LAST non-empty line is only the letter (C, **C**, (C)) after removing
#                 trailing closing tags such as </analysis>
#       option  : the LAST non-empty line repeats an option, "A) 14" or "(B) 3-bromo...": the
#                 model answered by copying the option line (seen in default mode)
#   - Still not counted: \boxed{X} (never seen in GPQA outputs) and lower-case letters
#     ("a) 0.85"), which can't be told apart from list items.
LENIENT_RULE_NAME = "any_answer_lenient_v1"
LENIENT_TAG = re.compile(r"<[^<>\n]*(?i:answer|choice)[^<>\n]*>\s*[*(]*\s*([A-Z])\s*[*)]*\s*</")
LENIENT_QUOTED = re.compile(r"""(?<![A-Za-z])(?i:answer)["'`*]*\s*[:=]?\s*["'`*]+([A-Z])["'`*]""")
LENIENT_KEY = re.compile(
    r"""(?m)(?<![A-Za-z])[`"'*]*(?i:(?:final\s+)?(?:choice|option))[`"'*]*[ \t]*:[`"'*( \t]*([A-Z])[`"'*)]*[ \t]*(?=$|<|[,}\]])""")
# "answer is C", "correct option is (B)", "choice is: D" -- the letter must stand alone (not "A good").
LENIENT_PHRASE = re.compile(
    r"""(?i:answer|choice|option)\s+(?i:is)\s*:?\s*[*"'`(]*\s*(?:(?i:option|choice)\s+)?[*"'`(]*([A-Z])[*"'`)]*(?=[\s.,;:!)]|$)""")
# "Option C is correct", "**Choice B** is the best answer"
LENIENT_OPTION_IS = re.compile(
    r"""(?i:option|choice)\s+[*"'`(]*([A-Z])[*"'`)]*\s+(?i:is)\s+(?i:the\s+)?(?i:correct|right|best|most)""")
# "I will select D", "So I choose C", "I'll pick option B"
LENIENT_SELECT = re.compile(r"""\b(?i:select|choose|pick)\s+(?:(?i:option|choice)\s+)?[*"'`(]*([A-Z])[*"'`)]*(?=[\s.,;:!)]|$)""")
# A phrase answer is ignored if one of these words comes shortly before it on the same line.
HYPOTHETICAL = re.compile(r"(?i)\b(?:if|whether|suppose|supposing|assume|assuming|unless|maybe|perhaps)\b")
HYPOTHETICAL_WINDOW = 40   # characters looked back on the same line (covers "What if the correct answer is")
LENIENT_BARE = re.compile(r"""^[`"'*( \t]*([A-Z])[`"'*). \t]*$""")
LENIENT_OPTION_LINE = re.compile(r"""^[`"'* \t]*\(?([A-Z])\)\s*\S""")
TRAILING_TAGS = re.compile(r"(?:\s*(?:</[a-z_]+>|<\|im_end\|>|```))+\s*$")


def find_lenient(text: str) -> list[tuple[int, str, str]]:
    """Every lenient answer in `text` as (position of the letter, letter, kind), in text order."""
    found = {pos: (letter, "field") for pos, letter in find_fields(text)}
    for kind, pat in (("tag", LENIENT_TAG), ("quoted", LENIENT_QUOTED), ("key", LENIENT_KEY),
                      ("phrase", LENIENT_PHRASE), ("phrase", LENIENT_OPTION_IS), ("phrase", LENIENT_SELECT)):
        for m in pat.finditer(text):
            if kind == "phrase":
                line_start = text.rfind("\n", 0, m.start()) + 1
                before = text[max(line_start, m.start() - HYPOTHETICAL_WINDOW):m.start()]
                if HYPOTHETICAL.search(before):
                    continue
            found.setdefault(m.start(1), (m.group(1), kind))
    body = TRAILING_TAGS.sub("", text)
    lines = body.rstrip().splitlines()
    if lines:
        for kind, pat in (("bare", LENIENT_BARE), ("option", LENIENT_OPTION_LINE)):
            m = pat.match(lines[-1])
            if m:
                found.setdefault(len(body.rstrip()) - len(lines[-1]) + m.start(1), (m.group(1), kind))
                break
    return [(p, l, k) for p, (l, k) in sorted(found.items())]


def extract_lenient(output_text: str | None, n_options: int = 4) -> dict:
    """Apply rule any_answer_lenient_v1 (see the block comment above). Mode-independent."""
    if not output_text or not output_text.strip():
        return {"letter": None, "rule": "empty_output", "n_answers": 0, "kind": None}
    hits = find_lenient(output_text)
    if not hits:
        return {"letter": None, "rule": "no_answer", "n_answers": 0, "kind": None}
    _, letter, kind = hits[-1]
    if letter not in string.ascii_uppercase[:n_options]:
        return {"letter": None, "rule": "invalid_letter", "n_answers": len(hits), "kind": kind}
    return {"letter": letter, "rule": "last_answer_anywhere", "n_answers": len(hits), "kind": kind}
