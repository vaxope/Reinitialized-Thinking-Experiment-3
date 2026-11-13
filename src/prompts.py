"""Build the exact model input for each mode (think, default).

Purpose : produce, by hand, the full string the model will see, so it can be logged,
          diffed and checked. We never let a server apply the chat template, because
          Qwen3.5's template silently turns thinking OFF unless enable_thinking=True.
Inputs  : a question (text + correct answer + 3 wrong answers, from data/gpqa_diamond.jsonl),
          a sample index, and the [prompt] section of configs/control.toml.
Outputs : - shuffle_options -> options in display order, gold letter, permutation
          - build_text      -> the complete prompt string (user turn + assistant prefix)
Pipeline: data/gpqa_diamond.jsonl -> [prompts] -> (later) src/generate.py -> SGLang.
          src/data.py uses it to print example prompts (milestone 1);
          check_against_tokenizer is the template-parity check (milestone 3).

Template source: Qwen/Qwen3.5-2B chat_template.jinja @ 15852e8c (read 2026-11-13):
    user turn       : '<|im_start|>user\\n' + content + '<|im_end|>\\n'
    generation start: '<|im_start|>assistant\\n' then
                      '<think>\\n'               if enable_thinking is true
                      '<think>\\n\\n</think>\\n\\n'  otherwise
    No system turn is inserted when no system message and no tools are given.
"""
from __future__ import annotations

import hashlib
import random
import string

MODES = ("think", "default")

# Start of the assistant turn per mode, copied character-for-character from the template.
ASSISTANT_PREFIX = {
    "think": "<think>\n",                 # open block: the model writes its reasoning next
    "default": "<think>\n\n</think>\n\n",  # closed empty block: the model answers directly
}

# The answer-format instruction recommended by Qwen for multiple-choice benchmarking,
# copied verbatim (backticks included) from the Qwen3.5-2B model card README @ 15852e8c,
# line 983, "Standardize Output Format": "Multiple-Choice Questions: Add the following
# JSON structure to the prompt to standardize responses: ...". Using the model's own
# recommended format makes "the model failed the format" less likely than a custom one.
MCQ_FORMAT_INSTRUCTION = (
    'Please show your choice in the `answer` field with only the choice letter, '
    'e.g., `"answer": "C"`.'
)

# The user-turn text. Identical for both modes: the ONLY difference between modes is
# the assistant prefix above. Any edit here changes every input => new run name.
# Layout: one-line task statement, the question, lettered options, then the format
# instruction last (the card gives no placement; last keeps it closest to generation).
QUESTION_PROMPT = (
    "Answer the following multiple-choice question.\n\n"
    "Question: {question}\n\n{options}\n\n"
    + MCQ_FORMAT_INSTRUCTION
)

# Hash of the prompt template + prefixes. Logged with every generation so two runs can
# be checked for identical prompting without diffing source code.
PROMPT_HASH = hashlib.sha256(
    (QUESTION_PROMPT + "".join(ASSISTANT_PREFIX[m] for m in MODES)).encode()
).hexdigest()[:16]


def shuffle_seed(salt: str, qid: str, sample_idx: int) -> str:
    """The seed string for one (question, sample). Both modes use the same seed, so
    think and default see the options in the same order and can be compared pairwise."""
    return f"{salt}|{qid}|{sample_idx}"


def shuffle_options(qid: str, sample_idx: int, correct: str, wrong: list[str],
                    salt: str) -> tuple[list[str], str, list[int]]:
    """Shuffle the 4 options deterministically.

    Returns (options, gold, order):
      options : option texts in display order (A, B, C, D)
      gold    : letter of the correct option, e.g. "C"
      order   : order[i] = index into [correct] + wrong of the option shown at position i.
                Index 0 is always the correct answer. Logged instead of option text, so
                the shuffle is recorded without copying GPQA text into logs.
    """
    pool = [correct] + list(wrong)
    order = list(range(len(pool)))
    # random.Random(str) hashes the string with SHA-512 (seed version 2): the result is
    # the same on every machine and Python version >= 3.2, unlike hash(str).
    random.Random(shuffle_seed(salt, qid, sample_idx)).shuffle(order)
    options = [pool[i] for i in order]
    gold = string.ascii_uppercase[order.index(0)]  # where the correct answer landed
    return options, gold, order


def user_text(question: str, options: list[str]) -> str:
    """Fill QUESTION_PROMPT: lettered options 'A) ...' one per line. (.format parses only
    the template, so braces inside question text, e.g. LaTeX, are inserted unchanged.)"""
    letters = string.ascii_uppercase[: len(options)]
    opt_block = "\n".join(f"{L}) {o}" for L, o in zip(letters, options))
    return QUESTION_PROMPT.format(question=question, options=opt_block)


def enable_thinking(mode: str) -> bool:
    """The template flag that each mode corresponds to (used only for the parity check)."""
    _check_mode(mode)
    return mode == "think"


def build_messages(question: str, options: list[str]) -> list[dict]:
    """The chat-message form of the same input (for tokenizer.apply_chat_template)."""
    return [{"role": "user", "content": user_text(question, options)}]


def build_text(mode: str, question: str, options: list[str]) -> str:
    """The complete raw input string for a mode: user turn + start of assistant turn."""
    _check_mode(mode)
    return (
        "<|im_start|>user\n" + user_text(question, options) + "<|im_end|>\n"
        "<|im_start|>assistant\n" + ASSISTANT_PREFIX[mode]
    )


def check_against_tokenizer(tokenizer, mode: str, question: str, options: list[str]) -> dict:
    """Milestone 3 (template parity): compare build_text with the official template.

    `tokenizer` is a transformers tokenizer loaded from Qwen/Qwen3.5-2B at the pinned
    revision. Returns a report; report["match"] must be True for both modes.
    """
    official = tokenizer.apply_chat_template(
        build_messages(question, options), tokenize=False,
        add_generation_prompt=True, enable_thinking=enable_thinking(mode),
    )
    ours = build_text(mode, question, options)
    ids = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    return {"mode": mode, "match": official == ours,
            "ids_match": ids(official) == ids(ours),
            "official_tail": official[-40:], "ours_tail": ours[-40:]}


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
