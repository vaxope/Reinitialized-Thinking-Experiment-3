"""Build the exact model input for each reasoning mode.

Why this exists: servers that apply chat templates for you can silently change
what the model sees (e.g. Qwen3.5 turns thinking OFF unless enable_thinking=True).
Here every mode's input is built explicitly, so it can be logged and diffed.

Pipeline position: data prep -> [modes.build_messages / build_text] -> rollout or eval.
"""
import random
import string

MODES = ("think", "nothink", "direct", "gar")

# Prefix of the assistant turn for each mode, copied from Qwen3.5's chat template.
THINK_OPEN = "<think>\n"                 # enable_thinking=True
THINK_EMPTY = "<think>\n\n</think>\n\n"  # enable_thinking False/undefined

QUESTION_PROMPT = (
    "Answer the following multiple-choice question. "
    "End your response with a final line of the form \"Answer: X\", "
    "where X is one of {letters}.\n\n"
    "Question: {question}\n\n{options}"
)

DIRECT_INSTRUCTION = "\n\nRespond with only the final answer line, with no explanation."

# GAR-style reinitialized thinking for solving (adapted from GAR's Figure 2 structure).
GAR_STEPS = (
    "\n\nBefore answering, reason inside <analysis></analysis> by following these steps:\n"
    "1. Identify the principles and facts the question depends on.\n"
    "2. Work through the problem step by step, and evaluate each option against them.\n"
    "3. Check the chosen answer: re-derive any key quantity and confirm consistency and units.\n"
    "After </analysis>, give the final answer line."
)

# Generation caps per mode. Only `direct` is capped tightly; others use the run config.
DIRECT_MAX_NEW_TOKENS = 16


def shuffle_options(qid: str, sample_idx: int, correct: str, wrong: list[str]):
    """Shuffle options with a seed fixed per (question, sample), shared by all modes.

    Returns (options_in_order, gold_letter, order) where `order` lists indices into
    [correct] + wrong, so the shuffle can be logged without storing option text.
    """
    opts = [correct] + list(wrong)
    order = list(range(len(opts)))
    random.Random(f"{qid}|{sample_idx}").shuffle(order)
    shuffled = [opts[i] for i in order]
    gold = string.ascii_uppercase[order.index(0)]
    return shuffled, gold, order


def user_text(mode: str, question: str, options: list[str]) -> str:
    """The user-turn text for a mode. Only the mode instruction differs between modes."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    letters = string.ascii_uppercase[: len(options)]
    opt_block = "\n".join(f"{L}) {o}" for L, o in zip(letters, options))
    text = QUESTION_PROMPT.format(letters=", ".join(letters), question=question, options=opt_block)
    if mode == "direct":
        text += DIRECT_INSTRUCTION
    elif mode == "gar":
        text += GAR_STEPS
    return text


def enable_thinking(mode: str) -> bool:
    """Only `think` turns thinking on; the template turns it off otherwise."""
    return mode == "think"


def build_messages(mode: str, question: str, options: list[str]) -> list[dict]:
    """Chat messages for tokenizer.apply_chat_template / slime's --apply-chat-template."""
    return [{"role": "user", "content": user_text(mode, question, options)}]


def build_text(mode: str, question: str, options: list[str]) -> str:
    """The full input string, built by hand to match Qwen3.5's template exactly.

    Gate 3 checks this equals tokenizer.apply_chat_template(build_messages(...),
    tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking(mode)).
    """
    prefix = THINK_OPEN if enable_thinking(mode) else THINK_EMPTY
    return (
        "<|im_start|>user\n" + user_text(mode, question, options) + "<|im_end|>\n"
        "<|im_start|>assistant\n" + prefix
    )


def check_against_tokenizer(tokenizer, mode: str, question: str, options: list[str]) -> dict:
    """Gate 3: compare the hand-built text with the official template. Returns a report."""
    official = tokenizer.apply_chat_template(
        build_messages(mode, question, options),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking(mode),
    )
    ours = build_text(mode, question, options)
    return {
        "mode": mode,
        "match": official == ours,
        "ours_tokens": len(tokenizer(ours, add_special_tokens=False)["input_ids"]),
        "official_tokens": len(tokenizer(official, add_special_tokens=False)["input_ids"]),
        "official_tail": official[-60:],
        "ours_tail": ours[-60:],
    }
