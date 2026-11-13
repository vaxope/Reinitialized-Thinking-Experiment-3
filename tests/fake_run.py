"""Write a SYNTHETIC generations.jsonl for testing src/score.py and the viewer.

Purpose : exercise scoring and the dashboard before any real run exists. Contains no
          GPQA text (questions are made up), so the file is safe to open and share.
Output  : JSONL with every field of src/schema.GENERATION_FIELDS, covering each
          outcome: correct, wrong, no answer line, unclosed think (truncated), error.
Usage   : python3 -m tests.fake_run runs/demo_fake/generations.jsonl [--questions 12 --samples 2]
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path

from src import prompts
from src.config import load_config


def fake_tokens(text: str) -> list[int]:
    """Stand-in tokenizer: one fake id per whitespace-separated word. crc32, not hash():
    Python's str hash changes between processes, which would make the fake run unreproducible."""
    return [zlib.crc32(w.encode()) % 150000 for w in text.split()]


def make_record(qid: str, sample_idx: int, mode: str, kind: str, cfg: dict, rng: random.Random) -> dict:
    question = f"Synthetic question {qid}: which value is the largest?"
    correct, wrong = "42", ["7", "13", "21"]
    options, gold, order = prompts.shuffle_options(qid, sample_idx, correct, wrong, cfg["prompt"]["shuffle_salt"])
    input_text = prompts.build_text(mode, question, options)
    wrong_letter = next(L for L in "ABCD" if L != gold)
    reasoning = " ".join(["Let me compare the options carefully."] * rng.randint(3, 40))
    answer_body = "Comparing the values, the largest is clearly the right one."
    finish, error = "stop", None
    if kind == "correct":
        visible = f'{answer_body}\n\n{{"answer": "{gold}"}}'
    elif kind == "wrong":
        visible = f'{answer_body}\n\n```json\n{{"answer": "{wrong_letter}"}}\n```'
    elif kind == "no_answer":
        visible = f"{answer_body}\n\nAnswer: {gold}"  # wrong format: a near miss
    elif kind == "truncated":
        visible, finish = None, "length"
    elif kind == "error":
        visible, finish, error = None, None, "ConnectionError: synthetic failure"
    if kind == "error":
        output_text = None
    elif mode == "think":
        output_text = reasoning + (f'\n"answer": "{gold}"' if kind == "truncated" else "")  # draft answer inside unfinished thinking
        if visible is not None:
            output_text += "\n</think>\n\n" + visible
    else:
        output_text = (reasoning + "\n" + visible) if visible else reasoning
    out_ids = fake_tokens(output_text) if output_text else []
    n_think = (len(fake_tokens(output_text.split("</think>")[0])) + 1) if (mode == "think" and output_text) else 0
    return {
        "run_id": "demo_fake", "mode": mode, "model": cfg["model"]["name"],
        "model_revision": cfg["model"]["revision"], "qid": qid, "sample_idx": sample_idx,
        "seed": prompts.shuffle_seed(cfg["prompt"]["shuffle_salt"], qid, sample_idx),
        "option_order": order, "chat_template_hash": prompts.PROMPT_HASH,
        "input_text": input_text, "input_token_ids": fake_tokens(input_text),
        "output_text": output_text, "output_token_ids": out_ids,
        # SGLang's format: [logprob, token_id, token_text]
        "output_logprobs": [[round(-rng.random() * 2, 4), t, None] for t in out_ids],
        "finish_reason": finish, "n_input_tokens": len(fake_tokens(input_text)),
        "n_output_tokens": len(out_ids), "n_think_tokens": n_think, "gold": gold,
        "sampling_params": {k: v for k, v in cfg["sampling"].items() if k != "status"},
        "sglang_version": "synthetic", "wall_time_s": None if error else round(len(out_ids) / 150 + rng.random(), 3),
        "error": error,
    }


KINDS = ["correct", "correct", "wrong", "no_answer", "truncated", "error"]


def make_run(n_questions: int, n_samples: int, cfg: dict, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    recs = []
    for i in range(n_questions):
        for s in range(n_samples):
            for mode in cfg["prompt"]["modes"]:
                kind = KINDS[(i + s + (mode == "think")) % len(KINDS)]
                recs.append(make_record(f"recFAKE{i:04d}", s, mode, kind, cfg, rng))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--questions", type=int, default=12)
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in make_run(a.questions, a.samples, cfg):
            f.write(json.dumps(r) + "\n")
    print(f"wrote synthetic run to {out}")


if __name__ == "__main__":
    main()
