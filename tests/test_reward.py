"""Unit tests for src/reward.py (GRPO reward + per-rollout log). Run: python3 tests/run_all.py

Fake slime Samples (types.SimpleNamespace) with synthetic text; no slime, GPU or GPQA needed.
"""
import asyncio
import enum
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from src import reward


class Status(enum.Enum):          # mirrors slime's Sample.Status values
    COMPLETED = "completed"
    TRUNCATED = "truncated"


def sample(response, label="B", status=Status.COMPLETED, mode="reinit", split=None, n=5):
    meta = {"qid": "syn0001", "prompt_mode": mode}
    if split:
        meta["_split"] = split
    return SimpleNamespace(response=response, label=label, status=status, metadata=meta,
                           tokens=list(range(100, 120)), response_length=n, rollout_id=3,
                           index=7, group_index=1, weight_versions=["3"])


def run(s):
    return asyncio.run(reward.reward(None, s))


def with_log_dir(fn):
    def wrapper():
        d = tempfile.mkdtemp()
        old = os.environ.get("GAR_ROLLOUT_LOG_DIR")
        os.environ["GAR_ROLLOUT_LOG_DIR"] = d
        try:
            return fn(Path(d))
        finally:
            if old is None:
                os.environ.pop("GAR_ROLLOUT_LOG_DIR", None)
            else:
                os.environ["GAR_ROLLOUT_LOG_DIR"] = old
    wrapper.__name__ = fn.__name__
    return wrapper


def logged(d):
    return [json.loads(l) for f in sorted(d.glob("rollouts_*.jsonl")) for l in f.read_text().splitlines()]


@with_log_dir
def test_correct_answer_after_analysis_gets_1(d):
    assert run(sample("steps...\n</analysis>\n\nanswer: B<|im_end|>")) == 1.0
    rec = logged(d)[0]
    assert rec["correct"] and rec["extracted_answer"] == "B" and rec["reward"] == 1.0
    assert rec["qid"] == "syn0001" and rec["gold"] == "B" and rec["split"] == "train"


@with_log_dir
def test_wrong_letter_and_no_answer_get_0(d):
    assert run(sample("x\n</analysis>\n\nanswer: C")) == 0.0
    assert run(sample("x\n</analysis>\n\nI am not sure.")) == 0.0


@with_log_dir
def test_analysis_never_closed_gets_0_even_if_lenient_finds_the_right_letter(d):
    # the control run's most common reinit failure: closed with </think> instead of </analysis>
    assert run(sample("work\nAnswer: B\n</think>\n\nanswer: B")) == 0.0
    rec = logged(d)[0]
    assert rec["extraction_rule"] == "unclosed_analysis" and rec["lenient_correct"] is True


@with_log_dir
def test_cut_off_answer_gets_0_even_with_the_right_letter(d):
    assert run(sample("x\n</analysis>\nanswer: B", status=Status.TRUNCATED)) == 0.0
    rec = logged(d)[0]
    assert rec["truncated"] and rec["status"] == "truncated" and not rec["lenient_correct"]


@with_log_dir
def test_log_has_the_answer_tokens_and_split(d):
    run(sample("x\n</analysis>\nanswer: B", split="eval", n=5))
    rec = logged(d)[0]
    assert rec["output_token_ids"] == list(range(115, 120))      # last response_length tokens
    assert rec["split"] == "eval" and rec["rollout_id"] == 3 and rec["n_output_tokens"] == 5
    assert rec["reward_rule"] == "last_answer_field_v3"


@with_log_dir
def test_list_input_and_default_mode(d):
    out = run([sample("x\n</analysis>\nanswer: B"), sample("x\n</analysis>\nanswer: A")])
    assert out == [1.0, 0.0] and len(logged(d)) == 2
    # the matched control (default mode): whole output is visible, no <analysis> needed
    assert run(sample("reasoning...\nanswer: B", mode="default")) == 1.0


def test_missing_log_dir_stops_the_run():
    old = os.environ.pop("GAR_ROLLOUT_LOG_DIR", None)
    try:
        run(sample("x\n</analysis>\nanswer: B"))
    except RuntimeError:
        return
    finally:
        if old is not None:
            os.environ["GAR_ROLLOUT_LOG_DIR"] = old
    raise AssertionError("expected RuntimeError without GAR_ROLLOUT_LOG_DIR")
