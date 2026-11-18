"""Tests for src/train_report.py on a synthetic rollout log (no model output, no GPQA text).
Run: python3 tests/run_all.py  (standard library only)."""
import json
import tempfile
from pathlib import Path

from src import train_report
from src.train_args import load_train_config

GROUP, BATCH = 8, 128


def _write(recs):
    d = Path(tempfile.mkdtemp())
    with (d / "rollouts_host_1.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return d


def _ans(t, split, version, group, correct, lenient=None, tokens=1000, truncated=False, answer="A"):
    return {"time": t, "split": split, "weight_versions": [] if version is None else [str(version)],
            "group_index": group, "qid": f"q{group}", "reward": 1.0 if correct else 0.0, "correct": correct,
            "lenient_correct": correct if lenient is None else lenient, "truncated": truncated,
            "n_output_tokens": tokens, "extracted_answer": answer, "analysis_closed": True}


def _run(versions=True):
    """One validation pass before training (300 answers, 90 right), then 2 training steps of
    16 groups x 8. Step 1: 4 groups all right, 4 all wrong, 8 half right. Step 2: 1 in 8 right by
    rule v3, 4 in 8 by the lenient reading, 1 in 8 truncated, 1 in 8 with no answer field."""
    recs, t = [], 0
    for i in range(300):
        recs.append(_ans(t, "eval", 1 if versions else None, 1000 + i, i < 90)); t += 1
    for step in (1, 2):
        for g in range(16):
            for j in range(GROUP):
                if step == 1:
                    ok = g < 4 or (g >= 8 and j < 4)          # groups 4-7 all wrong
                    recs.append(_ans(t, "train", step if versions else None, step * 100 + g, ok, tokens=2000 + j))
                else:
                    ok = j == 0                                # 1 in 8 right by rule v3 ...
                    recs.append(_ans(t, "train", step if versions else None, step * 100 + g, ok, lenient=j < 4,
                                     truncated=(j == 7), answer=None if j == 6 else "B"))
                t += 1
    return recs


def test_steps_from_weight_versions_and_group_signal():
    rep = train_report.report(_write(_run()), load_train_config())
    s1, s2 = rep["train"][1], rep["train"][2]
    assert s1["n"] == BATCH and s1["n_groups"] == 16 and s1["groups_full"] == 16
    assert s1["groups_no_signal"] == 8                      # 4 all-right + 4 all-wrong
    assert abs(s1["mean_reward"] - (32 + 32) / 128) < 1e-12 and s1["mean_reward"] == s1["accuracy"]
    assert s2["accuracy"] == 16 / 128 and s2["accuracy_lenient"] == 64 / 128
    assert abs(s2["format_gap"] - 48 / 128) < 1e-12 and s2["groups_no_signal"] == 0
    assert s2["truncation_rate"] == 16 / 128 and s2["answer_found"] == 112 / 128
    assert rep["validation"][0]["step"] == 0 and rep["validation"][0]["accuracy"] == 0.3


def test_fallback_without_versions_splits_by_batch():
    rep = train_report.report(_write(_run(versions=False)), load_train_config())
    assert sorted(rep["train"]) == [1, 2] and rep["train"][1]["groups_no_signal"] == 8


def test_markdown_shows_first_last_and_every_nth():
    rep = train_report.report(_write(_run()), load_train_config())
    md = train_report.markdown(rep, every=5)
    assert "| 1 | 128 |" in md and "| 2 | 128 |" in md and "8/16" in md
    assert "| 0 | untrained | 300 | 30.0%" in md
