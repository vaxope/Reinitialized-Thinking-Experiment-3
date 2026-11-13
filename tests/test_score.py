"""Unit tests for src/score.py (component 7). Run: python3 tests/run_all.py"""
import json
import math
import tempfile
from pathlib import Path

from src import score
from src.config import load_config
from tests.fake_run import make_record, make_run
import random

CFG = load_config()


def rec(kind, mode="think", qid="q1", s=0):
    return make_record(qid, s, mode, kind, CFG, random.Random(0))


def test_outcomes_for_each_kind():
    for kind, outcome in [("correct", "correct"), ("wrong", "wrong"), ("no_answer", "no_answer"),
                          ("truncated", "truncated"), ("error", "error")]:
        for mode in ("think", "default"):
            r = score.score_record(rec(kind, mode), CFG)
            assert r["outcome"] == outcome, (kind, mode, r["outcome"])
            assert r["correct"] == (outcome == "correct")


def test_truncated_think_with_draft_answer_is_not_correct():
    r = score.score_record(rec("truncated", "think"), CFG)
    assert r["extraction_rule"] == "unclosed_think" and r["correct"] is False


def test_truncated_with_answer_line_still_wrong_but_letter_kept():
    r = rec("correct", "default")
    r["finish_reason"] = "length"
    out = score.score_record(r, CFG)
    assert out["outcome"] == "truncated" and out["extracted_answer"] == r["gold"]


def test_input_record_is_not_modified():
    r = rec("correct")
    before = json.dumps(r, sort_keys=True)
    score.score_record(r, CFG)
    assert json.dumps(r, sort_keys=True) == before


def test_wrong_format_counts_as_near_miss():
    s = score.summarize([score.score_record(rec("no_answer", m), CFG) for m in ("think", "default")], [], CFG, "mem")
    assert all(d["n_near_miss_only"] == 1 for d in s["per_mode"].values())


def test_rule_name_in_config_matches_code():
    from src import extract
    assert CFG["scoring"]["extraction_rule"] == extract.RULE_NAME


def test_wilson_known_value():
    lo, hi = score.wilson_ci(50, 100, 0.95)  # textbook: [0.404, 0.596]
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert abs(score.wilson_ci(0, 10, 0.95)[0]) < 1e-12  # 0 up to float rounding


def test_mcnemar_exact():
    assert score.mcnemar_exact(0, 0) == 1.0
    assert abs(score.mcnemar_exact(0, 5) - 0.0625) < 1e-12  # 2 * (1/32)
    assert score.mcnemar_exact(3, 3) == 1.0


def test_describe():
    d = score.describe([1, 2, 3, 4, None])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["median"] == 2.5 and d["total"] == 10


def test_summary_counts_every_record():
    recs = make_run(12, 2, CFG)
    scored = [score.score_record(r, CFG) for r in recs]
    s = score.summarize(scored, [], CFG, "mem")
    for m, d in s["per_mode"].items():
        assert d["n_records"] == 24 and sum(d["outcomes"].values()) == 24
        assert d["ci_method"] == "clustered_by_question"
        assert d["n_correct"] == d["outcomes"].get("correct", 0)
    assert s["paired"]["n_pairs"] == 24 and s["missing_schema_fields"] == {}


def test_end_to_end_cli_and_overwrite_guard():
    with tempfile.TemporaryDirectory() as d:
        g = Path(d) / "generations.jsonl"
        lines = [json.dumps(r) for r in make_run(4, 1, CFG)] + ["{not json"]
        g.write_text("\n".join(lines) + "\n")
        score.main([str(g)])
        s = json.loads((Path(d) / "summary.json").read_text())
        assert s["corrupt_lines"][0]["line"] == 9 and s["n_records"] == 8
        assert s["per_mode"]["think"]["ci_method"] == "wilson"
        assert len((Path(d) / "scored.jsonl").read_text().splitlines()) == 8
        try:
            score.main([str(g)])
        except SystemExit:
            return
    raise AssertionError("expected refusal to overwrite")
