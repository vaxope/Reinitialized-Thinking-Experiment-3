"""Unit tests for src/data.py (component 4). Run: python3 tests/run_all.py

Uses a tiny made-up CSV, never GPQA text. The final test checks the real pinned file
only if it is already in the local Hugging Face cache (prints counts, no text).
"""
import csv
import hashlib
import io
import tempfile
from pathlib import Path

from src import data
from src.config import load_config

HEADER = ["Record ID", "Question", "Correct Answer", "Incorrect Answer 1",
          "Incorrect Answer 2", "Incorrect Answer 3", "High-level domain", "Subdomain", "Explanation"]


def fake_rows(n=3):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    for i in range(n):
        w.writerow([f"rec{i}", f"  Question {i}\nline two \n", " right ", "w1", "w2", "w3\n",
                    "Physics", "Optics", "not kept"])
    return list(csv.DictReader(io.StringIO(buf.getvalue(), newline="")))


def test_rows_to_questions_keeps_only_needed_fields_and_strips():
    q = data.rows_to_questions(fake_rows(1), strip=True)[0]
    assert set(q) == {"qid", "question", "correct", "wrong", "domain", "subdomain"}
    assert q["question"] == "Question 0\nline two"  # outer whitespace gone, inner newline kept
    assert q["correct"] == "right" and q["wrong"] == ["w1", "w2", "w3"]


def test_no_strip_keeps_raw_text():
    q = data.rows_to_questions(fake_rows(1), strip=False)[0]
    assert q["correct"] == " right "


def test_validate_counts_and_duplicates():
    qs = data.rows_to_questions(fake_rows(3), strip=True)
    assert data.validate(qs, 3) == []
    for bad, n in [(qs, 4), (qs + [qs[0]], 4)]:
        try:
            data.validate(bad, n)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_validate_warns_on_identical_options():
    qs = data.rows_to_questions(fake_rows(1), strip=True)
    qs[0]["wrong"][0] = qs[0]["correct"]
    assert len(data.validate(qs, 1)) == 1


def test_sha256_check():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.csv"
        p.write_bytes(b"abc")
        data.verify_sha256(p, hashlib.sha256(b"abc").hexdigest())
        try:
            data.verify_sha256(p, "0" * 64)
        except ValueError:
            return
    raise AssertionError("expected ValueError")


def test_real_pinned_csv_if_cached():
    cfg = load_config()
    path = data.hf_cache_path(cfg)
    if not path.exists():
        return "SKIP: pinned GPQA CSV not in local HF cache"
    data.verify_sha256(path, cfg["data"]["csv_sha256"])
    with path.open(newline="", encoding="utf-8") as f:
        qs = data.rows_to_questions(list(csv.DictReader(f)), cfg["data"]["strip_whitespace"])
    data.validate(qs, cfg["data"]["n_questions"])  # raises on any hard failure
