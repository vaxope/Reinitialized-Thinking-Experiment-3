"""Tests for src/generate.py against tests/fake_sglang.py (no GPU, no GPQA text).

Needs `transformers` (the .venv): skipped otherwise. Run: .venv/bin/python tests/run_all.py
Covers: every schema field written; token ids and decoded text agree; think-token counting;
truncation; HTTP errors logged (not dropped) and retried; resume skips finished keys;
config-change refusal; override guard for control/pilot run ids; score.py on the output.
"""
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = [{"qid": f"recTEST{i:02d}", "question": f"Synthetic question {i}: pick the largest.",
              "correct": "42", "wrong": ["7", "13", "21"], "domain": "Test", "subdomain": "Test"}
             for i in range(4)]


def _have_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeServer:
    def __init__(self, *extra):
        self.port = _free_port()
        self.proc = subprocess.Popen([sys.executable, "-m", "tests.fake_sglang", "--model-path", "Qwen/Qwen3.5-2B",
                                      "--revision", "15852e8c16360a2fea060d615a32b45270f8a8fc",
                                      "--port", str(self.port), "--fake-load-s", "0.5", *extra],
                                     cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.url = f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.proc.terminate()
        self.proc.wait(timeout=10)


def _run(tmp, server_url, *args):
    from src import generate
    data = Path(tmp) / "q.jsonl"
    data.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    return generate.main(["--data", str(data), "--server-url", server_url, "--runs-dir", str(Path(tmp) / "runs"), *args])


def _records(tmp, run_id):
    return [json.loads(l) for l in (Path(tmp) / "runs" / run_id / "generations.jsonl").read_text().splitlines()]


def test_end_to_end_records_resume_and_score():
    if not _have_transformers():
        return "SKIP: transformers not installed"
    from src import score
    from src.schema import GENERATION_FIELDS
    with tempfile.TemporaryDirectory() as tmp, FakeServer() as srv:
        assert _run(tmp, srv.url, "--run-id", "smoke_a", "--samples", "2", "--max-new-tokens", "4096") == 0
        recs = _records(tmp, "smoke_a")
        assert len(recs) == 4 * 2 * 2  # questions x samples x modes
        for r in recs:
            assert not [k for k in GENERATION_FIELDS if k not in r]
            assert r["error"] is None and r["finish_reason"] == "stop"
            assert r["response_checks"]["completion_tokens_match"] and r["response_checks"]["prompt_tokens_match"]
            assert r["response_checks"]["server_text_matches_decode"] is True and r["server_text"] is None
            assert r["output_text"].endswith("<|im_end|>")  # the stop token's id is kept, so its text is too
            assert r["n_input_tokens"] == len(r["input_token_ids"]) and r["n_output_tokens"] == len(r["output_token_ids"])
            assert r["sampling_params"]["max_new_tokens"] == 4096
            if r["mode"] == "think":
                assert r["think_closed"] and 0 < r["n_think_tokens"] < r["n_output_tokens"]
                assert r["output_text"].count("</think>") == 1
            else:
                assert r["n_think_tokens"] == 0
        # same (qid, sample) -> same option order and gold in both modes
        by = {(r["qid"], r["sample_idx"], r["mode"]): r for r in recs}
        for (q, s, m), r in by.items():
            other = by[(q, s, "default" if m == "think" else "think")]
            assert r["option_order"] == other["option_order"] and r["gold"] == other["gold"]
        # resume: nothing new is sent
        assert _run(tmp, srv.url, "--run-id", "smoke_a", "--samples", "2", "--max-new-tokens", "4096") == 0
        assert len(_records(tmp, "smoke_a")) == 16
        meta = [json.loads(l) for l in (Path(tmp) / "runs" / "smoke_a" / "run_meta.jsonl").read_text().splitlines()]
        assert [m["event"] for m in meta] == ["start", "end", "start", "end"] and meta[2]["n_to_send"] == 0
        # scoring works on the real generator's output
        score.main([str(Path(tmp) / "runs" / "smoke_a" / "generations.jsonl")])
        s = json.loads((Path(tmp) / "runs" / "smoke_a" / "summary.json").read_text())
        assert s["n_records"] == 16 and s["missing_schema_fields"] == {}
        assert all(d["extraction_rules"].get("last_answer_field") == 8 for d in s["per_mode"].values())
        assert s["run_timing"]["segments"] == 2


def test_truncation_is_recorded():
    if not _have_transformers():
        return "SKIP: transformers not installed"
    with tempfile.TemporaryDirectory() as tmp, FakeServer() as srv:
        _run(tmp, srv.url, "--run-id", "smoke_trunc", "--samples", "1", "--max-new-tokens", "20")
        recs = _records(tmp, "smoke_trunc")
        assert all(r["finish_reason"] == "length" and r["n_output_tokens"] == 20 for r in recs)
        assert all(r["think_closed"] is False and r["n_think_tokens"] == 20 for r in recs if r["mode"] == "think")


def test_errors_are_logged_then_retried():
    if not _have_transformers():
        return "SKIP: transformers not installed"
    with tempfile.TemporaryDirectory() as tmp:
        with FakeServer("--fake-error-every", "3") as srv:
            code = _run(tmp, srv.url, "--run-id", "smoke_err", "--samples", "1", "--concurrency", "1")
        assert code == 3  # non-zero exit when any request failed
        recs = _records(tmp, "smoke_err")
        errs = [r for r in recs if r["error"]]
        assert len(recs) == 8 and len(errs) == 2 and all("HTTP 500" in r["error"] for r in errs)
        with FakeServer() as srv:  # healthy server: --retry-errors re-sends only the 2 failures
            assert _run(tmp, srv.url, "--run-id", "smoke_err", "--samples", "1", "--retry-errors") == 0
        recs = _records(tmp, "smoke_err")
        assert len(recs) == 10  # the failed attempts stay in the log
        from src import score
        latest, superseded = score.latest_per_key(recs)
        assert len(latest) == 8 and len(superseded) == 2 and all(r["error"] is None for r in latest)


def test_overrides_refused_for_real_run_ids():
    if not _have_transformers():
        return "SKIP: transformers not installed"
    for rid in ("control_v1", "pilot_v1"):
        try:
            _run(tempfile.mkdtemp(), "http://127.0.0.1:1", "--run-id", rid, "--max-new-tokens", "10")
        except SystemExit as e:
            assert "not allowed" in str(e)
            continue
        raise AssertionError("expected refusal")


def test_config_change_refused_for_existing_run():
    if not _have_transformers():
        return "SKIP: transformers not installed"
    with tempfile.TemporaryDirectory() as tmp, FakeServer() as srv:
        _run(tmp, srv.url, "--run-id", "smoke_cfg", "--samples", "1", "--limit-questions", "1")
        cfg2 = Path(tmp) / "other.toml"
        cfg2.write_text((ROOT / "configs" / "control.toml").read_text() + "\n# changed\n")
        try:
            _run(tmp, srv.url, "--run-id", "smoke_cfg", "--config", str(cfg2), "--samples", "1", "--limit-questions", "1")
        except SystemExit as e:
            assert "Use a new --run-id" in str(e)
            return
        raise AssertionError("expected refusal")
