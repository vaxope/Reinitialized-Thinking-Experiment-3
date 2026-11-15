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
        assert len(recs) == 4 * 2 * 3  # questions x samples x modes
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
                assert r["output_text"].count("</think>") == 1 and r["n_analysis_tokens"] == 0
            else:
                assert r["n_think_tokens"] == 0
            if r["mode"] == "reinit":
                assert r["analysis_closed"] and 0 < r["n_analysis_tokens"] < r["n_output_tokens"]
                # the counted tokens end exactly at "</analysis>"
                head = r["output_token_ids"][:r["n_analysis_tokens"]]
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", revision=r["model_revision"])
                assert tok.decode(head).endswith("</analysis>")
                assert r["input_text"].startswith("<|im_start|>system\n")
        # same (qid, sample) -> same option order and gold in every mode
        by = {(r["qid"], r["sample_idx"], r["mode"]): r for r in recs}
        for (q, s, m), r in by.items():
            for other_mode in ("think", "default", "reinit"):
                other = by[(q, s, other_mode)]
                assert r["option_order"] == other["option_order"] and r["gold"] == other["gold"]
        # resume: nothing new is sent
        assert _run(tmp, srv.url, "--run-id", "smoke_a", "--samples", "2", "--max-new-tokens", "4096") == 0
        assert len(_records(tmp, "smoke_a")) == 24
        meta = [json.loads(l) for l in (Path(tmp) / "runs" / "smoke_a" / "run_meta.jsonl").read_text().splitlines()]
        assert [m["event"] for m in meta] == ["start", "end", "start", "end"] and meta[2]["n_to_send"] == 0
        # scoring works on the real generator's output
        score.main([str(Path(tmp) / "runs" / "smoke_a" / "generations.jsonl")])
        s = json.loads((Path(tmp) / "runs" / "smoke_a" / "summary.json").read_text())
        assert s["n_records"] == 24 and s["missing_schema_fields"] == {}
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
        assert all(r["analysis_closed"] is False and r["n_analysis_tokens"] == 20 for r in recs if r["mode"] == "reinit")


def test_errors_are_logged_then_retried():
    if not _have_transformers():
        return "SKIP: transformers not installed"
    with tempfile.TemporaryDirectory() as tmp:
        with FakeServer("--fake-error-every", "3") as srv:
            code = _run(tmp, srv.url, "--run-id", "smoke_err", "--samples", "1", "--concurrency", "1")
        assert code == 3  # non-zero exit when any request failed
        recs = _records(tmp, "smoke_err")
        errs = [r for r in recs if r["error"]]
        # 4 questions x 3 modes = 12 requests sent one at a time; every 3rd fails -> 4 errors
        assert len(recs) == 12 and len(errs) == 4 and all("HTTP 500" in r["error"] for r in errs)
        with FakeServer() as srv:  # healthy server: --retry-errors re-sends only the 4 failures
            assert _run(tmp, srv.url, "--run-id", "smoke_err", "--samples", "1", "--retry-errors") == 0
        recs = _records(tmp, "smoke_err")
        assert len(recs) == 16  # the failed attempts stay in the log
        from src import score
        latest, superseded = score.latest_per_key(recs)
        assert len(latest) == 12 and len(superseded) == 4 and all(r["error"] is None for r in latest)


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


def _fast_stall_config(tmp, check_s, probe_s):
    """A copy of control.toml with a short stall window, so the guard fires in seconds."""
    text = (ROOT / "configs" / "control.toml").read_text()
    assert "stall_check_s = 1200" in text and "stall_probe_timeout_s = 300" in text
    cfg = Path(tmp) / "fast_stall.toml"
    cfg.write_text(text.replace("stall_check_s = 1200", f"stall_check_s = {check_s}")
                       .replace("stall_probe_timeout_s = 300", f"stall_probe_timeout_s = {probe_s}"))
    return cfg


def test_stall_guard_stops_on_hung_server():
    # The server answers 3 requests, then hangs (also on /health_generate). generate.py must
    # notice, write a "stalled" end event, keep the 3 records and exit with code 4. It exits via
    # os._exit, so it runs in a subprocess here.
    if not _have_transformers():
        return "SKIP: transformers not installed"
    with tempfile.TemporaryDirectory() as tmp, FakeServer("--fake-hang-after", "3") as srv:
        data = Path(tmp) / "q.jsonl"
        data.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "src.generate", "--config", str(_fast_stall_config(tmp, 2, 2)),
                            "--data", str(data), "--server-url", srv.url, "--runs-dir", str(Path(tmp) / "runs"),
                            "--run-id", "smoke_stall", "--samples", "1", "--concurrency", "1"],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert p.returncode == 4, (p.returncode, p.stdout[-500:], p.stderr[-500:])
        assert "STALLED" in p.stdout and time.time() - t0 < 60
        assert len(_records(tmp, "smoke_stall")) == 3
        meta = [json.loads(l) for l in (Path(tmp) / "runs" / "smoke_stall" / "run_meta.jsonl").read_text().splitlines()]
        assert meta[-1]["event"] == "end" and meta[-1]["interrupted"] is True
        assert "/health_generate" in meta[-1]["stalled"] and meta[-1]["n_written"] == 3


def test_stall_guard_waits_for_slow_but_alive_server():
    # Each request takes 3 s, longer than the 1 s stall window, but /health_generate answers:
    # the guard must keep waiting and the run must finish normally.
    if not _have_transformers():
        return "SKIP: transformers not installed"
    with tempfile.TemporaryDirectory() as tmp, FakeServer("--fake-latency-s", "3") as srv:
        code = _run(tmp, srv.url, "--config", str(_fast_stall_config(tmp, 1, 2)), "--run-id", "smoke_slow",
                    "--samples", "1", "--limit-questions", "1")
        assert code == 0 and len(_records(tmp, "smoke_slow")) == 3


def test_pinned_sglang_version_enforced_for_real_runs():
    # The fake server reports version "fake-sglang": fine for smoke_* (warning), refused for pilot_*.
    if not _have_transformers():
        return "SKIP: transformers not installed"
    with tempfile.TemporaryDirectory() as tmp, FakeServer() as srv:
        try:
            _run(tmp, srv.url, "--run-id", "pilot_versiontest", "--pilot")
        except SystemExit as e:
            assert "sglang_version" in str(e) and "Refusing" in str(e)
            assert not (Path(tmp) / "runs" / "pilot_versiontest").exists()  # refused before the run dir
            return
        raise AssertionError("expected refusal")
