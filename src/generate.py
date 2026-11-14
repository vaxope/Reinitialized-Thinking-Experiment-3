"""Generate the control-run outputs by sending exact prompts to a running SGLang server.

Purpose : for every (question, sample, mode), build the exact input with src/prompts.py,
          tokenize it with the pinned Qwen tokenizer, send the TOKEN IDS to SGLang's native
          /generate endpoint, and append one record per generation to the run log BEFORE any
          scoring. Errors are recorded in their own field, never dropped.
Inputs  : data/gpqa_diamond.jsonl (src/data.py), configs/control.toml, a running SGLang
          server (started by scripts/deltaai/serve_and_generate.sh on a DeltaAI GPU node).
Outputs : runs/<run_id>/generations.jsonl  one record per generation (fields: src/schema.py)
          runs/<run_id>/run_meta.jsonl     one "start" and one "end" event per job segment
          runs/<run_id>/control.toml       copy of the config used (first segment only)
Pipeline: data.py -> prompts.py -> [generate.py <-> SGLang] -> score.py -> viewer

Safety checks before any request:
  1. the tokenizer's official chat template hash equals [model] chat_template_sha256;
  2. template parity: prompts.build_text == tokenizer.apply_chat_template, both modes;
  3. the server is healthy and serves the configured model;
  4. a run directory created with a different config is refused (comparability).

Resuming: records are keyed by (qid, sample_idx, mode). Re-running the same command skips
keys already in generations.jsonl, so a job killed at its time limit continues where it
stopped. --retry-errors also re-sends keys whose only records are errors; the new attempt is
appended and score.py keeps the latest record per key (the earlier one stays in the log).

Usage (normally via scripts/deltaai/*.sbatch, not by hand):
  python -m src.generate --run-id pilot_v1 --pilot
  python -m src.generate --run-id control_v1
  python -m src.generate --run-id smoke_123 --limit-questions 2 --samples 1 --max-new-tokens 2048
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src import prompts
from src.config import REPO_ROOT, load_config
from src.data import load_questions
from src.schema import GENERATION_FIELDS

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
HEALTH_POLL_S = 5          # seconds between /health checks while the server loads
PROGRESS_EVERY = 10        # print a progress line every N finished generations
# Run ids that may never use --max-new-tokens/--samples/--limit-questions overrides, so a
# real run can't be accidentally created with non-standard settings.
PROTECTED_PREFIXES = ("control", "pilot")


class Interrupted(Exception):
    """Raised in the main thread when Slurm (or the user) sends SIGTERM."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- HTTP helpers (stdlib only)
def http_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    """GET (payload None) or POST JSON; return the decoded JSON body. Raises on HTTP errors
    with the server's error text included, so it ends up in the record's error field."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="GET" if data is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from None


def wait_for_server(base: str, timeout_s: float) -> float:
    """Poll /health until it returns 200. Returns seconds waited."""
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"server at {base} not healthy after {timeout_s:.0f}s")
        time.sleep(HEALTH_POLL_S)


def server_facts(base: str) -> dict:
    """SGLang version and served model, from the server itself (endpoint names differ
    slightly between SGLang versions, so both spellings are tried)."""
    facts = {}
    for path in ("/get_server_info", "/server_info"):
        try:
            info = http_json(base + path)
            facts["sglang_version"] = info.get("version", "unknown")
            facts["server_args"] = {k: info.get(k) for k in (
                "model_path", "revision", "dtype", "context_length", "mem_fraction_static",
                "random_seed", "max_running_requests", "tp_size") if k in info}
            break
        except Exception as e:
            facts["server_info_error"] = repr(e)
    try:
        facts["model_info"] = http_json(base + "/get_model_info")
    except Exception as e:
        facts["model_info_error"] = repr(e)
    if "sglang_version" not in facts:  # fall back to the installed package, if any
        try:
            import sglang
            facts["sglang_version"] = sglang.__version__
        except Exception:
            facts["sglang_version"] = "unknown"
    return facts


# ---------------------------------------------------------------- tokenizer and checks
def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer  # inside the SGLang container / local .venv
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
    got = hashlib.sha256(tok.chat_template.encode()).hexdigest()
    if got != cfg["model"]["chat_template_sha256"]:
        sys.exit(f"Chat template hash {got} != pinned {cfg['model']['chat_template_sha256']}. "
                 "The tokenizer is not the pinned one; refusing to run.")
    return tok


def check_parity(tok, questions: list[dict], cfg: dict) -> None:
    """Gate 3 again, at run time, on real questions: our text must equal the official one."""
    for q in questions[:3]:
        opts, _, _ = prompts.shuffle_options(q["qid"], 0, q["correct"], q["wrong"], cfg["prompt"]["shuffle_salt"])
        for mode in cfg["prompt"]["modes"]:
            rep = prompts.check_against_tokenizer(tok, mode, q["question"], opts)
            if not (rep["match"] and rep["ids_match"]):
                sys.exit(f"Template parity FAILED for {q['qid']} mode={mode}: {rep}")


# ---------------------------------------------------------------- planning and resuming
def plan_jobs(questions: list[dict], n_samples: int, modes: list[str]) -> list[tuple]:
    """All (question, sample_idx, mode) to generate, in a stable order."""
    return [(q, s, m) for q in questions for s in range(n_samples) for m in modes]


def existing_keys(path: Path) -> tuple[set, set]:
    """(keys with a successful record, keys whose records are all errors) in a run log."""
    ok, err = set(), set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written last line from a killed job; score.py reports it
                key = (r.get("qid"), r.get("sample_idx"), r.get("mode"))
                (err if r.get("error") else ok).add(key)
    return ok, err - ok


# ---------------------------------------------------------------- one generation
def build_request(q: dict, sample_idx: int, mode: str, tok, cfg: dict, max_new: int) -> dict:
    """Everything known before the request is sent (it all goes into the record)."""
    salt = cfg["prompt"]["shuffle_salt"]
    options, gold, order = prompts.shuffle_options(q["qid"], sample_idx, q["correct"], q["wrong"], salt)
    text = prompts.build_text(mode, q["question"], options)
    ids = tok(text, add_special_tokens=False)["input_ids"]  # template text already has its markers
    sampling = {k: cfg["sampling"][k] for k in SAMPLING_KEYS}
    sampling["max_new_tokens"] = max_new
    # Keep every generated token in SGLang's own text (e.g. </think>) and don't add spaces
    # around special tokens; we still decode the ids ourselves (see run_one).
    sampling["skip_special_tokens"] = False
    sampling["spaces_between_special_tokens"] = False
    return {"qid": q["qid"], "sample_idx": sample_idx, "mode": mode, "gold": gold,
            "option_order": order, "seed": prompts.shuffle_seed(salt, q["qid"], sample_idx),
            "input_text": text, "input_token_ids": ids, "sampling_params": sampling}


def run_one(req: dict, base: str, tok, think_close_id: int, timeout: float) -> dict:
    """Send one request; return the response-derived fields (or the error)."""
    payload = {"input_ids": req["input_token_ids"], "sampling_params": req["sampling_params"],
               "return_logprob": True, "stream": False}
    t0 = time.perf_counter()
    started = utc_now()
    try:
        resp = http_json(base + "/generate", payload, timeout=timeout)
        wall = time.perf_counter() - t0
        meta = resp.get("meta_info", {})
        lps = meta.get("output_token_logprobs")  # [[logprob, token_id, text|None], ...]
        if resp.get("output_ids") is not None:
            out_ids, ids_source = list(resp["output_ids"]), "output_ids"
        elif lps is not None:
            out_ids, ids_source = [int(x[1]) for x in lps], "output_token_logprobs"
        else:
            raise RuntimeError(f"response has neither output_ids nor output_token_logprobs: keys={list(resp)}")
        # The token ids are the ground truth; the text is decoded from them with the same
        # pinned tokenizer, keeping special tokens, so text and ids can never disagree.
        out_text = tok.decode(out_ids, skip_special_tokens=False)
        fr = meta.get("finish_reason")
        finish = fr.get("type") if isinstance(fr, dict) else fr
        # SGLang's text omits the stop token it matched (e.g. <|im_end|>); ours keeps every id.
        # For the agreement check, compare against our decode without that final stop token.
        matched = fr.get("matched") if isinstance(fr, dict) else None
        trimmed = out_ids[:-1] if finish == "stop" and out_ids and out_ids[-1] == matched else out_ids
        text_for_check = tok.decode(trimmed, skip_special_tokens=False)
        if req["mode"] == "think":
            closed = think_close_id in out_ids
            n_think = out_ids.index(think_close_id) + 1 if closed else len(out_ids)
        else:
            closed, n_think = None, 0
        checks = {
            "ids_source": ids_source,
            "server_text_matches_decode": resp.get("text") == text_for_check if "text" in resp else None,
            "completion_tokens_match": meta.get("completion_tokens") in (None, len(out_ids)),
            "prompt_tokens_match": meta.get("prompt_tokens") in (None, len(req["input_token_ids"])),
        }
        return {
            "output_text": out_text, "output_token_ids": out_ids, "output_logprobs": lps,
            "finish_reason": finish, "n_output_tokens": len(out_ids), "n_think_tokens": n_think,
            "think_closed": closed, "wall_time_s": round(wall, 3), "request_started_at": started,
            "server_text": None if checks["server_text_matches_decode"] in (True, None) else resp.get("text"),
            "sglang_meta": {k: v for k, v in meta.items() if k not in ("output_token_logprobs", "input_token_logprobs")},
            "response_checks": checks, "error": None,
        }
    except Exception as e:
        return {
            "output_text": None, "output_token_ids": None, "output_logprobs": None,
            "finish_reason": None, "n_output_tokens": None, "n_think_tokens": None,
            "think_closed": None, "wall_time_s": round(time.perf_counter() - t0, 3),
            "request_started_at": started, "server_text": None, "sglang_meta": None,
            "response_checks": None,
            "error": f"{type(e).__name__}: {e}",
            "error_traceback": traceback.format_exc(limit=3),
        }


# ---------------------------------------------------------------- run metadata
def git_state() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def append_jsonl(path: Path, obj: dict) -> None:
    """Append one line and force it to disk, so a killed job loses at most the line in flight."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_run_dir(run_dir: Path, cfg: dict) -> None:
    """A run directory belongs to exactly one config: refuse to mix settings in one run."""
    meta = run_dir / "run_meta.jsonl"
    if not meta.exists():
        return
    with meta.open() as f:
        first = json.loads(f.readline())
    if first.get("config_sha256") != cfg["_meta"]["sha256"]:
        sys.exit(f"{run_dir} was started with config {first.get('config_sha256')}, current config is "
                 f"{cfg['_meta']['sha256']}. Use a new --run-id (CLAUDE.md Rule 3).")


# ---------------------------------------------------------------- main
def parse_args(argv):
    ap = argparse.ArgumentParser(description="Send control-run prompts to SGLang and log every generation.")
    ap.add_argument("--run-id", required=True, help="output folder name under runs/")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None, help="questions jsonl (default: [data] output_path)")
    ap.add_argument("--server-url", default=None, help="default http://[server] host:port")
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--pilot", action="store_true", help="first [generation] pilot_n_questions questions x pilot samples")
    ap.add_argument("--retry-errors", action="store_true", help="re-send keys whose only records are errors")
    ap.add_argument("--concurrency", type=int, default=None, help="override [generation] concurrency (speed only)")
    # Overrides that change WHAT is generated: only for smoke tests, refused for control/pilot runs.
    ap.add_argument("--limit-questions", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    gen = cfg["generation"]

    overrides = {k: v for k, v in (("limit_questions", args.limit_questions), ("samples", args.samples),
                                   ("max_new_tokens", args.max_new_tokens)) if v is not None}
    if overrides and args.run_id.startswith(PROTECTED_PREFIXES):
        sys.exit(f"Overrides {overrides} are not allowed for run id {args.run_id!r}; use e.g. smoke_<id>.")

    questions = load_questions(args.data or (REPO_ROOT / cfg["data"]["output_path"]))
    n_q = gen["pilot_n_questions"] if args.pilot else len(questions)
    n_q = min(n_q, args.limit_questions or n_q)
    n_samples = args.samples or (gen["pilot_samples_per_question"] if args.pilot else gen["samples_per_question"])
    max_new = args.max_new_tokens or gen["max_new_tokens"]
    concurrency = args.concurrency or gen["concurrency"]
    selected = questions[:n_q]

    # --- safety checks (before touching the run directory)
    tok = load_tokenizer(cfg)
    check_parity(tok, selected, cfg)
    think_close_id = tok.convert_tokens_to_ids("</think>")
    base = args.server_url or f"http://{cfg['server']['host']}:{cfg['server']['port']}"
    waited = wait_for_server(base, gen["server_wait_timeout_s"])
    facts = server_facts(base)
    served = (facts.get("model_info") or {}).get("model_path")
    if served and served.rstrip("/").split("/")[-1] != cfg["model"]["name"].split("/")[-1]:
        sys.exit(f"Server is serving {served!r}, config expects {cfg['model']['name']!r}.")

    run_dir = Path(args.runs_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    check_run_dir(run_dir, cfg)
    if not (run_dir / "control.toml").exists():
        shutil.copy(cfg["_meta"]["path"], run_dir / "control.toml")
    log_path = run_dir / "generations.jsonl"

    ok_keys, err_keys = existing_keys(log_path)
    skip = ok_keys | (set() if args.retry_errors else err_keys)
    jobs = [j for j in plan_jobs(selected, n_samples, cfg["prompt"]["modes"])
            if (j[0]["qid"], j[1], j[2]) not in skip]
    n_planned = len(selected) * n_samples * len(cfg["prompt"]["modes"])

    gpus = os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    t_start = time.time()
    append_jsonl(run_dir / "run_meta.jsonl", {
        "event": "start", "time": utc_now(), "run_id": args.run_id, "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpus": gpus,
        "git": git_state(), "config_sha256": cfg["_meta"]["sha256"], "prompt_hash": prompts.PROMPT_HASH,
        "chat_template_sha256": cfg["model"]["chat_template_sha256"], "server_url": base,
        "server_wait_s": round(waited, 1), **facts, "pilot": args.pilot, "overrides": overrides,
        "n_questions": n_q, "samples_per_question": n_samples, "max_new_tokens": max_new,
        "concurrency": concurrency, "n_planned": n_planned, "n_already_done": n_planned - len(jobs),
        "n_to_send": len(jobs), "qids": [q["qid"] for q in selected],
    })
    print(f"[generate] {args.run_id}: {len(jobs)} to send ({n_planned - len(jobs)} already done), "
          f"{n_q} questions x {n_samples} samples x {len(cfg['prompt']['modes'])} modes, "
          f"max_new_tokens={max_new}, concurrency={concurrency}, server {facts.get('sglang_version')}", flush=True)

    static = {"run_id": args.run_id, "model": cfg["model"]["name"], "model_revision": cfg["model"]["revision"],
              "chat_template_hash": cfg["model"]["chat_template_sha256"][:16], "prompt_hash": prompts.PROMPT_HASH,
              "sglang_version": facts.get("sglang_version"), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    n_done = n_err = out_tokens = 0

    def end_event(interrupted: bool) -> None:
        append_jsonl(run_dir / "run_meta.jsonl", {
            "event": "end", "time": utc_now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "interrupted": interrupted, "elapsed_s": round(time.time() - t_start, 1),
            "n_written": n_done, "n_errors": n_err, "output_tokens": out_tokens})

    # Slurm sends SIGTERM at the time limit. Turn it into an exception in the main thread so the
    # segment still gets an "end" event (its GPU time is then counted), then exit at once:
    # finished records are already on disk; requests in flight are simply re-sent on resume.
    def on_term(signum, _frame):
        raise Interrupted(signum)
    previous_handler = signal.signal(signal.SIGTERM, on_term)

    # Requests run in worker threads; only this main thread writes the log, so lines never interleave.
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = {}
        for q, s, m in jobs:
            req = build_request(q, s, m, tok, cfg, max_new)
            futures[pool.submit(run_one, req, base, tok, think_close_id, gen["request_timeout_s"])] = req
        for fut in as_completed(futures):
            req = futures[fut]
            res = fut.result()
            rec = {**static, **req, "n_input_tokens": len(req["input_token_ids"]), **res}
            missing = [k for k in GENERATION_FIELDS if k not in rec]
            assert not missing, f"record missing schema fields {missing}"  # programming error, not data
            append_jsonl(log_path, rec)
            n_done += 1
            n_err += rec["error"] is not None
            out_tokens += rec["n_output_tokens"] or 0
            if n_done % PROGRESS_EVERY == 0 or n_done == len(jobs):
                el = time.time() - t_start
                print(f"[generate] {n_done}/{len(jobs)} done, {n_err} errors, {out_tokens} output tokens, "
                      f"{el / 60:.1f} min, {out_tokens / max(el, 1e-9):.0f} tok/s", flush=True)
    except (Interrupted, KeyboardInterrupt):
        end_event(interrupted=True)
        print(f"[generate] interrupted after {n_done} records; re-run the same command to resume", flush=True)
        os._exit(143)  # don't wait for in-flight HTTP requests (they may take an hour)
    pool.shutdown()
    signal.signal(signal.SIGTERM, previous_handler)  # leave the process as we found it
    end_event(interrupted=False)
    print(f"[generate] finished: {n_done} written, {n_err} errors -> {log_path}", flush=True)
    return 0 if n_err == 0 else 3  # non-zero exit makes errors visible in the Slurm job state


if __name__ == "__main__":
    sys.exit(main())
