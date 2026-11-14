"""Check a generation run for errors and inconsistencies (the smoke test's pass/fail step).

Purpose : after a run (smoke test, pilot, full), verify mechanically that the log is sound
          BEFORE anyone looks at accuracy. Prints one line per check and PASS/FAIL; exits 1 on
          any FAIL. Prints no GPQA text (only ids, counts and the last characters of outputs).
Inputs  : runs/<run_id>/ (generations.jsonl, run_meta.jsonl), configs/control.toml,
          data/gpqa_diamond.jsonl (to rebuild every prompt and compare).
Pipeline: generate.py -> [check_run] -> score.py -> viewer
Checks  (FAIL):
  - run_meta: every job segment has a start and an end event; one config hash throughout
  - every record has all schema fields, a known mode, and no error
  - input_text == prompts.build_text(...) rebuilt from the data file with the logged option
    order, and gold letter consistent with that order
  - input_token_ids == tokenizer(input_text) (if transformers is available)
  - n_input_tokens / n_output_tokens equal the lengths of the logged id lists
  - finish_reason is "stop" or "length"; SGLang's own token counts agree with ours
  - think mode: n_think_tokens counts up to the first </think>; default mode: 0
  - every planned (qid, sample, mode) of the run is present exactly once
Warnings (do not fail): server text differs from our decode; think-mode output that stopped
without closing </think>; default-mode output containing </think>; truncations.
Usage: python -m src.check_run runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src import prompts
from src.config import REPO_ROOT, load_config
from src.data import load_questions
from src.schema import GENERATION_FIELDS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = load_config(args.config or (run_dir / "control.toml" if (run_dir / "control.toml").exists() else None) or REPO_ROOT / "configs" / "control.toml")
    questions = {q["qid"]: q for q in load_questions(args.data or REPO_ROOT / cfg["data"]["output_path"])}
    fails, warns = [], []

    def fail(msg): fails.append(msg)
    def warn(msg): warns.append(msg)

    # --- run metadata
    meta = [json.loads(l) for l in (run_dir / "run_meta.jsonl").read_text().splitlines() if l.strip()]
    starts = [m for m in meta if m["event"] == "start"]
    ends = [m for m in meta if m["event"] == "end"]
    if n_int := sum(bool(m.get("interrupted")) for m in ends):
        warn(f"{n_int} job segment(s) were interrupted by SIGTERM (time limit) and resumed")
    if len(starts) != len(ends):
        warn(f"{len(starts)} start events but {len(ends)} end events (a job segment was killed before finishing)")
    if len({m["config_sha256"] for m in starts}) != 1:
        fail("run_meta: more than one config hash in this run")
    last = starts[-1]
    print(f"run {last['run_id']}: {len(starts)} segment(s), sglang {last.get('sglang_version')}, "
          f"{last['n_questions']} questions x {last['samples_per_question']} samples, max_new_tokens {last['max_new_tokens']}")

    # --- records
    recs = []
    for i, line in enumerate((run_dir / "generations.jsonl").read_text().splitlines(), 1):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            fail(f"line {i}: not valid JSON")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
        think_close = tok.convert_tokens_to_ids("</think>")
    except Exception as e:
        tok, think_close = None, None
        warn(f"tokenizer not available ({type(e).__name__}); token-id checks skipped")

    salt = cfg["prompt"]["shuffle_salt"]
    for r in recs:
        k = f"{r.get('qid')}/s{r.get('sample_idx')}/{r.get('mode')}"
        if missing := [f for f in GENERATION_FIELDS if f not in r]:
            fail(f"{k}: missing fields {missing}"); continue
        if r["mode"] not in cfg["prompt"]["modes"]:
            fail(f"{k}: unknown mode")
        if r["error"]:
            fail(f"{k}: error: {r['error'][:200]}"); continue
        q = questions.get(r["qid"])
        if q is None:
            fail(f"{k}: qid not in data file")
        else:
            opts, gold, order = prompts.shuffle_options(r["qid"], r["sample_idx"], q["correct"], q["wrong"], salt)
            if order != r["option_order"] or gold != r["gold"]:
                fail(f"{k}: option order/gold differ from a fresh shuffle")
            if r["input_text"] != prompts.build_text(r["mode"], q["question"], opts):
                fail(f"{k}: input_text differs from prompts.build_text")
        if tok is not None and tok(r["input_text"], add_special_tokens=False)["input_ids"] != r["input_token_ids"]:
            fail(f"{k}: input_token_ids differ from tokenizing input_text")
        if r["n_input_tokens"] != len(r["input_token_ids"]) or r["n_output_tokens"] != len(r["output_token_ids"]):
            fail(f"{k}: token counts differ from id list lengths")
        if r["finish_reason"] not in ("stop", "length"):
            fail(f"{k}: finish_reason {r['finish_reason']!r}")
        chk = r.get("response_checks") or {}
        if chk.get("completion_tokens_match") is False or chk.get("prompt_tokens_match") is False:
            fail(f"{k}: SGLang token counts disagree with ours")
        if chk.get("server_text_matches_decode") is False:
            warn(f"{k}: SGLang text differs from our decode of the ids (stored in server_text)")
        if r["mode"] == "think" and think_close is not None:
            ids = r["output_token_ids"]
            expect = ids.index(think_close) + 1 if think_close in ids else len(ids)
            if r["n_think_tokens"] != expect:
                fail(f"{k}: n_think_tokens {r['n_think_tokens']} != {expect}")
            if r["finish_reason"] == "stop" and think_close not in ids:
                warn(f"{k}: stopped without closing </think>")
        if r["mode"] == "default":
            if r["n_think_tokens"] != 0:
                fail(f"{k}: default mode with n_think_tokens {r['n_think_tokens']}")
            if "</think>" in (r["output_text"] or ""):
                warn(f"{k}: default-mode output contains </think>")
        if r["finish_reason"] == "length":
            warn(f"{k}: truncated at {r['n_output_tokens']} tokens")

    # --- completeness: every planned key present (latest attempt counts)
    planned = {(qid, s, m) for qid in last["qids"] for s in range(last["samples_per_question"]) for m in cfg["prompt"]["modes"]}
    ok_keys = Counter((r["qid"], r["sample_idx"], r["mode"]) for r in recs if not r.get("error"))
    if missing := planned - set(ok_keys):
        fail(f"{len(missing)} planned generations have no successful record, e.g. {sorted(missing)[:3]}")
    if dup := [k for k, c in ok_keys.items() if c > 1]:
        fail(f"{len(dup)} keys have more than one successful record, e.g. {dup[:3]}")

    # --- short per-mode overview (no question text)
    for m in cfg["prompt"]["modes"]:
        rs = [r for r in recs if r["mode"] == m and not r.get("error")]
        if rs:
            n_out = sorted(r["n_output_tokens"] for r in rs)
            print(f"  {m:8s} {len(rs)} ok | output tokens median {n_out[len(n_out) // 2]}, max {n_out[-1]} | "
                  f"truncated {sum(r['finish_reason'] == 'length' for r in rs)} | "
                  f"ends: {rs[0]['output_text'][-60:]!r}")
    for w in warns[:30]:
        print("  WARN", w)
    for f in fails[:50]:
        print("  FAIL", f)
    print(f"{'PASS' if not fails else 'FAIL'}: {len(recs)} records, {len(fails)} failures, {len(warns)} warnings")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
