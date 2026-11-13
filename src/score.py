"""Score a run and summarize it: generations.jsonl -> scored.jsonl + summary.json + summary.md.

Purpose : the only place where correctness is decided. Reads raw generation records
          (never modifies them), applies src/extract.py, and computes every metric of the
          control run per mode, plus a paired think-vs-default comparison.
Inputs  : runs/<run>/generations.jsonl (schema: src/schema.py / CLAUDE.md "Logging"),
          configs/control.toml [scoring] and [prompt].
Outputs : in the same folder (or --out-dir):
          scored.jsonl  - each input record + extracted_answer, extraction_rule, correct,
                          extraction_detail, outcome, truncated, scoring_rule_version
          summary.json  - all metrics, machine-readable (the viewer can also recompute them)
          summary.md    - the same as a readable table
Pipeline: (later) generate.py on DeltaAI -> copy back -> [score] -> viewer/index.html, vault note.

Usage:  python3 -m src.score runs/<run>/generations.jsonl [--overwrite]

Every record is counted. Errors, truncations and unreadable lines are reported, never skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src import extract
from src.config import load_config
from src.schema import missing_fields

# Outcome of one record, in the order they are checked. Exactly one applies.
OUTCOMES = ("error", "truncated", "no_answer", "wrong", "correct")


# ---------------------------------------------------------------- reading
def read_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (records, corrupt). A line that is not valid JSON is kept as a 'corrupt'
    entry with its line number, so a damaged log is visible in the summary."""
    records, corrupt = [], []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                corrupt.append({"line": lineno, "error": str(e)})
    return records, corrupt


# ---------------------------------------------------------------- scoring one record
def score_record(rec: dict, cfg: dict) -> dict:
    """Return a copy of rec with the score fields added. Decision order = OUTCOMES."""
    n_opt = cfg["prompt"]["n_options"]
    out = dict(rec)
    truncated = rec.get("finish_reason") == "length"  # hit max_new_tokens
    if rec.get("error") or rec.get("output_text") is None:
        detail = {"letter": None, "rule": "generation_error", "n_answer_fields": 0,
                  "n_near_misses": 0, "think_closed": None}
    else:
        detail = extract.extract_answer(rec["output_text"], rec["mode"], n_opt)
    letter = detail["letter"]

    if detail["rule"] == "generation_error":
        outcome = "error"
    elif truncated and cfg["scoring"]["truncated_counts_as_wrong"]:
        outcome = "truncated"
    elif letter is None:
        outcome = "no_answer"
    else:
        outcome = "correct" if letter == rec.get("gold") else "wrong"

    out.update({
        "extracted_answer": letter,          # what the extractor found (kept even if truncated)
        "extraction_rule": detail["rule"],   # which branch of the rule decided it
        "correct": outcome == "correct",
        "extraction_detail": detail,
        "outcome": outcome,
        "truncated": truncated,
        "scoring_rule_version": cfg["scoring"]["extraction_rule"],
    })
    return out


# ---------------------------------------------------------------- statistics helpers
def z_value(level: float) -> float:
    """Two-sided normal quantile, e.g. 0.95 -> 1.96."""
    return statistics.NormalDist().inv_cdf(0.5 + level / 2)


def wilson_ci(k: int, n: int, level: float) -> tuple[float, float]:
    """Wilson score interval for k successes in n independent trials. Better than
    p +/- z*sqrt(p(1-p)/n) near 0 or 1 and for small n; never leaves [0, 1]."""
    if n == 0:
        return (math.nan, math.nan)
    z, p = z_value(level), k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def clustered_ci(per_question: list[float], level: float) -> tuple[float, float]:
    """CI when there are several samples per question: samples of the same question are
    correlated, so treat each QUESTION's mean accuracy as one observation (n = #questions)."""
    n = len(per_question)
    if n < 2:
        return (math.nan, math.nan)
    m, sd = statistics.fmean(per_question), statistics.stdev(per_question)
    half = z_value(level) * sd / math.sqrt(n)
    return (max(0.0, m - half), min(1.0, m + half))


def describe(values: list[float]) -> dict:
    """n, mean, median, p95, min, max, total of a numeric list (None values excluded)."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    def pct(q):  # linear interpolation between closest ranks
        i = (len(v) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)
    return {"n": len(v), "mean": statistics.fmean(v), "median": pct(0.5), "p95": pct(0.95),
            "min": v[0], "max": v[-1], "total": sum(v)}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: under 'no difference', each discordant pair is
    equally likely to favour either mode, so min(b, c) ~ Binomial(b + c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------- summary
def summarize_mode(recs: list[dict], level: float) -> dict:
    n = len(recs)
    n_correct = sum(r["correct"] for r in recs)
    by_q = defaultdict(list)
    for r in recs:
        by_q[r.get("qid")].append(1.0 if r["correct"] else 0.0)
    max_samples = max((len(v) for v in by_q.values()), default=0)
    if max_samples <= 1:
        ci, method = wilson_ci(n_correct, n, level), "wilson"
    else:
        ci, method = clustered_ci([statistics.fmean(v) for v in by_q.values()], level), "clustered_by_question"

    out_tok = [r.get("n_output_tokens") for r in recs]
    think_tok = [r.get("n_think_tokens") for r in recs]
    visible_tok = [(r["n_output_tokens"] - (r.get("n_think_tokens") or 0))
                   if r.get("n_output_tokens") is not None else None for r in recs]
    tok_per_s = [r["n_output_tokens"] / r["wall_time_s"] for r in recs
                 if r.get("n_output_tokens") and r.get("wall_time_s")]
    return {
        "n_records": n,
        "n_questions": len(by_q),
        "samples_per_question_max": max_samples,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else math.nan,  # denominator = ALL records, incl. errors/truncations
        "ci": list(ci), "ci_method": method, "ci_level": level,
        "outcomes": dict(Counter(r["outcome"] for r in recs)),
        "extraction_rules": dict(Counter(r["extraction_rule"] for r in recs)),
        # Format failures: answered, but not in the requested "answer": "X" form.
        "n_near_miss_only": sum(r["extraction_rule"] == "no_answer_field"
                                and r["extraction_detail"].get("n_near_misses", 0) > 0 for r in recs),
        # Letter bias check: the prompt's example is "C"; a model that over-picks C shows here.
        "extracted_letter_counts": dict(sorted(Counter(r["extracted_answer"] for r in recs
                                                       if r["extracted_answer"]).items())),
        "gold_letter_counts": dict(sorted(Counter(r.get("gold") for r in recs if r.get("gold")).items())),
        "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in recs)),
        "truncation_rate": sum(r["truncated"] for r in recs) / n if n else math.nan,
        "n_errors": sum(r["outcome"] == "error" for r in recs),
        "tokens_output": describe(out_tok),
        "tokens_think": describe(think_tok),
        "tokens_visible": describe(visible_tok),
        "tokens_input": describe([r.get("n_input_tokens") for r in recs]),
        "tokens_output_when_correct": describe([r.get("n_output_tokens") for r in recs if r["correct"]]),
        "tokens_output_when_not_correct": describe([r.get("n_output_tokens") for r in recs if not r["correct"]]),
        "wall_time_s": describe([r.get("wall_time_s") for r in recs]),
        # Per-request decode speed. With many concurrent requests, total GPU time is NOT the
        # sum of wall times; GPU-hours need the run's start/end time (from generate.py, later).
        "output_tokens_per_s_per_request": describe(tok_per_s),
    }


def paired(scored: list[dict], modes: list[str]) -> dict | None:
    """think vs default on the SAME (qid, sample_idx): same question, same option order."""
    if len(modes) != 2:
        return None
    a, b = modes
    cell = {(r.get("qid"), r.get("sample_idx"), r["mode"]): r["correct"] for r in scored}
    keys = {(q, s) for (q, s, m) in cell}
    both = [(cell[(q, s, a)], cell[(q, s, b)]) for (q, s) in keys
            if (q, s, a) in cell and (q, s, b) in cell]
    only_a = sum(x and not y for x, y in both)
    only_b = sum(y and not x for x, y in both)
    return {"modes": [a, b], "n_pairs": len(both),
            "both_correct": sum(x and y for x, y in both),
            f"only_{a}_correct": only_a, f"only_{b}_correct": only_b,
            "both_wrong": sum(not x and not y for x, y in both),
            "mcnemar_exact_p": mcnemar_exact(only_a, only_b),
            "note": "pairs share qid and sample_idx; with >1 sample per question, pairs are not independent"}


def summarize(scored: list[dict], corrupt: list[dict], cfg: dict, source: str) -> dict:
    level = cfg["scoring"]["ci_level"]
    modes = [m for m in cfg["prompt"]["modes"] if any(r.get("mode") == m for r in scored)]
    unknown_modes = sorted({str(r.get("mode")) for r in scored} - set(cfg["prompt"]["modes"]))
    dup = Counter((r.get("run_id"), r.get("qid"), r.get("mode"), r.get("sample_idx")) for r in scored)
    schema = Counter(f for r in scored for f in missing_fields(r))
    return {
        "source": source,
        "run_ids": sorted({str(r.get("run_id")) for r in scored}),
        "n_records": len(scored),
        "corrupt_lines": corrupt,
        "duplicate_keys": sum(c - 1 for c in dup.values() if c > 1),
        "missing_schema_fields": dict(schema),
        "unknown_modes": unknown_modes,
        "scoring": {"rule": cfg["scoring"]["extraction_rule"],
                    "truncated_counts_as_wrong": cfg["scoring"]["truncated_counts_as_wrong"],
                    "config_sha256": cfg["_meta"]["sha256"]},
        "per_mode": {m: summarize_mode([r for r in scored if r["mode"] == m], level) for m in modes},
        "paired": paired(scored, modes),
    }


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def to_markdown(s: dict) -> str:
    """A readable table of the summary. Every number comes from summary.json."""
    modes = list(s["per_mode"])
    pm = s["per_mode"]
    def row(label, f):
        return f"| {label} | " + " | ".join(f(pm[m]) for m in modes) + " |"
    lines = [
        f"# Summary: {', '.join(s['run_ids'])}",
        "",
        f"Source `{s['source']}` · {s['n_records']} records · rule `{s['scoring']['rule']}` · "
        f"truncated counts as wrong: {s['scoring']['truncated_counts_as_wrong']}",
        "",
        "| metric | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
        row("accuracy", lambda d: f"{d['accuracy']:.3f} [{fmt(d['ci'][0], 3)}, {fmt(d['ci'][1], 3)}] ({d['ci_method']})"),
        row("correct / records", lambda d: f"{d['n_correct']} / {d['n_records']}"),
        row("questions × max samples", lambda d: f"{d['n_questions']} × {d['samples_per_question_max']}"),
        row("outcomes", lambda d: ", ".join(f"{k} {v}" for k, v in sorted(d["outcomes"].items()))),
        row("extraction rules", lambda d: ", ".join(f"{k} {v}" for k, v in sorted(d["extraction_rules"].items()))),
        row("no field, but a look-alike answer", lambda d: str(d["n_near_miss_only"])),
        row("extracted letters (gold)", lambda d: ", ".join(f"{k} {v}" for k, v in d["extracted_letter_counts"].items()) + " (" + ", ".join(f"{k} {v}" for k, v in d["gold_letter_counts"].items()) + ")"),
        row("finish reasons", lambda d: ", ".join(f"{k} {v}" for k, v in sorted(d["finish_reasons"].items()))),
        row("truncation rate", lambda d: fmt(d["truncation_rate"], 3)),
        row("output tokens mean / median / p95 / max", lambda d: " / ".join(fmt(d["tokens_output"].get(k), 0) for k in ("mean", "median", "p95", "max"))),
        row("think tokens mean / median", lambda d: " / ".join(fmt(d["tokens_think"].get(k), 0) for k in ("mean", "median"))),
        row("visible tokens mean / median", lambda d: " / ".join(fmt(d["tokens_visible"].get(k), 0) for k in ("mean", "median"))),
        row("output tokens total", lambda d: fmt(d["tokens_output"].get("total"), 0)),
        row("output tokens mean: correct vs not", lambda d: f"{fmt(d['tokens_output_when_correct'].get('mean'), 0)} vs {fmt(d['tokens_output_when_not_correct'].get('mean'), 0)}"),
        row("input tokens mean", lambda d: fmt(d["tokens_input"].get("mean"), 0)),
        row("wall time s mean / median / max", lambda d: " / ".join(fmt(d["wall_time_s"].get(k), 1) for k in ("mean", "median", "max"))),
        row("tokens/s per request (median)", lambda d: fmt(d["output_tokens_per_s_per_request"].get("median"), 1)),
    ]
    p = s["paired"]
    if p:
        a, b = p["modes"]
        lines += ["", f"**Paired ({a} vs {b}, {p['n_pairs']} pairs):** both correct {p['both_correct']} · "
                  f"only {a} {p[f'only_{a}_correct']} · only {b} {p[f'only_{b}_correct']} · both wrong {p['both_wrong']} · "
                  f"exact McNemar p = {p['mcnemar_exact_p']:.3g}. _{p['note']}._"]
    problems = []
    if s["corrupt_lines"]: problems.append(f"{len(s['corrupt_lines'])} unreadable lines")
    if s["duplicate_keys"]: problems.append(f"{s['duplicate_keys']} duplicate (run, qid, mode, sample) keys")
    if s["missing_schema_fields"]: problems.append(f"missing schema fields: {s['missing_schema_fields']}")
    if s["unknown_modes"]: problems.append(f"records with unknown modes: {s['unknown_modes']}")
    lines += ["", "**Data problems:** " + ("; ".join(problems) if problems else "none")]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score generations.jsonl and write a summary.")
    ap.add_argument("generations", help="path to runs/<run>/generations.jsonl")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=None, help="default: same folder as the input")
    ap.add_argument("--overwrite", action="store_true", help="replace existing scored/summary files")
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    if cfg["scoring"]["extraction_rule"] != extract.RULE_NAME:  # config and code must agree
        sys.exit(f"config rule {cfg['scoring']['extraction_rule']!r} != src/extract.py {extract.RULE_NAME!r}")
    src = Path(args.generations)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    targets = [out_dir / n for n in ("scored.jsonl", "summary.json", "summary.md")]
    if any(t.exists() for t in targets) and not args.overwrite:
        sys.exit(f"Outputs exist in {out_dir}; pass --overwrite to replace them.")

    records, corrupt = read_jsonl(src)
    scored = [score_record(r, cfg) for r in records]
    summary = summarize(scored, corrupt, cfg, str(src))
    out_dir.mkdir(parents=True, exist_ok=True)
    with targets[0].open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    targets[1].write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    targets[2].write_text(to_markdown(summary))
    print(to_markdown(summary))
    print(f"Wrote {', '.join(str(t) for t in targets)}")


if __name__ == "__main__":
    main()
