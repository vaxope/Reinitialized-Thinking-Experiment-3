"""Compare a trained model's evaluation with the control run, mode by mode.

Purpose : the headline table of the training experiment: for each mode (think, default, reinit),
          the untrained baseline (control_v1) next to the trained model on the SAME questions,
          samples, option orders and settings:
            accuracy (rule v3, the official one) with its 95% CI, clustered by question as in
            src/score.py; accuracy (lenient) with CI; truncation rate; output tokens (median,
            p90, mean); median output tokens of correct answers; and the trained-minus-control
            change for each.
          Every number is recomputed from the two scored.jsonl files (src/score.py output),
          nothing is typed in by hand.
Checks  : both runs must cover the same (qid, sample_idx, mode) keys and use the same scoring
          rule; otherwise the comparison is refused.
Inputs  : runs/<baseline>/scored.jsonl, runs/<trained>/scored.jsonl, configs/control.toml.
Output  : a Markdown table on stdout (paste into the vault note); --json for machine use.
Pipeline: eval_trained.sbatch -> score.py -> [compare_runs] -> vault report.
Usage   : python3 -m src.compare_runs runs/control_v1 runs/eval_train_reinit_v1_step80 [--json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from src.config import load_config
from src.score import clustered_ci, wilson_ci

P90 = 0.90   # the token percentile reported beside the median (as in the report)


def load_scored(run_dir: Path) -> list[dict]:
    with (Path(run_dir) / "scored.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: list[float], q: float) -> float:
    v = sorted(values)
    if not v:
        return float("nan")
    i = q * (len(v) - 1)            # linear interpolation between closest ranks
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def accuracy(recs: list[dict], field: str, level: float) -> tuple[float, list[float]]:
    by_q = defaultdict(list)
    for r in recs:
        by_q[r["qid"]].append(1.0 if r.get(field) else 0.0)
    k, n = sum(sum(v) for v in by_q.values()), len(recs)
    if max(len(v) for v in by_q.values()) > 1:
        ci = clustered_ci([statistics.fmean(v) for v in by_q.values()], level)
    else:
        ci = wilson_ci(int(k), n, level)
    return k / n, list(ci)


def mode_stats(recs: list[dict], level: float) -> dict:
    toks = [r["n_output_tokens"] or 0 for r in recs]
    acc, ci = accuracy(recs, "correct", level)
    acc_l, ci_l = accuracy(recs, "lenient_correct", level)
    correct_toks = [r["n_output_tokens"] or 0 for r in recs if r["correct"]]
    return {"n": len(recs), "accuracy": acc, "ci": ci, "accuracy_lenient": acc_l, "ci_lenient": ci_l,
            "truncation_rate": sum(bool(r["truncated"]) for r in recs) / len(recs),
            "tokens_median": statistics.median(toks), "tokens_p90": percentile(toks, P90),
            "tokens_mean": statistics.fmean(toks),
            "tokens_median_correct": statistics.median(correct_toks) if correct_toks else float("nan")}


def compare(base: list[dict], trained: list[dict], modes: list[str], level: float) -> dict:
    key = lambda r: (r["qid"], r["sample_idx"], r["mode"])   # noqa: E731
    kb, kt = {key(r) for r in base}, {key(r) for r in trained}
    if kb != kt:
        raise ValueError(f"the runs cover different generations: {len(kb - kt)} only in the baseline, "
                         f"{len(kt - kb)} only in the trained run")
    rules = {r.get("scoring_rule_version") for r in base} | {r.get("scoring_rule_version") for r in trained}
    if len(rules) != 1:
        raise ValueError(f"the runs were scored with different rules: {sorted(map(str, rules))}")
    out = {}
    for m in modes:
        b = mode_stats([r for r in base if r["mode"] == m], level)
        t = mode_stats([r for r in trained if r["mode"] == m], level)
        out[m] = {"baseline": b, "trained": t, "change": {k: t[k] - b[k] for k in b if not isinstance(b[k], list)}}
    return out


def markdown(res: dict, names: tuple[str, str]) -> str:
    pct = lambda x: f"{100 * x:.1f}%"                         # noqa: E731
    ci = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"    # noqa: E731
    lines = [f"Baseline = `{names[0]}`, trained = `{names[1]}`. Change = trained minus baseline "
             "(accuracy in percentage points).", "",
             "| Mode | Run | Accuracy (v3) [95% CI] | Accuracy (lenient) [95% CI] | Truncated | Median tokens | p90 tokens | Mean tokens | Median tokens, correct |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, r in res.items():
        for label, s in (("baseline", r["baseline"]), ("trained", r["trained"])):
            lines.append(f"| {m} | {label} | {pct(s['accuracy'])} {ci(s['ci'])} | {pct(s['accuracy_lenient'])} "
                         f"{ci(s['ci_lenient'])} | {pct(s['truncation_rate'])} | {s['tokens_median']:,.0f} | "
                         f"{s['tokens_p90']:,.0f} | {s['tokens_mean']:,.0f} | {s['tokens_median_correct']:,.0f} |")
        c = r["change"]
        lines.append(f"| {m} | **change** | {100 * c['accuracy']:+.1f} pp | {100 * c['accuracy_lenient']:+.1f} pp | "
                     f"{100 * c['truncation_rate']:+.1f} pp | {c['tokens_median']:+,.0f} | {c['tokens_p90']:+,.0f} | "
                     f"{c['tokens_mean']:+,.0f} | {c['tokens_median_correct']:+,.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("baseline", help="runs/control_v1")
    ap.add_argument("trained", help="runs/eval_<run>_step<N>")
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config) if a.config else load_config()
    try:
        res = compare(load_scored(Path(a.baseline)), load_scored(Path(a.trained)), cfg["prompt"]["modes"],
                      cfg["scoring"]["ci_level"])
    except ValueError as e:
        sys.exit(f"[compare_runs] refused: {e}")
    print(json.dumps(res, indent=2) if a.json else markdown(res, (Path(a.baseline).name, Path(a.trained).name)))


if __name__ == "__main__":
    main()
