"""Compare HOW two runs answer, mode by mode: the trajectory side of baseline vs trained model.

Purpose : the accuracy table (src/compare_runs.py) says whether the trained model is better; this
          says where its trajectories go right or wrong and why they keep going, with the same
          definitions as the control analysis (Report section 5.5), so the two runs compare one to
          one. Per mode (think, default, reinit), baseline next to trained:
            A. outcomes: correct / wrong / no answer / truncated;
            B. the shape of the final answer line (finished outputs): the requested strict
               "answer": "X" field, other answer lines, <answer> tags, \\boxed{}, prose
               ("the answer is X"), a bare letter, none of these;
            C. commitments in the reasoning (correct and wrong finished outputs): how many outputs
               commit to a letter, commitments per output (median), letter switches (mean), first
               letter wrong but final right / first right but final wrong, and, for correct
               outputs, tokens spent after first committing to the final letter (median);
            D. invented sources ("I found a source", "according to a paper", ...: the model cannot
               look anything up), share of correct vs wrong outputs;
            E. length: median tokens when correct / wrong; per-question length rank correlation
               between the runs; median token change on the same (question, sample);
            F. the same (question, sample) in both runs: fixed (wrong -> right) and broken
               (right -> wrong), with example record ids to open in the viewer.
Heuristics: "reasoning" = the text before the first </think> (think) or </analysis> (reinit; also
          stops at a </think>), the whole output in default mode. A commitment = a phrase like "the
          answer is X", "I'll go with X", "final answer: X" (COMMIT below). Character positions
          become tokens by the output's own tokens-per-character ratio (approximate). These are
          pattern counts: phrasings they miss are not counted, so treat them as estimates.
Privacy : prints counts and record ids (qid + sample) only: never question, option or output
          text (CLAUDE.md Rule 5).
Inputs  : runs/<baseline>/scored.jsonl, runs/<trained>/scored.jsonl (src/score.py), configs/control.toml.
Output  : Markdown on stdout; --json for machine use.
Pipeline: eval_trained.sbatch -> score.py -> [compare_trajectories] (+ compare_runs) -> Report.
Usage   : python3 -m src.compare_trajectories runs/control_v1 runs/eval_train_reinit_v1_step<N> [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.compare_runs import load_scored
from src.config import load_config

COMMIT = re.compile(r"(?:the (?:correct |final |best )?answer (?:is|must be|should be|would be)|so the answer is|"
                    r"I(?:'ll| will) (?:go with|choose|pick|select)|final answer[:\s]*(?:is)?|"
                    r"correct (?:option|choice) is|the answer:)\s*(?:option\s*)?[\*\(`\"']*([A-D])\b", re.I)
SOURCE = re.compile(r"I (?:found|find|recall|remember) (?:a|the|one) (?:source|paper|study|reference|article|website)|"
                    r"according to (?:a|the|one) (?:source|paper|study|reference|article|website)|"
                    r"I (?:searched|looked (?:it )?up|checked online)", re.I)
FORMATS = [  # (name, pattern on the final answer line); first match wins
    ("strict field", re.compile(r'^\W*"answer"\s*:\s*"[A-D]"\W*$')),
    ("<answer> tag", re.compile(r"<answer>\s*\(?[A-D]\)?\s*</answer>", re.I)),
    ("\\boxed{}", re.compile(r"\\boxed\{\s*\(?(?:\\text\{)?[A-D]\}?\)?\s*\}")),
    ("answer line", re.compile(r"^[\W_]*answer[\W_]*:[\W_]*\(?[A-D]\)?[\W_]*$", re.I)),
    ("answer key + more text", re.compile(r"answer[\W_]*:.*\b[A-D]\b", re.I)),
    ("prose", re.compile(r"answer is\W*(?:option\s*)?\(?[A-D]\b", re.I)),
    ("bare letter", re.compile(r"^[\W_]*\(?[A-D]\)?[\W_]*$")),
]
ANSWERISH = re.compile(r"answer|boxed|\b[A-D]\b", re.I)
N_EXAMPLES = 3


def reasoning(r: dict) -> str:
    t = r.get("output_text") or ""
    if r["mode"] == "think":
        i = t.find("</think>"); return t[:i] if i >= 0 else t
    if r["mode"] == "reinit":
        ends = [x for x in (t.find("</analysis>"), t.find("</think>")) if x >= 0]
        return t[:min(ends)] if ends else t
    return t


def visible(r: dict) -> str:
    t = r.get("output_text") or ""
    tag = {"think": "</think>", "reinit": "</analysis>"}.get(r["mode"])
    if tag:
        i = t.find(tag)
        return t[i + len(tag):] if i >= 0 else ""
    return t


def answer_format(r: dict) -> str:
    lines = [ln.strip() for ln in visible(r).replace("<|im_end|>", "").splitlines() if ln.strip()]
    for ln in reversed(lines[-8:]):
        if ANSWERISH.search(ln):
            for name, pat in FORMATS:
                if pat.search(ln):
                    return name
            return "other"
    return "none"


def outcome(r: dict) -> str:
    return r.get("outcome") or ("correct" if r["correct"] else "wrong")


def rid(r: dict) -> str:
    return f"{r['qid']} s{r['sample_idx']}"


def commit_stats(rs: list[dict]) -> dict:
    n_with, counts, switches, wrong_first, right_first, after = 0, [], [], 0, 0, []
    for r in rs:
        t = reasoning(r)
        cs = [(m.start(), m.group(1)) for m in COMMIT.finditer(t)]
        counts.append(len(cs))
        if not cs:
            continue
        n_with += 1
        switches.append(sum(a[1] != b[1] for a, b in zip(cs, cs[1:])))
        final = r.get("lenient_answer")
        if final:
            wrong_first += cs[0][1] != r["gold"] and final == r["gold"]
            right_first += cs[0][1] == r["gold"] and final != r["gold"]
        if r["correct"]:
            hits = [p for p, letter in cs if letter == r["extracted_answer"]]
            if hits:
                tpc = (r["n_output_tokens"] or 0) / max(1, len(r.get("output_text") or ""))
                after.append((len(t) - hits[0]) * tpc)
    return {"n": len(rs), "with_commitment": n_with,
            "commitments_median": statistics.median(counts) if counts else None,
            "switches_mean": statistics.fmean(switches) if switches else None,
            "first_wrong_final_right": wrong_first, "first_right_final_wrong": right_first,
            "tokens_after_first_commit_median": statistics.median(after) if after else None}


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1   # average rank for ties
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def mode_view(rs: list[dict]) -> dict:
    finished = [r for r in rs if not r.get("truncated")]
    correct = [r for r in rs if r["correct"]]
    wrong = [r for r in finished if not r["correct"] and r.get("lenient_answer")]
    tok = lambda g: statistics.median([r["n_output_tokens"] or 0 for r in g]) if g else None   # noqa: E731
    src = lambda g: (sum(bool(SOURCE.search(reasoning(r))) for r in g), len(g))               # noqa: E731
    fmt = Counter(answer_format(r) for r in finished)
    return {"outcomes": dict(Counter(outcome(r) for r in rs)), "formats": dict(fmt.most_common()),
            "n_finished": len(finished), "commit_correct": commit_stats(correct),
            "commit_wrong": commit_stats(wrong), "source_correct": src(correct), "source_wrong": src(wrong),
            "tokens_correct_median": tok(correct), "tokens_wrong_median": tok(wrong)}


def compare(base: list[dict], trained: list[dict], modes: list[str]) -> dict:
    out = {}
    for m in modes:
        b = [r for r in base if r["mode"] == m]
        t = [r for r in trained if r["mode"] == m]
        bk = {(r["qid"], r["sample_idx"]): r for r in b}
        tk = {(r["qid"], r["sample_idx"]): r for r in t}
        both = sorted(set(bk) & set(tk))
        fixed = [rid(tk[k]) for k in both if not bk[k]["correct"] and tk[k]["correct"]]
        broken = [rid(tk[k]) for k in both if bk[k]["correct"] and not tk[k]["correct"]]
        per_q_b, per_q_t = defaultdict(list), defaultdict(list)
        for k in both:
            per_q_b[k[0]].append(bk[k]["n_output_tokens"] or 0)
            per_q_t[k[0]].append(tk[k]["n_output_tokens"] or 0)
        qs = sorted(per_q_b)
        rho = spearman([statistics.median(per_q_b[q]) for q in qs], [statistics.median(per_q_t[q]) for q in qs])
        deltas = [(tk[k]["n_output_tokens"] or 0) - (bk[k]["n_output_tokens"] or 0) for k in both]
        out[m] = {"baseline": mode_view(b), "trained": mode_view(t),
                  "same_sample": {"n": len(both), "fixed": len(fixed), "broken": len(broken),
                                  "fixed_examples": fixed[:N_EXAMPLES], "broken_examples": broken[:N_EXAMPLES],
                                  "length_rank_correlation": rho,
                                  "token_change_median": statistics.median(deltas) if deltas else None}}
    return out


def markdown(res: dict, names: tuple[str, str]) -> str:
    def n(x, fmt="{:,.0f}"):
        return "–" if x is None else fmt.format(x)

    def share(pair):
        k, tot = pair
        return f"{k}/{tot} ({100 * k / tot:.0f}%)" if tot else "–"

    out = [f"Baseline = `{names[0]}`, trained = `{names[1]}`. Counts from pattern matching: estimates.", ""]
    for m, r in res.items():
        b, t = r["baseline"], r["trained"]
        out += [f"### {m}", "", "| | Baseline | Trained |", "|---|---|---|"]
        for o in sorted(set(b["outcomes"]) | set(t["outcomes"])):
            out.append(f"| outcome: {o} | {b['outcomes'].get(o, 0)} | {t['outcomes'].get(o, 0)} |")
        for f in [name for name, _ in FORMATS] + ["other", "none"]:
            if f in b["formats"] or f in t["formats"]:
                out.append(f"| final line: {f} (of finished) | {b['formats'].get(f, 0)}/{b['n_finished']} | "
                           f"{t['formats'].get(f, 0)}/{t['n_finished']} |")
        for lab, key in (("correct", "commit_correct"), ("wrong", "commit_wrong")):
            cb, ct = b[key], t[key]
            out += [f"| {lab}: outputs with a commitment | {cb['with_commitment']}/{cb['n']} | {ct['with_commitment']}/{ct['n']} |",
                    f"| {lab}: commitments per output (median) | {n(cb['commitments_median'], '{:g}')} | {n(ct['commitments_median'], '{:g}')} |",
                    f"| {lab}: letter switches (mean) | {n(cb['switches_mean'], '{:.2f}')} | {n(ct['switches_mean'], '{:.2f}')} |"]
        out += [f"| correct: first letter wrong, final right | {b['commit_correct']['first_wrong_final_right']} | {t['commit_correct']['first_wrong_final_right']} |",
                f"| wrong: first letter right, final wrong | {b['commit_wrong']['first_right_final_wrong']} | {t['commit_wrong']['first_right_final_wrong']} |",
                f"| correct: tokens after first committing (median) | {n(b['commit_correct']['tokens_after_first_commit_median'])} | {n(t['commit_correct']['tokens_after_first_commit_median'])} |",
                f"| invented source: correct | {share(b['source_correct'])} | {share(t['source_correct'])} |",
                f"| invented source: wrong | {share(b['source_wrong'])} | {share(t['source_wrong'])} |",
                f"| median tokens: correct | {n(b['tokens_correct_median'])} | {n(t['tokens_correct_median'])} |",
                f"| median tokens: wrong | {n(b['tokens_wrong_median'])} | {n(t['tokens_wrong_median'])} |"]
        s = r["same_sample"]
        out += ["", f"Same question and sample ({s['n']} pairs): fixed {s['fixed']} (e.g. {', '.join(s['fixed_examples']) or '–'}), "
                f"broken {s['broken']} (e.g. {', '.join(s['broken_examples']) or '–'}). Per-question length rank "
                f"correlation {n(s['length_rank_correlation'], '{:.2f}')}; median token change {n(s['token_change_median'], '{:+,.0f}')}.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("baseline")
    ap.add_argument("trained")
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config) if a.config else load_config()
    res = compare(load_scored(Path(a.baseline)), load_scored(Path(a.trained)), cfg["prompt"]["modes"])
    print(json.dumps(res, indent=2) if a.json else markdown(res, (Path(a.baseline).name, Path(a.trained).name)))


if __name__ == "__main__":
    main()
