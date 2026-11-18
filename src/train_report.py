"""Per-step summary of a GRPO training run, from the answers src/reward.py logged.

Purpose : show what training did, step by step, with numbers only from the rollout log:
            training steps  answers, mean reward (= rule-v3 accuracy: reward is 1/0), lenient
                            accuracy, the gap between them (right letter in a shape rule v3
                            does not accept), groups with no learning signal (all 8 answers got
                            the same reward: GRPO's advantage is 0 for the whole group), answers
                            with an answer field at all, <analysis> closed, truncated, output
                            tokens (median, p90);
            validation      the same per pass on the held-out SuperGPQA slice (300 answers).
          Works on a partial log (a run still going), so it can be read mid-training.
Steps   : the weight version each answer was generated with (slime +1 per weight update, the first
          push = 1; see src/pick_checkpoint.py): training step k (1-based) is generated with
          version k; validation after k updates has version k + 1. Without versions, training
          records are split in time order into blocks of [rollout] global batch (128), between the
          validation passes.
Inputs  : runs/<run>/rollouts/rollouts_*.jsonl; configs/train_reinit.toml.
Output  : Markdown tables on stdout (--every N prints every N-th training step, plus the last);
          --json for machine use. Reads only.
Pipeline: training (slime + src/reward.py) -> [train_report] -> vault report (training section).
Usage   : python3 -m src.train_report runs/train_reinit_v1/rollouts [--every 5] [--json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.compare_runs import percentile
from src.pick_checkpoint import build_table, read_records, updates_done
from src.train_args import DEFAULT_TRAIN_CONFIG, load_train_config

EXTRA = ("group_index", "reward", "extracted_answer", "analysis_closed")


def training_steps(recs: list[dict], batch: int) -> dict[int, list[dict]]:
    """Training records by 1-based step: from the weight version, else blocks of `batch` in time order."""
    train = [r for r in recs if r["split"] != "eval"]
    if train and all(updates_done(r) is not None for r in train):
        steps = defaultdict(list)
        for r in train:
            steps[updates_done(r) + 1].append(r)
        return dict(steps)
    return {i // batch + 1: train[i:i + batch] for i in range(0, len(train), batch)}


def step_stats(rs: list[dict], group_size: int) -> dict:
    n = len(rs)
    toks = [r["n_output_tokens"] or 0 for r in rs]
    groups = defaultdict(list)
    for i, r in enumerate(rs):
        # slime's group_index identifies the 8 answers to one prompt; else question id
        groups[r.get("group_index") if r.get("group_index") is not None else r["qid"]].append(r["reward"] or 0)
    v3 = sum(bool(r["correct"]) for r in rs) / n
    len_acc = sum(bool(r["lenient_correct"]) for r in rs) / n
    closed = [r["analysis_closed"] for r in rs if r.get("analysis_closed") is not None]
    return {"n": n, "n_groups": len(groups),
            "groups_full": sum(len(g) == group_size for g in groups.values()),
            "mean_reward": statistics.fmean(r["reward"] or 0 for r in rs),
            "accuracy": v3, "accuracy_lenient": len_acc, "format_gap": len_acc - v3,
            "groups_no_signal": sum(len(set(g)) == 1 for g in groups.values()),
            "answer_found": sum(r.get("extracted_answer") is not None for r in rs) / n,
            "analysis_closed": (sum(map(bool, closed)) / len(closed)) if closed else None,
            "truncation_rate": sum(bool(r["truncated"]) for r in rs) / n,
            "tokens_median": statistics.median(toks), "tokens_p90": percentile(toks, 0.90)}


def report(rollout_dir: Path, cfg: dict) -> dict:
    recs = read_records(rollout_dir, EXTRA)
    k = cfg["rollout"]["n_samples_per_prompt"]
    batch = cfg["rollout"]["global_batch_size"]
    steps = {s: step_stats(rs, k) for s, rs in sorted(training_steps(recs, batch).items())}
    val = build_table(recs, cfg) if any(r["split"] == "eval" for r in recs) else []
    return {"group_size": k, "batch": batch, "train": steps, "validation": val}


def markdown(rep: dict, every: int) -> str:
    pct = lambda x: "–" if x is None else f"{100 * x:.1f}%"   # noqa: E731
    steps = sorted(rep["train"])
    shown = [s for s in steps if s % every == 0 or s in (steps[0], steps[-1])] if steps else []
    out = [f"**Training** ({len(steps)} steps logged; group size {rep['group_size']}, "
           f"{rep['batch']} answers per step; showing every {every}).", "",
           "| Step | Answers | Mean reward (= v3 acc.) | Lenient acc. | Format gap | Groups with no signal | "
           "Answer found | Analysis closed | Truncated | Median tokens | p90 tokens |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in shown:
        r = rep["train"][s]
        out.append(f"| {s} | {r['n']} | {r['mean_reward']:.3f} | {pct(r['accuracy_lenient'])} | "
                   f"{100 * r['format_gap']:+.1f} pp | {r['groups_no_signal']}/{r['n_groups']} | "
                   f"{pct(r['answer_found'])} | {pct(r['analysis_closed'])} | {pct(r['truncation_rate'])} | "
                   f"{r['tokens_median']:,.0f} | {r['tokens_p90']:,.0f} |")
    if rep["validation"]:
        out += ["", "**Validation** (held-out SuperGPQA, 1 answer per question; never GPQA).", "",
                "| After step | Checkpoint | Answers | Accuracy (v3) | Accuracy (lenient) | Truncated | Median tokens |",
                "|---|---|---|---|---|---|---|"]
        for r in rep["validation"]:
            out.append(f"| {r['step']} | {r['iter'] or 'untrained'} | {r['n']}{'' if r['complete'] else ' (incomplete)'} | "
                       f"{pct(r['accuracy'])} | {pct(r['accuracy_lenient'])} | {pct(r['truncation_rate'])} | "
                       f"{r['median_tokens']:,} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rollouts", help="runs/<run>/rollouts")
    ap.add_argument("--config", default=DEFAULT_TRAIN_CONFIG)
    ap.add_argument("--every", type=int, default=1, help="print every N-th training step (and the first/last)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = report(Path(a.rollouts), load_train_config(a.config))
    print(json.dumps(rep, indent=2) if a.json else markdown(rep, max(1, a.every)))


if __name__ == "__main__":
    main()
