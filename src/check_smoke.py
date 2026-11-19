"""Check the rollout log of the smoke run: did slime really train, validate and push weights?

Purpose : the smoke run (scripts/deltaai/smoke_train.sh) runs real slime training for [smoke] steps
          (2). train.py exiting with status 0 is not enough: it must also have produced what the
          long run relies on. This check reads the answers src/reward.py logged and requires:
            1. every training step 1..steps logged, each with the full batch: [rollout]
               global_batch_size answers (128) in rollout_batch_size groups (16) of
               n_samples_per_prompt (8) answers;
            2. every answer labelled with a weight version, and step k generated with the weights
               after k - 1 updates: step 2 then proves that the trainer's update reached SGLang
               (the weight push), the one part a single step cannot show;
            3. validation before training (step 0) and after the last step, each answering all
               [smoke] eval_questions once;
            4. every reward exactly [reward] correct or incorrect (1 or 0).
          It also prints the time each step took (from the log timestamps) and the answer lengths,
          as a first measurement for planning the long run.
Inputs  : runs/smoke_train_<job>/rollouts/ (rollouts_*.jsonl); configs/train_reinit.toml.
Output  : one line per check, then PASS or FAIL; exit code 1 on any failure. Reads only.
Pipeline: smoke_train.sh: slime train.py (2 steps) -> [check_smoke] -> pick_checkpoint -> convert
          -> check_trained -> serving test.
Usage   : python3 -m src.check_smoke runs/smoke_train_<job>/rollouts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pick_checkpoint import read_records, updates_done
from src.train_args import DEFAULT_TRAIN_CONFIG, load_train_config, smoke_config
from src.train_report import EXTRA, report


def check(rollout_dir: Path, cfg: dict) -> list[tuple[bool, str]]:
    """cfg = the REAL config; the smoke changes are applied here. Returns (ok, message) per check."""
    sc = smoke_config(cfg)
    steps, n_val = sc["run"]["num_steps"], cfg["smoke"]["eval_questions"]
    batch, k = sc["rollout"]["global_batch_size"], sc["rollout"]["n_samples_per_prompt"]
    n_groups = sc["rollout"]["rollout_batch_size"]
    rewards = {cfg["reward"]["correct"], cfg["reward"]["incorrect"]}
    recs = read_records(rollout_dir, EXTRA)
    out = []
    if not recs:
        return [(False, f"no answers logged in {rollout_dir}")]

    train = [r for r in recs if r["split"] != "eval"]
    unversioned = sum(updates_done(r) is None for r in recs)
    out.append((unversioned == 0, f"weight version on every answer ({unversioned} of {len(recs)} without)"))

    rep = report(rollout_dir, sc)
    got = sorted(rep["train"])
    out.append((got == list(range(1, steps + 1)), f"training steps logged {got} (from weight versions), "
                                                  f"expected 1..{steps}"))
    for s in range(1, steps + 1):
        st = rep["train"].get(s)
        ok = bool(st) and st["n"] == batch and st["n_groups"] == n_groups and st["groups_full"] == n_groups
        out.append((ok, f"step {s}: {st['n'] if st else 0} answers in {st['n_groups'] if st else 0} groups "
                        f"({st['groups_full'] if st else 0} full), expected {batch} in {n_groups} groups of {k}"))
    # (steps are numbered from the weight version, so "step 2 logged" with every answer versioned
    # means: 128 answers generated with the weights after 1 update, i.e. the push reached SGLang)

    val_steps = [(r["step"], r["n"]) for r in rep["validation"]]
    out.append((val_steps == [(0, n_val), (steps, n_val)],
                f"validation passes (step, answers) {val_steps}, expected [(0, {n_val}), ({steps}, {n_val})]"))

    bad = sum(r.get("reward") not in rewards for r in recs)
    out.append((bad == 0, f"rewards all in {sorted(rewards)} ({bad} other values)"))

    # timing and lengths: information for planning, never a failure (needs the versions)
    if unversioned or got != list(range(1, steps + 1)):
        return out
    ends = {s: max(r["time"] for r in train if updates_done(r) is not None and updates_done(r) + 1 == s)
            for s in got}
    first = min(r["time"] for r in recs)
    prev = first
    for s in got:
        st = rep["train"][s]
        print(f"[check_smoke] step {s}: done {ends[s] - prev:,.0f} s after the previous point; "
              f"answers median {st['tokens_median']:,.0f} / p90 {st['tokens_p90']:,.0f} tokens, "
              f"truncated {100 * st['truncation_rate']:.1f}%, mean reward {st['mean_reward']:.3f}")
        prev = ends[s]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rollouts", help="runs/smoke_train_<job>/rollouts")
    ap.add_argument("--config", default=DEFAULT_TRAIN_CONFIG)
    a = ap.parse_args(argv)
    results = check(Path(a.rollouts), load_train_config(a.config))
    for ok, msg in results:
        print(f"[check_smoke] {'PASS' if ok else 'FAIL'} {msg}")
    ok = all(ok for ok, _ in results)
    print(f"[check_smoke] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
