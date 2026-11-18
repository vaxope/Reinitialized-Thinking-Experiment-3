"""Pick the training checkpoint to evaluate, from the validation answers logged during training.

Purpose : the trained model evaluated on GPQA must be chosen WITHOUT looking at GPQA (CLAUDE.md:
          GPQA is never used to pick checkpoints). The choice uses only the held-out SuperGPQA
          validation slice (300 questions), whose every answer src/reward.py logged with
          split = "eval":
            chosen = the saved checkpoint with the highest validation rule-v3 accuracy;
                     ties -> the later step ([checkpoint] comment in configs/train_reinit.toml).
          The step-0 pass (before any update) is the untrained baseline: shown, never chosen.
How passes map to checkpoints (slime @ 79aa41a, train.py): validation runs before step 1 and
          after the update of every rollout r with (r + 1) % [eval] interval == 0; a checkpoint is
          saved after rollout r when (r + 1) % [checkpoint] save_interval == 0 or r is the last
          rollout, into Megatron's folder iter_<r, 7 digits>. With equal intervals (checked),
          validation pass k >= 1 measured exactly the weights saved as iter_{k * interval - 1:07d}.
          "Step" below = updates done = r + 1.
          Passes are told apart by time order: a pass is a run of consecutive eval records with no
          training record in between (rollout ids are cross-checked when the log has them).
Inputs  : runs/<run>/rollouts/rollouts_*.jsonl (src/reward.py), configs/train_reinit.toml,
          the checkpoint folder (to confirm the chosen iter_* exists).
Output  : a table of every validation pass, then one line `CHOSEN step=<n> iter=<dir>`; with
          --shell, only `STEP=<n> ITER_DIR=<path>` for the eval job to read. Exit 1 if nothing
          can be chosen. Reads only; writes nothing.
Pipeline: training (slime) -> [pick_checkpoint] -> convert to HF + src/check_trained.py ->
          the control evaluation (scripts/deltaai/eval_trained.sbatch).
Usage   : python3 -m src.pick_checkpoint --rollouts runs/train_reinit_v1/rollouts --ckpt-dir DIR
                                         [--step N] [--shell]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.train_args import DEFAULT_TRAIN_CONFIG, load_train_config


def read_records(rollout_dir: Path) -> list[dict]:
    """Every logged answer, oldest first (only the fields needed here, to keep memory small)."""
    keep = ("time", "split", "rollout_id", "correct", "lenient_correct", "truncated", "n_output_tokens", "qid")
    recs = []
    for f in sorted(Path(rollout_dir).glob("rollouts_*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    recs.append({k: r.get(k) for k in keep})
    recs.sort(key=lambda r: r["time"])
    return recs


def eval_passes(recs: list[dict]) -> list[list[dict]]:
    """Consecutive runs of eval records, split wherever a training record comes in between, and
    also wherever the logged rollout id changes (two passes with no training between them)."""
    passes, cur = [], []
    for r in recs:
        if r["split"] == "eval":
            if cur and r["rollout_id"] != cur[-1]["rollout_id"]:
                passes.append(cur); cur = []
            cur.append(r)
        elif cur:
            passes.append(cur); cur = []
    if cur:
        passes.append(cur)
    return passes


def summarize_pass(p: list[dict]) -> dict:
    n = len(p)
    toks = sorted(r["n_output_tokens"] or 0 for r in p)
    return {"n": n, "n_questions": len({r["qid"] for r in p}),
            "accuracy": sum(bool(r["correct"]) for r in p) / n,
            "accuracy_lenient": sum(bool(r["lenient_correct"]) for r in p) / n,
            "truncation_rate": sum(bool(r["truncated"]) for r in p) / n,
            "median_tokens": toks[n // 2], "rollout_ids": sorted({r["rollout_id"] for r in p} - {None})}


def build_table(recs: list[dict], cfg: dict) -> list[dict]:
    """One row per validation pass, with the step and checkpoint folder it belongs to."""
    interval, save = cfg["eval"]["interval"], cfg["checkpoint"]["save_interval"]
    if interval != save:
        raise ValueError(f"[eval] interval {interval} != [checkpoint] save_interval {save}: a validation "
                         "pass would not match a saved checkpoint")
    by_step = {}
    for k, p in enumerate(eval_passes(recs)):
        s = summarize_pass(p)
        if s["rollout_ids"]:
            # The log knows the rollout id: trust it. slime passes rollout_id 0 for the pass before
            # training and r for the pass after rollout r's update (r + 1 = interval, 2 x interval, ...).
            if len(s["rollout_ids"]) != 1:
                raise ValueError(f"validation pass {k} mixes rollout ids {s['rollout_ids']}")
            rid = s["rollout_ids"][0]
            step = 0 if rid == 0 else rid + 1
            if step % interval:
                raise ValueError(f"validation pass {k}: rollout id {rid} is not at a validation step")
        else:
            # No rollout id logged: time order. Only valid for an uninterrupted run (a resumed
            # run can redo or skip a pass), so say so.
            step = k * interval
            print(f"[pick_checkpoint] note: pass {k} has no rollout id; step {step} from time order", file=sys.stderr)
        # A resumed run may repeat a pass for the same step: the later one replaces it.
        by_step[step] = {"pass": k, "step": step, "iter": None if step == 0 else f"iter_{step - 1:07d}", **s}
    rows = [by_step[s] for s in sorted(by_step)]
    n_val = max((r["n"] for r in rows), default=0)   # every full pass answers the whole slice once
    for r in rows:
        r["complete"] = r["n"] == n_val
    return rows


def choose(rows: list[dict], ckpt_dir: Path | None, step: int | None) -> dict:
    cands = [r for r in rows if r["iter"] is not None and r["complete"]]
    if ckpt_dir is not None:
        cands = [r for r in cands if (Path(ckpt_dir) / r["iter"]).is_dir()]
    if step is not None:
        cands = [r for r in cands if r["step"] == step]
        if not cands:
            raise SystemExit(f"[pick_checkpoint] no complete validation pass with a saved checkpoint at step {step}")
        return cands[0]
    if not cands:
        raise SystemExit("[pick_checkpoint] no saved checkpoint with a complete validation pass yet")
    return max(cands, key=lambda r: (r["accuracy"], r["step"]))   # ties -> later step


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollouts", required=True, help="runs/<run>/rollouts")
    ap.add_argument("--ckpt-dir", default=None, help="Megatron checkpoint folder (iter_* inside)")
    ap.add_argument("--config", default=DEFAULT_TRAIN_CONFIG)
    ap.add_argument("--step", type=int, default=None, help="take this step instead of the best one")
    ap.add_argument("--shell", action="store_true", help="print only STEP=... ITER_DIR=...")
    a = ap.parse_args(argv)
    rows = build_table(read_records(Path(a.rollouts)), load_train_config(a.config))
    chosen = choose(rows, Path(a.ckpt_dir) if a.ckpt_dir else None, a.step)
    iter_dir = str(Path(a.ckpt_dir) / chosen["iter"]) if a.ckpt_dir else chosen["iter"]
    if a.shell:
        print(f"STEP={chosen['step']} ITER_DIR={iter_dir}")
        return
    print("pass  step  checkpoint     n    acc(v3)  acc(lenient)  truncated  median tokens")
    for r in rows:
        print(f"{r['pass']:>4}  {r['step']:>4}  {str(r['iter'] or 'untrained'):<13} {r['n']:>4}  "
              f"{r['accuracy']:.3f}    {r['accuracy_lenient']:.3f}         {r['truncation_rate']:.3f}      "
              f"{r['median_tokens']}{'' if r['complete'] else '  (incomplete pass)'}")
    how = f"step {a.step} requested" if a.step is not None else "highest validation v3 accuracy, ties -> later"
    print(f"CHOSEN step={chosen['step']} iter={iter_dir} ({how})")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as e:
        sys.exit(f"[pick_checkpoint] {e}")
