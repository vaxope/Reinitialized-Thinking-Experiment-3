"""Tests for src/check_smoke.py and pick_checkpoint --smoke on a synthetic smoke-run log (no model
output, no GPQA text). Run: python3 tests/run_all.py  (standard library only)."""
import json
import tempfile
from pathlib import Path

from src import check_smoke, pick_checkpoint
from src.train_args import load_train_config

CFG = load_train_config()
N_VAL = CFG["smoke"]["eval_questions"]
STEPS = CFG["smoke"]["steps"]
GROUPS, K = CFG["rollout"]["rollout_batch_size"], CFG["rollout"]["n_samples_per_prompt"]


def _rec(t, split, version, group, reward=1.0):
    return {"time": t, "split": split, "weight_versions": [] if version is None else [str(version)],
            "group_index": group, "qid": f"q{group}", "reward": reward, "correct": reward == 1.0,
            "lenient_correct": reward == 1.0, "truncated": False, "n_output_tokens": 1000 + group,
            "extracted_answer": "A", "analysis_closed": True}


def _log(steps=STEPS, val=N_VAL, versions=True, reward=1.0):
    """What a good smoke run logs: validation (weights after 0 updates), steps 1..n (step k with the
    weights after k-1 updates), validation after the last update."""
    v = (lambda x: x) if versions else (lambda x: None)
    recs, t = [], 0.0
    recs += [_rec(t + i, "eval", v(1), 1000 + i) for i in range(val)]; t += val
    for s in range(1, steps + 1):
        for g in range(GROUPS):
            for j in range(K):
                recs.append(_rec(t, "train", v(s), s * 100 + g, reward if j else 0.0)); t += 1
    recs += [_rec(t + i, "eval", v(STEPS + 1), 2000 + i) for i in range(val)]
    d = Path(tempfile.mkdtemp())
    with (d / "rollouts_host_1.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return d


def _ok(d):
    return all(ok for ok, _ in check_smoke.check(d, CFG))


def test_good_smoke_log_passes():
    assert _ok(_log())


def test_failures_are_caught():
    assert not _ok(_log(steps=1))                 # step 2 (the weight push) missing
    assert not _ok(_log(versions=False))          # no weight versions: the push can't be proven
    assert not _ok(_log(val=N_VAL - 1))           # incomplete validation
    assert not _ok(_log(reward=0.5))              # a reward that is not 0/1
    assert not _ok(Path(tempfile.mkdtemp()))      # nothing logged


def test_pick_checkpoint_smoke_picks_the_saved_last_step():
    d = _log()
    ckpt = Path(tempfile.mkdtemp())
    (ckpt / f"iter_{STEPS - 1:07d}").mkdir()
    (ckpt / pick_checkpoint.TRACKER).write_text(str(STEPS - 1))
    import contextlib
    import io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pick_checkpoint.main(["--smoke", "--rollouts", str(d), "--ckpt-dir", str(ckpt), "--shell"])
    assert out.getvalue().strip() == f"STEP={STEPS} ITER_DIR={ckpt / f'iter_{STEPS - 1:07d}'}"
    try:   # without --smoke the real intervals (10) reject a pass at step 2
        pick_checkpoint.main(["--rollouts", str(d), "--ckpt-dir", str(ckpt), "--shell"])
    except (ValueError, SystemExit):
        pass
    else:
        raise AssertionError("expected refusal without --smoke")
