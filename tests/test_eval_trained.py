"""Tests for the trained-model evaluation: src/pick_checkpoint.py, src/check_trained.py,
src/generate.py --checkpoint (against tests/fake_sglang.py) and src/compare_runs.py.

Synthetic data only (no GPQA text, no real weights). The generate test needs `transformers` and
the pinned tokenizer in the local HF cache (the .venv); it is skipped otherwise.
Run: .venv/bin/python tests/run_all.py
"""
import json
import shutil
import struct
import tempfile
from pathlib import Path

from src import check_trained, compare_runs, pick_checkpoint
from tests.test_check_roundtrip import f32, write_st

P = "model.language_model."
CFG = {"eval": {"interval": 20}, "checkpoint": {"save_interval": 20}}


# ---------------------------------------------------------------- pick_checkpoint
def _rec(t, split, correct, version=None, qid="q"):
    return {"time": t, "split": split, "weight_versions": [] if version is None else [str(version)],
            "correct": correct, "lenient_correct": correct, "truncated": False, "n_output_tokens": 100, "qid": qid}


def _log(accs, with_ids=True, n_val=4):
    """Validation passes with the given accuracies (pass 0 = before training), training in between."""
    recs, t = [], 0
    for k, acc in enumerate(accs):
        version = 20 * k + 1          # weights after 20k updates (the first push is version 1)
        for i in range(n_val):
            recs.append(_rec(t, "eval", i < acc * n_val, version if with_ids else None, qid=f"v{i}")); t += 1
        if k < len(accs) - 1:
            for _ in range(3):
                recs.append(_rec(t, "train", True)); t += 1
    return recs


def _ckpts(steps):
    d = Path(tempfile.mkdtemp())
    for s in steps:
        (d / f"iter_{s - 1:07d}").mkdir()
    return d


def test_pick_best_validation_ties_go_to_later_step_and_baseline_never_chosen():
    rows = pick_checkpoint.build_table(_log([1.0, 0.5, 0.75, 0.75, 0.25]), CFG)
    assert [r["step"] for r in rows] == [0, 20, 40, 60, 80]
    assert rows[0]["iter"] is None and rows[3]["iter"] == "iter_0000059"
    best = pick_checkpoint.choose(rows, _ckpts([20, 40, 60, 80]), None)
    assert best["step"] == 60   # 0.75 at steps 40 and 60 -> the later; the 1.0 baseline is not a checkpoint


def test_pick_without_weight_versions_uses_time_order():
    rows = pick_checkpoint.build_table(_log([0.0, 0.25, 0.5], with_ids=False), CFG)
    assert [(r["step"], r["iter"]) for r in rows] == [(0, None), (20, "iter_0000019"), (40, "iter_0000039")]


def test_pick_requested_step_missing_checkpoint_and_incomplete_pass():
    recs = _log([0.0, 0.5, 0.75])
    recs += [_rec(10_000, "eval", True, 61, qid="v0")]          # a pass cut off by a time limit (after 60)
    rows = pick_checkpoint.build_table(recs, CFG)
    assert rows[-1]["step"] == 60 and not rows[-1]["complete"]
    assert pick_checkpoint.choose(rows, _ckpts([20, 40, 60]), None)["step"] == 40   # incomplete never chosen
    assert pick_checkpoint.choose(rows, _ckpts([20, 40]), 20)["step"] == 20
    try:
        pick_checkpoint.choose(rows, _ckpts([20]), 40)          # step 40's folder is missing
    except SystemExit:
        pass
    else:
        raise AssertionError("expected refusal")


def test_pick_skips_a_save_newer_than_megatrons_tracker():
    rows = pick_checkpoint.build_table(_log([0.0, 0.25, 0.75]), CFG)
    d = _ckpts([20, 40])
    (d / "latest_checkpointed_iteration.txt").write_text("19\n")   # the step-40 save never finished
    assert pick_checkpoint.choose(rows, d, None)["step"] == 20


def test_pick_refuses_mismatched_intervals():
    try:
        pick_checkpoint.build_table(_log([0.0]), {"eval": {"interval": 20}, "checkpoint": {"save_interval": 10}})
    except ValueError:
        return
    raise AssertionError("expected refusal")


# ---------------------------------------------------------------- check_trained
def _base_tensors():
    return {P + "embed_tokens.weight": ("BF16", [2, 2], bytes(range(8))),
            P + "layers.0.mlp.up_proj.weight": ("BF16", [2, 2], bytes(range(8, 16))),
            P + "layers.0.linear_attn.A_log": ("F32", [2], f32(0.5, -1.25)),
            "model.visual.blocks.0.attn.qkv.weight": ("BF16", [2], b"\x01\x02\x03\x04"),
            "mtp.fc.weight": ("BF16", [2], b"\x05\x06\x07\x08")}


def _dirs(trained_tensors, tie=True, files=None):
    root = Path(tempfile.mkdtemp())
    base, trained = root / "base", root / "trained"
    write_st(base, _base_tensors()); write_st(trained, trained_tensors)
    for d in (base, trained):
        (d / "config.json").write_text(json.dumps({"tie_word_embeddings": tie}))
        (d / "tokenizer.json").write_text("{}")
    for name, text in (files or {}).items():
        (trained / name).write_text(text)
    return base, trained


def _trained(**changes):
    t = dict(_base_tensors())
    t[P + "layers.0.mlp.up_proj.weight"] = ("BF16", [2, 2], bytes(range(9, 17)))       # trained
    t[P + "layers.0.linear_attn.A_log"] = ("BF16", [2], b"\x00\x3f\xa0\xbf")           # 0.5, -1.25 in bf16
    t.update(changes)
    return t


def test_check_trained_passes_writes_manifest_and_detects_later_changes():
    base, trained = _dirs(_trained())
    rep = check_trained.compare(base, trained)
    assert rep["ok"] and rep["lm_changed"] == 1 and rep["lm_unchanged"] == 2 and rep["lm_rounded_to_bf16"] == 1
    assert rep["frozen_identical"] == 2
    check_trained.main([str(base), str(trained), "--source", '{"step": 80}'])
    m = check_trained.load_manifest(trained)
    assert m["source"] == {"step": 80} and len(m["model_sha256"]) == 64
    (trained / "tokenizer.json").write_text('{"changed": 1}')
    try:
        check_trained.load_manifest(trained)
    except ValueError as e:
        assert "tokenizer.json" in str(e)
    else:
        raise AssertionError("a changed file must be refused")


def test_check_trained_failures():
    # nothing trained
    assert not check_trained.compare(*_dirs(_trained(**{P + "layers.0.mlp.up_proj.weight":
                                                          _base_tensors()[P + "layers.0.mlp.up_proj.weight"]})))["ok"]
    # a frozen (vision) tensor changed
    assert not check_trained.compare(*_dirs(_trained(**{"model.visual.blocks.0.attn.qkv.weight":
                                                          ("BF16", [2], b"\x09\x02\x03\x04")})))["ok"]
    # a tensor missing (the converter was run without --add-missing-from-origin-hf)
    t = _trained(); del t["mtp.fc.weight"]
    assert check_trained.compare(*_dirs(t))["missing"] == ["mtp.fc.weight"]
    # a non-weight file differs
    base, trained = _dirs(_trained(), files={"config.json": "{}"})
    assert check_trained.compare(base, trained)["files_differ"] == ["config.json"]
    # shape change
    assert not check_trained.compare(*_dirs(_trained(**{P + "embed_tokens.weight": ("BF16", [4, 1], bytes(8))})))["ok"]


def test_check_trained_extra_lm_head_only_if_tied_and_equal():
    emb = _trained()[P + "embed_tokens.weight"]
    assert check_trained.compare(*_dirs(_trained(**{"lm_head.weight": emb})))["ok"]
    assert not check_trained.compare(*_dirs(_trained(**{"lm_head.weight": emb}), tie=False))["ok"]
    assert not check_trained.compare(*_dirs(_trained(**{"lm_head.weight": ("BF16", [2, 2], bytes(8))})))["ok"]


# ---------------------------------------------------------------- generate --checkpoint
def test_generate_checkpoint_needs_eval_run_id_and_a_checked_folder():
    from src import generate
    for argv in (["--run-id", "eval_x"], ["--run-id", "smoke_x", "--checkpoint", "/nowhere"]):
        try:
            generate.main(argv)
        except SystemExit as e:
            assert "go together" in str(e)
        else:
            raise AssertionError("expected refusal")
    try:
        generate.main(["--run-id", "eval_x", "--checkpoint", tempfile.mkdtemp()])
    except SystemExit as e:
        assert "Checkpoint refused" in str(e)
    else:
        raise AssertionError("expected refusal")


def _snapshot():
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download("Qwen/Qwen3.5-2B", revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
                                      local_files_only=True, allow_patterns=["*.json", "*.jinja", "*.txt"]))
    except Exception:
        return None


def test_generate_end_to_end_with_trained_checkpoint():
    snap = _snapshot()
    if snap is None:
        return "SKIP: transformers or the cached tokenizer not available"
    from src import generate
    from tests.test_generate import FakeServer, _records, _run
    with tempfile.TemporaryDirectory() as tmp:
        base, trained = Path(tmp) / "base", Path(tmp) / "step80"
        for d, tensors in ((base, _base_tensors()), (trained, _trained())):
            write_st(d, tensors)
            for f in snap.iterdir():    # the real tokenizer/config/template files
                if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
                    shutil.copy(f, d / f.name)
        check_trained.main([str(base), str(trained)])
        sha = check_trained.load_manifest(trained)["model_sha256"]
        # The fake server is not the pinned SGLang build; those two pins are tested elsewhere.
        saved = generate.check_sglang_version, generate.check_attention_backend
        generate.check_sglang_version = generate.check_attention_backend = lambda *a, **k: None
        try:
            srv = FakeServer.__new__(FakeServer)
            import subprocess, sys
            from tests.test_generate import ROOT, _free_port
            srv.port = _free_port()
            srv.proc = subprocess.Popen([sys.executable, "-m", "tests.fake_sglang", "--model-path", str(trained),
                                         "--port", str(srv.port), "--fake-load-s", "0.5"], cwd=ROOT,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            srv.url = f"http://127.0.0.1:{srv.port}"
            with srv:
                assert _run(tmp, srv.url, "--run-id", "eval_t_step80", "--checkpoint", str(trained)) == 0
                recs = _records(tmp, "eval_t_step80")
                assert len(recs) == 4 * 4 * 3   # all questions x control samples x modes, no overrides
                assert {r["model_revision"] for r in recs} == {f"trained:{sha}"}
                meta = json.loads((Path(tmp) / "runs" / "eval_t_step80" / "run_meta.jsonl").read_text().splitlines()[0])
                assert meta["checkpoint"]["model_sha256"] == sha
                # a folder without a passing check is refused before anything is sent
                try:
                    _run(tmp, srv.url, "--run-id", "eval_t_step80", "--checkpoint", str(base))
                except SystemExit as e:
                    assert "Checkpoint refused" in str(e)    # base has no manifest
                else:
                    raise AssertionError("expected refusal")
        finally:
            generate.check_sglang_version, generate.check_attention_backend = saved


# ---------------------------------------------------------------- compare_runs
def _scored(acc_by_mode, tokens=100):
    out = []
    for m, acc in acc_by_mode.items():
        for q in range(4):
            for s in range(2):
                ok = (q * 2 + s) < acc * 8
                out.append({"qid": f"q{q}", "sample_idx": s, "mode": m, "correct": ok, "lenient_correct": ok,
                            "truncated": False, "n_output_tokens": tokens, "scoring_rule_version": "v3"})
    return out


def test_compare_runs_changes_and_refusals():
    res = compare_runs.compare(_scored({"think": 0.5, "reinit": 0.25}), _scored({"think": 0.75, "reinit": 0.25}, 60),
                               ["think", "reinit"], 0.95)
    assert abs(res["think"]["change"]["accuracy"] - 0.25) < 1e-12 and res["reinit"]["change"]["accuracy"] == 0
    assert res["think"]["change"]["tokens_median"] == -40
    assert "| think | **change** | +25.0 pp" in compare_runs.markdown(res, ("a", "b"))
    try:
        compare_runs.compare(_scored({"think": 0.5}), _scored({"think": 0.5})[:-1], ["think"], 0.95)
    except ValueError as e:
        assert "different generations" in str(e)
    else:
        raise AssertionError("expected refusal")


def test_percentile_matches_linear_interpolation():
    assert compare_runs.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 9.1
