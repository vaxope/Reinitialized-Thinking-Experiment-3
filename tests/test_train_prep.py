"""Unit tests for src/train_prep.py (questions -> slime input lines). Run: python3 tests/run_all.py

Synthetic questions only (no SuperGPQA or GPQA text). The real-tokenizer parity check over all
5,733 training prompts is run by the module itself (python3 -m src.train_prep); here a fake
tokenizer checks that each guard fires.
"""
import copy
import hashlib

from src import prompts
from src.train_args import load_train_config
from src.train_prep import SAMPLE_IDX, build, check_with_tokenizer, make_record

CFG = load_train_config()


def q(i, domain="Physics"):
    return {"qid": f"syn{i:04d}", "question": f"Synthetic question {i}?", "correct": f"right {i}",
            "wrong": [f"wrong {i}a", f"wrong {i}b", f"wrong {i}c"], "domain": domain,
            "subdomain": "s", "difficulty": "hard", "source": "synthetic"}


def test_record_is_the_exact_reinit_prompt_and_gold_points_at_correct_option():
    rec = make_record(q(1), "reinit", "salt")
    options, gold, order = prompts.shuffle_options("syn0001", SAMPLE_IDX, "right 1", q(1)["wrong"], "salt")
    assert rec["text"] == prompts.build_text("reinit", "Synthetic question 1?", options)
    assert rec["text"].endswith("<think>\n\n</think>\n\n<analysis>\n")
    assert rec["label"] == gold and options["ABCD".index(gold)] == "right 1"
    assert rec["metadata"]["option_order"] == order and rec["metadata"]["qid"] == "syn0001"
    assert rec["metadata"]["prompt_mode"] == "reinit"


def test_one_fixed_option_order_per_question_and_deterministic():
    assert make_record(q(2), "reinit", "salt") == make_record(q(2), "reinit", "salt")
    # a different salt gives an independent shuffle (training vs GPQA)
    letters = {make_record(q(i), "reinit", "salt-a")["label"] for i in range(40)}
    assert len(letters) == 4                      # gold lands on every letter


def test_default_mode_for_the_matched_control():
    rec = make_record(q(3), "default", "salt")
    assert rec["text"].endswith("<think>\n\n</think>\n\n") and "<analysis>" not in rec["text"]
    assert "<|im_start|>system" not in rec["text"]          # no persona in default mode


def test_build_refuses_duplicate_questions_across_splits():
    rows = {"train": [q(1), q(2)], "val": [q(3)]}
    assert [len(v) for v in build(CFG, rows).values()] == [2, 1]
    try:
        build(CFG, {"train": [q(1)], "val": [q(1)]})
    except ValueError:
        return
    raise AssertionError("expected a duplicate-id error")


class FakeTok:
    """Mimics the parts of a transformers tokenizer that check_with_tokenizer uses."""
    def __init__(self, template="T", n_tokens=10, official_extra=""):
        self.chat_template, self.n, self.extra = template, n_tokens, official_extra

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        # rebuild what build_text makes, so parity holds unless official_extra is set
        sys_msg = [m["content"] for m in messages if m["role"] == "system"]
        user = [m["content"] for m in messages if m["role"] == "user"][0]
        head = f"<|im_start|>system\n{sys_msg[0]}<|im_end|>\n" if sys_msg else ""
        prefix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        return head + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + prefix + self.extra

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [0] * self.n}


def _cfg_with_template(t):
    c = copy.deepcopy(CFG)
    c["model"]["chat_template_sha256"] = hashlib.sha256(t.encode()).hexdigest()
    return c


def _raises(*a):
    try:
        check_with_tokenizer(*a)
    except ValueError:
        return True
    return False


def test_tokenizer_checks_pass_and_each_guard_fires():
    rows = {"train": [q(1), q(2)], "val": [q(3)]}
    cfg = _cfg_with_template("T")
    rec = build(cfg, rows)
    rep = check_with_tokenizer(FakeTok("T"), cfg, rows, rec)
    assert rep["n_parity_checked"] == 3 and rep["prompt_tokens_max"] == 10
    assert _raises(FakeTok("other template"), cfg, rows, rec)                 # wrong tokenizer
    assert _raises(FakeTok("T", official_extra="x"), cfg, rows, rec)          # parity broken
    assert _raises(FakeTok("T", n_tokens=CFG["rollout"]["max_prompt_len"] + 1), cfg, rows, rec)  # too long
