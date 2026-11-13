"""Unit tests for src/prompts.py (component 2). Run: python3 tests/run_all.py"""
from src import prompts
from src.config import load_config

Q, CORRECT, WRONG = "Which is largest?", "5", ["2", "3", "1"]
SALT = "test-salt"


def test_think_prefix_is_open_block():
    t = prompts.build_text("think", Q, ["a", "b", "c", "d"])
    assert t.endswith("<|im_start|>assistant\n<think>\n")


def test_default_prefix_is_closed_empty_block():
    t = prompts.build_text("default", Q, ["a", "b", "c", "d"])
    assert t.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_modes_differ_only_in_assistant_prefix():
    opts = ["a", "b", "c", "d"]
    think, default = prompts.build_text("think", Q, opts), prompts.build_text("default", Q, opts)
    common = "<|im_start|>user\n" + prompts.user_text(Q, opts) + "<|im_end|>\n<|im_start|>assistant\n"
    assert think == common + "<think>\n" and default == common + "<think>\n\n</think>\n\n"


def test_user_text_layout():
    u = prompts.user_text(Q, ["w", "x", "y", "z"])
    assert u == ("Answer the following multiple-choice question.\n\n"
                 "Question: Which is largest?\n\nA) w\nB) x\nC) y\nD) z\n\n"
                 'Please show your choice in the `answer` field with only the choice letter, '
                 'e.g., `"answer": "C"`.')


def test_format_instruction_is_qwen_verbatim():
    # Model card README @ 15852e8c, line 983 (Best Practices 3, "Standardize Output Format").
    assert prompts.MCQ_FORMAT_INSTRUCTION == (
        'Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.')


def test_braces_in_question_are_kept():
    assert "\\frac{1}{2}" in prompts.user_text("Compute \\frac{1}{2}", ["a", "b", "c", "d"])


def test_shuffle_is_deterministic():
    a = prompts.shuffle_options("q1", 0, CORRECT, WRONG, SALT)
    assert a == prompts.shuffle_options("q1", 0, CORRECT, WRONG, SALT)


def test_gold_points_at_correct_answer_for_many_seeds():
    for i in range(200):
        opts, gold, order = prompts.shuffle_options(f"q{i}", i % 4, CORRECT, WRONG, SALT)
        assert opts["ABCD".index(gold)] == CORRECT
        assert sorted(order) == [0, 1, 2, 3] and opts[order.index(0)] == CORRECT


def test_gold_positions_are_roughly_uniform():
    # A biased shuffle would make "always answer A" a winning strategy.
    golds = [prompts.shuffle_options(f"q{i}", 0, CORRECT, WRONG, SALT)[1] for i in range(4000)]
    for L in "ABCD":
        assert 850 < golds.count(L) < 1150, (L, golds.count(L))


def test_salt_and_sample_change_the_shuffle():
    base = [prompts.shuffle_options(f"q{i}", 0, CORRECT, WRONG, SALT)[2] for i in range(50)]
    other_salt = [prompts.shuffle_options(f"q{i}", 0, CORRECT, WRONG, "x")[2] for i in range(50)]
    other_sample = [prompts.shuffle_options(f"q{i}", 1, CORRECT, WRONG, SALT)[2] for i in range(50)]
    assert base != other_salt and base != other_sample


def test_unknown_mode_rejected():
    try:
        prompts.build_text("nothink", Q, ["a", "b", "c", "d"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_config_modes_match_module():
    assert tuple(load_config()["prompt"]["modes"]) == prompts.MODES


def test_template_parity_with_real_tokenizer():
    """Milestone 3. Skipped unless `transformers` is installed and the tokenizer files
    for the pinned revision can be loaded (local cache or network)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return "SKIP: transformers not installed"
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
    opts, _, _ = prompts.shuffle_options("q1", 0, CORRECT, WRONG, SALT)
    for mode in prompts.MODES:
        rep = prompts.check_against_tokenizer(tok, mode, Q, opts)
        assert rep["match"] and rep["ids_match"], rep
