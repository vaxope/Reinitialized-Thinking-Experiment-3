import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import modes

Q, OPTS = "Which is largest?", ["2", "3", "5", "1"]


def test_think_prefix_open():
    assert modes.build_text("think", Q, OPTS).endswith("<|im_start|>assistant\n<think>\n")


def test_other_modes_empty_think():
    for m in ("nothink", "direct", "gar"):
        assert modes.build_text(m, Q, OPTS).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_only_instruction_differs():
    base = modes.user_text("nothink", Q, OPTS)
    assert modes.user_text("think", Q, OPTS) == base
    assert modes.user_text("direct", Q, OPTS) == base + modes.DIRECT_INSTRUCTION
    assert modes.user_text("gar", Q, OPTS) == base + modes.GAR_STEPS


def test_shuffle_deterministic_and_gold_correct():
    a = modes.shuffle_options("q1", 0, "right", ["w1", "w2", "w3"])
    b = modes.shuffle_options("q1", 0, "right", ["w1", "w2", "w3"])
    assert a == b
    shuffled, gold, order = a
    assert shuffled["ABCD".index(gold)] == "right" and sorted(order) == [0, 1, 2, 3]


def test_enable_thinking_only_for_think():
    assert [modes.enable_thinking(m) for m in modes.MODES] == [True, False, False, False]
