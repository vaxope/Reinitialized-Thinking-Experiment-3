import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.reward import extract_answer, reward


def test_plain():
    assert extract_answer("work...\nAnswer: B")["letter"] == "B"


def test_markdown_variants():
    assert extract_answer("Answer: **C**")["letter"] == "C"
    assert extract_answer("**Answer: (D)**")["letter"] == "D"


def test_last_line_wins():
    r = "Answer: A\nwait, recheck...\nAnswer: C"
    out = extract_answer(r)
    assert out["letter"] == "C" and out["n_answer_lines"] == 2


def test_decoy_letters_ignored():
    # slime's built-in gpqa extractor would read "answer is a bit" as A
    assert extract_answer("I think the answer is a bit unclear. Option B looks right.")["letter"] is None


def test_answer_inside_blocks_ignored():
    r = "<analysis>\nAnswer: A\n</analysis>\nAnswer: B"
    assert extract_answer(r)["letter"] == "B"
    r2 = "<think>\nAnswer: A\n</think>\n\nAnswer: D"
    assert extract_answer(r2)["letter"] == "D"


def test_unclosed_think_is_no_answer():
    out = extract_answer("<think>\nlong reasoning... Answer: A")
    assert out["letter"] is None and out["rule"] == "unclosed_think"


def test_invalid_letter():
    assert extract_answer("Answer: E", n_options=4)["rule"] == "invalid_letter"


def test_reward_and_truncation():
    assert reward("Answer: B", "B") == 1.0
    assert reward("Answer: B", "C") == 0.0
    assert reward("Answer: B", "B", truncated=True) == 0.0
