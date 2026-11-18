"""Tests for src/compare_trajectories.py on synthetic records (no GPQA text).
Run: python3 tests/run_all.py  (standard library only)."""
from src import compare_trajectories as ct


def _r(qid, s, mode, text, gold="B", correct=None, letter=None, lenient=None, tokens=100, truncated=False):
    letter = letter if letter is not None else lenient
    return {"qid": qid, "sample_idx": s, "mode": mode, "output_text": text, "gold": gold,
            "correct": bool(correct), "extracted_answer": letter, "lenient_answer": lenient,
            "n_output_tokens": tokens, "truncated": truncated,
            "outcome": "truncated" if truncated else ("correct" if correct else "wrong")}


def test_answer_formats():
    f = lambda text, mode="default": ct.answer_format(_r("q", 0, mode, text))   # noqa: E731
    assert f('work\n"answer": "C"') == "strict field"
    assert f("work\n**Answer: C**") == "answer line"
    assert f("work\nAnswer: C because the energy is lowest") == "answer key + more text"
    assert f("work\nSo the answer is C.") == "prose"
    assert f("work\n<answer>C</answer>") == "<answer> tag"
    assert f("work\n\\boxed{C}") == "\\boxed{}"
    assert f("work\n(C)") == "bare letter"
    assert f("no letters here at all") == "none"
    # think mode: only the text after </think> counts; an unclosed block has no visible answer
    assert f('the answer is A\n</think>\n"answer": "C"', "think") == "strict field"
    assert f("the answer is A, still thinking", "think") == "none"


def test_commitments_switches_and_tokens_after():
    text = "maybe the answer is A. hmm, wait. I'll go with B. " + "x" * 50 + '</think>\n"answer": "B"'
    r = _r("q1", 0, "think", text, correct=True, letter="B", lenient="B", tokens=len(text))   # 1 token per char
    s = ct.commit_stats([r])
    assert s["with_commitment"] == 1 and s["commitments_median"] == 2 and s["switches_mean"] == 1
    assert s["first_wrong_final_right"] == 1 and s["first_right_final_wrong"] == 0
    reasoning = ct.reasoning(r)
    assert s["tokens_after_first_commit_median"] == len(reasoning) - reasoning.index("I'll go with B")


def test_compare_fixed_broken_and_privacy_of_output():
    base = [_r("q1", 0, "reinit", "a</analysis>\nanswer: A", correct=False, lenient="A", tokens=300),
            _r("q2", 0, "reinit", "the answer is B</analysis>\nanswer: B", correct=True, lenient="B", tokens=200),
            _r("q3", 0, "reinit", "I found a source that says A</analysis>\nanswer: A", correct=False, lenient="A")]
    trained = [_r("q1", 0, "reinit", "the answer is B</analysis>\nanswer: B", correct=True, lenient="B", tokens=100),
               _r("q2", 0, "reinit", "the answer is B</analysis>\nanswer: C", correct=False, lenient="C", tokens=150),
               _r("q3", 0, "reinit", "x</analysis>\nanswer: A", correct=False, lenient="A")]
    res = ct.compare(base, trained, ["reinit"])["reinit"]
    assert res["same_sample"]["fixed"] == 1 and res["same_sample"]["fixed_examples"] == ["q1 s0"]
    assert res["same_sample"]["broken"] == 1 and res["same_sample"]["token_change_median"] == -50
    assert res["baseline"]["source_wrong"] == (1, 2) and res["trained"]["source_wrong"] == (0, 2)
    md = ct.markdown({"reinit": res}, ("a", "b"))
    assert "found a source" not in md and "answer: B" not in md   # never any output text


def test_spearman():
    assert ct.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert ct.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert ct.spearman([1, 2], [1, 2]) is None
