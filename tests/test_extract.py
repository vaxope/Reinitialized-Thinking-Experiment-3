"""Unit tests for src/extract.py (component 3). Run: python3 tests/run_all.py"""
from src.extract import extract_answer


def ex(text, mode="default", n=4):
    return extract_answer(text, mode, n)


def test_plain_answer_field():
    assert ex('reasoning...\n"answer": "B"')["letter"] == "B"


def test_field_inside_json_and_code_block():
    assert ex('{"answer": "C"}')["letter"] == "C"
    assert ex('Here it is:\n```json\n{\n  "answer": "D"\n}\n```')["letter"] == "D"
    assert ex('"answer" :"A"')["letter"] == "A"           # spacing around the colon allowed


def test_last_field_wins_after_self_correction():
    r = ex('"answer": "A"\nWait, let me recheck.\n{"answer": "C"}')
    assert r["letter"] == "C" and r["n_answer_fields"] == 2


def test_other_formats_are_not_accepted_but_counted_as_near_misses():
    for s in ["Answer: B", "The answer is B.", "**Answer:** B", 'answer: "B"',
              '"Answer": "B"', '"answer": "b"']:
        r = ex(s)
        assert r["letter"] is None and r["rule"] == "no_answer_field", s
        assert r["n_near_misses"] >= 1, s


def test_boxed_is_not_accepted():
    r = ex("\\boxed{B}")
    assert r["letter"] is None and r["n_near_misses"] == 0


def test_example_text_from_prompt_shape_is_still_last_field_rule():
    # The prompt's own example ("answer": "C") only appears in the INPUT, never in output_text,
    # so it cannot be picked up. If the model echoes it and then answers, the last one wins.
    assert ex('As instructed, e.g. "answer": "C". My choice:\n"answer": "A"')["letter"] == "A"


def test_invalid_letter():
    assert ex('"answer": "E"')["rule"] == "invalid_letter"
    assert ex('"answer": "E"', n=5)["letter"] == "E"


def test_empty_output():
    assert ex("")["rule"] == "empty_output" and ex("   \n")["rule"] == "empty_output"
    assert ex(None)["rule"] == "empty_output"


def test_think_mode_reads_only_after_close():
    r = ex('reasoning\n"answer": "A"\n</think>\n\nfinal text\n"answer": "D"', mode="think")
    assert r["letter"] == "D" and r["n_answer_fields"] == 1 and r["think_closed"] is True


def test_think_mode_unclosed_is_no_answer():
    # The prompt opened <think>, so the output has no "<think>", only (maybe) "</think>".
    # A truncated output with a draft answer must NOT count.
    r = ex('long reasoning...\n"answer": "A"\nhmm, maybe', mode="think")
    assert r["letter"] is None and r["rule"] == "unclosed_think" and r["think_closed"] is False


def test_think_mode_closed_but_no_answer_after():
    r = ex('reasoning\n"answer": "A"\n</think>\n\nI am not sure.', mode="think")
    assert r["letter"] is None and r["rule"] == "no_answer_field"


def test_think_mode_splits_at_first_close():
    r = ex('r</think>\n"answer": "B"\n</think>\n"answer": "C"', mode="think")
    assert r["letter"] == "C" and r["n_answer_fields"] == 2


def test_unknown_mode_rejected():
    try:
        ex('"answer": "A"', mode="nothink")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
