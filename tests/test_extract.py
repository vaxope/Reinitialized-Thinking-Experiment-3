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


def test_v3_accepts_answer_lines_seen_in_pilot():
    # Every final-line shape observed in pilot_v1 that names one letter after an `answer` key.
    for s in ["answer: B", "Answer: B", 'answer: "B"', 'answer": "B"', '`answer": "B"`',
              '`answer`: "B"', '`"answer": "B"`', "**Answer:** B", "Answer: **B**",
              'answer": "B"</answer>', 'answer": "B"\n</solution>', "Final Answer: (B)",
              "    Answer: B", 'answer: "answer": "B"', '```\nanswer": "B"\n```',
              '"Answer": "B"', "answer: B\n</chain_of_thought>"]:
        r = ex("work...\n\n" + s)
        assert r["letter"] == "B" and r["rule"] == "last_answer_field", s


def test_v3_still_rejects_prose_and_other_shapes():
    for s in ["The answer is B.", "B", "A) B", "\\boxed{B}", "<answer>B</answer>", 'answer: "b"',
              '"choice": "B"', "Answer: B because the energy is lower", "Therefore, **Option B** is right",
              "answer = B", "ananswer: B"]:
        r = ex("work...\n\n" + s)
        assert r["letter"] is None and r["rule"] == "no_answer_field", s


def test_v3_prose_counts_as_near_miss():
    for s in ["The answer is B.", "Answer: B because the energy is lower", "answer = B"]:
        assert ex(s)["n_near_misses"] >= 1, s


def test_v3_answer_line_does_not_double_count_strict_field():
    r = ex('"answer": "B"')
    assert r["n_answer_fields"] == 1 and r["n_strict_fields"] == 1 and r["n_near_misses"] == 0


def test_v3_last_field_wins_across_both_shapes():
    assert ex('"answer": "A"\nWait, recheck.\nanswer: C')["letter"] == "C"
    assert ex('answer: A\nWait.\n"answer": "D" is my final choice')["letter"] == "D"


def test_strict_letter_reports_v2_decision():
    assert ex("answer: C")["strict_letter"] is None
    assert ex('"answer": "C"')["strict_letter"] == "C"
    assert ex('"answer": "A"\nanswer: C')["strict_letter"] == "A"     # v2 would have picked A


def test_v3_answer_line_inside_reasoning_block_is_ignored():
    assert ex("answer: A\n</think>\nI am unsure.", mode="think")["rule"] == "no_answer_field"
    assert ex("answer: A\nstill going", mode="reinit")["rule"] == "unclosed_analysis"
    assert ex("x</think>\nanswer: A\n</analysis>\nanswer: D", mode="reinit")["letter"] == "D"


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


def test_reinit_reads_only_after_analysis_close():
    r = ex('step 1...\n"answer": "A"\n</analysis>\n\nfinal\n"answer": "D"', mode="reinit")
    assert r["letter"] == "D" and r["n_answer_fields"] == 1 and r["rule"] == "last_answer_field"
    assert r["analysis_closed"] is True and r["think_closed"] is None


def test_reinit_unclosed_analysis_is_no_answer():
    r = ex('step 1...\n"answer": "A"\nstill working', mode="reinit")
    assert r["letter"] is None and r["rule"] == "unclosed_analysis" and r["analysis_closed"] is False


def test_reinit_splits_at_first_close():
    r = ex('a</analysis>\n"answer": "B"\n</analysis>\n"answer": "C"', mode="reinit")
    assert r["letter"] == "C" and r["n_answer_fields"] == 2


def test_reinit_ignores_think_close():
    # </think> means nothing in reinit mode: only </analysis> splits.
    r = ex('</think>\n"answer": "A"', mode="reinit")
    assert r["rule"] == "unclosed_analysis"


# ---------------------------------------------------------------- lenient reading (beside v3)
from src.extract import extract_lenient


def test_lenient_counts_answers_v3_rejects():
    # The control_v1 reinit failure shapes: closed with </think>, never closed, invented tag,
    # answer inside the analysis, bare final letter, prose.
    for s, want in [('analysis...\nAnswer: A\n</think>\n\nre-solve...\n\nanswer": "B"', "B"),
                    ("steps...\nAnswer: D\n<answer>D</answer>", "D"),
                    ('work...\nanswer": "A"\n</analysis>', "A"),
                    ("work...\n\n**C**\n</analysis>", "C"),
                    ("work...\n\nC", "C"),
                    ("Therefore the answer is B.", "B"),
                    ("Option C is correct.", "C"),
                    ('done\nchoice: "D"</choice>', "D"),
                    ("<final_answer>(A)</final_answer>", "A")]:
        r = extract_lenient(s)
        assert r["letter"] == want and r["rule"] == "last_answer_anywhere", (s, r)


def test_lenient_still_rejects_non_answers():
    for s in ["work...\na) 0.85", "\\boxed{B}", "answer: A good start", "the answer is b", "no letter here", "So **C**\n</analysis>"]:
        assert extract_lenient(s)["letter"] is None, s
    assert extract_lenient("")["rule"] == "empty_output" and extract_lenient(None)["rule"] == "empty_output"


def test_lenient_last_answer_wins_and_invalid_letter():
    assert extract_lenient("answer: A\nwait\nThe answer is C.")["letter"] == "C"
    assert extract_lenient("answer: E")["rule"] == "invalid_letter"


def test_lenient_accepts_everything_v3_accepts():
    for s in ['"answer": "B"', "answer: B", 'answer": "B"', "**Answer:** B", 'answer": "B"</answer>']:
        assert extract_lenient(s)["letter"] == extract_answer(s, "default")["letter"] == "B", s


def test_lenient_ignores_hypotheticals_and_reads_final_answer():
    # Both cases seen in control_v1: a hypothesis in the reasoning, then a final answer in an
    # unusual shape. The final answer must win; the hypothesis must not count.
    assert extract_lenient("If Option D is correct then...\n</analysis>\n\nanswer\nanswer\"A\"")["letter"] == "A"
    assert extract_lenient("What if the answer is C (0.91c)?\nI will select D.\n\n<div class=\"answer\">D</div>")["letter"] == "D"
    assert extract_lenient("Suppose the answer is B. Then x fails.")["letter"] is None
    assert extract_lenient("<spananswer>D</answer>")["letter"] == "D"
    assert extract_lenient('answer": "B"\\')["letter"] == "B"


def test_lenient_counts_a_final_option_line():
    assert extract_lenient("A) 14")["letter"] == "A"
    assert extract_lenient("work...\n(B) 3-bromo-4'-fluoro-1,1'-biphenyl")["letter"] == "B"
    assert extract_lenient("A) 14\nB) 15\nmore text")["letter"] is None   # not the last line
