"""Unit tests for src/train_data.py (training-data build). Run: python3 tests/run_all.py

Uses made-up SuperGPQA-shaped rows and made-up "GPQA" questions, never real text. The
last test checks the real config file loads (no data needed).
"""
from src import prompts, train_data as td

FIELDS = ["Physics", "Chemistry", "Biology"]


def row(uuid, field="Physics", q=None, n_opt=10, gold=2, discipline="Science"):
    opts = [f"{uuid} option {i}" for i in range(n_opt)]
    return {"uuid": uuid, "question": q or f"Question text number {uuid} about something",
            "options": opts, "answer": opts[gold], "answer_letter": td.LETTERS[gold],
            "discipline": discipline, "field": field, "subfield": "Sub", "difficulty": "hard",
            "is_calculation": True}


def cfg(val_per_field=2):
    return {"filter": {"discipline": "Science", "fields": FIELDS},
            "options": {"n_options": 4, "distractor_salt": "s1"},
            "decontam": {"ngram_drop": 13, "ngram_review": 8},
            "split": {"val_per_field": val_per_field, "split_salt": "s2"},
            "source": {"hf_repo": "r", "hf_file": "f", "hf_revision": "v", "file_sha256": "h"},
            "_meta": {"path": "x", "sha256": "y"}}


def test_filter_is_exact_label_match():
    rows = [row("a"), row("b", field="Astronomy"), row("c", discipline="Engineering"),
            row("d", field="physics")]  # wrong case is not a match
    assert [r["uuid"] for r in td.filter_rows(rows, "Science", FIELDS)] == ["a"]


def test_cut_keeps_gold_and_is_deterministic():
    r = row("u1", n_opt=10, gold=7)
    a, b = td.cut_options(r, 4, "s"), td.cut_options(r, 4, "s")
    assert a == b and a["correct"] == r["answer"] and len(a["wrong"]) == 3
    assert r["answer"] not in a["wrong"] and len(set(a["wrong"])) == 3
    assert set(a["wrong"]) <= set(r["options"]) and a["n_options_original"] == 10
    # Same shape as data/gpqa_diamond.jsonl, so prompts.py accepts it unchanged.
    options, gold, _ = prompts.shuffle_options(a["qid"], 0, a["correct"], a["wrong"], "salt")
    assert options["ABCD".index(gold)] == a["correct"]


def test_cut_spreads_over_distractors():
    # Over many questions every non-gold position should get picked sometimes (not "first 3").
    picked = set()
    for i in range(200):
        r = row(f"q{i}", n_opt=10, gold=0)
        picked |= {r["options"].index(w) for w in td.cut_options(r, 4, "s")["wrong"]}
    assert picked == set(range(1, 10))


def test_four_option_rows_keep_all_options():
    r = row("f4", n_opt=4, gold=1)
    assert sorted(td.cut_options(r, 4, "s")["wrong"]) == sorted(o for o in r["options"] if o != r["answer"])


def test_row_checks():
    bad_letter = row("x"); bad_letter["answer_letter"] = "C"; bad_letter["answer"] = "something else"
    dup = row("y"); dup["options"][5] = dup["options"][4] + " "
    short = row("z", n_opt=4); short["answer_letter"] = "F"
    assert td.check_row(bad_letter) == "gold_letter_text_mismatch"
    assert td.check_row(dup) == "duplicate_options"
    assert td.check_row(short) == "answer_letter_out_of_range"
    assert td.check_row(row("ok")) is None


def test_dedup_keeps_first_and_drops_conflicts():
    a = row("a", q="Same question")
    b = dict(a, uuid="b", question="Same question ")  # same text after strip, same gold
    c, d = row("c", q="Other q"), row("d", q="Other q")
    d["answer"] = "different gold"
    kept, dropped = td.dedup([a, b, c, d])
    assert [r["uuid"] for r in kept] == ["a"]
    assert {(x["uuid"], x["reason"]) for x in dropped} == {
        ("b", "duplicate_question"), ("c", "duplicate_question_conflicting_gold"),
        ("d", "duplicate_question_conflicting_gold")}


def test_decontam_drops_planted_overlap_and_only_reviews_short_ones():
    gpqa = [{"qid": "g1", "question": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"},
            {"qid": "g2", "question": "The Exact Same Question, here!"},
            {"qid": "g3", "question": "one two three four five six seven eight"}]
    rows = [row("long", q="Intro: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu end"),
            row("exact", q="the exact same question here"),
            row("short", q="So one two three four five six seven eight, then more words"),
            row("clean")]
    kept, dropped, review = td.decontaminate(rows, gpqa, 13, 8)
    assert [r["uuid"] for r in kept] == ["short", "clean"]
    assert {(d["uuid"], d["reason"], tuple(d["gpqa_qids"])) for d in dropped} == {
        ("long", "gpqa_13gram_overlap", ("g1",)), ("exact", "gpqa_exact_match", ("g2",))}
    assert review == [{"uuid": "short", "ngram": 8, "gpqa_qids": ["g3"]}]
    assert "question" not in str(dropped + review)  # ids only, no text


def test_split_is_stratified_disjoint_and_order_independent():
    recs = [td.cut_options(row(f"{f[0]}{i}", field=f), 4, "s") for f in FIELDS for i in range(10)]
    train, val = td.split(recs, FIELDS, 3, "salt")
    assert td.per_field(val, FIELDS) == {f: 3 for f in FIELDS}
    assert not {q["qid"] for q in train} & {q["qid"] for q in val}
    assert len(train) + len(val) == len(recs)
    _, val_rev = td.split(list(reversed(recs)), FIELDS, 3, "salt")
    assert {q["qid"] for q in val_rev} == {q["qid"] for q in val}


def test_split_refuses_too_small_field():
    recs = [td.cut_options(row(f"p{i}"), 4, "s") for i in range(2)]
    try:
        td.split(recs, FIELDS, 2, "salt")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_build_accounts_for_every_row():
    rows = [row(f"{f[0]}{i}", field=f) for f in FIELDS for i in range(6)]
    rows += [row("astro", field="Astronomy"), dict(rows[0], uuid="dup"),
             row("contam", field="Chemistry", q="w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14")]
    gpqa = [{"qid": "g", "question": "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13"}]
    train, val, m = td.build(cfg(), rows, gpqa)
    c = m["counts"]
    assert c["source_rows"] == 21 and c["after_filter"] == 20 and c["after_dedup"] == 19
    assert c["after_decontam"] == 18 and c["val"] == 6 and c["train"] == 12
    assert {d["uuid"]: d["reason"] for d in m["dropped"]} == {"dup": "duplicate_question",
                                                            "contam": "gpqa_13gram_overlap"}
    assert all(len(q["wrong"]) == 3 for q in train + val)
    assert td.build(cfg(), rows, gpqa)[:2] == (train, val)  # deterministic


def test_real_config_loads():
    c = td.load_train_config()
    assert c["filter"]["fields"] == FIELDS and c["options"]["n_options"] == 4
    assert c["split"]["val_per_field"] == 100 and len(c["source"]["file_sha256"]) == 64
