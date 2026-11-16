"""Build the GRPO training data: pinned SuperGPQA -> physics/chemistry/biology, 4 options.

Purpose : turn one exact, hash-verified SuperGPQA file into a training file and a held-out
          validation file whose records have the SAME shape as data/gpqa_diamond.jsonl, so
          src/prompts.py (prompt + per-sample option shuffle) and src/extract.py (answer
          rule, later the reward) work on them unchanged.
Inputs  : configs/train_data.toml (every parameter); the SuperGPQA jsonl at the pinned
          revision (downloaded if missing; public); data/gpqa_diamond.jsonl (decontamination).
Outputs : <[output] dir>/train.jsonl and val.jsonl, one question per line:
            {"qid": <SuperGPQA uuid>, "question", "correct", "wrong": [3 texts],
             "domain": <field>, "subdomain": <subfield>, "difficulty", "is_calculation",
             "n_options_original", "source": "SuperGPQA"}
          and manifest.json: source revision + hash, config hash, counts after every step,
          every dropped uuid with its reason, review-only overlap hits (ids only), output hashes.
Steps   : verify -> filter (exact labels) -> drop duplicate questions -> drop GPQA overlaps
          -> cut to 4 options (gold + seeded random wrong ones) -> split (val per field).
Pipeline: [this] -> data/train/... -> (later) difficulty filter -> slime training.
          GPQA is only READ here, to remove overlaps; it is never written to the outputs.

Usage:
  python3 -m src.train_data                 # build (refuses to overwrite)
  python3 -m src.train_data --show 3        # print 3 training prompts (reinit mode) as sent
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

from src.config import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "train_data.toml"
REQUIRED = {
    "source": ["hf_repo", "hf_file", "hf_revision", "file_sha256", "local_path", "n_rows"],
    "filter": ["discipline", "fields"],
    "options": ["n_options", "distractor_salt"],
    "decontam": ["gpqa_path", "ngram_drop", "ngram_review"],
    "split": ["val_per_field", "split_salt"],
    "output": ["dir", "train_file", "val_file", "manifest_file"],
}
LETTERS = "ABCDEFGHIJ"  # SuperGPQA answer letters (up to 10 options)
SOURCE_FIELDS = ("uuid", "question", "options", "answer", "answer_letter", "discipline",
                 "field", "subfield", "difficulty", "is_calculation")


def load_train_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Parse the TOML file, check required keys, record the file's hash."""
    path = Path(path)
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    missing = [f"{s}.{k}" for s, keys in REQUIRED.items() for k in keys if k not in cfg.get(s, {})]
    if missing:
        raise KeyError(f"{path}: missing config keys: {missing}")
    if cfg["decontam"]["ngram_review"] >= cfg["decontam"]["ngram_drop"]:
        raise ValueError("[decontam] ngram_review must be shorter than ngram_drop")
    cfg["_meta"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return cfg


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_source(cfg: dict) -> Path:
    """The pinned file: local copy if present, else download it (public, no token)."""
    s = cfg["source"]
    path = REPO_ROOT / s["local_path"]
    if not path.exists():
        url = f"https://huggingface.co/datasets/{s['hf_repo']}/resolve/{s['hf_revision']}/{s['hf_file']}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    got = sha256_file(path)
    if got != s["file_sha256"]:
        raise ValueError(f"{path}: sha256 {got} != pinned {s['file_sha256']}. Wrong file or revision.")
    return path


def read_rows(path: Path, n_expected: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if n_expected is not None and len(rows) != n_expected:
        raise ValueError(f"expected {n_expected} rows, got {len(rows)}")
    return rows


# ---------------------------------------------------------------- steps (pure functions)

def check_row(r: dict) -> str | None:
    """Why a source row is unusable, or None. These are data defects, not filtering."""
    if any(k not in r for k in SOURCE_FIELDS):
        return "missing_field"
    if not r["question"].strip() or any(not o.strip() for o in r["options"]):
        return "empty_text"
    if r["answer_letter"] not in LETTERS[:len(r["options"])]:
        return "answer_letter_out_of_range"
    if r["options"][LETTERS.index(r["answer_letter"])] != r["answer"]:
        return "gold_letter_text_mismatch"  # the letter must point at the gold text
    if len({o.strip() for o in r["options"]}) < len(r["options"]):
        return "duplicate_options"          # would make the gold ambiguous after the cut
    return None


def filter_rows(rows: list[dict], discipline: str, fields: list[str]) -> list[dict]:
    """Exact label match: discipline == X and field in the list. Nothing else is looked at."""
    return [r for r in rows if r["discipline"] == discipline and r["field"] in fields]


def dedup(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the first copy of each question text (outer whitespace ignored).

    If the copies disagree on the gold answer text, the question is ambiguous and every
    copy is dropped. Returns (kept, dropped_records).
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["question"].strip(), []).append(r)
    kept, dropped = [], []
    for copies in groups.values():
        if len({c["answer"].strip() for c in copies}) > 1:
            dropped += [{"uuid": c["uuid"], "reason": "duplicate_question_conflicting_gold"} for c in copies]
            continue
        kept.append(copies[0])
        dropped += [{"uuid": c["uuid"], "reason": "duplicate_question",
                     "detail": f"same question as {copies[0]['uuid']}"} for c in copies[1:]]
    order = {id(r): i for i, r in enumerate(rows)}  # keep original file order
    kept.sort(key=lambda r: order[id(r)])
    return kept, dropped


def words(text: str) -> list[str]:
    """Normalised words: lower case, letters and digits only (punctuation/spacing ignored)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def decontaminate(rows: list[dict], gpqa: list[dict], n_drop: int, n_review: int
                  ) -> tuple[list[dict], list[dict], list[dict]]:
    """Drop rows that match a GPQA question exactly or share an n_drop-word run with one.

    Rows sharing only an n_review-word run are kept and listed for review. Only ids are
    recorded, never text (GPQA text must not leave data/). Returns (kept, dropped, review).
    """
    exact = {" ".join(words(g["question"])): g["qid"] for g in gpqa}
    index: dict[int, dict] = {n_drop: {}, n_review: {}}
    for g in gpqa:
        for n in index:
            for gram in ngrams(g["question"], n):
                index[n].setdefault(gram, set()).add(g["qid"])

    def hits(r: dict, n: int) -> list[str]:
        return sorted(set().union(*(index[n].get(gr, set()) for gr in ngrams(r["question"], n))))

    kept, dropped, review = [], [], []
    for r in rows:
        norm = " ".join(words(r["question"]))
        if norm in exact:
            dropped.append({"uuid": r["uuid"], "reason": "gpqa_exact_match", "gpqa_qids": [exact[norm]]})
        elif q := hits(r, n_drop):
            dropped.append({"uuid": r["uuid"], "reason": f"gpqa_{n_drop}gram_overlap", "gpqa_qids": q})
        else:
            kept.append(r)
            if q := hits(r, n_review):
                review.append({"uuid": r["uuid"], "ngram": n_review, "gpqa_qids": q})
    return kept, dropped, review


def cut_options(r: dict, n_options: int, salt: str) -> dict:
    """Gold + (n_options - 1) wrong options chosen at random, seeded by "<salt>|<uuid>".

    The output order of `wrong` does not matter: src/prompts.shuffle_options reshuffles
    every (question, sample) anyway. The record has the gpqa_diamond.jsonl shape plus
    SuperGPQA metadata.
    """
    gold_i = LETTERS.index(r["answer_letter"])
    pool = [o for i, o in enumerate(r["options"]) if i != gold_i]
    if len(pool) < n_options - 1:
        raise ValueError(f"{r['uuid']}: only {len(pool) + 1} options, need {n_options}")
    # random.Random(str) seeds from the string's SHA-512, so this is stable across
    # machines and Python versions (same mechanism as prompts.shuffle_options).
    wrong = random.Random(f"{salt}|{r['uuid']}").sample(pool, n_options - 1)
    return {"qid": r["uuid"], "question": r["question"], "correct": r["answer"], "wrong": wrong,
            "domain": r["field"], "subdomain": r["subfield"], "difficulty": r["difficulty"],
            "is_calculation": r["is_calculation"], "n_options_original": len(r["options"]),
            "source": "SuperGPQA"}


def split(records: list[dict], fields: list[str], val_per_field: int, salt: str
          ) -> tuple[list[dict], list[dict]]:
    """Per field, the val_per_field questions with the smallest sha256("<salt>|<qid>") go to
    validation; the rest to training. Deterministic and independent of file order."""
    rank = lambda q: hashlib.sha256(f"{salt}|{q['qid']}".encode()).hexdigest()
    val_ids = set()
    for f in fields:
        group = sorted((q for q in records if q["domain"] == f), key=rank)
        if len(group) <= val_per_field:
            raise ValueError(f"field {f}: {len(group)} questions, cannot hold out {val_per_field}")
        val_ids |= {q["qid"] for q in group[:val_per_field]}
    return ([q for q in records if q["qid"] not in val_ids],
            [q for q in records if q["qid"] in val_ids])


def per_field(records: list[dict], fields: list[str]) -> dict:
    return {f: sum(q["domain"] == f for q in records) for f in fields}


# ---------------------------------------------------------------- the whole build

def build(cfg: dict, rows: list[dict], gpqa: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """All steps in order. Returns (train, val, manifest-without-output-hashes)."""
    fields = cfg["filter"]["fields"]
    counts, dropped = {"source_rows": len(rows)}, []

    selected = filter_rows(rows, cfg["filter"]["discipline"], fields)
    counts["after_filter"] = len(selected)
    usable = []
    for r in selected:
        if reason := check_row(r):
            dropped.append({"uuid": r.get("uuid"), "reason": reason})
        else:
            usable.append(r)
    counts["after_row_checks"] = len(usable)

    unique, d = dedup(usable)
    dropped += d
    counts["after_dedup"] = len(unique)

    clean, d, review = decontaminate(unique, gpqa, cfg["decontam"]["ngram_drop"],
                                     cfg["decontam"]["ngram_review"])
    dropped += d
    counts["after_decontam"] = len(clean)

    records = [cut_options(r, cfg["options"]["n_options"], cfg["options"]["distractor_salt"])
               for r in clean]
    train, val = split(records, fields, cfg["split"]["val_per_field"], cfg["split"]["split_salt"])
    counts.update(train=len(train), val=len(val))

    manifest = {
        "source": {k: cfg["source"][k] for k in ("hf_repo", "hf_file", "hf_revision", "file_sha256")},
        "config": cfg["_meta"],
        "gpqa_checked": {"n_questions": len(gpqa)},
        "counts": counts,
        "train_per_field": per_field(train, fields),
        "val_per_field": per_field(val, fields),
        "n_options_original": {str(k): sum(q["n_options_original"] == k for q in records)
                               for k in sorted({q["n_options_original"] for q in records})},
        "dropped": dropped,
        "review_overlaps": review,
    }
    # Every source row of the selected fields is either output or listed as dropped.
    assert counts["after_filter"] == len(train) + len(val) + len(dropped), "rows lost"
    return train, val, manifest


def write_jsonl(path: Path, records: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as f:
        for q in records:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return sha256_file(path)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def show(out_dir: Path, cfg: dict, n: int) -> None:
    """Print n training prompts exactly as they would be sent in reinit mode (sample 0)."""
    from src import prompts
    from src.config import load_config
    salt = load_config()["prompt"]["shuffle_salt"]  # same option-shuffle salt as the control run
    path = out_dir / cfg["output"]["train_file"]
    with path.open(encoding="utf-8") as f:
        qs = [json.loads(line) for _, line in zip(range(n), f)]
    for q in qs:
        options, gold, order = prompts.shuffle_options(q["qid"], 0, q["correct"], q["wrong"], salt)
        assert options["ABCD".index(gold)] == q["correct"], q["qid"]
        print(f"===== qid={q['qid']} field={q['domain']} gold={gold} order={order} =====")
        print(repr(prompts.build_text("reinit", q["question"], options)))
    print(f"\nGold letters verified for {len(qs)} questions.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--show", type=int, default=0, help="print N training prompts (reinit) from the built file")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = ap.parse_args(argv)
    cfg = load_train_config(args.config)
    out_dir = REPO_ROOT / cfg["output"]["dir"]

    if args.show:
        show(out_dir, cfg, args.show)
        return
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        sys.exit(f"{out_dir} is not empty; pass --overwrite to replace it (CLAUDE.md Rule 3: ask first).")

    rows = read_rows(locate_source(cfg), cfg["source"]["n_rows"])
    gpqa = read_rows(REPO_ROOT / cfg["decontam"]["gpqa_path"], None)
    train, val, manifest = build(cfg, rows, gpqa)

    out_dir.mkdir(parents=True, exist_ok=True)
    o = cfg["output"]
    manifest["outputs"] = {o["train_file"]: write_jsonl(out_dir / o["train_file"], train),
                           o["val_file"]: write_jsonl(out_dir / o["val_file"], val)}
    manifest["code_git_commit"] = git_commit()
    (out_dir / o["manifest_file"]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    c = manifest["counts"]
    print(" -> ".join(f"{k} {v}" for k, v in c.items() if k not in ("train", "val")))
    print(f"train {c['train']} {manifest['train_per_field']}, val {c['val']} {manifest['val_per_field']}")
    print(f"dropped {len(manifest['dropped'])}: "
          f"{sorted({d['reason'] for d in manifest['dropped']})}; review-only overlaps {len(manifest['review_overlaps'])}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
