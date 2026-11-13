"""Prepare GPQA Diamond: pinned CSV -> data/gpqa_diamond.jsonl, and show example prompts.

Purpose : get the 198 questions from one exact, verified file and write them in the
          minimal form the rest of the pipeline needs. Milestone 1 of CLAUDE.md.
Inputs  : the GPQA Diamond CSV at the revision pinned in configs/control.toml, found
          (in this order) at --csv PATH, in the local Hugging Face cache, or downloaded
          from Hugging Face with the token in $HF_TOKEN (the dataset is gated).
Output  : data/gpqa_diamond.jsonl, one line per question:
            {"qid": "rec06pnAkLOr2t2mp", "question": ..., "correct": ...,
             "wrong": [..., ..., ...], "domain": "Physics", "subdomain": ...}
          data/ is gitignored: GPQA text is test-only and must never be committed.
Pipeline: [data] -> data/gpqa_diamond.jsonl -> src/prompts.py -> (later) src/generate.py.

Usage:
  python3 -m src.data                 # build the jsonl (refuses to overwrite)
  python3 -m src.data --show 5        # print 5 prompts per mode exactly as they will be sent
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

from src import prompts
from src.config import REPO_ROOT, load_config

# CSV column names in the official GPQA file. "Record ID" is GPQA's own stable id,
# used as our qid so results can always be traced back to the source row.
COL_ID, COL_Q, COL_CORRECT = "Record ID", "Question", "Correct Answer"
COL_WRONG = ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
COL_DOMAIN, COL_SUB = "High-level domain", "Subdomain"


def hf_cache_path(cfg: dict) -> Path:
    """Where huggingface_hub stores this exact file revision, if it was downloaded before."""
    repo = cfg["data"]["hf_repo"].replace("/", "--")
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return root / "hub" / f"datasets--{repo}" / "snapshots" / cfg["data"]["hf_revision"] / cfg["data"]["hf_file"]


def download_csv(cfg: dict, dest: Path) -> Path:
    """Download the pinned revision with the standard library (no huggingface_hub needed)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("No --csv given, file not in HF cache, and $HF_TOKEN unset. GPQA is gated: "
                 "accept its terms at https://huggingface.co/datasets/Idavidrein/gpqa first.")
    d = cfg["data"]
    url = f"https://huggingface.co/datasets/{d['hf_repo']}/resolve/{d['hf_revision']}/{d['hf_file']}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req) as r:
        dest.write_bytes(r.read())
    return dest


def locate_csv(cfg: dict, csv_arg: str | None) -> Path:
    if csv_arg:
        return Path(csv_arg)
    cached = hf_cache_path(cfg)
    if cached.exists():
        return cached
    return download_csv(cfg, REPO_ROOT / "data" / "raw" / cfg["data"]["hf_file"])


def verify_sha256(path: Path, expected: str) -> None:
    """Refuse any file that is not byte-identical to the pinned one."""
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != expected:
        raise ValueError(f"{path}: sha256 {got} != pinned {expected}. Wrong file or revision.")


def rows_to_questions(rows: list[dict], strip: bool) -> list[dict]:
    """Keep only the fields we need; optionally strip outer whitespace (see config)."""
    clean = (lambda s: s.strip()) if strip else (lambda s: s)
    return [{
        "qid": r[COL_ID].strip(),
        "question": clean(r[COL_Q]),
        "correct": clean(r[COL_CORRECT]),
        "wrong": [clean(r[c]) for c in COL_WRONG],
        "domain": r[COL_DOMAIN].strip(),
        "subdomain": r[COL_SUB].strip(),
    } for r in rows]


def validate(questions: list[dict], n_expected: int) -> list[str]:
    """Hard checks raise; soft findings are returned as warnings and printed."""
    if len(questions) != n_expected:
        raise ValueError(f"expected {n_expected} questions, got {len(questions)}")
    qids = [q["qid"] for q in questions]
    if len(set(qids)) != len(qids):
        raise ValueError("duplicate Record IDs")
    warnings = []
    for q in questions:
        texts = [q["question"], q["correct"], *q["wrong"]]
        if any(not t for t in texts):
            raise ValueError(f"{q['qid']}: empty question or option")
        opts = [q["correct"], *q["wrong"]]
        if len(set(opts)) != len(opts):  # identical options would make the gold ambiguous
            warnings.append(f"{q['qid']}: two options have identical text")
    return warnings


def load_questions(path: str | Path) -> list[dict]:
    """Read data/gpqa_diamond.jsonl back (used by --show and later by generate.py)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def show_prompts(questions: list[dict], cfg: dict, n: int) -> None:
    """Milestone 1: print the first n prompts per mode exactly as sent, and check gold."""
    salt = cfg["prompt"]["shuffle_salt"]
    for q in questions[:n]:
        options, gold, order = prompts.shuffle_options(q["qid"], 0, q["correct"], q["wrong"], salt)
        # Gold check: the option under the gold letter must be the correct answer text.
        assert options["ABCD".index(gold)] == q["correct"], q["qid"]
        for mode in cfg["prompt"]["modes"]:
            print(f"===== qid={q['qid']} sample=0 mode={mode} gold={gold} order={order} =====")
            print(repr(prompts.build_text(mode, q["question"], options)))  # repr shows \n exactly
    print(f"\nGold letters verified for {min(n, len(questions))} questions.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, help="config path (default configs/control.toml)")
    ap.add_argument("--csv", default=None, help="path to gpqa_diamond.csv (else HF cache, else download)")
    ap.add_argument("--show", type=int, default=0, help="print N prompts per mode from the built jsonl")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    out = REPO_ROOT / cfg["data"]["output_path"]

    if args.show:
        show_prompts(load_questions(out), cfg, args.show)
        return

    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; pass --overwrite to replace it (CLAUDE.md Rule 3: ask first).")
    src = locate_csv(cfg, args.csv)
    verify_sha256(src, cfg["data"]["csv_sha256"])
    with src.open(newline="", encoding="utf-8") as f:  # newline="" keeps multi-line cells intact
        rows = list(csv.DictReader(f))
    questions = rows_to_questions(rows, cfg["data"]["strip_whitespace"])
    for w in validate(questions, cfg["data"]["n_questions"]):
        print("WARNING:", w)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {len(questions)} questions to {out} (source {src}, sha256 verified).")


if __name__ == "__main__":
    main()
