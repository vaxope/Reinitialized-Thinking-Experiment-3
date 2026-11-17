"""Build slime's input files: each training question posed in the training prompt mode.

Purpose : turn the SuperGPQA training/validation questions (src/train_data.py output) into the
          files slime reads, one line per question:
              {"text": <exact prompt string>, "label": <gold letter>, "metadata": {...}}
          The prompt is built by src/prompts.build_text, the SAME code that built the control
          run's prompts, so a training question looks to the model exactly like a GPQA question
          in that mode. slime sends `text` unchanged (no chat template; src/slime_hooks.py keeps
          it a plain string), tokenized with add_special_tokens=False as in src/generate.py.
Checks  : before writing anything, with the pinned tokenizer:
            - the tokenizer's chat-template hash equals [model] chat_template_sha256;
            - template parity for EVERY prompt (build_text == the official template, as text and
              token ids; milestone 3 of the control run, now on the training questions);
            - no prompt is longer than [rollout] max_prompt_len (slime would drop it silently).
Inputs  : configs/train_reinit.toml [data] [rollout] [model]; <source_dir>/{train,val}.jsonl.
Outputs : <output_dir>/train.jsonl, val.jsonl, manifest.json (hashes, counts, checks, git commit).
Pipeline: data/train/supergpqa_pcb_v1 --[train_prep]--> slime_reinit/{train,val}.jsonl
          --> slime --prompt-data / --eval-prompt-data (src/train_args.py).

Usage   : python3 -m src.train_prep [--config configs/train_reinit.toml] [--overwrite]
          (needs `transformers` and the pinned tokenizer: run with .venv/bin/python locally, or
          inside the container on DeltaAI)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

from src import prompts
from src.config import REPO_ROOT
from src.train_args import DEFAULT_TRAIN_CONFIG, load_train_config
from src.train_data import git_commit, read_rows, sha256_file, write_jsonl

SPLITS = ("train", "val")
# Sample index used in the option-shuffle seed. Every answer in a GRPO group must see the SAME
# prompt, so each question gets one fixed option order (index 0), unlike the control run, which
# reshuffles per sample.
SAMPLE_IDX = 0


def make_record(q: dict, mode: str, salt: str) -> dict:
    """One slime input line for one question (q has the gpqa_diamond.jsonl shape)."""
    options, gold, order = prompts.shuffle_options(q["qid"], SAMPLE_IDX, q["correct"], q["wrong"], salt)
    return {
        "text": prompts.build_text(mode, q["question"], options),
        "label": gold,
        "metadata": {
            "qid": q["qid"], "prompt_mode": mode, "option_order": order,
            "shuffle_seed": prompts.shuffle_seed(salt, q["qid"], SAMPLE_IDX),
            "domain": q.get("domain"), "subdomain": q.get("subdomain"),
            "difficulty": q.get("difficulty"), "source": q.get("source"),
        },
    }


def build(cfg: dict, rows: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Records for every split. Pure: no files, no tokenizer."""
    mode, salt = cfg["data"]["prompt_mode"], cfg["data"]["shuffle_salt"]
    out = {split: [make_record(q, mode, salt) for q in rows[split]] for split in SPLITS}
    seen = collections.Counter(r["metadata"]["qid"] for recs in out.values() for r in recs)
    dup = [q for q, n in seen.items() if n > 1]
    if dup:
        raise ValueError(f"{len(dup)} question ids appear more than once across train/val, e.g. {dup[:3]}")
    return out


def check_with_tokenizer(tokenizer, cfg: dict, rows: dict[str, list[dict]],
                         records: dict[str, list[dict]]) -> dict:
    """Template hash, parity for every prompt, and prompt lengths. Raises on any failure."""
    want = cfg["model"]["chat_template_sha256"]
    got = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if got != want:
        raise ValueError(f"tokenizer chat template sha256 {got} != pinned {want}")
    mode, salt, limit = cfg["data"]["prompt_mode"], cfg["data"]["shuffle_salt"], cfg["rollout"]["max_prompt_len"]
    lengths, bad = [], []
    for split in SPLITS:
        for q, rec in zip(rows[split], records[split]):
            options, _, _ = prompts.shuffle_options(q["qid"], SAMPLE_IDX, q["correct"], q["wrong"], salt)
            rep = prompts.check_against_tokenizer(tokenizer, mode, q["question"], options)
            if not (rep["match"] and rep["ids_match"]):
                bad.append(q["qid"])
            lengths.append(len(tokenizer(rec["text"], add_special_tokens=False)["input_ids"]))
    if bad:
        raise ValueError(f"template parity failed for {len(bad)} prompts, e.g. {bad[:3]}")
    too_long = sum(n > limit for n in lengths)
    if too_long:
        raise ValueError(f"{too_long} prompts exceed max_prompt_len={limit} (longest {max(lengths)})")
    return {"n_parity_checked": len(lengths), "prompt_tokens_max": max(lengths),
            "prompt_tokens_median": sorted(lengths)[len(lengths) // 2], "max_prompt_len": limit}


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer  # only needed here; tests use a fake tokenizer
    return AutoTokenizer.from_pretrained(cfg["model"]["hf_repo"], revision=cfg["model"]["revision"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build slime input files for training.")
    ap.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG))
    ap.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = ap.parse_args(argv)
    cfg = load_train_config(args.config)
    src_dir, out_dir = REPO_ROOT / cfg["data"]["source_dir"], REPO_ROOT / cfg["data"]["output_dir"]
    targets = [out_dir / f"{s}.jsonl" for s in SPLITS] + [out_dir / "manifest.json"]
    if any(t.exists() for t in targets) and not args.overwrite:
        sys.exit(f"{out_dir} already has output files; pass --overwrite to replace them")

    rows = {s: read_rows(src_dir / f"{s}.jsonl", None) for s in SPLITS}
    records = build(cfg, rows)
    checks = check_with_tokenizer(load_tokenizer(cfg), cfg, rows, records)

    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {s: write_jsonl(out_dir / f"{s}.jsonl", records[s]) for s in SPLITS}
    manifest = {
        "config": cfg["_meta"]["path"], "config_sha256": sha256_file(Path(args.config)),
        "prompt_mode": cfg["data"]["prompt_mode"], "prompt_hash": prompts.PROMPT_HASH,
        "shuffle_salt": cfg["data"]["shuffle_salt"], "sample_idx": SAMPLE_IDX,
        "source": {s: {"path": str(src_dir / f"{s}.jsonl"), "sha256": sha256_file(src_dir / f"{s}.jsonl")}
                   for s in SPLITS},
        "counts": {s: len(records[s]) for s in SPLITS},
        "gold_letters": {s: dict(sorted(collections.Counter(r["label"] for r in records[s]).items()))
                         for s in SPLITS},
        "checks": checks, "outputs_sha256": hashes, "git_commit": git_commit(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("prompt_mode", "counts", "gold_letters", "checks")}, indent=2))


if __name__ == "__main__":
    main()
