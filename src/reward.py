"""Reward for GRPO training, plus the per-rollout log. Called by slime for every answer.

Purpose : score one generated answer exactly as evaluation scores it, and record it.
            reward = [reward] correct (1.0)   if rule v3 (src/extract.extract_answer) finds the
                                              gold letter AND the answer was not cut off;
                     [reward] incorrect (0.0) otherwise: wrong letter, no answer, <analysis>
                                              never closed, or cut off at the response limit.
          No format reward and no partial credit (user, 2026-11-15). The lenient reading
          (any_answer_lenient_v1) is recorded beside it as a diagnostic, never rewarded.
Log     : every scored answer is appended, BEFORE slime uses the reward (Rule 4), to
          $GAR_ROLLOUT_LOG_DIR/rollouts_<host>_<pid>.jsonl: question id, step, group, split
          (train/eval), the answer text and its token ids, status, letters, reward. The viewer
          and the analysis scripts read these. A logging failure stops the run (never dropped).
Inputs  : a slime Sample (sample.response, .label, .status, .metadata, .tokens, ...).
          Config: $GAR_TRAIN_CONFIG (default configs/train_reinit.toml).
Pipeline: slime rollout -> src/slime_hooks.generate -> SGLang -> [reward] -> GRPO advantages.
          Wired in by src/train_args.py as --custom-rm-path src.reward.reward.
slime API: slime awaits `rm(args, sample)` (slime/rollout/rm_hub/__init__.py async_rm), so the
          entry point is async; with --group-rm it would pass a list, which is handled too.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

from src import extract
from src.train_args import DEFAULT_TRAIN_CONFIG, load_train_config

N_OPTIONS = 4  # SuperGPQA cut to 4 options (configs/train_data.toml [options])
_CFG = None


def config() -> dict:
    """The training config, loaded once per process; checks the rule matches the code."""
    global _CFG
    if _CFG is None:
        cfg = load_train_config(os.environ.get("GAR_TRAIN_CONFIG", DEFAULT_TRAIN_CONFIG))
        if cfg["reward"]["rule"] != extract.RULE_NAME:
            raise ValueError(f"[reward] rule {cfg['reward']['rule']!r} != src/extract.py {extract.RULE_NAME!r}")
        _CFG = cfg
    return _CFG


def status_name(sample) -> str:
    """slime's Sample.Status enum value as text ('completed', 'truncated', ...)."""
    s = getattr(sample, "status", None)
    return str(getattr(s, "value", s))


def score(response: str, label: str, mode: str, truncated: bool, cfg: dict) -> dict:
    """Pure scoring: the reward and everything recorded about it."""
    v3 = extract.extract_answer(response, mode, N_OPTIONS)
    lenient = extract.extract_lenient(response, N_OPTIONS)
    correct = v3["letter"] is not None and v3["letter"] == label and not truncated
    return {
        "reward": cfg["reward"]["correct"] if correct else cfg["reward"]["incorrect"],
        "correct": correct, "truncated": truncated,
        "extracted_answer": v3["letter"], "extraction_rule": v3["rule"],
        "analysis_closed": v3.get("analysis_closed"), "think_closed": v3.get("think_closed"),
        "lenient_answer": lenient["letter"],
        "lenient_correct": lenient["letter"] == label and not truncated,
    }


def log_record(sample, result: dict, mode: str) -> dict:
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    n = int(getattr(sample, "response_length", 0) or 0)
    tokens = list(getattr(sample, "tokens", None) or [])
    return {
        "time": time.time(), "rollout_id": getattr(sample, "rollout_id", None),
        "index": getattr(sample, "index", None), "group_index": getattr(sample, "group_index", None),
        "split": meta.get("_split", "train"), "qid": meta.get("qid"), "prompt_mode": mode,
        "gold": sample.label, "status": status_name(sample), "n_output_tokens": n,
        "output_text": sample.response, "output_token_ids": tokens[-n:] if n else [],
        "weight_versions": list(getattr(sample, "weight_versions", None) or []),
        "reward_rule": config()["reward"]["rule"], **result,
    }


def write_log(rec: dict) -> None:
    log_dir = os.environ.get("GAR_ROLLOUT_LOG_DIR")
    if not log_dir:
        raise RuntimeError("GAR_ROLLOUT_LOG_DIR is not set: every rollout must be logged (Rule 4)")
    path = Path(log_dir) / f"rollouts_{socket.gethostname()}_{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:   # one process appends to its own file
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def reward_one(sample) -> float:
    cfg = config()
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    mode = meta.get("prompt_mode", cfg["data"]["prompt_mode"])
    truncated = status_name(sample) == "truncated"
    result = score(sample.response or "", sample.label, mode, truncated, cfg)
    write_log(log_record(sample, result, mode))
    return result["reward"]


async def reward(args, sample, **kwargs):
    """slime's custom reward entry point (--custom-rm-path src.reward.reward)."""
    if isinstance(sample, list):
        return [reward_one(s) for s in sample]
    return reward_one(sample)
