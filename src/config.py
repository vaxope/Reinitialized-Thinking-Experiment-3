"""Load configs/control.toml.

Purpose : one function, `load_config`, so every module reads parameters the same way
          and nothing hard-codes a value that belongs in the config.
Input   : a TOML file path (default: configs/control.toml at the repo root).
Output  : a plain nested dict, e.g. cfg["prompt"]["n_options"] -> 4.
Pipeline: used by src/data.py, src/score.py (and later src/generate.py).
"""
from __future__ import annotations

import hashlib
import tomllib  # standard library since Python 3.11; no install needed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "control.toml"

# Sections and keys every consumer relies on. Checked at load time so a typo in the
# config fails immediately instead of deep inside a run.
REQUIRED = {
    "model": ["name", "revision", "dtype"],
    "data": ["hf_repo", "hf_file", "hf_revision", "csv_sha256", "n_questions",
             "output_path", "strip_whitespace"],
    "prompt": ["modes", "n_options", "shuffle_salt"],
    "sampling": ["temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "repetition_penalty"],
    "generation": ["max_new_tokens", "samples_per_question", "pilot_n_questions",
                   "pilot_samples_per_question"],
    "scoring": ["extraction_rule", "truncated_counts_as_wrong", "ci_level"],
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Parse the TOML file and verify that every required key is present."""
    path = Path(path)
    with path.open("rb") as f:  # tomllib requires a binary file handle
        cfg = tomllib.load(f)
    missing = [f"{sec}.{key}" for sec, keys in REQUIRED.items()
               for key in keys if key not in cfg.get(sec, {})]
    if missing:
        raise KeyError(f"{path}: missing config keys: {missing}")
    # Record which exact config produced a run: hash of the file's bytes.
    cfg["_meta"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return cfg
