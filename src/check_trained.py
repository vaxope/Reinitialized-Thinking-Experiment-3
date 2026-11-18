"""Check a trained checkpoint converted back to Hugging Face format, and fingerprint it.

Purpose : the evaluation must serve exactly "the base model with trained language-model weights",
          and nothing else may differ. slime's tools/convert_torch_dist_to_hf.py writes the trained
          language-model tensors; with --add-missing-from-origin-hf it copies the tensors training
          never touches (vision tower, multi-token-prediction head) from the base snapshot, and with
          --origin-hf-dir it copies config, tokenizer and chat template. This check requires:
            - tensor names: exactly the base model's (an extra lm_head.weight is allowed only when
              the base ties embeddings AND it equals the trained embedding bytes exactly);
            - every tensor outside --prefix (vision, MTP): byte-identical to the base;
            - every tensor under --prefix: same shape; same dtype, or fp32 -> bf16 (Megatron trains
              in bf16, as in src/check_roundtrip.py); AT LEAST ONE must differ from the base
              (otherwise training did not reach these weights, and the evaluation would just
              repeat the control run);
            - every non-weight file of the base (config.json, tokenizer, chat_template.jinja, ...):
              byte-identical (so prompts, template hash and architecture are unchanged).
          On success it writes <dir>/gar_checkpoint.json: SHA-256 of every file, one model_sha256
          over all of them, the counts above, and where the weights came from (--source).
Inputs  : BASE_DIR (pinned HF snapshot), TRAINED_DIR (converter output).
Output  : report lines; gar_checkpoint.json; exit 1 on any failure (nothing written then).
Pipeline: training -> src/pick_checkpoint.py -> convert -> [check_trained] -> src/generate.py
          --checkpoint TRAINED_DIR (refuses a folder without a passing gar_checkpoint.json).
Standard library only (safetensors read directly, as in src/check_roundtrip.py).
Usage   : python3 -m src.check_trained BASE_DIR TRAINED_DIR [--source '{"run": ..., "step": ...}']
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from src.check_roundtrip import DEFAULT_PREFIX, f32_to_bf16_bytes, read_bytes, read_index

MANIFEST = "gar_checkpoint.json"
WEIGHT_FILES = (".safetensors",)
INDEX_FILE = "model.safetensors.index.json"   # rewritten by the converter; its own content may differ
EMBED = "model.language_model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def non_weight_files(folder: Path) -> list[str]:
    return sorted(p.name for p in Path(folder).iterdir()
                  if p.is_file() and not p.name.endswith(WEIGHT_FILES) and p.name not in (INDEX_FILE, MANIFEST))


def compare(base: Path, trained: Path, prefix: str = DEFAULT_PREFIX) -> dict:
    a, b = read_index(base), read_index(trained)
    rep = {"n_tensors_base": len(a), "n_tensors_trained": len(b), "missing": sorted(set(a) - set(b)),
           "extra": [], "shape": [], "dtype": [], "frozen_changed": [], "lm_changed": 0, "lm_unchanged": 0,
           "lm_rounded_to_bf16": 0, "frozen_identical": 0, "files_differ": [], "files_missing": []}
    tied = json.loads((Path(base) / "config.json").read_text()).get("tie_word_embeddings", False)
    for name in sorted(set(b) - set(a)):
        ok_head = (name == LM_HEAD and tied and EMBED in b and b[name]["shape"] == b[EMBED]["shape"]
                   and read_bytes(b[name]) == read_bytes(b[EMBED]))
        rep["extra"].append({"name": name, "allowed": ok_head})
    for name in sorted(set(a) & set(b)):
        x, y = a[name], b[name]
        if x["shape"] != y["shape"]:
            rep["shape"].append(f"{name}: {x['shape']} -> {y['shape']}"); continue
        if not name.startswith(prefix):                      # frozen: must be the base's bytes
            if x["dtype"] == y["dtype"] and read_bytes(x) == read_bytes(y):
                rep["frozen_identical"] += 1
            else:
                rep["frozen_changed"].append(name)
            continue
        if x["dtype"] == y["dtype"]:
            same = read_bytes(x) == read_bytes(y)
        elif x["dtype"] == "F32" and y["dtype"] == "BF16":
            rep["lm_rounded_to_bf16"] += 1
            same = f32_to_bf16_bytes(read_bytes(x)) == read_bytes(y)
        else:
            rep["dtype"].append(f"{name}: {x['dtype']} -> {y['dtype']}"); continue
        rep["lm_unchanged" if same else "lm_changed"] += 1
    for f in non_weight_files(base):
        if not (Path(trained) / f).is_file():
            rep["files_missing"].append(f)
        elif (Path(base) / f).read_bytes() != (Path(trained) / f).read_bytes():
            rep["files_differ"].append(f)
    rep["ok"] = bool(a) and rep["lm_changed"] > 0 and not (
        rep["missing"] or rep["shape"] or rep["dtype"] or rep["frozen_changed"] or rep["files_differ"]
        or rep["files_missing"] or any(not e["allowed"] for e in rep["extra"]))
    return rep


def fingerprint(folder: Path) -> dict:
    files = {p.name: sha256_file(p) for p in sorted(Path(folder).iterdir()) if p.is_file() and p.name != MANIFEST}
    joined = "\n".join(f"{n} {h}" for n, h in sorted(files.items()))
    return {"files": files, "model_sha256": hashlib.sha256(joined.encode()).hexdigest()}


def load_manifest(folder: Path) -> dict:
    """The manifest of a checked folder; raises if missing, failed, or the files changed since."""
    path = Path(folder) / MANIFEST
    if not path.is_file():
        raise ValueError(f"{folder} has no {MANIFEST}: run src.check_trained on it first")
    m = json.loads(path.read_text())
    if not m.get("ok"):
        raise ValueError(f"{path} does not record a passing check")
    now = fingerprint(Path(folder))
    if now["model_sha256"] != m["model_sha256"]:
        changed = sorted(n for n in set(now["files"]) | set(m["files"]) if now["files"].get(n) != m["files"].get(n))
        raise ValueError(f"{folder}: files changed since the check: {changed}")
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base")
    ap.add_argument("trained")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--source", default="{}", help="JSON: where the weights came from (run, step, iter, job)")
    a = ap.parse_args(argv)
    rep = compare(Path(a.base), Path(a.trained), a.prefix)
    print(f"[check_trained] {'PASS' if rep['ok'] else 'FAIL'}: {rep['n_tensors_trained']} tensors "
          f"(base {rep['n_tensors_base']}); language model: {rep['lm_changed']} changed, {rep['lm_unchanged']} "
          f"unchanged, {rep['lm_rounded_to_bf16']} fp32->bf16; frozen (vision/MTP): {rep['frozen_identical']} "
          f"identical; non-weight files identical: {len(non_weight_files(Path(a.base))) - len(rep['files_differ']) - len(rep['files_missing'])}"
          f"/{len(non_weight_files(Path(a.base)))}")
    for key in ("missing", "extra", "shape", "dtype", "frozen_changed", "files_differ", "files_missing"):
        if rep[key]:
            print(f"  {key} ({len(rep[key])}): {rep[key][:10]}")
    if rep["lm_changed"] == 0:
        print("  no language-model tensor differs from the base: training did not change these weights")
    if not rep["ok"]:
        sys.exit(1)
    fp = fingerprint(Path(a.trained))
    manifest = {"ok": True, "base_dir": str(a.base), "source": json.loads(a.source), "check": rep, **fp}
    (Path(a.trained) / MANIFEST).write_text(json.dumps(manifest, indent=2))
    print(f"[check_trained] model_sha256 {fp['model_sha256']} -> {Path(a.trained) / MANIFEST}")


if __name__ == "__main__":
    main()
