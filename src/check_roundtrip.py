"""Check a HF -> Megatron -> HF weight round trip: every language-model weight must come back.

Purpose : slime trains Megatron-format weights converted from the Hugging Face checkpoint
          (tools/convert_hf_to_torch_dist.py). If that conversion scrambled a tensor (e.g. the new
          linear-attention layers of Qwen3.5), training would still run, from a broken model. This
          check converts the Megatron copy back (tools/convert_torch_dist_to_hf.py) and compares it
          with the original, tensor by tensor:
            - same dtype            -> the bytes must be identical;
            - fp32 -> bf16          -> must equal the original rounded to bf16 (round-to-nearest-
                                       even); Megatron trains in bf16, so this rounding is expected
                                       and is counted, not hidden;
            - anything else, a missing tensor or a shape change -> FAIL.
          Only tensors under --prefix (the language model) are compared: slime does not convert
          the vision tower or the multi-token-prediction head, which training does not use.
Inputs  : two folders of .safetensors files (original snapshot, round-trip output).
Output  : a report; exit code 1 on any failure. Standard library only (reads the safetensors
          format directly: 8-byte header length, JSON header, raw tensor bytes).
Pipeline: scripts/deltaai/slime_setup.sbatch, after the conversion, before any training job.

Usage   : python3 -m src.check_roundtrip ORIGINAL_DIR ROUNDTRIP_DIR [--prefix model.language_model.]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

DEFAULT_PREFIX = "model.language_model."
BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def read_index(folder: Path) -> dict[str, dict]:
    """name -> {file, dtype, shape, start, end} for every tensor in every .safetensors file."""
    out = {}
    for f in sorted(Path(folder).glob("*.safetensors")):
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        base = 8 + n
        for name, v in header.items():
            if name == "__metadata__":
                continue
            start, end = v["data_offsets"]
            out[name] = {"file": f, "dtype": v["dtype"], "shape": v["shape"],
                         "start": base + start, "end": base + end}
    return out


def read_bytes(t: dict) -> bytes:
    with t["file"].open("rb") as fh:
        fh.seek(t["start"])
        return fh.read(t["end"] - t["start"])


def f32_to_bf16_bytes(raw: bytes) -> bytes:
    """Round little-endian fp32 values to bf16 (round to nearest, ties to even), as PyTorch does."""
    n = len(raw) // 4
    out = bytearray(2 * n)
    for i, (u,) in enumerate(struct.iter_unpack("<I", raw)):
        if (u & 0x7F800000) == 0x7F800000 and (u & 0x007FFFFF):   # NaN: keep it a quiet NaN
            r = (u >> 16) | 0x0040
        else:
            r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        struct.pack_into("<H", out, 2 * i, r & 0xFFFF)
    return bytes(out)


def compare(original: Path, roundtrip: Path, prefix: str = DEFAULT_PREFIX) -> dict:
    a, b = read_index(original), read_index(roundtrip)
    names = sorted(n for n in a if n.startswith(prefix))
    rep = {"compared": 0, "identical": 0, "rounded_to_bf16": [], "missing": [], "shape": [],
           "dtype": [], "different": [], "extra": sorted(n for n in b if n.startswith(prefix) and n not in a)}
    for name in names:
        if name not in b:
            rep["missing"].append(name); continue
        x, y = a[name], b[name]
        rep["compared"] += 1
        if x["shape"] != y["shape"]:
            rep["shape"].append(f"{name}: {x['shape']} -> {y['shape']}"); continue
        if x["dtype"] == y["dtype"]:
            if read_bytes(x) == read_bytes(y):
                rep["identical"] += 1
            else:
                rep["different"].append(name)
        elif x["dtype"] == "F32" and y["dtype"] == "BF16":
            if f32_to_bf16_bytes(read_bytes(x)) == read_bytes(y):
                rep["rounded_to_bf16"].append(name)
            else:
                rep["different"].append(name)
        else:
            rep["dtype"].append(f"{name}: {x['dtype']} -> {y['dtype']}")
    rep["n_original_under_prefix"] = len(names)
    rep["ok"] = bool(names) and not (rep["missing"] or rep["shape"] or rep["dtype"] or rep["different"])
    return rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("original")
    ap.add_argument("roundtrip")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    a = ap.parse_args(argv)
    rep = compare(Path(a.original), Path(a.roundtrip), a.prefix)
    print(f"[check_roundtrip] {'PASS' if rep['ok'] else 'FAIL'}: {rep['compared']}/{rep['n_original_under_prefix']} "
          f"tensors compared, {rep['identical']} identical, {len(rep['rounded_to_bf16'])} fp32->bf16 rounded exactly")
    for key in ("missing", "shape", "dtype", "different", "extra", "rounded_to_bf16"):
        if rep[key]:
            print(f"  {key} ({len(rep[key])}): {rep[key][:10]}")
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
