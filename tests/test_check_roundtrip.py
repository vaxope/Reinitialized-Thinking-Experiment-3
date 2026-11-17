"""Unit tests for src/check_roundtrip.py on small synthetic .safetensors files.
Run: python3 tests/run_all.py  (standard library only)."""
import json
import struct
import tempfile
from pathlib import Path

from src.check_roundtrip import compare, f32_to_bf16_bytes

P = "model.language_model."


def write_st(folder: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """Write a .safetensors file: {name: (dtype, shape, raw bytes)}."""
    header, blob, off = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blob += raw; off += len(raw)
    h = json.dumps(header).encode()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + blob)


def f32(*vals):
    return struct.pack(f"<{len(vals)}f", *vals)


def base():
    return {P + "layers.0.mlp.up_proj.weight": ("BF16", [2, 2], bytes(range(8))),
            P + "layers.0.linear_attn.A_log": ("F32", [3], f32(0.1, -2.5, 3.14159)),
            "model.visual.blocks.0.attn.qkv.weight": ("BF16", [2], b"\x01\x02\x03\x04")}


def run(orig, rt):
    d = Path(tempfile.mkdtemp())
    write_st(d / "a", orig); write_st(d / "b", rt)
    return compare(d / "a", d / "b", P)


def test_identical_language_model_passes_and_vision_is_ignored():
    rt = {k: v for k, v in base().items() if not k.startswith("model.visual")}   # vision not converted
    rep = run(base(), rt)
    assert rep["ok"] and rep["identical"] == 2 and rep["compared"] == 2


def test_fp32_returned_as_exactly_rounded_bf16_passes_and_is_counted():
    rt = dict(base())
    name = P + "layers.0.linear_attn.A_log"
    rt[name] = ("BF16", [3], f32_to_bf16_bytes(base()[name][2]))
    rep = run(base(), rt)
    assert rep["ok"] and rep["rounded_to_bf16"] == [name]


def test_wrongly_rounded_or_changed_values_fail():
    rt = dict(base()); name = P + "layers.0.linear_attn.A_log"
    rt[name] = ("BF16", [3], b"\x00\x00\x00\x00\x00\x00")
    assert not run(base(), rt)["ok"]
    rt = dict(base()); name = P + "layers.0.mlp.up_proj.weight"
    rt[name] = ("BF16", [2, 2], bytes([0, 1, 2, 3, 4, 5, 6, 99]))
    rep = run(base(), rt)
    assert not rep["ok"] and rep["different"] == [name]


def test_missing_tensor_or_shape_change_fails():
    rt = dict(base()); rt.pop(P + "layers.0.mlp.up_proj.weight")
    assert run(base(), rt)["missing"]
    rt = dict(base()); rt[P + "layers.0.mlp.up_proj.weight"] = ("BF16", [4, 1], bytes(range(8)))
    assert run(base(), rt)["shape"]


def test_bf16_rounding_matches_known_values():
    # 1.0 is exact in bf16 (0x3F80); 1 + 2^-8 is a tie -> rounds to even (0x3F80);
    # 1 + 3*2^-9 rounds up to 0x3F81
    raw = struct.pack("<3f", 1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -9)
    assert f32_to_bf16_bytes(raw) == struct.pack("<3H", 0x3F80, 0x3F80, 0x3F81)
