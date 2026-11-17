"""Unit tests for src/slime_hooks.py. Run: python3 tests/run_all.py

slime is not installed locally, so small stand-in modules imitate the two pieces of slime the
hooks touch: slime.data.data_source (whose Dataset asserts list prompts when a processor exists,
as slime/utils/data.py does) and slime.rollout.sglang_rollout (whose generate receives the final
sampling params). The real thing is exercised on DeltaAI in gate G4.
"""
import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

import src.slime_hooks as hooks

SAMPLING = hooks.config()["sampling"]


def _install_fake_slime(calls):
    ds = types.ModuleType("slime.data.data_source")
    ds.load_processor = lambda *a, **k: "an image processor"

    class Buffer:                       # imitates RolloutDataSourceWithBuffer's dataset build
        def __init__(self, args):
            processor = ds.load_processor(args.hf_checkpoint)
            prompts = ["<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<analysis>\n"]
            if processor:               # slime/utils/data.py: processor => prompt must be a list
                assert all(isinstance(p, list) for p in prompts), "prompt must be a list"
            self.dataset = SimpleNamespace(origin_samples=[SimpleNamespace(prompt=p) for p in prompts])
    ds.RolloutDataSourceWithBuffer = Buffer

    sr = types.ModuleType("slime.rollout.sglang_rollout")
    sr.load_processor = lambda *a, **k: "an image processor"

    async def generate(args, sample, sampling_params):
        calls.append(dict(sampling_params))
        return sample
    sr.generate = generate
    for name in ("slime", "slime.data", "slime.rollout"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["slime.data.data_source"], sys.modules["slime.rollout.sglang_rollout"] = ds, sr


def _remove_fake_slime():
    for name in ("slime.data.data_source", "slime.rollout.sglang_rollout", "slime.data", "slime.rollout", "slime"):
        sys.modules.pop(name, None)
    importlib.reload(hooks)


def slime_params(**over):
    p = {"temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"], "top_k": SAMPLING["top_k"],
         "max_new_tokens": 32768, "no_stop_trim": True}
    p.update(over)
    return p


def test_the_three_missing_sampling_values_are_added():
    out = hooks.control_sampling_params(slime_params(), SAMPLING)
    assert out["presence_penalty"] == 1.5 and out["min_p"] == 0.0 and out["repetition_penalty"] == 1.0
    assert out["max_new_tokens"] == 32768 and out["no_stop_trim"] is True   # slime's own keys kept


def test_mismatching_sampling_is_refused():
    for bad in ({"temperature": 0.6}, {"top_p": 1.0}, {"top_k": -1}, {"presence_penalty": 0.0}):
        try:
            hooks.control_sampling_params(slime_params(**bad), SAMPLING)
        except ValueError:
            continue
        raise AssertionError(f"expected refusal for {bad}")


def test_input_params_are_not_modified():
    p = slime_params()
    hooks.control_sampling_params(p, SAMPLING)
    assert "presence_penalty" not in p


def test_generate_sends_control_sampling_and_marks_split():
    calls = []
    _install_fake_slime(calls)
    try:
        importlib.reload(hooks)
        s = SimpleNamespace(metadata={"qid": "syn"})
        asyncio.run(hooks.generate(None, s, slime_params(), evaluation=True))
        assert calls[0]["presence_penalty"] == 1.5 and s.metadata["_split"] == "eval"
        asyncio.run(hooks.generate(None, s, slime_params()))
        assert s.metadata["_split"] == "train"
    finally:
        _remove_fake_slime()


def test_text_only_data_source_keeps_plain_string_prompts():
    calls = []
    _install_fake_slime(calls)
    try:
        # without our data source, slime's dataset build rejects our string prompts
        try:
            sys.modules["slime.data.data_source"].RolloutDataSourceWithBuffer(SimpleNamespace(hf_checkpoint="x"))
            raise AssertionError("the stand-in should reject string prompts while a processor exists")
        except AssertionError as e:
            assert "must be a list" in str(e)
        importlib.reload(hooks)
        src = hooks.TextOnlyDataSource(SimpleNamespace(hf_checkpoint="x", apply_chat_template=False, multimodal_keys=None))
        assert src.dataset.origin_samples[0].prompt.endswith("<analysis>\n")
        assert sys.modules["slime.rollout.sglang_rollout"].load_processor() is None   # eval builder too
    finally:
        _remove_fake_slime()


def test_chat_template_or_images_are_refused():
    calls = []
    _install_fake_slime(calls)
    try:
        importlib.reload(hooks)
        for args in (SimpleNamespace(hf_checkpoint="x", apply_chat_template=True, multimodal_keys=None),
                     SimpleNamespace(hf_checkpoint="x", apply_chat_template=False, multimodal_keys={"image": "img"})):
            try:
                hooks.TextOnlyDataSource(args)
            except ValueError:
                continue
            raise AssertionError("expected refusal")
    finally:
        _remove_fake_slime()
