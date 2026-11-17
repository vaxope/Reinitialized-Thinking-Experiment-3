"""Plug-ins that make slime's rollouts match the control run. Loaded by slime by import path.

Purpose : two adjustments, through slime's documented extension points (no slime code edited):

  1. `generate` (--custom-generate-function-path src.slime_hooks.generate)
     slime sends only temperature, top_p and top_k to SGLang (slime/rollout/sglang_rollout.py,
     GenerateState.sampling_params). The control run also used min_p 0.0, presence_penalty 1.5
     and repetition_penalty 1.0. This wrapper adds those three from [sampling], checks the other
     three equal [sampling] too, marks the sample as train or eval (for the rollout log), then
     calls slime's own generate. Everything else (token-id prompt, logprobs, abort handling) is
     slime's unchanged code.

  2. `TextOnlyDataSource` (--data-source-path src.slime_hooks.TextOnlyDataSource)
     Qwen3.5-2B ships an image processor. When slime finds one it assumes prompts are chat-message
     lists (slime/utils/data.py Dataset: `if processor: assert isinstance(prompt, list)`), but our
     prompts are finished strings from src/prompts.build_text: the reinit prompt ends with
     "<analysis>\\n" after the assistant prefix, which a chat template cannot produce. Our prompts
     are text only, so this data source makes slime's processor lookup return None (in the data
     source and in the eval-dataset builder). slime then keeps each prompt string and tokenizes
     it with tokenizer.encode(text, add_special_tokens=False), exactly as src/generate.py did for
     the control run. It is otherwise slime's default RolloutDataSourceWithBuffer, unchanged.

Inputs  : the training config ($GAR_TRAIN_CONFIG, default configs/train_reinit.toml).
Pipeline: slime rollout -> [generate] -> SGLang /generate -> src/reward.py -> GRPO.
Testing : tests/test_slime_hooks.py (with stand-in slime modules; slime is not installed locally).
          On DeltaAI the first generate call prints the exact sampling params it sends (gate G4).
"""
from __future__ import annotations

import os
import sys

from src.train_args import DEFAULT_TRAIN_CONFIG, load_train_config

# Sent to SGLang by slime itself (from slime flags built out of [sampling]): checked here.
CHECKED_KEYS = ("temperature", "top_p", "top_k")
# Not sent by slime: added here from [sampling].
ADDED_KEYS = ("min_p", "presence_penalty", "repetition_penalty")

_CFG = None
_ANNOUNCED = False


def config() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = load_train_config(os.environ.get("GAR_TRAIN_CONFIG", DEFAULT_TRAIN_CONFIG))
    return _CFG


def control_sampling_params(params: dict, sampling: dict) -> dict:
    """A copy of slime's sampling params with the control run's full sampling. Pure."""
    out = dict(params)
    for key in CHECKED_KEYS:
        if out.get(key) != sampling[key]:
            raise ValueError(f"slime sent {key}={out.get(key)!r}, but [sampling] {key}={sampling[key]!r}")
    for key in ADDED_KEYS:
        if key in out and out[key] != sampling[key]:
            raise ValueError(f"slime sent {key}={out[key]!r}, but [sampling] {key}={sampling[key]!r}")
        out[key] = sampling[key]
    return out


async def generate(args, sample, sampling_params, evaluation=False):
    """slime's generate, with the control run's sampling. slime passes `evaluation` because
    this function's signature names it (slime/rollout/sglang_rollout.py generate_and_rm)."""
    global _ANNOUNCED
    params = control_sampling_params(sampling_params, config()["sampling"])
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    meta["_split"] = "eval" if evaluation else "train"   # read by src/reward.py's rollout log
    sample.metadata = meta
    if not _ANNOUNCED:
        shown = {k: params.get(k) for k in CHECKED_KEYS + ADDED_KEYS + ("max_new_tokens",)}
        print(f"[slime_hooks] sampling params sent to SGLang: {shown}", flush=True)
        _ANNOUNCED = True
    from slime.rollout.sglang_rollout import generate as slime_generate  # slime's own function
    return await slime_generate(args, sample, params)


# ---------------------------------------------------------------- text-only data source
def _no_processor(*args, **kwargs):
    """Stand-in for slime's load_processor: our prompts never contain images."""
    return None


def use_text_only_prompts() -> list[str]:
    """Make slime's processor lookup return None where datasets and requests are built.
    Returns the module names patched (for the job log)."""
    patched = []
    for name in ("slime.data.data_source", "slime.rollout.sglang_rollout"):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                __import__(name)
                mod = sys.modules[name]
            except ImportError:
                continue
        if hasattr(mod, "load_processor"):
            mod.load_processor = _no_processor
            patched.append(name)
    return patched


try:  # slime is only importable inside the slime container
    from slime.data.data_source import RolloutDataSourceWithBuffer as _Base
except ImportError:  # local tests: a placeholder that explains itself if ever used
    class _Base:  # type: ignore[no-redef]
        def __init__(self, args):
            raise ImportError("slime is not installed: TextOnlyDataSource only runs inside the slime container")


class TextOnlyDataSource(_Base):
    """slime's default data source, with plain-string prompts kept exactly as written."""

    def __init__(self, args):
        if getattr(args, "apply_chat_template", False) or getattr(args, "multimodal_keys", None):
            raise ValueError("TextOnlyDataSource needs plain-text prompts: no --apply-chat-template, "
                             "no --multimodal-keys")
        patched = use_text_only_prompts()
        print(f"[slime_hooks] text-only prompts: processor disabled in {patched}", flush=True)
        super().__init__(args)
        dataset = getattr(self, "dataset", None)
        if dataset is not None:
            bad = sum(not isinstance(s.prompt, str) for s in dataset.origin_samples)
            if bad:
                raise ValueError(f"{bad} prompts are not plain strings")
            print(f"[slime_hooks] {len(dataset.origin_samples)} training prompts loaded as plain text",
                  flush=True)
