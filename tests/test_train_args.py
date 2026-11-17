"""Unit tests for src/train_args.py (training config -> slime flags). Run: python3 tests/run_all.py

They check that the confirmed parameter sheet reaches slime unchanged, and that every guard
refuses what it should: sampling or reward differing from the control run, anything outside
plain GRPO, batch sizes that don't give one update per step, and architecture numbers that
differ from the model's config.json. No slime, GPU or network needed.
"""
import copy
import json
import tempfile
from pathlib import Path

from src.config import load_config
from src.train_args import (FORBIDDEN_FLAGS, HF_CONFIG_KEYS, build_args, check_against_control,
                            check_flags, check_hf_config, check_plain_grpo, load_train_config)

CFG = load_train_config()
CONTROL = load_config()
PATHS = dict(hf_checkpoint="HF", ref_load="REF", save="SAVE", prompt_data="P.jsonl", eval_data="E.jsonl")


def value(flags, flag):
    """The value that follows `flag` (one occurrence expected)."""
    assert flags.count(flag) == 1, (flag, flags.count(flag))
    return flags[flags.index(flag) + 1]


def raises(fn, *a):
    try:
        fn(*a)
    except (ValueError, KeyError):
        return True
    return False


def test_real_config_passes_all_checks():
    check_against_control(CFG, CONTROL)
    check_plain_grpo(CFG)


def test_confirmed_parameter_sheet_reaches_slime():
    f = build_args(CFG, PATHS)
    expected = {
        "--n-samples-per-prompt": "8", "--rollout-batch-size": "16", "--global-batch-size": "128",
        "--rollout-max-response-len": "32768", "--eval-max-response-len": "32768",
        "--advantage-estimator": "grpo", "--eps-clip": "0.2", "--kl-loss-type": "k3",
        "--kl-loss-coef": "0.001", "--entropy-coef": "0.0",
        "--optimizer": "adam", "--lr": "1e-06", "--lr-decay-style": "constant", "--lr-warmup-iters": "0",
        "--adam-beta1": "0.9", "--adam-beta2": "0.999", "--adam-eps": "1e-08",
        "--weight-decay": "0.01", "--clip-grad": "1.0",
        "--rollout-temperature": "1.0", "--rollout-top-p": "0.95", "--rollout-top-k": "20",
        "--eval-temperature": "1.0", "--eval-top-p": "0.95", "--eval-top-k": "20",
        "--actor-num-gpus-per-node": "1", "--attention-dropout": "0.0", "--hidden-dropout": "0.0",
        "--save-interval": "20", "--eval-interval": "20", "--num-rollout": "25",
    }
    for flag, want in expected.items():
        assert value(f, flag) == want, (flag, value(f, flag), want)
    for switch in ("--colocate", "--use-kl-loss", "--rollout-shuffle", "--use-dynamic-batch-size"):
        assert switch in f, switch


def test_resume_folder_and_reference_are_wired():
    f = build_args(CFG, PATHS)
    assert value(f, "--load") == value(f, "--save") == "SAVE"   # a resubmitted job resumes
    assert value(f, "--ref-load") == "REF"                       # frozen KL reference = start model
    assert value(f, "--hf-checkpoint") == "HF"


def test_our_plugins_are_used_and_no_chat_template():
    f = build_args(CFG, PATHS)
    assert value(f, "--custom-rm-path") == "src.reward.reward"
    assert value(f, "--custom-generate-function-path") == "src.slime_hooks.generate"
    assert value(f, "--data-source-path") == "src.slime_hooks.TextOnlyDataSource"
    assert "--apply-chat-template" not in f


def test_no_flag_outside_plain_grpo_is_ever_emitted():
    f = build_args(CFG, PATHS)
    assert not set(f) & set(FORBIDDEN_FLAGS)
    for bad in FORBIDDEN_FLAGS:
        assert raises(check_flags, ["--lr", "1e-6", bad])


def test_sampling_or_reward_differing_from_control_is_refused():
    for key, new in [("temperature", 0.7), ("top_p", 1.0), ("top_k", 50), ("min_p", 0.05),
                     ("presence_penalty", 0.0), ("repetition_penalty", 1.1)]:
        c = copy.deepcopy(CFG); c["sampling"][key] = new
        assert raises(check_against_control, c, CONTROL), key
    c = copy.deepcopy(CFG); c["reward"]["rule"] = "last_answer_field_v2"
    assert raises(check_against_control, c, CONTROL)
    c = copy.deepcopy(CFG); c["model"]["revision"] = "main"
    assert raises(check_against_control, c, CONTROL)


def test_anything_outside_plain_grpo_is_refused():
    for section, key, new in [("grpo", "per_token_loss", True), ("grpo", "use_tis", True),
                              ("grpo", "dynamic_sampling", True), ("grpo", "entropy_coef", 0.01),
                              ("grpo", "std_normalization", False), ("grpo", "advantage_estimator", "gspo"),
                              ("rollout", "global_batch_size", 256), ("eval", "max_response_len", 16384)]:
        c = copy.deepcopy(CFG); c[section][key] = new
        assert raises(check_plain_grpo, c), (section, key)


def _config_json(text_config: dict) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(json.dumps({"text_config": text_config}))
    return d / "config.json"


def test_architecture_must_match_config_json():
    good = {theirs: CFG["megatron_model"][ours] for ours, theirs in HF_CONFIG_KEYS.items()}
    check_hf_config(CFG, _config_json(good))                      # matches -> no error
    bad = dict(good, hidden_size=1024)
    assert raises(check_hf_config, CFG, _config_json(bad))
    # rope values may sit in rope_parameters (transformers 5 layout)
    moved = {k: v for k, v in good.items() if k not in ("rope_theta", "partial_rotary_factor")}
    moved["rope_parameters"] = {"rope_theta": good["rope_theta"], "partial_rotary_factor": good["partial_rotary_factor"]}
    check_hf_config(CFG, _config_json(moved))


def test_real_model_config_json_if_cached():
    snaps = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/"
                                  + CFG["model"]["revision"] + "/config.json"))
    if not snaps:
        return "SKIP: Qwen3.5-2B config.json not in the local HF cache"
    check_hf_config(CFG, snaps[0])
