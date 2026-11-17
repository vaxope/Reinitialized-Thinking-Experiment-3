"""Turn configs/train_reinit.toml into slime's command-line flags, after checking it.

Purpose : the ONLY place where training parameters become slime flags, so the flags slime
          sees always equal the confirmed parameter sheet. Before producing anything it checks:
            1. sampling and the reward rule equal the control run's (configs/control.toml);
            2. questions x answers per question = answers per optimizer update (one update
               per step, fully on-policy);
            3. plain GRPO: no per-token loss, TIS, dynamic sampling or entropy bonus, and no
               flag for any of them can appear in the output;
            4. (on DeltaAI) the Megatron architecture numbers equal the model's config.json.
Inputs  : configs/train_reinit.toml, configs/control.toml, and paths that differ per user/run
          (HF checkpoint, Megatron checkpoint folders, slime input files) given on the CLI.
Output  : the flag list, one flag or value per line (the Slurm script reads it into an array),
          also written to runs/<run>/slime_args.txt so every run records what it ran with.
Pipeline: configs/train_reinit.toml --[train_args]--> flags --> `python train.py <flags>`
          (slime, inside the slime container, scripts/deltaai/train_reinit.sbatch).

Usage   : python3 -m src.train_args --hf-checkpoint DIR --ref-load DIR --save DIR \\
              [--config configs/train_reinit.toml] [--out FILE]
          python3 -m src.train_args --hf-checkpoint DIR --model-args   (conversion flags only)
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from src.config import REPO_ROOT, load_config

DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train_reinit.toml"

# Our plug-ins, as import paths slime loads with its load_function (the repo root must be on
# PYTHONPATH inside the Ray workers; the Slurm script sets it).
DATA_SOURCE = "src.slime_hooks.TextOnlyDataSource"   # keeps our prompt text exactly as built
GENERATE_HOOK = "src.slime_hooks.generate"           # adds min_p / presence / repetition penalty
REWARD_HOOK = "src.reward.reward"                     # rule v3 reward + per-rollout log

# slime flags that would switch on something other than plain GRPO. build_args never emits
# them; check_flags refuses any list that contains one (protects against later edits).
FORBIDDEN_FLAGS = (
    "--calculate-per-token-loss",       # DAPO per-token averaging
    "--use-tis",                         # truncated importance sampling
    "--eps-clip-high",                   # DAPO clip-higher
    "--dynamic-sampling-filter-path",    # DAPO dynamic sampling
    "--over-sampling-batch-size",        # only used with dynamic sampling
    "--use-score-centering",
    "--apply-chat-template",             # our prompt text must reach the model unchanged
    "--disable-grpo-std-normalization",  # GRPO divides by the group std (Dr. GRPO removes it)
)

# Our [megatron_model] key -> key in the model's config.json (text_config).
HF_CONFIG_KEYS = {
    "num_layers": "num_hidden_layers",
    "hidden_size": "hidden_size",
    "ffn_hidden_size": "intermediate_size",
    "num_attention_heads": "num_attention_heads",
    "num_query_groups": "num_key_value_heads",
    "kv_channels": "head_dim",
    "vocab_size": "vocab_size",
    "rotary_base": "rope_theta",
    "rotary_percent": "partial_rotary_factor",
    "norm_epsilon": "rms_norm_eps",
    "max_position_embeddings": "max_position_embeddings",
}

REQUIRED = {
    "run": ["name", "seed", "num_steps"],
    "model": ["hf_repo", "revision", "chat_template_sha256"],
    "megatron_model": ["spec", "flags", "normalization", "position_embedding_type",
                       "attention_dropout", "hidden_dropout", *HF_CONFIG_KEYS],
    "numerics": ["accumulate_allreduce_grads_in_fp32", "attention_softmax_in_fp32"],
    "data": ["source_dir", "prompt_mode", "shuffle_salt", "output_dir", "input_key",
             "label_key", "metadata_key"],
    "reward": ["rule", "correct", "incorrect"],
    "sampling": ["temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "repetition_penalty"],
    "rollout": ["n_samples_per_prompt", "rollout_batch_size", "global_batch_size",
                "max_response_len", "max_prompt_len", "shuffle", "shuffle_seed"],
    "grpo": ["advantage_estimator", "std_normalization", "eps_clip", "use_kl_loss",
             "kl_loss_type", "kl_loss_coef", "per_token_loss", "entropy_coef", "use_tis",
             "dynamic_sampling"],
    "optimizer": ["optimizer", "lr", "lr_decay_style", "lr_warmup_iters", "adam_beta1",
                  "adam_beta2", "adam_eps", "weight_decay", "clip_grad"],
    "eval": ["interval", "n_samples_per_prompt", "max_response_len", "name"],
    "checkpoint": ["save_interval"],
    "hardware": ["num_gpus", "rollout_num_gpus_per_engine", "sglang_mem_fraction_static",
                 "max_tokens_per_gpu", "log_probs_chunk_size", "recompute_granularity",
                 "recompute_method", "recompute_num_layers"],
}


def load_train_config(path: str | Path = DEFAULT_TRAIN_CONFIG) -> dict:
    """Parse the training config and verify every required key is present."""
    path = Path(path)
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    missing = [f"{s}.{k}" for s, ks in REQUIRED.items() for k in ks if k not in cfg.get(s, {})]
    if missing:
        raise KeyError(f"{path}: missing config keys: {missing}")
    cfg["_meta"] = {"path": str(path)}
    return cfg


# ---------------------------------------------------------------- checks
def check_against_control(cfg: dict, control: dict) -> None:
    """Training must sample, score and start from exactly what the baseline used."""
    for key in REQUIRED["sampling"]:
        if cfg["sampling"][key] != control["sampling"][key]:
            raise ValueError(f"[sampling] {key} = {cfg['sampling'][key]!r} differs from the control "
                             f"run's {control['sampling'][key]!r} (configs/control.toml)")
    if cfg["reward"]["rule"] != control["scoring"]["extraction_rule"]:
        raise ValueError(f"[reward] rule {cfg['reward']['rule']!r} != control extraction_rule "
                         f"{control['scoring']['extraction_rule']!r}")
    for key in ("revision", "chat_template_sha256"):
        if cfg["model"][key] != control["model"][key]:
            raise ValueError(f"[model] {key} differs from configs/control.toml")
    if cfg["model"]["hf_repo"] != control["model"]["name"]:
        raise ValueError("[model] hf_repo differs from configs/control.toml [model] name")


def check_plain_grpo(cfg: dict) -> None:
    """The user's rule (2026-11-17): plain GRPO only."""
    g, r = cfg["grpo"], cfg["rollout"]
    if g["advantage_estimator"] != "grpo":
        raise ValueError("[grpo] advantage_estimator must be 'grpo'")
    if not g["std_normalization"]:
        raise ValueError("[grpo] std_normalization must be true (GRPO divides by the group std)")
    for key in ("per_token_loss", "use_tis", "dynamic_sampling"):
        if g[key]:
            raise ValueError(f"[grpo] {key} must be false (plain GRPO only)")
    if g["entropy_coef"] != 0:
        raise ValueError("[grpo] entropy_coef must be 0 (plain GRPO only)")
    if r["rollout_batch_size"] * r["n_samples_per_prompt"] != r["global_batch_size"]:
        raise ValueError("questions per step x answers per question must equal global_batch_size "
                         "(exactly one optimizer update per step)")
    if cfg["eval"]["max_response_len"] != r["max_response_len"]:
        raise ValueError("validation must use the training response limit")


def check_hf_config(cfg: dict, config_json: Path) -> None:
    """Every Megatron architecture number must equal the model's own config.json."""
    raw = json.loads(Path(config_json).read_text())
    text = raw.get("text_config", raw)          # Qwen3.5 nests the language model config
    rope = text.get("rope_parameters") or {}     # transformers 5 keeps rope values here
    bad = []
    for ours, theirs in HF_CONFIG_KEYS.items():
        got = text.get(theirs, rope.get(theirs))
        if got != cfg["megatron_model"][ours]:
            bad.append(f"{ours}={cfg['megatron_model'][ours]!r} but config.json {theirs}={got!r}")
    if bad:
        raise ValueError("[megatron_model] does not match the model's config.json: " + "; ".join(bad))


def check_flags(flags: list[str]) -> None:
    present = [f for f in flags if f in FORBIDDEN_FLAGS]
    if present:
        raise ValueError(f"flags outside plain GRPO in the output: {present}")


# ---------------------------------------------------------------- flag building
def _flag_value(v) -> str:
    if isinstance(v, bool):
        raise TypeError("booleans are switches, not values")
    return repr(v) if isinstance(v, float) else str(v)


def model_args(cfg: dict) -> list[str]:
    """The Megatron architecture flags alone. Used by build_args AND by the one-time weight
    conversion (tools/convert_hf_to_torch_dist.py), so both always describe the same model."""
    m = cfg["megatron_model"]
    a = ["--spec", *m["spec"], *m["flags"]]
    for key in HF_CONFIG_KEYS:
        a += ["--" + key.replace("_", "-"), _flag_value(m[key])]
    a += ["--normalization", m["normalization"], "--position-embedding-type", m["position_embedding_type"],
          "--attention-dropout", _flag_value(m["attention_dropout"]),
          "--hidden-dropout", _flag_value(m["hidden_dropout"])]
    return a


def build_args(cfg: dict, paths: dict) -> list[str]:
    """The slime command line, grouped as in slime's own run scripts.

    paths: hf_checkpoint (HF model folder), ref_load (Megatron copy of the start model, also the
    frozen KL reference), save (checkpoint folder; also --load, so a resubmitted job resumes),
    prompt_data and eval_data (slime input files from src/train_prep.py).
    """
    m, d, ro, g, o = cfg["megatron_model"], cfg["data"], cfg["rollout"], cfg["grpo"], cfg["optimizer"]
    ev, hw, nu, s = cfg["eval"], cfg["hardware"], cfg["numerics"], cfg["sampling"]
    a: list[str] = []

    def opt(flag, value):
        a.extend([flag, _flag_value(value)])

    # -- hardware: 1 node, training and generation share the GPU(s) and take turns
    a += ["--actor-num-nodes", "1"]
    opt("--actor-num-gpus-per-node", hw["num_gpus"])
    opt("--num-gpus-per-node", hw["num_gpus"])   # slime: set when using < 8 GPUs with --colocate
    a.append("--colocate")
    opt("--rollout-num-gpus-per-engine", hw["rollout_num_gpus_per_engine"])
    opt("--sglang-mem-fraction-static", hw["sglang_mem_fraction_static"])

    # -- model architecture (slime scripts/models/qwen3.5-*.sh format)
    a += model_args(cfg)

    # -- checkpoints: start = ref = converted base model; save and resume in one folder
    opt("--hf-checkpoint", paths["hf_checkpoint"])
    opt("--ref-load", paths["ref_load"])
    opt("--load", paths["save"])
    opt("--save", paths["save"])
    opt("--save-interval", cfg["checkpoint"]["save_interval"])
    opt("--seed", cfg["run"]["seed"])

    # -- data and our plug-ins
    opt("--prompt-data", paths["prompt_data"])
    opt("--input-key", d["input_key"])
    opt("--label-key", d["label_key"])
    opt("--metadata-key", d["metadata_key"])
    opt("--data-source-path", DATA_SOURCE)
    opt("--custom-generate-function-path", GENERATE_HOOK)
    opt("--custom-rm-path", REWARD_HOOK)
    if ro["shuffle"]:
        a.append("--rollout-shuffle")
    opt("--rollout-seed", ro["shuffle_seed"])

    # -- rollout: questions, answers, lengths, sampling (min_p/penalties added by the hook)
    opt("--num-rollout", cfg["run"]["num_steps"])
    opt("--rollout-batch-size", ro["rollout_batch_size"])
    opt("--n-samples-per-prompt", ro["n_samples_per_prompt"])
    opt("--global-batch-size", ro["global_batch_size"])
    opt("--rollout-max-response-len", ro["max_response_len"])
    opt("--rollout-max-prompt-len", ro["max_prompt_len"])
    opt("--rollout-temperature", s["temperature"])
    opt("--rollout-top-p", s["top_p"])
    opt("--rollout-top-k", s["top_k"])

    # -- validation: same sampling and limit, held-out SuperGPQA slice
    opt("--eval-interval", ev["interval"])
    a += ["--eval-prompt-data", ev["name"], str(paths["eval_data"])]
    opt("--eval-input-key", d["input_key"])
    opt("--eval-label-key", d["label_key"])
    opt("--n-samples-per-eval-prompt", ev["n_samples_per_prompt"])
    opt("--eval-max-response-len", ev["max_response_len"])
    opt("--eval-max-prompt-len", ro["max_prompt_len"])
    opt("--eval-temperature", s["temperature"])
    opt("--eval-top-p", s["top_p"])
    opt("--eval-top-k", s["top_k"])

    # -- plain GRPO
    opt("--advantage-estimator", g["advantage_estimator"])   # std normalisation is slime's default
    opt("--eps-clip", g["eps_clip"])
    if g["use_kl_loss"]:
        a.append("--use-kl-loss")
        opt("--kl-loss-type", g["kl_loss_type"])
        opt("--kl-loss-coef", g["kl_loss_coef"])
    opt("--entropy-coef", g["entropy_coef"])

    # -- optimizer (AdamW; every value explicit)
    for key in ("optimizer", "lr", "lr_decay_style", "lr_warmup_iters", "adam_beta1",
                "adam_beta2", "adam_eps", "weight_decay", "clip_grad"):
        opt("--" + key.replace("_", "-"), o[key])

    # -- memory / parallelism on one GPU
    a += ["--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
          "--context-parallel-size", "1"]
    a.append("--use-dynamic-batch-size")
    opt("--max-tokens-per-gpu", hw["max_tokens_per_gpu"])
    opt("--log-probs-chunk-size", hw["log_probs_chunk_size"])
    opt("--recompute-granularity", hw["recompute_granularity"])
    opt("--recompute-method", hw["recompute_method"])
    opt("--recompute-num-layers", hw["recompute_num_layers"])

    # -- numerics
    if nu["accumulate_allreduce_grads_in_fp32"]:
        a.append("--accumulate-allreduce-grads-in-fp32")
    if nu["attention_softmax_in_fp32"]:
        a.append("--attention-softmax-in-fp32")

    check_flags(a)
    return a


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG))
    ap.add_argument("--control-config", default=None, help="default: configs/control.toml")
    ap.add_argument("--hf-checkpoint", required=True, help="HF model folder (snapshot @ revision)")
    ap.add_argument("--ref-load", help="Megatron copy of the start model")
    ap.add_argument("--save", help="checkpoint folder (also used to resume)")
    ap.add_argument("--out", default=None, help="also write the flags here (one per line)")
    ap.add_argument("--model-args", action="store_true",
                    help="print only the architecture flags (for the weight conversion)")
    args = ap.parse_args(argv)

    cfg = load_train_config(args.config)
    if args.model_args:   # conversion: architecture checked against config.json, nothing else needed
        check_hf_config(cfg, Path(args.hf_checkpoint) / "config.json")
        sys.stdout.write("\n".join(model_args(cfg)) + "\n")
        return
    if not (args.ref_load and args.save):
        ap.error("--ref-load and --save are required unless --model-args")
    control = load_config(args.control_config) if args.control_config else load_config()
    check_against_control(cfg, control)
    check_plain_grpo(cfg)
    check_hf_config(cfg, Path(args.hf_checkpoint) / "config.json")
    out_dir = REPO_ROOT / cfg["data"]["output_dir"]
    paths = {"hf_checkpoint": args.hf_checkpoint, "ref_load": args.ref_load, "save": args.save,
             "prompt_data": out_dir / "train.jsonl", "eval_data": out_dir / "val.jsonl"}
    for key in ("prompt_data", "eval_data"):
        if not Path(paths[key]).exists():
            sys.exit(f"{paths[key]} not found: run `python3 -m src.train_prep` first")
    flags = build_args(cfg, paths)
    print(f"[train_args] {len(flags)} flags for run {cfg['run']['name']!r}", file=sys.stderr)
    text = "\n".join(flags) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
