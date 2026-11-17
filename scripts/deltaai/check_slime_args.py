"""Check the training flags against slime's REAL parser, inside the slime image (setup job, seconds).

Purpose : a training job must not start with a flag slime ignores or a value slime changes.
          slime parses Megatron's flags with ignore_unknown_args=True (slime/backends/
          megatron_utils/arguments.py), so a flag that neither SGLang's nor Megatron's/slime's
          parser knows is DROPPED SILENTLY, and the run would train with that setting's default.
          This script:
            1. strict pass: SGLang's parser takes its --sglang-* flags, then Megatron's parser
               with slime's arguments must accept every remaining flag (unknown flag = FAIL);
            2. full pass: slime's own parse_args() with all its validation (HF config vs
               architecture, ref-load present, batch arithmetic, ...), exactly as train.py runs it;
            3. every value slime ends up with equals src.train_args.expected_resolved(config);
            4. our three plug-ins load through slime's own loader, and the generate hook takes
               `evaluation` (slime only passes it when the signature names it).
Inputs  : the flags file written by src/train_args.py (one flag or value per line);
          configs/train_reinit.toml (via src.train_args).
Output  : one line per check; exit 1 on any failure. Nothing is started or written.
Run     : inside the slime image with the GPU visible (slime_setup.sbatch step 4):
            python3 scripts/deltaai/check_slime_args.py runs/<run>/slime_args_setup.txt
"""
import argparse
import inspect
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))   # our plug-ins (src.*) must import as they will in training

from src.train_args import DATA_SOURCE, GENERATE_HOOK, REWARD_HOOK, expected_resolved, load_train_config  # noqa: E402

MISSING = object()


def report(name, ok, detail=""):
    print(f"[check_slime_args] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    return ok


def strict_unknown_flags(flags):
    """Flags no parser in slime's chain accepts (the ones slime would drop silently)."""
    from megatron.training.arguments import parse_args as megatron_parse_args
    from slime.backends.sglang_utils.arguments import add_sglang_arguments
    from slime.utils.arguments import get_slime_extra_args_provider

    sglang_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_sglang_arguments(sglang_parser)
    _, rest = sglang_parser.parse_known_args(flags)   # what slime's SGLang phase leaves over
    sys.argv = ["train.py"] + rest
    try:
        megatron_parse_args(extra_args_provider=get_slime_extra_args_provider(), ignore_unknown_args=False)
    except SystemExit as e:   # argparse prints "unrecognized arguments: ..." to stderr
        return f"Megatron/slime parser rejected the flags (exit {e.code}; see the message above)"
    return None


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
        except (TypeError, ValueError):
            return False
    return a == b


def main():
    flags = [line.rstrip("\n") for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    results = []

    problem = strict_unknown_flags(flags)
    results.append(report("every flag known to slime's parsers", problem is None, problem or f"({len(flags)} tokens)"))

    from slime.utils.arguments import parse_args
    sys.argv = ["train.py"] + flags
    try:
        args = parse_args()
        results.append(report("slime parse_args + validation", True))
    except BaseException as e:   # SystemExit / AssertionError / ValueError from slime's validation
        results.append(report("slime parse_args + validation", False, f"{type(e).__name__}: {e}"))
        args = None

    if args is not None:
        exp = expected_resolved(load_train_config())
        bad = {k: (getattr(args, k, MISSING), v) for k, v in exp.items() if not same(getattr(args, k, MISSING), v)}
        bad = {k: ("<missing>" if got is MISSING else got, want) for k, (got, want) in bad.items()}
        results.append(report(f"{len(exp)} resolved values = configs/train_reinit.toml", not bad,
                              f"mismatches (got, expected): {bad}" if bad else ""))

    from slime.utils.misc import load_function
    try:
        gen = load_function(GENERATE_HOOK)
        load_function(REWARD_HOOK)
        load_function(DATA_SOURCE)
        has_eval = "evaluation" in inspect.signature(gen).parameters
        results.append(report("plug-ins load (generate, reward, data source)", has_eval,
                              "" if has_eval else "generate hook lost its `evaluation` parameter"))
    except Exception as e:
        results.append(report("plug-ins load (generate, reward, data source)", False, f"{type(e).__name__}: {e}"))

    print(f"[check_slime_args] {'ALL PASS' if all(results) else 'FAILED'} ({sum(results)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
