"""Tests for the slime image pieces: configs/slime_container.toml, src/slime_container.py,
containers/slime_gh200.def and the build scripts. Run: python3 tests/run_all.py

The image itself can only be built on DeltaAI; these checks make sure the recipe and the config
agree exactly before any paid build job: every placeholder in the recipe has a value from the
config, every pin in the config is used, and no version is hard-coded in the recipe.
"""
import py_compile
import re
import shutil
import subprocess

from src.config import REPO_ROOT
from src.slime_container import RECIPE, build_args, load

CFG = load()
PLACEHOLDER = re.compile(r"\{\{\s*([A-Z0-9_]+)\s*\}\}")


def recipe_lines():
    return RECIPE.read_text().splitlines()


def code_lines():
    """Recipe lines that are not comments (the python heredocs' lines count as code)."""
    return [l for l in recipe_lines() if not l.lstrip().startswith("#")]


def test_every_placeholder_has_a_value_and_every_value_is_used():
    used = set(PLACEHOLDER.findall(RECIPE.read_text()))
    given = set(build_args(CFG))
    assert used == given, (sorted(used - given), sorted(given - used))


def test_no_placeholder_inside_a_comment():
    # Apptainer substitutes placeholders everywhere, comments included
    bad = [l for l in recipe_lines() if l.lstrip().startswith("#") and "{{" in l]
    assert not bad, bad


def test_no_version_or_commit_hard_coded_in_the_recipe():
    version = re.compile(r"(?<![\w.])\d+\.\d+\.\d+(?![\w.])")
    commit = re.compile(r"\b[0-9a-f]{40}\b")
    bad = [l for l in code_lines() if version.search(l) or commit.search(l) or "sha256:" in l]
    assert not bad, bad


def test_build_args_are_complete_and_lists_are_flattened():
    a = build_args(CFG)
    assert a["BASE_DIGEST"].startswith("sha256:") and len(a["SLIME_COMMIT"]) == 40
    assert a["MEGATRON_PATCHES"].split()[0] == "megatron.patch"     # required patch first
    assert a["SGLANG_PATCHES"].split()[0] == "sglang.patch"
    assert a["TORCH_CUDA_ARCH_LIST"] == "9.0"                        # GH200 = sm90 only


def test_pins_follow_the_plan():
    assert CFG["base"]["tag"] == "v0.5.21" and CFG["slime"]["patch_set"] == "v0.5.21"
    assert CFG["slime"]["commit"].startswith("79aa41a")             # the slime commit studied
    # the image must hold what Qwen3.5 GRPO needs: linear-attention kernels and colocation
    assert CFG["pins"]["flash_linear_attention"] and CFG["pins"]["torch_memory_saver_commit"]


def test_scripts_parse():
    py_compile.compile(str(REPO_ROOT / "scripts" / "deltaai" / "check_slime_gpu.py"), doraise=True)
    bash = shutil.which("bash")
    for name in ("build_slime.sbatch", "submit.sh", "slime_common.sh", "slime_setup.sbatch",
                 "train_reinit.sbatch", "train_inside.sh"):
        r = subprocess.run([bash, "-n", str(REPO_ROOT / "scripts" / "deltaai" / name)], capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stderr)
