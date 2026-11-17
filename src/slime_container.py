"""Map configs/slime_container.toml to the build arguments of containers/slime_gh200.def.

Purpose : the recipe (containers/slime_gh200.def) contains no versions; it reads every pin as
          an Apptainer build argument ({{ NAME }}). This module is the single mapping from the
          config to those arguments, used by scripts/deltaai/build_slime.sbatch and checked by
          tests/test_slime_container.py (recipe and config must agree exactly).
Inputs  : configs/slime_container.toml.
Output  : NAME=VALUE lines (one per build argument), or one config value (--sif-name, ...).
Pipeline: configs/slime_container.toml --[slime_container]--> apptainer build --build-arg ...
          containers/slime_gh200.def --> the slime .sif used by the training job.

Usage   : python3 -m src.slime_container --build-args | --sif-name | --sif-sha256 | --fakeroot-test-image
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from src.config import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "slime_container.toml"
RECIPE = REPO_ROOT / "containers" / "slime_gh200.def"


def load(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def build_args(cfg: dict) -> dict[str, str]:
    """Recipe variable -> value. Lists become space-separated strings."""
    b, s, p, k = cfg["base"], cfg["slime"], cfg["pins"], cfg["build"]
    args = {
        "BASE_IMAGE": b["image"], "BASE_DIGEST": b["digest_arm64"],
        "SLIME_REPO": s["repo"], "SLIME_COMMIT": s["commit"], "PATCH_SET": s["patch_set"],
        "MEGATRON_PATCHES": " ".join(s["megatron_patches"]),
        "SGLANG_PATCHES": " ".join(s["sglang_patches"]),
        "MEGATRON_REPO": p["megatron_repo"], "MEGATRON_COMMIT": p["megatron_commit"],
        "TE_VERSION": p["transformer_engine"], "FLA_VERSION": p["flash_linear_attention"],
        "NRE_VERSION": p["nvidia_resiliency_ext"],
        "TMS_REPO": p["torch_memory_saver_repo"], "TMS_COMMIT": p["torch_memory_saver_commit"],
        "NCCL_VERSION": p["nccl"], "ROUTER_VERSION": p["sglang_router"],
        "NUMPY_VERSION": p["numpy"], "SCIPY_VERSION": p["scipy"], "PYARROW_VERSION": p["pyarrow"],
        "KERNELS_BELOW": p["kernels_below"],
        "TORCH_CUDA_ARCH_LIST": k["torch_cuda_arch_list"], "TE_CUDA_ARCHS": k["transformer_engine_cuda_archs"],
        "MAX_JOBS": str(k["max_jobs"]),
    }
    empty = [name for name, v in args.items() if not str(v).strip()]
    if empty:
        raise ValueError(f"empty build arguments: {empty}")
    return args


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build-args", action="store_true")
    g.add_argument("--sif-name", action="store_true")
    g.add_argument("--sif-sha256", action="store_true")
    g.add_argument("--fakeroot-test-image", action="store_true")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if a.build_args:
        sys.stdout.write("".join(f"{k}={v}\n" for k, v in build_args(cfg).items()))
    elif a.sif_name:
        print(cfg["output"]["sif_name"])
    elif a.sif_sha256:
        print(cfg["output"]["sif_sha256"])
    else:
        print(cfg["build"]["fakeroot_test_image"])


if __name__ == "__main__":
    main()
