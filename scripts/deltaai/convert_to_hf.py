"""Convert a Megatron torch_dist checkpoint to Hugging Face format: slime's converter, made to read
the checkpoint format its own pinned Megatron writes.

Purpose : slime's tools/convert_torch_dist_to_hf.py (@ 79aa41a) reads the checkpoint's common data
          (the training `args` it needs: number of layers, experts, ...) from a separate
          `common.pt` file. The Megatron slime pins (ae76c60e) no longer writes that file: it stores
          the common data as one object ("common_state") inside the torch_dist checkpoint
          (megatron/core/dist_checkpointing/serialization.py, save(): sharded_state_dict
          ["common_state"]; load_common_state_dict() reads both formats). So slime's tool fails on
          every checkpoint this Megatron writes (setup job 3631056: "No such file or directory:
          './common.pt'"). This wrapper changes that read, and which entries are loaded:
            1. the args come from Megatron's own load_common_state_dict() (both formats); if that
               refuses (it unpickles with weights_only=True), the same object is read directly
               from the checkpoint with torch.load (our own checkpoint, so trusted);
            2. the weight tensors are loaded with the tool's own storage reader, but planned by
               TensorsOnlyPlanner: slime's planner skips only keys containing "optimizer" or
               "_state", so the non-tensor "content_metadata" entry this Megatron also writes
               (training/checkpointing.py) would be loaded as if it were a weight. Here slime's
               skips are kept and every other non-tensor entry is skipped too (keys printed);
            3. everything else is slime's tool itself, imported as a module (its `__main__` block
               does not run): its save_tensors() converts and writes the tensors, its
               copy_assets() copies tokenizer/config files.
          Same command-line flags as slime's tool (the ones we use).
Inputs  : --input-dir <checkpoint iteration folder (.metadata inside)>, --output-dir (must not exist),
          --origin-hf-dir <pinned HF snapshot>, [--add-missing-from-origin-hf], [--vocab-size N].
Output  : .safetensors files + copied assets in --output-dir. Never writes into --input-dir.
Pipeline: slime_setup.sbatch step 3 (round trip), smoke_train.sh step 4, eval_trained.sbatch step 2.
Run     : inside the slime image, PYTHONPATH=/opt/Megatron-LM:/opt/slime
            python3 scripts/deltaai/convert_to_hf.py --input-dir ... --output-dir ... --origin-hf-dir ...
"""
import argparse
import importlib.util
import io
import os
import sys
import time

SLIME_TOOL = "/opt/slime/tools/convert_torch_dist_to_hf.py"
COMMON_KEY = "common_state"        # megatron.core.dist_checkpointing.serialization.save()
CHUNK_SIZE = 5 * 1024 ** 3         # slime's tool default (--chunk-size), bytes per .safetensors file


def load_megatron_args(input_dir: str):
    """The training args saved in the checkpoint's common data (either checkpoint format)."""
    import torch
    import torch.distributed.checkpoint as dist_cp
    from megatron.core.dist_checkpointing.mapping import ShardedObject
    from megatron.core.dist_checkpointing.serialization import load_common_state_dict  # registers safe globals

    try:
        common = load_common_state_dict(input_dir)
        how = "Megatron load_common_state_dict"
    except Exception as e:   # e.g. a class not on Megatron's weights_only allow-list
        print(f"[convert_to_hf] Megatron's loader refused ({type(e).__name__}: {e}); reading {COMMON_KEY} directly")
        key = ShardedObject(COMMON_KEY, None, (1,), (0,)).unique_key
        buf = {key: io.BytesIO()}
        dist_cp.load(buf, storage_reader=dist_cp.FileSystemReader(input_dir), no_dist=True)
        obj = buf[key]
        if isinstance(obj, io.BytesIO):
            obj.seek(0)
            obj = torch.load(obj, weights_only=False)   # our own checkpoint
        common = obj[0] if isinstance(obj, (list, tuple)) else obj
        how = f"{COMMON_KEY} read directly"
    if "args" not in common:
        sys.exit(f"[convert_to_hf] no 'args' in the checkpoint's common data (keys: {sorted(common)})")
    print(f"[convert_to_hf] args from {how}: num_layers={common['args'].num_layers}")
    return common["args"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--origin-hf-dir", required=True)
    ap.add_argument("--add-missing-from-origin-hf", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=None)
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.input_dir, ".metadata")):
        sys.exit(f"[convert_to_hf] {a.input_dir} is not a torch_dist checkpoint (no .metadata)")
    if os.path.exists(a.output_dir):
        sys.exit(f"[convert_to_hf] {a.output_dir} already exists; not overwriting")

    # 1. args, BEFORE importing slime's tool: it replaces pickle.Unpickler globally
    megatron_args = load_megatron_args(a.input_dir)

    # 2. slime's tool as a module
    spec = importlib.util.spec_from_file_location("slime_convert_torch_dist_to_hf", SLIME_TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    import torch
    import torch.distributed.checkpoint as dist_cp
    from transformers import AutoConfig

    class TensorsOnlyPlanner(dist_cp.default_planner.DefaultLoadPlanner):
        """Load every weight tensor in the checkpoint and nothing else."""
        def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
            skipped = []
            for k, v in metadata.state_dict_metadata.items():
                # slime's own skips ("optimizer", "_state": TE's _extra_state, rng and common state),
                # plus anything that is not a tensor (e.g. content_metadata)
                if "optimizer" in k or "_state" in k or not isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                    skipped.append(k)
                    continue
                state_dict[k] = torch.empty(v.size, dtype=v.properties.dtype)
            opt = sum("optimizer" in k for k in skipped)
            other = [k for k in skipped if "optimizer" not in k]
            print(f"[convert_to_hf] loading {len(state_dict)} tensors; skipping {opt} optimizer entries and "
                  f"{len(other)} others (e.g. {[k for k in other if '_extra_state' not in k][:10]})")
            super().set_up_planner(state_dict, metadata, is_coordinator)

    model_name = type(AutoConfig.from_pretrained(a.origin_hf_dir, trust_remote_code=True)).__name__.lower()
    state_dict, t = {}, time.time()
    dist_cp.state_dict_loader._load_state_dict(state_dict, storage_reader=tool.WrappedStorageReader(a.input_dir),
                                               planner=TensorsOnlyPlanner(), no_dist=True)
    print(f"[convert_to_hf] {len(state_dict)} tensors loaded in {time.time() - t:.1f} s (model {model_name})")
    tool.save_tensors(megatron_args, model_name, state_dict, a.output_dir, CHUNK_SIZE, a.vocab_size,
                      a.origin_hf_dir if a.add_missing_from_origin_hf else None)
    tool.copy_assets(a.origin_hf_dir, a.output_dir)
    print(f"[convert_to_hf] done -> {a.output_dir}")


if __name__ == "__main__":
    main()
