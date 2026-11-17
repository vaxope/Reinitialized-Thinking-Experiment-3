"""GPU check of a freshly built slime image (runs at the end of build_slime.sbatch, ~1-2 min).

Purpose : catch a broken GPU stack before any paid training job: the parts of the image that
          training actually runs on the GPU, each checked against a plain reference.
            1. CUDA visible, GPU is sm90 (GH200).
            2. TransformerEngine: te.pytorch.Linear forward + backward = torch matmul.
            3. flash-linear-attention: chunk_gated_delta_rule (the fast kernel Qwen3.5's linear-
               attention layers train with) = fused_recurrent_gated_delta_rule (fla's step-by-step
               reference) on small random inputs, forward and gradients.
            4. torch_memory_saver imports (it is exercised for real in the first training step).
Inputs  : none (random tensors, fixed seed). No model weights, no data.
Output  : one line per check; exit 1 if any check fails. Run INSIDE the slime image:
            apptainer exec --nv <slime.sif> python3 scripts/deltaai/check_slime_gpu.py
"""
import sys

import torch

# Tolerances for bf16 kernels vs references computed from the same bf16 inputs. bf16 keeps
# ~3 significant digits, so element-wise differences of a few 1e-2 relative are normal; these
# bounds catch wrong results (orders of magnitude off), not rounding.
TE_ATOL, TE_RTOL = 5e-2, 5e-2
FLA_MAX_REL_ERR = 5e-2
SEED = 0


def report(name, ok, detail=""):
    print(f"[check_slime_gpu] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    return ok


def check_cuda():
    ok = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability() if ok else None
    return report("cuda", ok and cap == (9, 0), f"available={ok} capability={cap} "
                  f"device={torch.cuda.get_device_name() if ok else None}")


def check_te():
    import transformer_engine.pytorch as te
    torch.manual_seed(SEED)
    lin = te.Linear(256, 512, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = lin(x)
    ref = x @ lin.weight.t()
    y.float().sum().backward()
    ok = torch.allclose(y.float(), ref.float(), atol=TE_ATOL, rtol=TE_RTOL) and x.grad is not None
    return report("transformer_engine Linear", ok, f"max|diff|={(y - ref).abs().max().item():.3e}")


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def check_fla():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    torch.manual_seed(SEED)
    B, T, H, K, V = 1, 256, 2, 64, 64   # small shapes; the layout is [batch, time, heads, dim]
    dev, dt = "cuda", torch.bfloat16

    def inputs():
        torch.manual_seed(SEED)
        q = torch.randn(B, T, H, K, device=dev, dtype=dt)
        k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, dtype=torch.float32), dim=-1).to(dt)
        v = torch.randn(B, T, H, V, device=dev, dtype=dt)
        g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32))  # log decay < 0
        beta = torch.rand(B, T, H, device=dev, dtype=torch.float32).sigmoid().to(dt)
        return [t.requires_grad_(True) for t in (q, k, v, g, beta)]

    a, b = inputs(), inputs()
    o_chunk, _ = chunk_gated_delta_rule(*a)
    o_ref, _ = fused_recurrent_gated_delta_rule(*b)
    o_chunk.float().sum().backward()
    o_ref.float().sum().backward()
    errs = [rel_err(o_chunk, o_ref)] + [rel_err(x.grad, y.grad) for x, y in zip(a, b)]
    return report("fla chunk_gated_delta_rule vs recurrent", max(errs) < FLA_MAX_REL_ERR,
                  f"rel. errors out/dq/dk/dv/dg/dbeta = {[f'{e:.1e}' for e in errs]}")


def check_tms():
    import torch_memory_saver  # noqa: F401
    return report("torch_memory_saver import", True)


def main():
    results = []
    for check in (check_cuda, check_te, check_fla, check_tms):
        try:
            results.append(check())
        except Exception as e:   # report every failure, keep checking the rest
            results.append(report(check.__name__, False, f"{type(e).__name__}: {e}"))
    print(f"[check_slime_gpu] {'ALL PASS' if all(results) else 'FAILED'} ({sum(results)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
