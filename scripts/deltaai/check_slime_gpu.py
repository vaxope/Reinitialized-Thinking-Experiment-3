"""GPU check of a freshly built slime image (runs at the end of build_slime.sbatch, ~1-2 min).

Purpose : catch a broken GPU stack before any paid training job: the parts of the image that
          training actually runs on the GPU, each checked against a plain reference.
            0. Every module the image must provide imports WITH the GPU driver (the build's own
               import check runs without one and defers driver-only failures to here).
               IMPORT_MODULES must equal the recipe's step-12 list (tests/test_slime_container.py).
            1. CUDA visible, GPU is sm90 (GH200).
            2. TransformerEngine: te.pytorch.Linear forward + backward = torch matmul.
            3. flash-linear-attention, exactly as Qwen3.5's linear-attention layers call it in
               training (slime_plugins/models/qwen3_5.py Qwen3_5GatedDeltaNet):
               a. chunk_gated_delta_rule (the fast kernel) = fused_recurrent_gated_delta_rule
                  (fla's step-by-step version), on two packed sequences of different lengths
                  (cu_seqlens), with use_qk_l2norm_in_kernel=True; outputs and all gradients;
               b. ShortConvolution (causal depthwise conv + SiLU, varlen) = torch conv1d per sequence;
               c. FusedRMSNormGated (RMSNorm * weight * SiLU(gate)) = the plain formula.
            4. torch_memory_saver imports (it is exercised for real in the first training step).
Inputs  : none (random tensors, fixed seed). No model weights, no data.
Output  : one line per check; exit 1 if any check fails. Run INSIDE the slime image:
            apptainer exec --nv <slime.sif> python3 scripts/deltaai/check_slime_gpu.py
"""
import importlib
import sys

import torch

# Modules the image must provide (same list as containers/slime_gh200.def step 12).
IMPORT_MODULES = ["torch", "sglang", "sglang_router", "megatron.core", "transformer_engine", "fla",
                  "torch_memory_saver", "ray", "transformers", "slime", "slime_plugins.models.qwen3_5"]

# Tolerances for bf16 kernels vs references computed from the same bf16 inputs. bf16 keeps
# ~3 significant digits, so element-wise differences of a few 1e-2 relative are normal; these
# bounds catch wrong results (orders of magnitude off), not rounding.
TE_ATOL, TE_RTOL = 5e-2, 5e-2
FLA_MAX_REL_ERR = 5e-2
SEED = 0
# Small test shapes. SEQ_LENS: two packed sequences of different lengths, so the varlen path
# (cu_seqlens) that slime uses for every training batch is exercised, including a length that is
# not a multiple of fla's 64-token chunk. CONV_KERNEL = Qwen3.5's linear_conv_kernel_dim (4,
# config.json). NORM_EPS = Qwen3.5's rms_norm_eps (1e-6, config.json).
SEQ_LENS = (100, 156)
HEADS, HEAD_DIM = 2, 64
CONV_CHANNELS, CONV_KERNEL = 128, 4
NORM_EPS = 1e-6


def report(name, ok, detail=""):
    print(f"[check_slime_gpu] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    return ok


def check_imports():
    failed = {}
    for name in IMPORT_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"
    return report("imports", not failed, f"{len(IMPORT_MODULES) - len(failed)}/{len(IMPORT_MODULES)} "
                  + (f"failed: {failed}" if failed else ""))


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


def cu_seqlens():
    bounds = [0]
    for n in SEQ_LENS:
        bounds.append(bounds[-1] + n)
    return torch.tensor(bounds, device="cuda", dtype=torch.long)


def check_fla_gated_delta_rule():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    T, H, K, dev, dt = sum(SEQ_LENS), HEADS, HEAD_DIM, "cuda", torch.bfloat16

    def inputs():
        # the dtypes slime's layer passes: q/k/v/beta in bf16, the log-decay g in fp32
        torch.manual_seed(SEED)
        q = torch.randn(1, T, H, K, device=dev, dtype=dt)
        k = torch.randn(1, T, H, K, device=dev, dtype=dt)
        v = torch.randn(1, T, H, K, device=dev, dtype=dt)
        g = torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
        beta = torch.randn(1, T, H, device=dev, dtype=dt).sigmoid()
        return [t.requires_grad_(True) for t in (q, k, v, g, beta)]

    cu = cu_seqlens()
    a, b = inputs(), inputs()
    o_chunk, _ = chunk_gated_delta_rule(*a[:3], g=a[3], beta=a[4], initial_state=None, output_final_state=False,
                                        use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    o_ref, _ = fused_recurrent_gated_delta_rule(*b[:3], g=b[3], beta=b[4], initial_state=None,
                                                output_final_state=False, use_qk_l2norm_in_kernel=True,
                                                cu_seqlens=cu)
    o_chunk.float().sum().backward()
    o_ref.float().sum().backward()
    errs = [rel_err(o_chunk, o_ref)] + [rel_err(x.grad, y.grad) for x, y in zip(a, b)]
    return report("fla chunk_gated_delta_rule vs recurrent (varlen, qk l2norm)", max(errs) < FLA_MAX_REL_ERR,
                  f"rel. errors out/dq/dk/dv/dg/dbeta = {[f'{e:.1e}' for e in errs]}")


def check_fla_short_conv():
    from fla.modules import ShortConvolution
    torch.manual_seed(SEED)
    conv = ShortConvolution(hidden_size=CONV_CHANNELS, kernel_size=CONV_KERNEL, bias=False).cuda().to(torch.bfloat16)
    x = torch.randn(1, sum(SEQ_LENS), CONV_CHANNELS, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y, _ = conv(x=x, cu_seqlens=cu_seqlens())
    refs, start = [], 0
    for n in SEQ_LENS:   # each sequence on its own: causal = pad K-1 on the left, keep the first n
        seg = x[0, start:start + n].t().unsqueeze(0).float()
        out = torch.nn.functional.conv1d(seg, conv.weight.float(), padding=CONV_KERNEL - 1, groups=CONV_CHANNELS)
        refs.append(torch.nn.functional.silu(out[..., :n])[0].t())
        start += n
    ref = torch.cat(refs).unsqueeze(0)
    y.float().sum().backward()
    err = rel_err(y, ref)
    return report("fla ShortConvolution vs torch conv1d (varlen)", err < FLA_MAX_REL_ERR and x.grad is not None,
                  f"rel. error {err:.1e}")


def check_fla_gated_norm():
    from fla.modules import FusedRMSNormGated
    torch.manual_seed(SEED)
    norm = FusedRMSNormGated(HEAD_DIM, eps=NORM_EPS, activation="silu", device="cuda", dtype=torch.bfloat16)
    x = torch.randn(sum(SEQ_LENS) * HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    z = torch.randn_like(x, requires_grad=True)
    y = norm(x, z)
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + NORM_EPS) * norm.weight.float() \
        * torch.nn.functional.silu(z.float())
    y.float().sum().backward()
    err = rel_err(y, ref)
    return report("fla FusedRMSNormGated vs formula", err < FLA_MAX_REL_ERR and z.grad is not None,
                  f"rel. error {err:.1e}")


def check_tms():
    import torch_memory_saver  # noqa: F401
    return report("torch_memory_saver import", True)


def main():
    results = []
    for check in (check_imports, check_cuda, check_te, check_fla_gated_delta_rule, check_fla_short_conv,
                  check_fla_gated_norm, check_tms):
        try:
            results.append(check())
        except Exception as e:   # report every failure, keep checking the rest
            results.append(report(check.__name__, False, f"{type(e).__name__}: {e}"))
    print(f"[check_slime_gpu] {'ALL PASS' if all(results) else 'FAILED'} ({sum(results)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
