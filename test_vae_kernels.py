"""Test suite for VAE NKI kernels — conv2d_k1, conv2d_k3_shifted, vae_self_attention.

Run on Neuron instance:
    python test_vae_kernels.py

Tests each kernel against a CPU PyTorch reference implementation.

Production shapes from Wan 2.1 VAE decoder (dim=96, dim_mult=[1,2,4,4], 480p):
  Decoder stages:
    Stage 0 (bottleneck): C=384, H=30, W=52   (1560 spatial tokens)
    Stage 1 (after up2d):  C=192, H=60, W=104  (6240 spatial tokens)
    Stage 2 (after up3d):  C=96,  H=120, W=208 (24960 spatial tokens)
    Stage 3 (no upsample): C=96,  H=240, W=416 (99840 spatial tokens)
  Attention: d=384 (middle block), seq=1560
"""
import os
import sys
import math
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "models", "wan", "kernels"))

DEVICE = torch.device("neuron")
TOL = 0.07  # bf16 tolerance

results = []


def record(name, max_diff, err=None):
    ok = err is None and max_diff is not None and max_diff < TOL
    status = "PASS" if ok else "FAIL"
    if err:
        msg = f"  [{status}] {name:<60s}  ERROR: {err[:120]}"
    else:
        msg = f"  [{status}] {name:<60s}  max_diff = {max_diff:.6f}"
    print(msg)
    results.append((name, ok, max_diff, err))


# ════════════════════════════════════════════════════════════════════════
# Helper: build shifted inputs for conv2d_k3
# ════════════════════════════════════════════════════════════════════════
def build_shifted_inputs(x_2d, H, W, K=3, padding=1):
    C = x_2d.shape[0]
    x_3d = x_2d.reshape(C, H, W)
    x_padded = F.pad(x_3d, (padding, padding, padding, padding))
    shifts = []
    for kh in range(K):
        for kw in range(K):
            window = x_padded[:, kh:kh + H, kw:kw + W]
            shifts.append(window.reshape(C, H * W))
    stacked = torch.stack(shifts, dim=0)
    result = stacked.reshape(9 * C, H * W)
    HW = H * W
    pad_hw = (512 - HW % 512) % 512
    if pad_hw > 0:
        result = F.pad(result, (0, pad_hw))
    return result


def build_weight_slices_T(weight_4d):
    C_out, C_in, K, K2 = weight_4d.shape
    assert K == 3 and K2 == 3
    slices = []
    for kh in range(K):
        for kw in range(K):
            slices.append(weight_4d[:, :, kh, kw].T)
    return torch.cat(slices, dim=0)


# ════════════════════════════════════════════════════════════════════════
# TEST: Conv2d K=1 (pointwise)
# ════════════════════════════════════════════════════════════════════════
def test_conv2d_k1():
    print("\n── Conv2d K=1 (vae_conv2d_k1) ───────────────────────────────────")
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from vae_conv2d import vae_conv2d_k1
    except Exception as e:
        record("conv2d_k1: import", None, f"{type(e).__name__}: {e}")
        return

    # Wan 2.1 shapes: (name, C_in, C_out, H, W)
    configs = [
        ("shortcut_96→192_120x208",      128, 192,  120, 208),
        ("shortcut_192→384_60x104",       192, 384,  60, 104),
        ("attn_qkv_384→1152_30x52",      384, 1152, 30, 52),
        ("attn_proj_384→384_30x52",       384, 384,  30, 52),
        ("shortcut_384→192_60x104",       384, 192,  60, 104),
        ("shortcut_192→96_120x208",       192, 128,  120, 208),
    ]

    wrapped = wrap_nki(vae_conv2d_k1)

    for name, C_in, C_out, H, W in configs:
        try:
            torch.manual_seed(0)
            C_in_padded = ((C_in + 127) // 128) * 128
            C_out_padded = ((C_out + 127) // 128) * 128
            HW = H * W
            HW_padded = ((HW + 511) // 512) * 512

            conv = nn.Conv2d(C_in, C_out, 1, bias=True)
            nn.init.normal_(conv.weight, std=0.02)
            nn.init.zeros_(conv.bias)

            x = torch.randn(1, C_in, H, W, dtype=torch.float32)
            with torch.no_grad():
                ref = conv(x)
            ref_flat = ref.reshape(C_out, HW).to(torch.bfloat16)

            x_flat = x.reshape(C_in, HW).to(torch.bfloat16)
            if C_in < C_in_padded:
                x_flat = F.pad(x_flat, (0, 0, 0, C_in_padded - C_in))
            if HW < HW_padded:
                x_flat = F.pad(x_flat, (0, HW_padded - HW))

            w = conv.weight.data.reshape(C_out, C_in).to(torch.bfloat16)
            w_T = w.T.contiguous()
            w_T_padded = torch.zeros(C_in_padded, C_out_padded, dtype=torch.bfloat16)
            w_T_padded[:C_in, :C_out] = w_T

            b = conv.bias.data.to(torch.bfloat16)
            b_padded = torch.zeros(C_out_padded, 1, dtype=torch.bfloat16)
            b_padded[:C_out, 0] = b

            out = wrapped(
                x_flat.to(DEVICE), w_T_padded.to(DEVICE),
                b_padded.to(DEVICE), HW
            ).cpu()
            out = out[:C_out, :HW]

            diff = (out.float() - ref_flat.float()).abs().max().item()
            record(f"conv2d_k1: {name}", diff)
        except Exception as e:
            record(f"conv2d_k1: {name}", None, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# TEST: Conv2d K=3 (shifted inputs approach)
# ════════════════════════════════════════════════════════════════════════
def test_conv2d_k3():
    print("\n── Conv2d K=3 (vae_conv2d_k3_shifted) ──────────────────────────")
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from vae_conv2d import vae_conv2d_k3_shifted
    except Exception as e:
        record("conv2d_k3: import", None, f"{type(e).__name__}: {e}")
        return

    # Wan 2.1 shapes
    configs = [
        ("resblock_384→384_30x52",    384, 384,  30, 52),
        ("resblock_384→384_60x104",   384, 384,  60, 104),
        ("resblock_192→192_120x208",  192, 192,  120, 208),
        ("resblock_96→96_240x416",    128, 128,  30, 52),  # padded to 128
        ("head_conv_96→3_240x416",    128, 128,  30, 52),
    ]

    wrapped = wrap_nki(vae_conv2d_k3_shifted)

    for name, C_in, C_out, H, W in configs:
        try:
            torch.manual_seed(0)
            C_in_padded = ((C_in + 127) // 128) * 128
            C_out_padded = ((C_out + 127) // 128) * 128
            HW = H * W
            HW_padded = ((HW + 511) // 512) * 512

            conv = nn.Conv2d(C_in, C_out, 3, padding=1, bias=True)
            nn.init.normal_(conv.weight, std=0.02)
            nn.init.zeros_(conv.bias)

            x = torch.randn(1, C_in, H, W, dtype=torch.float32)
            with torch.no_grad():
                ref = conv(x)
            ref_flat = ref.reshape(C_out, HW).to(torch.bfloat16)

            x_flat = x.reshape(C_in, HW)
            shifted = build_shifted_inputs(x_flat.to(torch.bfloat16), H, W)
            if C_in < C_in_padded:
                chunks = shifted.reshape(9, C_in, -1)
                padded_chunks = F.pad(chunks, (0, 0, 0, C_in_padded - C_in))
                shifted = padded_chunks.reshape(9 * C_in_padded, -1)

            w_slices_T = build_weight_slices_T(conv.weight.data).to(torch.bfloat16)
            w_T_padded = torch.zeros(C_in_padded * 9, C_out_padded, dtype=torch.bfloat16)
            for k_idx in range(9):
                src_start = k_idx * C_in
                dst_start = k_idx * C_in_padded
                w_T_padded[dst_start:dst_start + C_in, :C_out] = w_slices_T[src_start:src_start + C_in, :]

            b = conv.bias.data.to(torch.bfloat16)
            b_padded = torch.zeros(C_out_padded, 1, dtype=torch.bfloat16)
            b_padded[:C_out, 0] = b

            out = wrapped(
                shifted.to(DEVICE), w_T_padded.to(DEVICE),
                b_padded.to(DEVICE), HW
            ).cpu()
            out = out[:C_out, :HW]

            diff = (out.float() - ref_flat.float()).abs().max().item()
            record(f"conv2d_k3: {name}", diff)
        except Exception as e:
            record(f"conv2d_k3: {name}", None, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# TEST: VAE Self-Attention
# ════════════════════════════════════════════════════════════════════════
def test_vae_attention():
    print("\n── VAE Self-Attention (vae_self_attention) ──────────────────────")
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from vae_attention import vae_self_attention
    except Exception as e:
        record("vae_attn: import", None, f"{type(e).__name__}: {e}")
        return

    def _sdpa_ref(q, k, v, scale):
        qa = q.permute(0, 2, 1).float()
        ka = k.permute(0, 2, 1).float()
        va = v.float()
        scores = torch.matmul(qa, ka.transpose(-1, -2)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, va)
        return out.permute(1, 0, 2).to(q.dtype)

    SEQ_PAD = 512

    # Wan 2.1: attention dim = 384 (middle block)
    configs = [
        ("dec_middle_d384_30x52",    384, 30, 52),
        ("enc_middle_d384_30x52",    384, 30, 52),
        ("small_d256_16x16",          256, 16, 16),
        ("med_d384_20x32",            384, 20, 32),
    ]

    wrapped = wrap_nki(vae_self_attention)

    for name, d, H, W in configs:
        try:
            torch.manual_seed(0)
            seq_raw = H * W
            pad = (SEQ_PAD - seq_raw % SEQ_PAD) % SEQ_PAD
            seq = seq_raw + pad

            # d must be multiple of 128 for NKI
            d_padded = ((d + 127) // 128) * 128
            scale = 1.0 / math.sqrt(d_padded)

            q = torch.randn(1, d_padded, seq_raw, dtype=torch.bfloat16)
            k = torch.randn(1, d_padded, seq_raw, dtype=torch.bfloat16)
            v = torch.randn(1, seq_raw, d_padded, dtype=torch.bfloat16)
            identity = torch.eye(128, dtype=torch.bfloat16)

            if pad > 0:
                q_padded = F.pad(q, (0, pad))
                k_padded = F.pad(k, (0, pad))
                v_padded = F.pad(v, (0, 0, 0, pad))
            else:
                q_padded = q
                k_padded = k
                v_padded = v

            ref_padded = _sdpa_ref(q_padded, k_padded, v_padded, scale)

            out = wrapped(
                q_padded.to(DEVICE), k_padded.to(DEVICE),
                v_padded.to(DEVICE), identity.to(DEVICE),
                softmax_scale=scale
            ).cpu()

            out = out[:seq_raw]
            ref = ref_padded[:seq_raw]

            diff = (out.float() - ref.float()).abs().max().item()
            record(f"vae_attn: {name} (d={d_padded},seq={seq_raw}→{seq})", diff)
        except Exception as e:
            record(f"vae_attn: {name}", None, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("VAE NKI Kernel Test Suite (Wan 2.1 shapes)")
    print(f"Device: {DEVICE}  Tolerance: max_diff < {TOL}")
    print("=" * 72)

    test_conv2d_k1()
    test_conv2d_k3()
    test_vae_attention()

    print("\n" + "=" * 72)
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_total = len(results)
    all_ok = n_pass == n_total
    print(f"SUMMARY: {n_pass}/{n_total} passed  {'✅ ALL PASS' if all_ok else '❌ FAILURES'}")
    print("=" * 72)
    sys.exit(0 if all_ok else 1)
