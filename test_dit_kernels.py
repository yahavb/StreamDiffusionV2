"""Accuracy test for the DiT-path NKI kernels — self-attention + RoPE — vs CPU PyTorch.

These are the kernels that run DURING DENOISING (30 layers x N steps x every block),
NOT the VAE (see test_vae_kernels.py for those). If either has meaningful per-element
error, it COMPOUNDS across 30 layers and produces blur — which is why a single-kernel
"pass" at loose tolerance can still soften the full pipeline. So we report max AND mean
abs diff at the real Wan-1.3B shapes and flag anything above a tight bf16 bar.

Run on a Neuron instance:
    python test_dit_kernels.py

Wan2.1-T2V-1.3B DiT: dim=1536, num_heads=12, head_dim=128.
Self-attn seqlen = num_frame_per_block(3) * frame_seq_len; 480x640 -> latent 60x80 ->
patch/2 -> 30x40 = 1200 tokens/frame -> ~3600, padded to a multiple of 8192.
"""
import os
import sys
import math
import traceback

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "kernels"))

DEVICE = torch.device("neuron")
# bf16 round-off for a single op is ~4e-3. A CORRECT kernel should be near that.
# 0.07 (the VAE suite's bar) is far too loose for a 30-layer-compounding attn op — a
# kernel at 0.07 per layer can blur the output. Flag the real number; judge by it.
TOL_TIGHT = 0.02
results = []


def record(name, max_diff, mean_diff=None, err=None):
    ok = err is None and max_diff is not None and max_diff < TOL_TIGHT
    status = "PASS" if ok else ("HIGH" if err is None else "FAIL")
    if err:
        msg = f"  [{status}] {name:<52s}  ERROR: {err[:110]}"
    else:
        msg = f"  [{status}] {name:<52s}  max={max_diff:.6f}  mean={mean_diff:.6f}"
    print(msg, flush=True)
    results.append((name, ok, max_diff, err))


# ════════════════════════════════════════════════════════════════════════
# DiT self-attention: NKI wan_flash_self_attn vs CPU SDPA
# ════════════════════════════════════════════════════════════════════════
def test_self_attn():
    print("\n── DiT Self-Attention (wan_flash_self_attn) ─────────────────────", flush=True)
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from self_attention import wan_flash_self_attn
    except Exception as e:
        record("self_attn: import", None, err=f"{type(e).__name__}: {e}")
        return

    HEAD_DIM = 128
    scale = 1.0 / math.sqrt(HEAD_DIM)
    SECTION = 8192

    # (name, seq_q raw) — head_dim fixed 128; single head (bs=head count folded to bs=1 here)
    configs = [
        ("1frame_1200tok", 1200),
        ("3frame_3600tok", 3600),
    ]
    wrapped = wrap_nki(wan_flash_self_attn)

    for name, seq_raw in configs:
        try:
            torch.manual_seed(0)
            seq_q = ((seq_raw + 127) // 128) * 128          # q: multiple of 128
            seq_k = ((seq_raw + SECTION - 1) // SECTION) * SECTION  # k: multiple of 8192
            d = HEAD_DIM

            # reference tensors (valid region only), fp32 SDPA
            q = torch.randn(1, d, seq_raw, dtype=torch.float32)
            k = torch.randn(1, d, seq_raw, dtype=torch.float32)
            v = torch.randn(1, seq_raw, d, dtype=torch.float32)

            qa = q.permute(0, 2, 1)                          # (1, seq, d)
            scores = torch.matmul(qa, k) * scale             # (1, seq, seq)
            attn = torch.softmax(scores, dim=-1)
            ref = torch.matmul(attn, v)                      # (1, seq, d)
            ref = ref[0]                                     # (seq_raw, d)

            # NKI inputs: q (1,d,seq_q), k (1,d,seq_k), v (1,seq_k,d), mask (128,seq_k)
            qn = F.pad(q, (0, seq_q - seq_raw)).to(torch.bfloat16)
            kn = F.pad(k, (0, seq_k - seq_raw)).to(torch.bfloat16)
            vn = F.pad(v, (0, 0, 0, seq_k - seq_raw)).to(torch.bfloat16)
            identity = torch.eye(128, dtype=torch.bfloat16)
            mask = torch.zeros(128, seq_k, dtype=torch.bfloat16)
            mask[:, seq_raw:] = float("-inf")                # mask padded keys
            num_sections = seq_k // SECTION

            out = wrapped(qn.to(DEVICE), kn.to(DEVICE), vn.to(DEVICE),
                          identity.to(DEVICE), mask.to(DEVICE),
                          softmax_scale=scale, num_sections=num_sections).cpu()
            # returns (seq_q, bs, d) -> take valid rows
            out = out[:seq_raw, 0, :].float()

            diff = (out - ref).abs()
            record(f"self_attn: {name} (seq {seq_raw}->{seq_q}/{seq_k})",
                   diff.max().item(), diff.mean().item())
        except Exception as e:
            record(f"self_attn: {name}", None, err=f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# RoPE: NKI causal_rope_rotation vs CPU reference
# ════════════════════════════════════════════════════════════════════════
def test_rope():
    print("\n── DiT RoPE (causal_rope_rotation) ──────────────────────────────", flush=True)
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from rope import causal_rope_rotation
    except Exception as e:
        record("rope: import", None, err=f"{type(e).__name__}: {e}")
        return

    NUM_HEADS, HEAD_DIM = 12, 128
    D = HEAD_DIM
    configs = [("1frame_1200tok", 1200), ("3frame_3600tok", 3600)]
    wrapped = wrap_nki(causal_rope_rotation)

    for name, seq_raw in configs:
        try:
            torch.manual_seed(0)
            seq = ((seq_raw + 127) // 128) * 128
            x = torch.randn(seq, NUM_HEADS, HEAD_DIM, dtype=torch.float32)

            # build cos/sin (interleaved-pair RoPE, the pipeline's float32 cos/sin form)
            pos = torch.arange(seq, dtype=torch.float32)
            inv = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
            ang = pos[:, None] * inv[None, :]               # (seq, D/2)
            cos_h = torch.cos(ang); sin_h = torch.sin(ang)
            cos_exp = torch.repeat_interleave(cos_h, 2, dim=1)   # (seq, D)
            sin_exp = torch.repeat_interleave(sin_h, 2, dim=1)
            cos_sin = torch.cat([cos_exp, sin_exp], dim=1)       # (seq, 2D)

            # CPU reference: rotate-half on interleaved pairs
            def rope_ref(xh):
                x1 = xh[..., 0::2]; x2 = xh[..., 1::2]
                xswap = torch.stack([-x2, x1], dim=-1).reshape_as(xh)
                return xh * cos_exp[:, None, :] + xswap * sin_exp[:, None, :]
            ref = rope_ref(x)[:seq_raw]

            out = wrapped(x.to(torch.bfloat16).to(DEVICE),
                          cos_sin.to(DEVICE),
                          num_heads=NUM_HEADS, head_dim=HEAD_DIM).cpu()
            out = out[:seq_raw].float()

            diff = (out - ref[:seq_raw]).abs()
            record(f"rope: {name} (seq {seq_raw}->{seq})",
                   diff.max().item(), diff.mean().item())
        except Exception as e:
            record(f"rope: {name}", None, err=f"{type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    print("=" * 72)
    print("DiT NKI Kernel Accuracy Test (Wan 2.1-1.3B shapes) — NKI vs CPU PyTorch")
    print(f"Device: {DEVICE}  tight bf16 bar: max_diff < {TOL_TIGHT}")
    print("A kernel above the bar COMPOUNDS across 30 layers -> blur. Judge by the number.")
    print("=" * 72)
    test_self_attn()
    test_rope()
    print("\n" + "=" * 72)
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    print(f"SUMMARY: {n_pass}/{len(results)} within tight bar. "
          f"HIGH max_diff => that kernel is the blur (fix it or run exact).")
    print("=" * 72)
