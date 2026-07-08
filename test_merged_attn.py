"""Numerically validate forward_merged vs a CPU full-attention reference — SINGLE process
(world=1, so SP collectives are no-ops). If it MATCHES here, the core attention/RoPE math
is right and the noise is in the SP collective ordering (test multi-rank next). If it FAILS
here, the bug is in forward_merged's core math (RoPE offset, head layout, reshape) — found
without needing 16 ranks.

Run: python test_merged_attn.py   (CPU or single neuron core)
"""
import os
import sys
import math
import torch
import torch.nn.functional as F

os.environ.setdefault("USE_NKI_KERNELS", "false")  # force the SDPA fallback path (CPU-able)
SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT)
sys.path.insert(0, os.path.join(SCRIPT, "models", "wan", "wan_base"))

from models.wan.neuron_layers import CausalWanSelfAttention

DIM, HEADS, HD = 1536, 12, 128
FRAME_SEQ = 720          # 320x576 latent 40x72 patch2 -> 30x36? use small: f=3,h=?,w=?
# pick a small frame grid: h=8,w=8 -> frame_seq 64, 3 frames -> 192 tokens
H, W, F_ = 8, 8, 3
FRAME_SEQ = H * W
SEQ = F_ * FRAME_SEQ


def cpu_ref_attention(attn, x, grid, freqs_cos, freqs_sin, current_start):
    """Plain full self-attention with the SAME RoPE the module uses, on CPU."""
    b, s, _ = x.shape
    q = attn.norm_q(attn.q(x)).view(b, s, HEADS, HD)
    k = attn.norm_k(attn.k(x)).view(b, s, HEADS, HD)
    v = attn.v(x).view(b, s, HEADS, HD)
    sf = torch.tensor(current_start // (grid[1] * grid[2]))
    rq = attn._nki_rope_apply(q, grid, freqs_cos, freqs_sin, start_frame=sf)
    rk = attn._nki_rope_apply(k, grid, freqs_cos, freqs_sin, start_frame=sf)
    o = F.scaled_dot_product_attention(
        rq.permute(0, 2, 1, 3), rk.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
    o = o.permute(0, 2, 1, 3).flatten(2)
    return attn.o(o)


def main():
    torch.manual_seed(0)
    from models.wan.neuron_layers import rope_params
    attn = CausalWanSelfAttention(DIM, HEADS, -1, 0, True, 1e-6, 0, FRAME_SEQ).eval()
    # freqs
    def _mk(dh):
        c = dh // 2; s0 = c - 2 * (c // 3); s1 = c // 3
        cf, sf = rope_params(1024, 2 * s0); ch, sh = rope_params(1024, 2 * s1)
        cw, sw = rope_params(1024, 2 * s1)
        return torch.cat([cf, ch, cw], 1), torch.cat([sf, sh, sw], 1)
    fc, fs = _mk(HD)
    grid = (F_, H, W)
    x = torch.randn(1, SEQ, DIM)

    with torch.no_grad():
        ref = cpu_ref_attention(attn, x, grid, fc, fs, current_start=0)
        # forward_merged with world=1 (SP no-ops). Needs get_* to return 1/0 — not
        # distributed, so the accessors return defaults (sp=1, tp=1, world=1).
        merged = attn.forward_merged(
            x, grid, fc, fs, kv_cache=None, cache_update_start=0, current_start=0,
            cu_shared_buffers=None, dn_shared_buffers=None,
            num_valid_frames_dn=F_, nfpb_cu=F_)

    diff = (ref - merged).abs()
    print(f"ref {tuple(ref.shape)} merged {tuple(merged.shape)}")
    print(f"max_diff={diff.max().item():.6f}  mean={diff.mean().item():.6f}")
    print("MATCH" if diff.max().item() < 1e-2 else "MISMATCH -> core math bug in forward_merged")


if __name__ == "__main__":
    main()
