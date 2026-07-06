"""VAE decode accuracy: WHOLE-CLIP decode vs BLOCK-BY-BLOCK streaming decode.

Inference decodes the video one streaming block (num_frame_per_block latent frames) at a
time. A whole-clip decode processes ALL latent frames in one .decode() call with the
temporal feat_cache maintained throughout. If the two DIVERGE, the per-block streaming
loses temporal context between blocks -> softens/blurs the output. This numerically
isolates whether OUR decode path (USE_RF_VAE=0, original WanVAE_.decode) is the blur.

Run on Neuron:  USE_RF_VAE=0 python test_vae_decode.py
"""
import os
import sys
import math

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "models", "wan", "wan_base"))

DEVICE = torch.device("neuron")

# Wan2.1-1.3B VAE: z_dim=16, 8x spatial, 4x temporal. 480x640 -> latent 60x80.
Z_DIM = 16
LAT_H, LAT_W = 60, 80
NPB = 3                 # num_frame_per_block (latent frames per streaming block)
N_BLOCKS = 5            # decode 5 blocks -> 15 latent frames
VAE_PTH = os.environ.get("VAE_PTH", "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")


def main():
    from modules.vae import _video_vae
    torch.manual_seed(0)

    mean = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517,
                         1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497,
                         0.2503, -0.2921], dtype=torch.bfloat16)
    std = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160],
                       dtype=torch.bfloat16)
    scale = [mean.to(DEVICE), (1.0 / std).to(DEVICE)]

    model = _video_vae(pretrained_path=VAE_PTH, z_dim=Z_DIM).eval().requires_grad_(False)
    model = model.to(dtype=torch.bfloat16, device=DEVICE)

    T = NPB * N_BLOCKS
    # latent [b, c, t, h, w]  (decode() expects b c t h w)
    z = torch.randn(1, Z_DIM, T, LAT_H, LAT_W, dtype=torch.bfloat16, device=DEVICE)

    # ── WHOLE-CLIP: one decode() over all T frames (cache maintained throughout) ──
    with torch.no_grad():
        whole = model.decode(z.clone(), scale)     # [1,3,T_pix,H,W]
    whole = whole.float().cpu()

    # ── BLOCK-BY-BLOCK: how inference does it — decode() per block of NPB frames ──
    # (each call does clear_cache() at start -> NO cross-block temporal context; this is
    #  exactly the non-RF-VAE inference path). Concatenate the per-block pixels.
    outs = []
    with torch.no_grad():
        for b in range(N_BLOCKS):
            zb = z[:, :, b * NPB:(b + 1) * NPB, :, :].clone()
            outs.append(model.decode(zb, scale).float().cpu())
    block = torch.cat(outs, dim=2)

    # ── STREAMED block decode (THE FIX): decode_stream keeps the cache across blocks ──
    souts = []
    with torch.no_grad():
        for b in range(N_BLOCKS):
            zb = z[:, :, b * NPB:(b + 1) * NPB, :, :].clone()
            souts.append(model.decode_stream(zb, scale, first_chunk=(b == 0)).float().cpu())
    stream = torch.cat(souts, dim=2)

    def _cmp(a, tag):
        tmin = min(whole.shape[2], a.shape[2])
        d = (whole[:, :, :tmin] - a[:, :, :tmin]).abs()
        print(f"  {tag:20s} shape {tuple(a.shape)}  max_diff={d.max().item():.6f}  mean={d.mean().item():.6f}")
        return d.max().item()

    print("=" * 64)
    print(f"VAE decode vs WHOLE-CLIP (T_lat={T}, npb={NPB})  whole={tuple(whole.shape)}")
    m_block = _cmp(block, "block(clear/blk)")
    m_stream = _cmp(stream, "stream(cache kept)")
    print(f"  VERDICT: block {'BLUR' if m_block>=0.05 else 'ok'}; "
          f"stream {'STILL DIVERGES' if m_stream>=0.05 else 'MATCHES whole-clip = FIXED'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
