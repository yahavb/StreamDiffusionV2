"""2-way channel tensor-parallel sharding for the RF VAE decoder (vae_rf.py).

Adapted from wan2-i2v-14b/models/vae_tp.py to OUR vae_rf classes, whose
CausalConv3d manages its own temporal `self.cache` + `spatial_temporal_padding`
via `_causal_conv3d_core` (not the cache_x-arg style of the source). The TP
pattern is identical (Megatron column->row):
  - ResidualBlock: first CausalConv3d column-parallel (shard out-ch, no comm),
    second row-parallel (shard in-ch, all-reduce). RMS_norm after the first
    conv is sharded to the local out-ch.
  - AttentionBlock: to_qkv column-parallel, proj row-parallel.
  - conv1 / head / shortcut / Resample: replicated (cheap or full<->full).

Gated by VAE_TP_DEGREE (default 1 = no sharding). Decoder-only (decode path).
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from modules.vae_rf import (
    CausalConv3d, RMS_norm, _causal_conv3d_core, CACHE_T,
    vae_scaled_dot_product_attention, _split_qkv,
)

logger = logging.getLogger(__name__)

_VAE_TP_GROUP: Optional[dist.ProcessGroup] = None
_VAE_TP_RANK: int = 0
_VAE_TP_WORLD_SIZE: int = 1


def create_vae_tp_group(vae_ranks: List[int]) -> Optional[dist.ProcessGroup]:
    """Process group for VAE channel-TP across the given global ranks."""
    global _VAE_TP_GROUP, _VAE_TP_RANK, _VAE_TP_WORLD_SIZE
    if len(vae_ranks) <= 1:
        _VAE_TP_RANK, _VAE_TP_WORLD_SIZE, _VAE_TP_GROUP = 0, 1, None
        return None
    group = dist.new_group(vae_ranks)
    gr = dist.get_rank()
    if gr in vae_ranks:
        _VAE_TP_GROUP = group
        _VAE_TP_RANK = vae_ranks.index(gr)
        _VAE_TP_WORLD_SIZE = len(vae_ranks)
    logger.info(f"[VAE-TP] group ranks={vae_ranks} global={gr} vae_rank={_VAE_TP_RANK}")
    return group


def get_vae_tp_world_size():
    return _VAE_TP_WORLD_SIZE


def get_vae_tp_group():
    return _VAE_TP_GROUP


@torch.compiler.disable
def _vae_all_reduce_sum(x):
    if _VAE_TP_WORLD_SIZE <= 1:
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_VAE_TP_GROUP)
    return x


# ── sharded CausalConv3d (preserves our cache + spatial_temporal_padding) ──

class _ShardedCausalConv3d(CausalConv3d):
    """Base: copy a CausalConv3d's config, replace weight/bias with a shard.
    Keeps the exact forward/cache logic of the parent; subclasses set how the
    weight is sliced and whether the output is all-reduced."""

    def __init__(self, orig: CausalConv3d, weight, bias, row_parallel: bool):
        nn.Module.__init__(self)
        # mirror parent config
        self.in_channels = orig.in_channels
        self.out_channels = orig.out_channels
        self.kernel_size = orig.kernel_size
        self.stride = orig.stride
        self.dilation = orig.dilation
        self.groups = orig.groups
        self.original_padding = orig.original_padding
        self.spatial_temporal_padding = orig.spatial_temporal_padding
        self.cache = None
        self._row_parallel = row_parallel
        self.weight = nn.Parameter(weight.contiguous())
        # row-parallel: bias added AFTER all-reduce (full out-ch, one copy);
        # column-parallel: bias is the local out-ch shard.
        self.bias = nn.Parameter(bias.contiguous()) if bias is not None else None

    def forward(self, x):
        T = x.shape[2]
        if self.cache is None:
            B, C, _, H, W = x.shape
            self.cache = torch.zeros(B, C, CACHE_T, H, W, dtype=x.dtype, device=x.device)
        # row-parallel adds bias after all-reduce, so run the conv core with bias=None
        conv_bias = None if self._row_parallel else self.bias
        out = _causal_conv3d_core(
            x, self.cache, self.weight, conv_bias, self.stride,
            self.dilation, self.groups, self.spatial_temporal_padding)
        if self._row_parallel:
            out = _vae_all_reduce_sum(out)
            if self.bias is not None:
                out = out + self.bias.view(1, -1, 1, 1, 1)
        # identical cache update to the parent
        if T >= CACHE_T:
            self.cache = x[:, :, -CACHE_T:, :, :].detach().clone()
        else:
            self.cache = torch.cat([self.cache[:, :, T:, :, :], x], dim=2).detach().clone()
        return out


def _col_conv3d(orig, r, n):
    """Column-parallel: shard OUTPUT channels (dim 0). No comm."""
    oc = orig.weight.shape[0]; c = oc // n; s = r * c
    b = orig.bias.data[s:s + c] if orig.bias is not None else None
    return _ShardedCausalConv3d(orig, orig.weight.data[s:s + c], b, row_parallel=False)


def _row_conv3d(orig, r, n):
    """Row-parallel: shard INPUT channels (dim 1). All-reduce output."""
    ic = orig.weight.shape[1]; c = ic // n; s = r * c
    b = orig.bias.data.clone() if orig.bias is not None else None
    return _ShardedCausalConv3d(orig, orig.weight.data[:, s:s + c], b, row_parallel=True)


# ── sharded Conv2d (AttentionBlock qkv/proj) ──

class _ColConv2d(nn.Module):
    def __init__(self, orig, r, n):
        super().__init__()
        oc = orig.weight.shape[0]; c = oc // n; s = r * c
        self.weight = nn.Parameter(orig.weight.data[s:s + c].contiguous())
        self.bias = nn.Parameter(orig.bias.data[s:s + c].contiguous()) if orig.bias is not None else None
        self.stride, self.padding, self.dilation = orig.stride, orig.padding, orig.dilation

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


class _RowConv2d(nn.Module):
    def __init__(self, orig, r, n):
        super().__init__()
        ic = orig.weight.shape[1]; c = ic // n; s = r * c
        self.weight = nn.Parameter(orig.weight.data[:, s:s + c].contiguous())
        self.bias = nn.Parameter(orig.bias.data.clone()) if orig.bias is not None else None
        self.stride, self.padding, self.dilation = orig.stride, orig.padding, orig.dilation

    def forward(self, x):
        out = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation)
        out = _vae_all_reduce_sum(out)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# ── block sharders ──

def _shard_residual(block, r, n):
    """residual = [RMS_norm(in), SiLU, Conv(in->out), RMS_norm(out), SiLU, Id, Conv(out->out)].
    Conv[2] column-parallel, Conv[6] row-parallel, RMS_norm[3] sharded to local out-ch."""
    convs = [i for i, l in enumerate(block.residual) if type(l).__name__ == 'CausalConv3d']
    if len(convs) < 2:
        return block
    i0, i1 = convs[0], convs[1]
    block.residual[i0] = _col_conv3d(block.residual[i0], r, n)
    block.residual[i1] = _row_conv3d(block.residual[i1], r, n)
    # shard the RMS_norm that sits AFTER the column-parallel conv (operates on local out-ch)
    for i, l in enumerate(block.residual):
        if isinstance(l, RMS_norm) and i > i0:
            full = l.gamma.shape[0]; c = full // n; s = r * c
            l.gamma = nn.Parameter(l.gamma.data[s:s + c].contiguous())
            if isinstance(l.bias, nn.Parameter):
                l.bias = nn.Parameter(l.bias.data[s:s + c].contiguous())
            l.scale = c ** 0.5
    # shortcut stays replicated (full->full)
    return block


def _shard_attention(block, r, n):
    """to_qkv column-parallel, proj row-parallel; attention runs on local channels."""
    block.to_qkv = _ColConv2d(block.to_qkv, r, n)
    block.proj = _RowConv2d(block.proj, r, n)
    local = block.dim // n

    def tp_forward(x):
        identity = x
        b, c, t, h, w = x.size()
        xw = x.transpose(1, 2).reshape(b * t, c, h, w)
        xw = block.norm(xw)                 # norm on FULL channels (input is full)
        qkv = block.to_qkv(xw)              # -> local 3*local channels
        q, k, v = _split_qkv(qkv)
        xw = vae_scaled_dot_product_attention(q, k, v)
        xw = xw.squeeze(1).permute(0, 2, 1).reshape(b * t, local, h, w)
        xw = block.proj(xw)                 # row-parallel -> all-reduce -> full
        xw = xw.reshape(b, t, c, h, w).transpose(1, 2)
        return xw + identity

    block.forward = tp_forward
    return block


def shard_vae_decoder_tp(decoder, r: int, n: int):
    """In-place 2-way channel TP on the RF Decoder3d. conv1/head/Resample replicated."""
    logger.info(f"[VAE-TP] sharding RF decoder rank={r}/{n}")
    for seq in (decoder.middle, decoder.upsamples):
        for i, l in enumerate(seq):
            t = type(l).__name__
            if t == 'ResidualBlock':
                seq[i] = _shard_residual(l, r, n)
            elif t == 'AttentionBlock':
                seq[i] = _shard_attention(l, r, n)
            # Resample / others: replicated
    return decoder


def shard_vae_model_tp(vae_model, r: int, n: int):
    """Shard the WanVAE_ decoder (decode path only). Encoder replicated."""
    shard_vae_decoder_tp(vae_model.decoder, r, n)
    vae_model._vae_tp_degree = n
    vae_model._vae_tp_rank = r
    return vae_model
