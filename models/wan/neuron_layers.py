"""Neuron/Trainium-compatible layers for StreamDiffusionV2.

Ported from aws-neuron-eks-samples/rolling-forcing/app/models/layers.py.
These replace GPU-only ops (Conv3d, flex_attention, float64, complex) with
Neuron-safe equivalents (matmul patch embed, SDPA, float32, cos/sin RoPE).
"""
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# No-op jit wrapper (torch_neuronx.jit not always available)
def jit(fn=None, **kwargs):
    if fn is None:
        return lambda f: f
    return fn


# ── Kernel fusion via torch.compile(backend='neuron') ────────────────────
# The Neuron SDK provides 'neuron' as a torch.compile backend that fuses
# sequences of PyTorch ops into single compiled NEFFs, eliminating per-op
# kernel launch overhead (model_switch + DMA copyin/copyout).
#
# Reference: rolling-forcing/app/inference_neuron_tp.py uses:
#   torch.compile(module, backend='neuron', dynamic=False)
# for FFN, patch_embedding, text_embedding, time_embedding, head, VAE, T5.

USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "false").lower() == "true"
_COMPILE_BACKEND = os.environ.get("NEURON_COMPILE_BACKEND", "neuron")


def neuron_compile(module_or_fn, **kwargs):
    """Compile a module with torch.compile(backend='neuron') for NEFF fusion.
    
    Fuses sequences of small ops (Linear→GELU→Linear, norm+scale+shift)
    into single compiled NEFFs, dramatically reducing kernel launch overhead.
    
    Enable with USE_TORCH_COMPILE=true environment variable.
    The 'neuron' backend is provided by torch_neuronx and compiles the graph
    into optimized Neuron executables.
    """
    if not USE_TORCH_COMPILE:
        return module_or_fn
    
    try:
        compiled = torch.compile(module_or_fn, backend=_COMPILE_BACKEND, dynamic=False)
        return compiled
    except Exception as e:
        print(f"[neuron_compile] WARNING: torch.compile(backend='{_COMPILE_BACKEND}') "
              f"failed, falling back to eager: {e}")
        return module_or_fn

# ── NKI kernel loading ──────────────────────────────────────────────────
USE_NKI_KERNELS = os.environ.get("USE_NKI_KERNELS", "true").lower() == "true"

NKI_AVAILABLE = False
wan_cross_attn = None
ROPE_NKI_AVAILABLE = False
causal_rope_rotation_nki = None
SELF_ATTN_NKI_AVAILABLE = False
wan_flash_self_attn_nki = None
ATTN_SEQLEN_MULTIPLE = 8192

if USE_NKI_KERNELS:
    # NOTE: do NOT swallow these errors. A silent `except: pass` here is what
    # caused the kernels to fall back to eager for the whole baseline run
    # (self_nki=False rope_nki=False) with no diagnostic. Log every failure.
    import logging as _logging
    _klog = _logging.getLogger(__name__)
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from kernels.cross_attention import wan_cross_attn as _wan_cross_attn
        wan_cross_attn = wrap_nki(_wan_cross_attn)
        NKI_AVAILABLE = True
        _klog.info("[nki] cross_attention: LOADED")
    except Exception as e:
        _klog.warning("[nki] cross_attention: FAILED to load: %r", e)
    try:
        from torch_neuronx.nki_hop import wrap_nki as _wrap_rope
        from kernels.rope import causal_rope_rotation as _causal_rope_rotation
        causal_rope_rotation_nki = _wrap_rope(_causal_rope_rotation)
        ROPE_NKI_AVAILABLE = True
        _klog.info("[nki] rope: LOADED")
    except Exception as e:
        _klog.warning("[nki] rope: FAILED to load: %r", e)
    try:
        from torch_neuronx.nki_hop import wrap_nki as _wrap_sa
        from kernels.self_attention import wan_flash_self_attn as _wan_flash_self_attn
        wan_flash_self_attn_nki = _wrap_sa(_wan_flash_self_attn)
        SELF_ATTN_NKI_AVAILABLE = True
        _klog.info("[nki] self_attention: LOADED")
    except Exception as e:
        _klog.warning("[nki] self_attention: FAILED to load: %r", e)


# ── Activations (explicit math for Neuron tracing) ─────────────────────

@jit
class GELU(nn.Module):
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        result = 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        return result.to(dtype)


@jit
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ── Norms ───────────────────────────────────────────────────────────────

@jit
class WanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))

    def norm_(self, x):
        x = x.float()
        mean = torch.sum(x, dim=-1, keepdim=True) / self.dim
        diff = x - mean
        variance = torch.sum(diff * diff, dim=-1, keepdim=True) / self.dim
        return (diff * torch.rsqrt(variance + self.eps)).to(x.dtype)

    def forward(self, x):
        output = self.norm_(x)
        if hasattr(self, 'weight'):
            output = output * self.weight + self.bias
        return output.type_as(x)


@jit
class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


# ── Patch Embedding (Conv3d replacement via matmul) ─────────────────────

@jit
class WanPatchEmbed(nn.Module):
    """Matmul-based patch embedding (Neuron does not support Conv3d)."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.patch_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, C, F, H, W = x.shape
        pT, pH, pW = self.patch_size
        x = x.reshape(B, C, F // pT, pT, H // pH, pH, W // pW, pW)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, (F // pT) * (H // pH) * (W // pW), C * pT * pH * pW)
        # Neuron requires matching dtypes for matmul
        w = self.weight.flatten(1).to(x.dtype).t()
        b = self.bias.to(x.dtype)
        out = torch.matmul(x, w) + b
        out = out.transpose(1, 2).reshape(
            B, self.out_channels, F // pT, H // pH, W // pW)
        return out


# ── FFN ─────────────────────────────────────────────────────────────────

@jit
class WanFFN(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.gelu = GELU()
        self.fc2 = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


# ── RoPE (float32 cos/sin, no complex/float64) ─────────────────────────

def rope_params(max_seq_len, dim, theta=10000):
    """Precompute (cos, sin) each [max_seq_len, dim//2] float32."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.cos(freqs).float(), torch.sin(freqs).float()


def sinusoidal_embedding_1d(dim, position):
    """1-D sinusoidal embeddings (float32, no float64)."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.float()
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def causal_rope_apply(x, grid_sizes, freqs_cos, freqs_sin, start_frame=torch.tensor(0)):
    """Apply 3D rotary position embeddings using cos/sin (no complex)."""
    n, c = x.size(2), x.size(3) // 2
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    f, h, w = grid_sizes
    seq_len = f * h * w
    frame_idx = start_frame + torch.arange(f, device=start_frame.device)

    cos = torch.cat([
        torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_cos[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_cos[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)

    sin = torch.cat([
        torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_sin[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_sin[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)

    x_0 = x[:, :seq_len].to(torch.float32)
    x_pairs = x_0.reshape(1, seq_len, n, c, 2)
    x_re = x_pairs[:, :, :, :, 0:1].reshape(1, seq_len, n, c)
    x_im = x_pairs[:, :, :, :, 1:2].reshape(1, seq_len, n, c)
    out_re = x_re * cos - x_im * sin
    out_im = x_re * sin + x_im * cos
    x_0 = torch.cat([out_re.unsqueeze(-1), out_im.unsqueeze(-1)], dim=-1)
    x_0 = x_0.reshape(1, seq_len, n, c * 2)
    return x_0.type_as(x)


# ── Unpatchify ──────────────────────────────────────────────────────────

def unpatchify(x, out_dim, patch_size, grid_sizes):
    f, h, w = grid_sizes
    pT, pH, pW = patch_size
    u = x.squeeze(0).view(f, h, w, pT, pH, pW, out_dim)
    u = u.permute(6, 0, 3, 1, 4, 2, 5).contiguous()
    u = u.reshape(out_dim, f * pT, h * pH, w * pW)
    return u


# ── Flow pred → x0 conversion (float32) ────────────────────────────────

def convert_flow_pred_to_x0(flow_pred, xt, sigma_t):
    dtype = flow_pred.dtype
    flow_pred = flow_pred.float()
    xt = xt.float()
    sigma_t = sigma_t.float().reshape(-1, 1, 1, 1)
    x0_pred = xt - sigma_t * flow_pred
    return x0_pred.to(dtype)


# ── Modulation helpers ──────────────────────────────────────────────────

def modulation_chunk(modulation, e):
    e = modulation.unsqueeze(1) + e
    return e[:, :, 0:1], e[:, :, 1:2], e[:, :, 2:3], e[:, :, 3:4], e[:, :, 4:5], e[:, :, 5:6]


def modulated_norm_scale(norm_x, scale, ones, num_frames, frame_seqlen):
    y = norm_x.unflatten(1, (num_frames, frame_seqlen))
    return y * (ones + scale)


def modulated_norm_shift(y, shift):
    return (y + shift).flatten(1, 2)


def modulated_residual(x, y, scale, num_frames, frame_seqlen):
    return x + (y.unflatten(1, (num_frames, frame_seqlen)) * scale).flatten(1, 2)


def causal_head_modulate(x, e, modulation):
    e = modulation.unsqueeze(1) + e
    e_shift = e[:, :, 0:1]
    e_scale = e[:, :, 1:2]
    return x * (1 + e_scale) + e_shift


# ── CausalHead ──────────────────────────────────────────────────────────

class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        out_channels = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = jit(nn.Linear(dim, out_channels))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self._modulate = jit(causal_head_modulate)

    def forward(self, x, e):
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        x = self.norm(x).unflatten(1, (num_frames, frame_seqlen))
        x = self._modulate(x, e, self.modulation)
        return self.head(x)


# ── Cross Attention ─────────────────────────────────────────────────────

class WanT2VCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True,
                 eps=1e-6, layer_idx=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_idx = layer_idx
        self.q = jit(nn.Linear(dim, dim))
        self.k = jit(nn.Linear(dim, dim))
        self.v = jit(nn.Linear(dim, dim))
        self.o = jit(nn.Linear(dim, dim))
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)
        self.register_buffer('identity', torch.eye(self.head_dim), persistent=False)
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, context, context_lens, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        assert crossattn_cache is not None
        if not crossattn_cache["is_init"]:
            crossattn_cache["is_init"] = True
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            crossattn_cache["k"] = k
            crossattn_cache["v"] = v
        else:
            k = crossattn_cache["k"]
            v = crossattn_cache["v"]

        if q.device.type == "neuron" and NKI_AVAILABLE:
            q_nki = q[0].permute(1, 2, 0).contiguous()
            k_nki = k[0].permute(1, 2, 0).contiguous()
            v_nki = v[0].permute(1, 0, 2).contiguous()
            seqlen_q = q_nki.shape[2]
            P = 128
            pad = (P - seqlen_q % P) % P
            if pad > 0:
                q_nki = F.pad(q_nki, (0, pad))
            x_nki = wan_cross_attn(q_nki, k_nki, v_nki, self.identity,
                                   softmax_scale=self.softmax_scale)
            x = x_nki[:seqlen_q].unsqueeze(0).flatten(2)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attn_out = F.scaled_dot_product_attention(q, k, v)
            x = attn_out.permute(0, 2, 1, 3).flatten(2)

        x = self.o(x)
        return x


# ── Self Attention ──────────────────────────────────────────────────────

class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=1,
                 qk_norm=True, eps=1e-6, layer_idx=0, frame_length=1560):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_idx = layer_idx
        self.frame_length = frame_length
        self.max_attention_size = 21 * frame_length
        self.block_length = 3 * frame_length
        self.kv_cache_logical_size = 24 * frame_length

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)
        self._rope_nki_available = ROPE_NKI_AVAILABLE
        self._rope_kernel = causal_rope_rotation_nki
        self._nki_available = SELF_ATTN_NKI_AVAILABLE
        self._self_attn_kernel = wan_flash_self_attn_nki
        self.register_buffer('identity', torch.eye(self.head_dim), persistent=False)
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        # #20 KV-prefix cache: the anchor+window prefix assembled into the attention
        # buffer is IDENTICAL across all denoise steps of a block (kv_cache only
        # updates on the block's first step). Caching it per-layer and rebuilding
        # only the small current-block tail per step removes the dominant per-step
        # copy NEFFs (41% DMA in the profiler). Gated; persists across steps.
        self._kv_prefix_cache = os.environ.get("USE_KV_PREFIX_CACHE", "").lower() in ("1", "true")
        self._pfx_k = None        # per-layer persistent attention buffers (k, v)
        self._pfx_v = None
        self._pfx_start = -1      # current_start the cached prefix was built for
        self._pfx_offset = 0      # prefix length (tail written at [offset:offset+valid])

    def cache_copy_inplace(self, k_dst, k_src, v_dst=None, v_src=None):
        k_dst.copy_(k_src)
        if v_dst is not None:
            v_dst.copy_(v_src)

    def _nki_rope_apply(self, x, grid_sizes, freqs_cos, freqs_sin, start_frame):
        if x.device.type != "neuron" or not self._rope_nki_available:
            return causal_rope_apply(
                x, grid_sizes, freqs_cos, freqs_sin, start_frame=start_frame
            ).type_as(x)

        b, s, n, d = x.shape
        f, h, w = grid_sizes
        seq_len = f * h * w
        c = d // 2
        s0 = c - 2 * (c // 3)
        s1 = c // 3
        frame_idx = start_frame + torch.arange(f, device=x.device)

        cos_half = torch.cat([
            torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_cos[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_cos[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, c)

        sin_half = torch.cat([
            torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_sin[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_sin[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, c)

        cos_expanded = cos_half.repeat_interleave(2, dim=-1)
        sin_expanded = sin_half.repeat_interleave(2, dim=-1)
        sign = torch.ones(d, device=x.device, dtype=sin_expanded.dtype)
        sign[0::2] = -1.0
        sin_signed = sin_expanded * sign.unsqueeze(0)
        cos_sin = torch.cat([cos_expanded, sin_signed], dim=-1).contiguous()

        P = 128
        pad = (P - seq_len % P) % P
        if pad > 0:
            cos_sin = F.pad(cos_sin, (0, 0, 0, pad))
            x_nki = F.pad(x[0, :seq_len], (0, 0, 0, 0, 0, pad))
        else:
            x_nki = x[0, :seq_len]

        out = self._rope_kernel(x_nki, cos_sin, num_heads=n, head_dim=d)
        return out[:seq_len].unsqueeze(0).type_as(x)

    def forward(self, x, grid_sizes, freqs_cos, freqs_sin,
                kv_cache=None, current_start=0, cache_start=None,
                updating_cache=False, num_valid_frames=None, shared_buffers=None):
        assert kv_cache is not None
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert b == 1
        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        f, h, w = grid_sizes
        frame_seqlen = h * w
        current_start_frame = current_start // frame_seqlen
        current_start_frame_t = torch.tensor(current_start_frame, device=x.device)
        roped_query = self._nki_rope_apply(q, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t)
        roped_key = self._nki_rope_apply(k, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t)

        num_frames_per_block = self.block_length // self.frame_length
        grid_sizes_one_block = (num_frames_per_block, h, w)

        if num_valid_frames is not None:
            valid_tokens = num_valid_frames * frame_seqlen
        else:
            valid_tokens = f * h * w

        # ── TRAINING (grad-on) FUNCTIONAL PATH — no in-place cache writes ──
        # Autograd forbids in-place modification of slice-views (the .copy_ into
        # buffer_k/kv_cache slices), which is fine for inference (no_grad) but fatal
        # for backprop (SliceBackward inplace error under FSDP). For DISTILLATION the
        # student runs 1-step / single-block: local_start_index==0, no eviction, so
        # attention K/V = the CURRENT block's own roped_key/v — no cache assembly
        # needed. Build it functionally (out-of-place) and attend directly. Matches
        # the clean-forward shape of the reference Neuron training examples.
        if self.training:  # set once on the student DiT; stable across ckpt fwd/recompute
            k_len_int = valid_tokens
            kf = roped_key[:, :valid_tokens]              # [1, valid, n, d]
            vf = v[:, :valid_tokens]
            q_attn = roped_query.permute(0, 2, 1, 3)
            k_attn = kf.permute(0, 2, 1, 3)
            v_attn = vf.permute(0, 2, 1, 3)
            attn_out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
            x = attn_out.permute(0, 2, 1, 3).flatten(2)
            return self.o(x)

        # Cache management
        cache_end = cache_start + self.block_length
        global_end_index = kv_cache["global_end_index"]
        local_end_index_current = kv_cache["local_end_index"]
        num_new_tokens = cache_end - global_end_index
        kv_cache_size = self.kv_cache_logical_size

        buffer_k, buffer_v = shared_buffers
        sink_tokens = self.block_length

        num_evicted = 0
        if (num_new_tokens > 0) and (num_new_tokens + local_end_index_current > kv_cache_size):
            num_evicted = num_new_tokens + local_end_index_current - kv_cache_size
            evict_rolled = kv_cache_size - 2 * sink_tokens
            src_start = sink_tokens + num_evicted
            self.cache_copy_inplace(
                buffer_k[0, :evict_rolled], kv_cache["k"][0, src_start:src_start + evict_rolled],
                buffer_v[0, :evict_rolled], kv_cache["v"][0, src_start:src_start + evict_rolled])
            self.cache_copy_inplace(
                kv_cache["k"][0, sink_tokens:sink_tokens + evict_rolled], buffer_k[0, :evict_rolled],
                kv_cache["v"][0, sink_tokens:sink_tokens + evict_rolled], buffer_v[0, :evict_rolled])

        local_end_index = local_end_index_current + num_new_tokens - num_evicted
        local_start_index = local_end_index - self.block_length

        if local_start_index == 0:
            self.cache_copy_inplace(
                kv_cache["k"][0, :self.block_length], k[0, :self.block_length],
                kv_cache["v"][0, :self.block_length], v[0, :self.block_length])
        else:
            self.cache_copy_inplace(
                kv_cache["k"][0, local_start_index:local_end_index], roped_key[0, :self.block_length],
                kv_cache["v"][0, local_start_index:local_end_index], v[0, :self.block_length])

        if num_new_tokens > 0:
            kv_cache["global_end_index"] = cache_end
            kv_cache["local_end_index"] = local_end_index

        # Assemble KV into buffers
        if updating_cache:
            cache_len = min(local_end_index, self.max_attention_size)
            cache_start_pos = max(0, local_end_index - self.max_attention_size)
            self.cache_copy_inplace(
                buffer_k[0, :cache_len], kv_cache["k"][0, cache_start_pos:cache_start_pos + cache_len],
                buffer_v[0, :cache_len], kv_cache["v"][0, cache_start_pos:cache_start_pos + cache_len])
            if cache_start_pos == 0:
                anchor_roped = self._nki_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin,
                    start_frame=torch.tensor(0, device=v.device))
                self.cache_copy_inplace(buffer_k[0, :self.block_length], anchor_roped[0])
            k_len_int = cache_len
        else:
            # #20: the anchor+window PREFIX (built into buffer_k/v at [0:offset]) is
            # identical across all denoise steps of a block — kv_cache only changes on
            # the block's first step. When USE_KV_PREFIX_CACHE, assemble the prefix
            # into a PER-LAYER persistent buffer once per block (detected by a change
            # in current_start), and on later steps skip the big prefix copies and
            # write only the small current-block tail. Same math, ~1/num_steps the DMA.
            use_pfx = self._kv_prefix_cache
            if use_pfx:
                if (self._pfx_k is None) or (self._pfx_k.shape != buffer_k.shape):
                    self._pfx_k = torch.zeros_like(buffer_k)
                    self._pfx_v = torch.zeros_like(buffer_v)
                    self._pfx_start = -1
                dst_k, dst_v = self._pfx_k, self._pfx_v
                prefix_fresh = (self._pfx_start == int(current_start))
            else:
                dst_k, dst_v = buffer_k, buffer_v
                prefix_fresh = False

            offset = 0
            if local_start_index > 0:
                if not prefix_fresh:
                    wc_max = self.max_attention_size - valid_tokens - self.block_length
                    wc_end = local_start_index
                    wc_start = max(self.block_length, wc_end - wc_max)
                    wc_len = wc_end - wc_start
                    wc_frame_length = wc_len // self.frame_length
                    rope_start_frame = current_start_frame - wc_frame_length - num_frames_per_block
                    anchor_roped = self._nki_rope_apply(
                        kv_cache["k"][0, :self.block_length].unsqueeze(0),
                        grid_sizes_one_block, freqs_cos, freqs_sin,
                        start_frame=torch.tensor(rope_start_frame, device=v.device))
                    self.cache_copy_inplace(
                        dst_k[0, :self.block_length], anchor_roped[0],
                        dst_v[0, :self.block_length], kv_cache["v"][0, :self.block_length])
                    offset = self.block_length
                    if wc_len > 0:
                        self.cache_copy_inplace(
                            dst_k[0, offset:offset + wc_len], kv_cache["k"][0, wc_start:wc_start + wc_len],
                            dst_v[0, offset:offset + wc_len], kv_cache["v"][0, wc_start:wc_start + wc_len])
                    offset += wc_len
                    if use_pfx:
                        self._pfx_offset = offset
                else:
                    # Prefix already in dst_k/v from this block's first step — reuse it.
                    offset = self._pfx_offset

            self.cache_copy_inplace(
                dst_k[0, offset:offset + valid_tokens], roped_key[0, :valid_tokens],
                dst_v[0, offset:offset + valid_tokens], v[0, :valid_tokens])
            k_len_int = offset + valid_tokens
            if use_pfx:
                self._pfx_start = int(current_start)
                buffer_k, buffer_v = dst_k, dst_v  # attention reads from the per-layer buffer

        # Attention
        if roped_query.device.type == "neuron" and self._nki_available:
            q_kern = roped_query[0].permute(1, 2, 0).contiguous()
            k_kern = buffer_k[0].permute(1, 2, 0).contiguous()
            v_kern = buffer_v[0].permute(1, 0, 2).contiguous()
            seqlen_k = k_kern.shape[2]
            seqlen_q_orig = q_kern.shape[2]
            P = 128
            pad_q = (P - seqlen_q_orig % P) % P
            if pad_q > 0:
                q_kern = F.pad(q_kern, (0, pad_q))
            mask = torch.zeros((P, seqlen_k), dtype=torch.bfloat16, device=q_kern.device)
            if k_len_int < seqlen_k:
                mask[:, k_len_int:] = float('-inf')
            num_sections = seqlen_k // ATTN_SEQLEN_MULTIPLE
            x = self._self_attn_kernel(
                q_kern, k_kern, v_kern, self.identity, mask,
                softmax_scale=self.softmax_scale, num_sections=num_sections)
            x = x[:seqlen_q_orig].unsqueeze(0).flatten(2)
        else:
            q_attn = roped_query.permute(0, 2, 1, 3)
            k_attn = buffer_k[:, :k_len_int].permute(0, 2, 1, 3)
            v_attn = buffer_v[:, :k_len_int].permute(0, 2, 1, 3)
            attn_out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
            x = attn_out.permute(0, 2, 1, 3).flatten(2)

        x = self.o(x)
        return x


# ── Attention Block ─────────────────────────────────────────────────────

class CausalWanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads,
                 local_attn_size=-1, sink_size=0, qk_norm=True,
                 cross_attn_norm=False, eps=1e-6, layer_idx=0,
                 frame_length=1560):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.norm2 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps,
            layer_idx, frame_length)
        self.cross_attn = WanT2VCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps, layer_idx=layer_idx)
        # FFN: Linear→GELU→Linear (compiled AFTER TP sharding in tp_utils.py)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), GELU(), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # Modulation helpers (plain functions — compiled after sharding)
        self._modulation_chunk = modulation_chunk
        self._modulated_norm_scale = modulated_norm_scale
        self._modulated_norm_shift = modulated_norm_shift
        self._modulated_residual = modulated_residual

    def forward(self, x, e, grid_sizes, freqs_cos, freqs_sin, context,
                context_lens, updating_cache=False, kv_cache=None,
                crossattn_cache=None, current_start=0, cache_start=None,
                num_valid_frames=None, shared_buffers=None):
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e0, e1, e2, e3, e4, e5 = self._modulation_chunk(self.modulation, e)

        norm_ones = torch.ones_like(e1)
        y = self.self_attn(
            self._modulated_norm_shift(
                self._modulated_norm_scale(self.norm1(x), e1, norm_ones, num_frames, frame_seqlen),
                e0),
            grid_sizes, freqs_cos, freqs_sin, kv_cache, current_start,
            cache_start, updating_cache=updating_cache,
            num_valid_frames=num_valid_frames, shared_buffers=shared_buffers)
        x = self._modulated_residual(x, y, e2, num_frames, frame_seqlen)

        x = x + self.cross_attn(self.norm3(x), context, context_lens,
                                crossattn_cache=crossattn_cache)

        # FFN. (Old inner FFN-checkpoint removed — FSDP's apply_activation_checkpointing
        # wraps the WHOLE block; a second inner checkpoint double-recomputes -> tensor
        # count mismatch in NO_REENTRANT.)
        y = self.ffn(
            self._modulated_norm_shift(
                self._modulated_norm_scale(self.norm2(x), e4, norm_ones, num_frames, frame_seqlen),
                e3))
        x = self._modulated_residual(x, y, e5, num_frames, frame_seqlen)
        return x
