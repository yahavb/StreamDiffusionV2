"""Neuron-compatible CausalWanModel for StreamDiffusionV2.

Ported from aws-neuron-eks-samples/rolling-forcing/app/models/causal_model.py.
Uses Neuron-safe layers (no Conv3d, no flex_attention, no float64).
"""
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from models.wan.neuron_layers import (
    GELU, SiLU, WanPatchEmbed, CausalHead, CausalWanAttentionBlock,
    rope_params, sinusoidal_embedding_1d, unpatchify, jit, ATTN_SEQLEN_MULTIPLE,
    neuron_compile,
)


def _init_rope_freqs(dim, num_heads):
    assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
    d = dim // num_heads
    cos_0, sin_0 = rope_params(1024, d - 4 * (d // 6))
    cos_1, sin_1 = rope_params(1024, 2 * (d // 6))
    cos_2, sin_2 = rope_params(1024, 2 * (d // 6))
    return torch.cat([cos_0, cos_1, cos_2], dim=1), torch.cat([sin_0, sin_1, sin_2], dim=1)


class NeuronCausalWanModel(ModelMixin, ConfigMixin):
    """Neuron-compatible CausalWanModel using Neuron-safe layers."""

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['CausalWanAttentionBlock']

    @register_to_config
    def __init__(self, model_type='t2v', patch_size=(1, 2, 2), text_len=512,
                 in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256,
                 text_dim=4096, out_dim=16, num_heads=16, num_layers=32,
                 local_attn_size=-1, sink_size=0, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6, frame_length=1560):
        super().__init__()
        assert model_type == 't2v'
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # DiT submodules (compiled AFTER TP sharding via tp_utils.py)
        self.patch_embedding = WanPatchEmbed(in_dim, dim, patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), GELU(), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                't2v_cross_attn', dim, ffn_dim, num_heads,
                local_attn_size, sink_size, qk_norm, cross_attn_norm,
                eps, layer_idx, frame_length)
            for layer_idx in range(num_layers)
        ])

        self.head = CausalHead(dim, out_dim, patch_size, eps)
        self._sinusoidal_embedding_1d = jit(sinusoidal_embedding_1d)
        self._unpatchify = jit(unpatchify)
        self.freqs_cos, self.freqs_sin = _init_rope_freqs(dim, num_heads)
        self.num_frame_per_block = 1

    def _update_frame_length(self, new_frame_length, num_frame_per_block=3, num_kv_cache=6):
        """Update self-attention dims to match pipeline's actual cache allocation."""
        block_length = num_frame_per_block * new_frame_length
        kv_cache_logical_size = num_kv_cache * new_frame_length
        max_attention_size = min(num_kv_cache, 21) * new_frame_length
        for block in self.blocks:
            attn = block.self_attn
            attn.frame_length = new_frame_length
            attn.block_length = block_length
            attn.max_attention_size = max_attention_size
            attn.kv_cache_logical_size = kv_cache_logical_size

    def _forward_inference(self, x, t, context, updating_cache=False,
                           kv_cache=None, crossattn_cache=None,
                           current_start=0, cache_start=None,
                           num_valid_frames=None, shared_buffers=None):
        assert self.model_type == 't2v'
        assert x.shape[0] == 1
        # Inference asserts no-grad (the rolling KV cache uses in-place .copy_ writes,
        # fine for eval). For DISTILLATION we backprop through this forward on Neuron
        # (proven: eager loss.backward() works on device — see private-torch-neuronx
        # gpt2-train-loop). The .copy_ targets are pre-allocated BUFFERS, not autograd
        # leaves, so grads flow through attention normally. Allow grad when the DiT is
        # in training mode; keep the inference guard otherwise.
        if not self.training:
            assert not torch.is_grad_enabled(), \
                "grad enabled in inference forward — set model.eval() or wrap in no_grad"

        # Access device from a raw parameter (compiled modules may not expose .weight)
        device = next(self.parameters()).device
        if self.freqs_cos.device != device:
            self.freqs_cos = self.freqs_cos.to(device)
            self.freqs_sin = self.freqs_sin.to(device)

        x = self.patch_embedding(x)
        grid_sizes = tuple(int(d) for d in x.shape[2:])
        x = x.flatten(2).transpose(1, 2)

        e = self.time_embedding(
            self._sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        assert context.size(1) == self.text_len
        context = self.text_embedding(context)

        kwargs = dict(
            e=e0, grid_sizes=grid_sizes,
            freqs_cos=self.freqs_cos, freqs_sin=self.freqs_sin,
            context=context, context_lens=context_lens,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
        )

        for block_index, block in enumerate(self.blocks):
            kwargs.update({
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index],
                "current_start": current_start,
                "cache_start": cache_start,
            })
            x = block(x, **kwargs)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = x.flatten(1, 2)
        result = self._unpatchify(x, self.out_dim, self.patch_size, grid_sizes).unsqueeze(0)
        return result

    def forward(self, *args, **kwargs):
        assert kwargs.get('kv_cache', None) is not None
        return self._forward_inference(*args, **kwargs)
