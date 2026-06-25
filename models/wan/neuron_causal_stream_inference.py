"""Neuron-compatible CausalStreamInferencePipeline for StreamDiffusionV2.

Adapted from causal_stream_inference.py for Trainium/Neuron devices.
Key differences from GPU version:
  - KV cache indices are Python int (not tensor) for Neuron tracing
  - Per-rank model placement: T5 on rank 2, VAE on rank 0, DiT on all
  - Shared attention buffers for NKI kernels
  - Uses NeuronCausalWanDiffusionWrapper, NeuronWanTextEncoder, NeuronWanVAEWrapper
"""
import logging
import os
import sys
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from models.wan.neuron_wan_wrapper import (
    NeuronCausalWanDiffusionWrapper,
    NeuronWanTextEncoder,
    NeuronWanVAEWrapper,
)
from models.wan.neuron_layers import ATTN_SEQLEN_MULTIPLE

LOGGER = logging.getLogger(__name__)

# Per-rank model placement (matches rolling-forcing layout):
# T5 on rank 2 (ND1) — keeps T5 off the same HBM bank as VAE
# VAE on rank 0 (ND0) — lightweight, only needed at output time
# DiT on ALL ranks — TP-4 sharded
T5_RANK = int(os.environ.get("T5_RANK", "2"))
VAE_RANK = int(os.environ.get("VAE_RANK", "0"))


def add_noise(original_samples, noise, sigma):
    """Diffusion forward process: mix clean samples with noise.

    Ported from rolling-forcing causal_inference_pipeline.add_noise. Replaces
    FlowMatchScheduler.add_noise()'s argmin lookup with a precomputed sigma.

    Args:
        original_samples: [B*F, C, H, W] clean latents
        noise:            [B*F, C, H, W] random noise
        sigma:            [B*F, 1, 1, 1] precomputed sigma values
    Returns: [B*F, C, H, W] noisy samples, same dtype as noise
    """
    return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)


class _ContiguousWrapper(nn.Module):
    """Wraps a compiled module to ensure all tensor inputs are contiguous.
    
    Neuron's torch.compile(backend='neuron') requires contiguous tensors.
    When num_frame_per_block > 1, various ops (attention reshape, unflatten,
    expand, etc.) produce non-contiguous views. This wrapper calls .contiguous()
    on all tensor args/kwargs before forwarding to the compiled module.
    
    Also proxies attribute access (e.g. .weight, .bias) to the inner module
    so that code like `self.patch_embedding.weight.device` still works.
    """
    def __init__(self, compiled_module):
        super().__init__()
        self.compiled_module = compiled_module

    def forward(self, *args, **kwargs):
        args = tuple(a.contiguous() if isinstance(a, torch.Tensor) and not a.is_contiguous() else a
                     for a in args)
        kwargs = {k: v.contiguous() if isinstance(v, torch.Tensor) and not v.is_contiguous() else v
                  for k, v in kwargs.items()}
        return self.compiled_module(*args, **kwargs)

    def __getattr__(self, name):
        if name == 'compiled_module':
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.compiled_module, name)


def _contiguous_compile(module):
    """Compile a module with neuron backend and wrap with contiguous guard."""
    compiled = torch.compile(module, backend='neuron', dynamic=False)
    return _ContiguousWrapper(compiled)


class NeuronCausalStreamInferencePipeline(nn.Module):
    """StreamDiffusionV2 streaming inference pipeline on Neuron/Trainium."""

    def __init__(self, args, device="neuron"):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        # TP config
        self.tp_degree = getattr(args, "tp_degree", 4)

        # Model config
        model_path = getattr(args, "model_path", "wan_models/Wan2.1-T2V-1.3B")
        checkpoint_path = getattr(args, "generator_ckpt", None)
        vae_path = getattr(args, "vae_path",
                           f"{model_path}/Wan2.1_VAE.pth")
        use_ema = getattr(args, "use_ema", False)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        # Denoising schedule
        denoising_step_list = list(args.denoising_step_list)
        timestep_shift = getattr(args, "timestep_shift", 8.0)
        warp = getattr(args, "warp_denoising_step", False)

        # Model type
        model_type = getattr(args, "model_type", "T2V-1.3B")
        if model_type == "T2V-1.3B":
            self.num_transformer_blocks = 30
            self.num_heads = 12
            self.dim = 1536
        elif model_type == "T2V-14B":
            self.num_transformer_blocks = 40
            self.num_heads = 40
            self.dim = 5120
        else:
            raise ValueError(f"Model type {model_type} not supported")

        # With TP, each rank only holds num_heads / tp_degree heads
        self.num_heads_per_rank = self.num_heads // self.tp_degree
        self.head_dim = self.dim // self.num_heads

        # Spatial dims
        scale_size = 16
        self.height = args.height // scale_size * 2
        self.width = args.width // scale_size * 2
        self.frame_seq_length = (args.height // scale_size) * (args.width // scale_size)
        self.num_kv_cache = getattr(args, "num_kv_cache", 6)
        self.kv_cache_length = self.frame_seq_length * self.num_kv_cache

        # Initialize generator (DiT) with TP
        self.generator = NeuronCausalWanDiffusionWrapper(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            use_ema=use_ema,
            denoising_step_list=denoising_step_list,
            timestep_shift=timestep_shift,
            num_frame_per_block=self.num_frame_per_block,
            device=device,
            tp_degree=self.tp_degree,
        )

        # Update frame length in model (must match pipeline's kv_cache allocation)
        self.generator.model._update_frame_length(
            self.frame_seq_length, self.num_frame_per_block, self.num_kv_cache)

        # Per-rank model placement (same as rolling-forcing):
        # T5 (~9.6 GB) on T5_RANK only (rank 2, ND0-NC2) with torch.compile
        # VAE (~0.66 GB) on VAE_RANK only (rank 0, ND0-NC0) with torch.compile
        # DiT TP-4 sharded on all ranks with torch.compile on sub-modules
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # All ranks need the tokenizer (lightweight, CPU-only)
        wan_base = os.path.join(os.path.dirname(__file__), "wan_base")
        if wan_base not in sys.path:
            sys.path.insert(0, wan_base)
        from modules.tokenizers import HuggingfaceTokenizer
        tokenizer_path = os.path.join(model_path, "google/umt5-xxl/")
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace')

        # Initialize T5 (only on T5_RANK — on Neuron with torch.compile)
        if self.rank == T5_RANK:
            LOGGER.info(f"Loading T5 text encoder on Neuron (rank {T5_RANK})...")
            self.text_encoder = NeuronWanTextEncoder(
                model_path=model_path, device=device)
        else:
            self.text_encoder = None

        # Initialize VAE (only on VAE_RANK — with torch.compile)
        if self.rank == VAE_RANK:
            LOGGER.info(f"Loading VAE decoder on rank {VAE_RANK}...")
            self.vae = NeuronWanVAEWrapper(
                vae_pth=vae_path, device=device)
            # Compile VAE for fast decode
            self.vae._ensure_model()
            self.vae._model = torch.compile(
                self.vae._model, backend='neuron', dynamic=False)
            LOGGER.info("VAE compiled with torch.compile(backend='neuron')")
        else:
            self.vae = None

        # Compile DiT sub-modules (same as rolling-forcing inference_neuron_tp.py:229-237)
        # Wrap with ContiguousWrapper to ensure all tensor inputs are contiguous
        # before passing to compiled NEFF — required when num_frame_per_block > 1
        # produces non-contiguous views from attention/reshape ops.
        dit_model = self.generator.model
        dit_model.patch_embedding = _contiguous_compile(dit_model.patch_embedding)
        dit_model.text_embedding = _contiguous_compile(dit_model.text_embedding)
        dit_model.time_embedding = _contiguous_compile(dit_model.time_embedding)
        dit_model.time_projection = _contiguous_compile(dit_model.time_projection)
        dit_model.head = _contiguous_compile(dit_model.head)
        # Fine-grained submodule compilation — MATCH rolling-forcing (8.09 fps).
        # RF compiles q/k/v/o (self+cross), ffn, AND norm1/2/3 per block; NKI
        # kernels (attn/rope) stay eager BETWEEN compiled ops. SD previously
        # compiled ONLY ffn, leaving the 8 attention Linears + 3 norms in eager
        # op-by-op dispatch — the dominant per-block DiT cost (4.3s vs RF 1.1s).
        for block in dit_model.blocks:
            block.self_attn.q = _contiguous_compile(block.self_attn.q)
            block.self_attn.k = _contiguous_compile(block.self_attn.k)
            block.self_attn.v = _contiguous_compile(block.self_attn.v)
            block.self_attn.o = _contiguous_compile(block.self_attn.o)
            block.cross_attn.q = _contiguous_compile(block.cross_attn.q)
            block.cross_attn.k = _contiguous_compile(block.cross_attn.k)
            block.cross_attn.v = _contiguous_compile(block.cross_attn.v)
            block.cross_attn.o = _contiguous_compile(block.cross_attn.o)
            block.ffn = _contiguous_compile(block.ffn)
            block.norm1 = _contiguous_compile(block.norm1)
            block.norm2 = _contiguous_compile(block.norm2)
            block.norm3 = _contiguous_compile(block.norm3)
        LOGGER.info(f"DiT compiled: patch/text/time/head + per-block "
                    f"q/k/v/o(self+cross)+ffn+norm1/2/3 × {len(dit_model.blocks)}")

        # Scheduler
        self.scheduler = self.generator.scheduler

        # Denoising step list
        self.denoising_step_list = torch.tensor(
            denoising_step_list, dtype=torch.long, device=self.device)

        t2v = getattr(args, "t2v", True)
        if not t2v and self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        if warp:
            # Map step indices to actual scheduler timesteps (flow-matching sigma schedule)
            timesteps = torch.cat((
                self.scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32)
            )).to(self.device)
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
            LOGGER.info(f"Warped denoising steps: {self.denoising_step_list.tolist()}")
        else:
            LOGGER.info(f"Raw denoising steps: {self.denoising_step_list.tolist()}")

        self.denoising_steps = len(self.denoising_step_list)

        # Rolling-forcing pattern machinery (ported from RF
        # CausalInferencePipeline). context_noise/context_sigma drive the
        # dedicated 3-frame cache-update call; timestep/sigma_patterns drive the
        # per-window padded denoise input.
        self.context_noise = float(getattr(args, "context_noise", 0.0))
        self.timestep_patterns = self._build_timestep_patterns()
        self.sigma_patterns = self._build_sigma_patterns()
        self.context_sigma = self._timestep_to_sigma(self.context_noise)
        self._add_noise = add_noise

        # State
        self.conditional_dict = None
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.hidden_states = None
        self.shared_buffers = None
        self.args = args

        LOGGER.info("NeuronCausalStreamInferencePipeline initialized "
                     "(%d blocks, %d heads, frame_seq=%d, kv_cache=%d)",
                     self.num_transformer_blocks, self.num_heads,
                     self.frame_seq_length, self.kv_cache_length)

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize KV cache with Python int indices (Neuron-safe).
        
        With TP-4, each rank only stores num_heads_per_rank heads.
        """
        kv_cache = []
        for i in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, self.kv_cache_length,
                                  self.num_heads_per_rank, self.head_dim],
                                 dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_length,
                                  self.num_heads_per_rank, self.head_dim],
                                 dtype=dtype, device=device),
                "global_end_index": 0,  # Python int for Neuron
                "local_end_index": 0,   # Python int for Neuron
            })
        self.kv_cache1 = kv_cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize cross-attention cache (TP-aware: local heads only)."""
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, self.num_heads_per_rank, self.head_dim],
                                 dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_heads_per_rank, self.head_dim],
                                 dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache = crossattn_cache

    def _initialize_shared_buffers(self, batch_size, dtype, device):
        """Allocate shared K/V buffers for NKI attention kernels (TP-aware)."""
        max_attn_size = 21 * self.frame_seq_length
        # Round up to ATTN_SEQLEN_MULTIPLE for NKI
        padded = ((max_attn_size + ATTN_SEQLEN_MULTIPLE - 1)
                  // ATTN_SEQLEN_MULTIPLE) * ATTN_SEQLEN_MULTIPLE
        self.shared_buffers = (
            torch.zeros([batch_size, padded, self.num_heads_per_rank, self.head_dim],
                        dtype=dtype, device=device),
            torch.zeros([batch_size, padded, self.num_heads_per_rank, self.head_dim],
                        dtype=dtype, device=device),
        )

    def prepare(self, text_prompts: List[str], device=None,
                dtype=None, noise=None, current_start=0, current_end=None,
                batch_denoise=True, **kwargs):
        """Encode prompt, initialize caches, run first-block anchor denoising.
        
        T5 runs on T5_RANK (rank 2) on Neuron with torch.compile.
        Embeddings (bf16) are broadcast to all ranks for DiT.
        Same pattern as rolling-forcing encode_prompt_distributed().
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        batch_size = noise.shape[0]

        # Step 1: Tokenize on all ranks (CPU, fast)
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)

        # Step 2: T5_RANK encodes with compiled T5 on Neuron
        if self.rank == T5_RANK:
            seq_len = mask.gt(0).sum(dim=1).long()
            enc_result = self.text_encoder(text_prompts)
            prompt_embeds = enc_result['prompt_embeds'].to(device=device, dtype=dtype)
            # Zero-out padding (already done in encoder, but ensure contiguous)
            prompt_embeds = prompt_embeds.contiguous()
        else:
            # Allocate buffer to receive embeddings: [batch, 512, 4096] for umt5-xxl
            prompt_embeds = torch.zeros(
                batch_size, 512, 4096, dtype=dtype, device=device)

        # Step 3: Broadcast bf16 embeddings from T5_RANK to all ranks
        if dist.is_initialized():
            dist.broadcast(prompt_embeds, src=T5_RANK)

        self.conditional_dict = {
            'prompt_embeds': prompt_embeds,
        }

        # Initialize caches
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size, dtype, device)
            self._initialize_crossattn_cache(batch_size, dtype, device)
            self._initialize_shared_buffers(batch_size, dtype, device)
        else:
            for i in range(self.num_transformer_blocks):
                self.crossattn_cache[i]["is_init"] = False

        # Run anchor block denoising
        current_start_int = int(current_start)
        current_end_int = int(current_end) if current_end is not None else self.frame_seq_length

        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=device,
                dtype=torch.int64) * current_timestep

            denoised_pred = self.generator(
                noisy_image_or_video=noise,
                conditional_dict=self.conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_int,
                current_end=current_end_int,
                updating_cache=True,
                shared_buffers=self.shared_buffers,
            )

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1]
                noise = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones([batch_size], device=device,
                                                dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])

        if not batch_denoise:
            return denoised_pred

        # Set up batch denoising state
        self.batch_size = len(self.denoising_step_list)
        self.hidden_states = torch.zeros(
            (self.batch_size, self.num_frame_per_block, *noise.shape[2:]),
            dtype=dtype, device=device)

        # Expand caches for batch denoising
        for i in range(self.num_transformer_blocks):
            self.kv_cache1[i]['k'] = self.kv_cache1[i]['k'].repeat(self.batch_size, 1, 1, 1)
            self.kv_cache1[i]['v'] = self.kv_cache1[i]['v'].repeat(self.batch_size, 1, 1, 1)
            self.crossattn_cache[i]['k'] = self.crossattn_cache[i]['k'].expand(self.batch_size, -1, -1, -1)
            self.crossattn_cache[i]['v'] = self.crossattn_cache[i]['v'].expand(self.batch_size, -1, -1, -1)

        self.kv_cache_starts = (torch.ones(self.batch_size, dtype=torch.long, device=device)
                                * current_end_int)
        self.kv_cache_ends = self.kv_cache_starts + self.frame_seq_length
        self.timestep = self.denoising_step_list.clone()
        self.conditional_dict['prompt_embeds'] = (
            self.conditional_dict['prompt_embeds'].repeat(self.batch_size, 1, 1))

        return denoised_pred

    def prepare_prompt_only(self, text_prompts: List[str], device=None,
                            dtype=None, batch_size=1):
        """Encode the prompt and set conditional_dict — no anchor denoising.

        The rolling-forcing window loop manages its own caches and handles the
        first block through the ramp-up patterns, so (unlike prepare()) it must
        NOT have an anchor block pre-denoised into the KV cache. This mirrors RF,
        where conditional_dict is built by encode_prompt_distributed() and the
        window loop starts from empty caches.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)

        if self.rank == T5_RANK:
            enc_result = self.text_encoder(text_prompts)
            prompt_embeds = enc_result['prompt_embeds'].to(device=device, dtype=dtype).contiguous()
        else:
            prompt_embeds = torch.zeros(batch_size, 512, 4096, dtype=dtype, device=device)

        if dist.is_initialized():
            dist.broadcast(prompt_embeds, src=T5_RANK)

        self.conditional_dict = {'prompt_embeds': prompt_embeds}
        return self.conditional_dict

    def inference_stream(self, noise, current_start, current_end,
                         current_step=None):
        """Run one streaming inference step (single frame block)."""
        self.hidden_states[1:] = self.hidden_states[:-1].clone()
        self.hidden_states[0] = noise[0]

        self.kv_cache_starts[1:] = self.kv_cache_starts[:-1].clone()
        self.kv_cache_starts[0] = current_start

        self.kv_cache_ends[1:] = self.kv_cache_ends[:-1].clone()
        self.kv_cache_ends[0] = current_end

        if current_step is not None:
            self.timestep[0] = current_step

        self.hidden_states = self.generator(
            noisy_image_or_video=self.hidden_states,
            conditional_dict=self.conditional_dict,
            timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=int(self.kv_cache_starts[0]),
            current_end=int(self.kv_cache_ends[0]),
            shared_buffers=self.shared_buffers,
        )

        for i in range(len(self.denoising_step_list) - 1):
            self.hidden_states[[i]] = self.scheduler.add_noise(
                self.hidden_states[[i]],
                torch.randn_like(self.hidden_states[[i]]),
                self.denoising_step_list[i + 1] * torch.ones(
                    [1], device=self.hidden_states.device, dtype=torch.long))

        return self.hidden_states

    def inference_wo_batch(self, noise, current_start, current_end,
                           current_step=None):
        """Run denoising without batch parallelism (sequential steps)."""
        batch_size = noise.shape[0]
        current_start_int = int(current_start)
        current_end_int = int(current_end)

        self.denoising_step_list[0] = current_step
        num_steps = len(self.denoising_step_list)
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=noise.device,
                dtype=torch.int64) * current_timestep

            # MATCH rolling-forcing pipeline: the expensive KV-cache assembly
            # (pad/concat to max_attention_size=8190 — the 35ms+26ms copy NEFFs)
            # is gated by updating_cache. RF runs the denoise steps WITHOUT it
            # (cheap windowed read) and updates the cache ONCE at the end. SD was
            # passing updating_cache=True on ALL 5 steps -> the big assembly ran
            # 5x/block, the dominant per-block cost. Only update on the last step.
            is_last_step = (index == num_steps - 1)
            denoised_pred = self.generator(
                noisy_image_or_video=noise,
                conditional_dict=self.conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_int,
                current_end=current_end_int,
                updating_cache=is_last_step,
                shared_buffers=self.shared_buffers,
            )

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1]
                noise = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones(
                        [batch_size], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])

        return denoised_pred

    # ── Rolling-forcing pattern machinery (ported from RF) ────────────────

    def _build_timestep_patterns(self):
        """Build unique timestep patterns for all window types (CPU, float32).

        Ported verbatim from RF CausalInferencePipeline._build_timestep_patterns.
        Returns a [2*nds-1, max_frames] tensor where:
          - Pattern 0: steady-state (full window)
          - Patterns 1..nds-1: ramp-up (window growing from start)
          - Patterns nds..2*nds-2: ramp-down (window shrinking from end)
        """
        nds = len(self.denoising_step_list)
        nfpb = self.num_frame_per_block
        max_frames = nds * nfpb

        steady = []
        for ts in reversed(self.denoising_step_list):
            steady.extend([ts.item()] * nfpb)

        patterns = [steady]
        for i in range(1, nds):
            cnf = i * nfpb
            patterns.append(steady[-cnf:] + [0.0] * (max_frames - cnf))
        for i in range(1, nds):
            cnf = i * nfpb
            patterns.append(steady[:cnf] + [0.0] * (max_frames - cnf))

        return torch.tensor(patterns, dtype=torch.float32)

    def _timestep_to_sigma(self, timestep_val):
        """Map a single timestep value to its corresponding sigma.

        Ported from RF; uses argmin against the scheduler timestep grid.
        """
        timesteps = self.scheduler.timesteps.to("cpu")
        sigmas = self.scheduler.sigmas.to("cpu")
        idx = torch.argmin((timesteps - timestep_val).abs())
        return sigmas[idx].item()

    def _build_sigma_patterns(self):
        """Precompute sigma values for each timestep pattern (RF layout)."""
        sigma_patterns = torch.zeros_like(self.timestep_patterns)
        for i, pattern in enumerate(self.timestep_patterns):
            for j, t in enumerate(pattern):
                sigma_patterns[i, j] = self._timestep_to_sigma(t.item())
        return sigma_patterns

    @torch.no_grad()
    def inference_rolling_forcing(self, noise, num_output_frames=None):
        """Rolling-forcing windowed denoise loop (ported from RF).

        Replaces the per-block 5-step sequential loop. One windowed denoise
        forward + one 3-frame cache-update per window, amortizing the denoise
        steps across the rolling window.

        Args:
            noise: [B, F, C, H, W] full-clip input noise (on device).
            num_output_frames: number of frames to return (defaults to F).
        Returns:
            [B, num_output_frames, C, H, W] denoised latents.
        """
        device = noise.device
        dtype = noise.dtype
        batch_size, num_frames, num_channels, height, width = noise.shape

        nfpb = self.num_frame_per_block
        requested_frames = num_frames if num_output_frames is None else num_output_frames

        # Round up to next multiple of nfpb
        if num_frames % nfpb != 0:
            padded_total = ((num_frames // nfpb) + 1) * nfpb
            pad_count = padded_total - num_frames
            pad_noise = torch.randn(
                batch_size, pad_count, num_channels, height, width,
                dtype=dtype, device=device)
            noise = torch.cat([noise, pad_noise], dim=1)
            num_frames = padded_total

        num_blocks = num_frames // nfpb
        num_output_frames = requested_frames

        # Caches: initialize or reset
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size, dtype, device)
            self._initialize_crossattn_cache(batch_size, dtype, device)
            self._initialize_shared_buffers(batch_size, dtype, device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = 0
                self.kv_cache1[block_index]["local_end_index"] = 0

        # Construct rolling-forcing windows (RF index bookkeeping verbatim)
        nds = len(self.denoising_step_list)
        rolling_window_length_blocks = nds
        window_start_blocks = []
        window_end_blocks = []
        pattern_indices = []
        window_num = num_blocks + rolling_window_length_blocks - 1

        for window_index in range(window_num):
            start_block = max(0, window_index - rolling_window_length_blocks + 1)
            end_block = min(num_blocks - 1, window_index)
            window_start_blocks.append(start_block)
            window_end_blocks.append(end_block)
            num_blks = end_block - start_block + 1
            if num_blks == nds:
                pattern_indices.append(0)                  # steady-state
            elif start_block == 0:
                pattern_indices.append(num_blks)            # ramp-up
            else:
                pattern_indices.append(nds - 1 + num_blks)  # ramp-down

        max_frames = rolling_window_length_blocks * nfpb

        output = torch.zeros(
            [batch_size, num_output_frames + max_frames - nfpb,
             num_channels, height, width], device=device, dtype=dtype)
        noisy_cache = torch.zeros(
            [batch_size, num_output_frames + max_frames,
             num_channels, height, width], device=device, dtype=dtype)

        if self.timestep_patterns.device != device:
            self.timestep_patterns = self.timestep_patterns.to(device)
            self.sigma_patterns = self.sigma_patterns.to(device)

        padded_input = torch.zeros(
            [batch_size, max_frames, num_channels, height, width],
            device=device, dtype=dtype)
        padded_timestep = torch.zeros(
            [batch_size, max_frames], device=device, dtype=torch.float32)
        padded_sigma = torch.zeros(
            [batch_size, max_frames], device=device, dtype=torch.float32)

        # 3-frame buffers for the dedicated cache-update call (constant t/sigma)
        cache_input = torch.zeros(
            [batch_size, nfpb, num_channels, height, width],
            device=device, dtype=dtype)
        cache_timestep = torch.full(
            [batch_size, nfpb], self.context_noise, device=device, dtype=torch.float32)
        cache_sigma = torch.full(
            [batch_size, nfpb], self.context_sigma, device=device, dtype=torch.float32)

        block_sigma_list = []
        for step in self.denoising_step_list:
            sigma_val = self._timestep_to_sigma(step.item())
            block_sigma_list.append(
                sigma_val * torch.ones(
                    [batch_size * nfpb, 1, 1, 1], dtype=torch.float32, device=device))

        for window_index in range(window_num):
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]

            current_start_frame = start_block * nfpb
            current_end_frame = (end_block + 1) * nfpb
            current_num_frames = current_end_frame - current_start_frame

            padded_input.copy_(
                noisy_cache[:, current_start_frame:current_start_frame + max_frames])

            if current_num_frames == max_frames or current_start_frame == 0:
                noise_offset = current_num_frames - nfpb
                padded_input[:, noise_offset:noise_offset + nfpb].copy_(
                    noise[:, current_end_frame - nfpb:current_end_frame])

            padded_timestep[:] = self.timestep_patterns[pattern_indices[window_index]]
            padded_sigma[:] = self.sigma_patterns[pattern_indices[window_index]]

            num_valid_frames = current_num_frames

            # Windowed denoise call — sigma-driven, tuple return.
            # current_start passed exactly as RF: start_frame * frame_seq_length.
            _, denoised_pred = self.generator(
                noisy_image_or_video=padded_input,
                conditional_dict=self.conditional_dict,
                timestep=padded_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                num_valid_frames=num_valid_frames,
                shared_buffers=self.shared_buffers,
                sigma=padded_sigma,
            )

            copy_end = min(current_start_frame + max_frames, output.shape[1])
            copy_len = copy_end - current_start_frame
            output[:, current_start_frame:copy_end].copy_(denoised_pred[:, :copy_len])

            # Re-noising loop (RF verbatim)
            num_blks = end_block - start_block + 1
            step_base = (num_blks - 1) if (
                start_block == 0 and num_blks < nds) else (nds - 1)

            for block_idx in range(start_block, end_block + 1):
                local_offset = block_idx - start_block
                step_index = step_base - local_offset
                if step_index == nds - 1:
                    continue

                full_noise = torch.randn(
                    batch_size * num_valid_frames, *denoised_pred.shape[2:],
                    dtype=denoised_pred.dtype).to(device)
                block_pred = denoised_pred[
                    :, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                block_noise = full_noise.unflatten(
                    0, (batch_size, num_valid_frames)
                )[:, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                block_sigma = block_sigma_list[step_index + 1]

                noisy_cache[:, block_idx * nfpb:(block_idx + 1) * nfpb] = \
                    self._add_noise(block_pred, block_noise, block_sigma) \
                    .unflatten(0, (batch_size, nfpb))

            # Dedicated 3-frame cache-update call (updating_cache=True).
            cache_input.copy_(denoised_pred[:, :nfpb])
            self.generator(
                noisy_image_or_video=cache_input,
                conditional_dict=self.conditional_dict,
                timestep=cache_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                updating_cache=True,
                num_valid_frames=nfpb,
                shared_buffers=self.shared_buffers,
                sigma=cache_sigma,
            )

        return output[:, :num_output_frames]

    @torch.no_grad()
    def inference_rolling_forcing_streaming(self, noise, num_output_frames=None):
        """Generator version of inference_rolling_forcing (ported from RF).

        Same computation, but yields finalized blocks as they complete all nds
        denoising steps, enabling true streaming overlap with VAE decode.

        Yields:
            (start_frame_index, latent_block [B, nfpb, C, H, W] on CPU)
        """
        device = noise.device
        dtype = noise.dtype
        batch_size, num_frames, num_channels, height, width = noise.shape

        nfpb = self.num_frame_per_block
        requested_frames = num_frames if num_output_frames is None else num_output_frames

        if num_frames % nfpb != 0:
            padded_total = ((num_frames // nfpb) + 1) * nfpb
            pad_count = padded_total - num_frames
            pad_noise = torch.randn(
                batch_size, pad_count, num_channels, height, width,
                dtype=dtype, device=device)
            noise = torch.cat([noise, pad_noise], dim=1)
            num_frames = padded_total

        num_blocks = num_frames // nfpb

        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size, dtype, device)
            self._initialize_crossattn_cache(batch_size, dtype, device)
            self._initialize_shared_buffers(batch_size, dtype, device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = 0
                self.kv_cache1[block_index]["local_end_index"] = 0

        nds = len(self.denoising_step_list)
        max_frames = nds * nfpb
        window_num = num_blocks + nds - 1

        output = torch.zeros(
            [batch_size, num_frames + max_frames - nfpb,
             num_channels, height, width], device=device, dtype=dtype)
        noisy_cache = torch.zeros(
            [batch_size, num_frames + max_frames,
             num_channels, height, width], device=device, dtype=dtype)

        if self.timestep_patterns.device != device:
            self.timestep_patterns = self.timestep_patterns.to(device)
            self.sigma_patterns = self.sigma_patterns.to(device)

        padded_input = torch.zeros(
            [batch_size, max_frames, num_channels, height, width],
            device=device, dtype=dtype)
        padded_timestep = torch.zeros(
            [batch_size, max_frames], device=device, dtype=torch.float32)
        padded_sigma = torch.zeros(
            [batch_size, max_frames], device=device, dtype=torch.float32)

        cache_input = torch.zeros(
            [batch_size, nfpb, num_channels, height, width],
            device=device, dtype=dtype)
        cache_timestep = torch.full(
            [batch_size, nfpb], self.context_noise, device=device, dtype=torch.float32)
        cache_sigma = torch.full(
            [batch_size, nfpb], self.context_sigma, device=device, dtype=torch.float32)

        block_sigma_list = []
        for step in self.denoising_step_list:
            sigma_val = self._timestep_to_sigma(step.item())
            block_sigma_list.append(
                sigma_val * torch.ones(
                    [batch_size * nfpb, 1, 1, 1], dtype=torch.float32, device=device))

        window_start_blocks = []
        window_end_blocks = []
        pattern_indices = []
        for window_index in range(window_num):
            start_block = max(0, window_index - nds + 1)
            end_block = min(num_blocks - 1, window_index)
            window_start_blocks.append(start_block)
            window_end_blocks.append(end_block)
            num_blks = end_block - start_block + 1
            if num_blks == nds:
                pattern_indices.append(0)
            elif start_block == 0:
                pattern_indices.append(num_blks)
            else:
                pattern_indices.append(nds - 1 + num_blks)

        last_finalized_block = -1

        for window_index in range(window_num):
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]

            current_start_frame = start_block * nfpb
            current_end_frame = (end_block + 1) * nfpb
            current_num_frames = current_end_frame - current_start_frame

            padded_input.copy_(
                noisy_cache[:, current_start_frame:current_start_frame + max_frames])

            if current_num_frames == max_frames or current_start_frame == 0:
                noise_offset = current_num_frames - nfpb
                padded_input[:, noise_offset:noise_offset + nfpb].copy_(
                    noise[:, current_end_frame - nfpb:current_end_frame])

            padded_timestep[:] = self.timestep_patterns[pattern_indices[window_index]]
            padded_sigma[:] = self.sigma_patterns[pattern_indices[window_index]]

            num_valid_frames = current_num_frames

            _, denoised_pred = self.generator(
                noisy_image_or_video=padded_input,
                conditional_dict=self.conditional_dict,
                timestep=padded_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                num_valid_frames=num_valid_frames,
                shared_buffers=self.shared_buffers,
                sigma=padded_sigma,
            )

            output[:, current_start_frame:current_start_frame + max_frames].copy_(
                denoised_pred)

            num_blks = end_block - start_block + 1
            step_base = (num_blks - 1) if (
                start_block == 0 and num_blks < nds) else (nds - 1)

            for block_idx in range(start_block, end_block + 1):
                local_offset = block_idx - start_block
                step_index = step_base - local_offset
                if step_index == nds - 1:
                    continue

                full_noise = torch.randn(
                    batch_size * num_valid_frames, *denoised_pred.shape[2:],
                    dtype=denoised_pred.dtype).to(device)
                block_pred = denoised_pred[
                    :, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                block_noise = full_noise.unflatten(
                    0, (batch_size, num_valid_frames)
                )[:, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                block_sigma = block_sigma_list[step_index + 1]

                noisy_cache[:, block_idx * nfpb:(block_idx + 1) * nfpb] = \
                    self._add_noise(block_pred, block_noise, block_sigma) \
                    .unflatten(0, (batch_size, nfpb))

            cache_input.copy_(denoised_pred[:, :nfpb])
            self.generator(
                noisy_image_or_video=cache_input,
                conditional_dict=self.conditional_dict,
                timestep=cache_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                updating_cache=True,
                num_valid_frames=nfpb,
                shared_buffers=self.shared_buffers,
                sigma=cache_sigma,
            )

            # Yield finalized blocks (passed all nds denoising steps).
            finalized_block = window_index - nds + 1
            if finalized_block > last_finalized_block and finalized_block >= 0:
                for blk in range(last_finalized_block + 1, finalized_block + 1):
                    if blk < num_blocks:
                        sf = blk * nfpb
                        ef = min((blk + 1) * nfpb, requested_frames)
                        if sf < requested_frames:
                            yield (sf, output[:, sf:ef].clone().cpu())
                last_finalized_block = finalized_block

        # Ramp-down: yield remaining blocks.
        for blk in range(last_finalized_block + 1, num_blocks):
            sf = blk * nfpb
            ef = min((blk + 1) * nfpb, requested_frames)
            if sf < requested_frames:
                yield (sf, output[:, sf:ef].clone().cpu())

    def decode_latents(self, latents):
        """Decode latents to pixel space using VAE (only on VAE_RANK).
        
        Returns decoded video on VAE_RANK, None on other ranks.
        """
        if self.rank == VAE_RANK:
            return self.vae.decode_to_pixel(latents)
        else:
            # Other ranks don't have VAE loaded — return None
            return None
