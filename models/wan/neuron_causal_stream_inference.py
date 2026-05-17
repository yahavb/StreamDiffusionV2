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

        # Per-rank model placement to avoid OOM:
        # T5 (~9.6 GB) only on T5_RANK, VAE (~0.66 GB) only on VAE_RANK
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Initialize text encoder (only on T5_RANK)
        if self.rank == T5_RANK:
            LOGGER.info(f"Loading T5 text encoder on rank {T5_RANK}...")
            self.text_encoder = NeuronWanTextEncoder(
                model_path=model_path, device=device)
        else:
            self.text_encoder = None

        # Initialize VAE (only on VAE_RANK)
        if self.rank == VAE_RANK:
            LOGGER.info(f"Loading VAE decoder on rank {VAE_RANK}...")
            self.vae = NeuronWanVAEWrapper(
                vae_pth=vae_path, device=device)
        else:
            self.vae = None

        # Scheduler
        self.scheduler = self.generator.scheduler

        # Denoising step list
        self.denoising_step_list = torch.tensor(
            denoising_step_list, dtype=torch.long, device=self.device)
        assert self.denoising_step_list[-1] == 0

        t2v = getattr(args, "t2v", True)
        if not t2v:
            self.denoising_step_list = self.denoising_step_list[:-1]

        if warp:
            timesteps = torch.cat((
                self.scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32)
            )).to(self.device)
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

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
        
        T5 runs only on T5_RANK; embeddings are broadcast to all ranks.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        batch_size = noise.shape[0]

        # Text encoding — only T5_RANK runs T5, then broadcast to all
        if self.rank == T5_RANK:
            enc_result = self.text_encoder(text_prompts)
            prompt_embeds = enc_result['prompt_embeds'].to(device=device, dtype=dtype)
            # Also get context_lens if present
            context_lens = enc_result.get('context_lens',
                torch.tensor([prompt_embeds.shape[1]], dtype=torch.long, device=device))
        else:
            # Allocate placeholder — will be filled by broadcast
            # T5 output: [batch, seq_len=512, dim=2048] for umt5-xxl
            prompt_embeds = torch.zeros(
                [batch_size, 512, 2048], dtype=dtype, device=device)
            context_lens = torch.tensor([512], dtype=torch.long, device=device)

        # Broadcast T5 embeddings from T5_RANK to all ranks
        if dist.is_initialized():
            dist.broadcast(prompt_embeds, src=T5_RANK)
            dist.broadcast(context_lens, src=T5_RANK)

        self.conditional_dict = {
            'prompt_embeds': prompt_embeds,
            'context_lens': context_lens,
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
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=noise.device,
                dtype=torch.int64) * current_timestep

            denoised_pred = self.generator(
                noisy_image_or_video=noise,
                conditional_dict=self.conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_int,
                current_end=current_end_int,
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

    def decode_latents(self, latents):
        """Decode latents to pixel space using VAE (only on VAE_RANK).
        
        Returns decoded video on VAE_RANK, None on other ranks.
        """
        if self.rank == VAE_RANK:
            return self.vae.decode_to_pixel(latents)
        else:
            # Other ranks don't have VAE loaded — return None
            return None
