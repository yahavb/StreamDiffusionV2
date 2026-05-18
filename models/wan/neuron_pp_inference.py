"""Pipeline-Parallel (PP-4) inference for StreamDiffusionV2 on Trainium.

Instead of TP-4 (all ranks share each layer via all-reduce), PP-4 gives each
rank a FULL copy of the model and assigns each rank a different denoising step.
Frames flow through the pipeline in a ring: rank0→rank1→rank2→rank3→output.

At steady state, every micro-step produces 1 clean frame block (3 frames).
Throughput is ~4× sequential because all ranks work concurrently on different
frames at different noise levels.

Architecture (Wan2.1-1.3B, 5 denoising steps, 4 ranks):
  Rank 0: step 0 (t=1000→800) + T5 encode
  Rank 1: step 1 (t=800→600)
  Rank 2: step 2 (t=600→400)  
  Rank 3: steps 3+4 (t=400→200→0) + VAE decode

Communication: P2P send/recv of latent tensor between adjacent ranks.
No all-reduce needed — eliminates TP communication overhead entirely.
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

# PP rank assignments
T5_RANK = 0     # T5 on first rank (receives input first)
VAE_RANK = 3    # VAE on last rank (produces final output)


class _ContiguousWrapper(nn.Module):
    """Wraps a compiled module to ensure all tensor inputs are contiguous."""
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


class NeuronPPInferencePipeline(nn.Module):
    """Pipeline-Parallel streaming inference for StreamDiffusionV2.
    
    Each rank holds the FULL model (no TP sharding) and is responsible for
    specific denoising step(s). Latents flow rank0→rank1→rank2→rank3 via P2P.
    """

    def __init__(self, args, device="neuron"):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        # PP config
        self.pp_degree = getattr(args, "pp_degree", 4)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Model config
        model_path = getattr(args, "model_path", "wan_models/Wan2.1-T2V-1.3B")
        checkpoint_path = getattr(args, "generator_ckpt", None)
        vae_path = getattr(args, "vae_path", f"{model_path}/Wan2.1_VAE.pth")
        use_ema = getattr(args, "use_ema", False)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)

        # Denoising schedule
        denoising_step_list = list(args.denoising_step_list)
        timestep_shift = getattr(args, "timestep_shift", 5.0)
        warp = getattr(args, "warp_denoising_step", True)

        # Model type
        model_type = getattr(args, "model_type", "T2V-1.3B")
        if model_type == "T2V-1.3B":
            self.num_transformer_blocks = 30
            self.num_heads = 12
            self.dim = 1536
        else:
            raise ValueError(f"PP mode only supports T2V-1.3B currently")

        self.head_dim = self.dim // self.num_heads

        # Spatial dims — VAE compresses 8x, giving latent H/W at scale_size=8
        scale_size = 8
        self.height = args.height // scale_size   # 60
        self.width = args.width // scale_size     # 104
        # After patch embed (patch_size=2×2), frame_seq_length = (H/2)*(W/2)
        self.frame_seq_length = (self.height // 2) * (self.width // 2)
        self.num_kv_cache = getattr(args, "num_kv_cache", 6)
        self.kv_cache_length = self.frame_seq_length * self.num_kv_cache

        # Latent shape for P2P communication
        # [batch=1, num_frame_per_block, channels=16, h, w]
        self.latent_shape = (1, self.num_frame_per_block, 16,
                            self.height, self.width)

        # Initialize generator — FULL model per rank (tp_degree=1)
        LOGGER.info(f"[PP-Rank {self.rank}] Loading full DiT model (no TP sharding)...")
        self.generator = NeuronCausalWanDiffusionWrapper(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            use_ema=use_ema,
            denoising_step_list=denoising_step_list,
            timestep_shift=timestep_shift,
            num_frame_per_block=self.num_frame_per_block,
            device=device,
            tp_degree=1,  # NO TP — full model
        )

        # Update frame length
        self.generator.model._update_frame_length(
            self.frame_seq_length, self.num_frame_per_block, self.num_kv_cache)

        # Tokenizer (all ranks need it for prompt processing)
        wan_base = os.path.join(os.path.dirname(__file__), "wan_base")
        if wan_base not in sys.path:
            sys.path.insert(0, wan_base)
        from modules.tokenizers import HuggingfaceTokenizer
        tokenizer_path = os.path.join(model_path, "google/umt5-xxl/")
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace')

        # T5 on T5_RANK only
        if self.rank == T5_RANK:
            LOGGER.info(f"[PP-Rank {self.rank}] Loading T5 text encoder...")
            self.text_encoder = NeuronWanTextEncoder(
                model_path=model_path, device=device)
        else:
            self.text_encoder = None

        # VAE on VAE_RANK only
        if self.rank == VAE_RANK:
            LOGGER.info(f"[PP-Rank {self.rank}] Loading VAE decoder...")
            self.vae = NeuronWanVAEWrapper(vae_pth=vae_path, device=device)
            self.vae._ensure_model()
            self.vae._model = torch.compile(
                self.vae._model, backend='neuron', dynamic=False)
        else:
            self.vae = None

        # Compile DiT sub-modules
        dit_model = self.generator.model
        dit_model.patch_embedding = _contiguous_compile(dit_model.patch_embedding)
        dit_model.text_embedding = _contiguous_compile(dit_model.text_embedding)
        dit_model.time_embedding = _contiguous_compile(dit_model.time_embedding)
        dit_model.time_projection = _contiguous_compile(dit_model.time_projection)
        dit_model.head = _contiguous_compile(dit_model.head)
        for block in dit_model.blocks:
            block.ffn = _contiguous_compile(block.ffn)
        LOGGER.info(f"[PP-Rank {self.rank}] DiT compiled (full 30 blocks)")

        # Scheduler
        self.scheduler = self.generator.scheduler

        # Denoising step list (warped)
        self.denoising_step_list = torch.tensor(
            denoising_step_list, dtype=torch.long, device=self.device)
        if warp:
            timesteps = torch.cat((
                self.scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32)
            )).to(self.device)
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
            LOGGER.info(f"[PP-Rank {self.rank}] Warped steps: {self.denoising_step_list.tolist()}")

        self.num_denoising_steps = len(self.denoising_step_list)

        # Assign denoising steps to ranks (round-robin with last rank getting extras)
        self._assign_steps_to_ranks()

        # State
        self.conditional_dict = None
        self.kv_cache = None
        self.crossattn_cache = None
        self.shared_buffers = None
        self.args = args

        LOGGER.info(f"[PP-Rank {self.rank}] PP pipeline initialized. "
                    f"My steps: {self.my_step_indices} "
                    f"(timesteps: {[self.denoising_step_list[i].item() for i in self.my_step_indices]})")

    def _assign_steps_to_ranks(self):
        """Assign denoising steps to PP ranks.
        
        With 5 steps and 4 ranks:
          Rank 0: step 0
          Rank 1: step 1
          Rank 2: step 2
          Rank 3: steps 3, 4  (last rank gets remainder)
        """
        steps_per_rank = self.num_denoising_steps // self.pp_degree
        remainder = self.num_denoising_steps % self.pp_degree

        self.rank_step_assignments = []
        idx = 0
        for r in range(self.pp_degree):
            n = steps_per_rank + (1 if r >= self.pp_degree - remainder else 0)
            self.rank_step_assignments.append(list(range(idx, idx + n)))
            idx += n

        self.my_step_indices = self.rank_step_assignments[self.rank]

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize KV cache — full heads (no TP splitting)."""
        kv_cache = []
        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, self.kv_cache_length,
                                  self.num_heads, self.head_dim],
                                 dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_length,
                                  self.num_heads, self.head_dim],
                                 dtype=dtype, device=device),
                "global_end_index": 0,
                "local_end_index": 0,
            })
        self.kv_cache = kv_cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize cross-attention cache — full heads."""
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, self.num_heads, self.head_dim],
                                 dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_heads, self.head_dim],
                                 dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache = crossattn_cache

    def _initialize_shared_buffers(self, batch_size, dtype, device):
        """Allocate shared K/V buffers for attention."""
        max_attn_size = 21 * self.frame_seq_length
        padded = ((max_attn_size + ATTN_SEQLEN_MULTIPLE - 1)
                  // ATTN_SEQLEN_MULTIPLE) * ATTN_SEQLEN_MULTIPLE
        self.shared_buffers = (
            torch.zeros([batch_size, padded, self.num_heads, self.head_dim],
                        dtype=dtype, device=device),
            torch.zeros([batch_size, padded, self.num_heads, self.head_dim],
                        dtype=dtype, device=device),
        )

    def encode_prompt(self, text_prompts: List[str]):
        """Encode text prompt. T5 runs on T5_RANK, broadcast to all."""
        batch_size = len(text_prompts)

        if self.rank == T5_RANK:
            enc_result = self.text_encoder(text_prompts)
            prompt_embeds = enc_result['prompt_embeds'].to(
                device=self.device, dtype=self.dtype).contiguous()
        else:
            prompt_embeds = torch.zeros(
                batch_size, 512, 4096, dtype=self.dtype, device=self.device)

        if dist.is_initialized():
            dist.broadcast(prompt_embeds, src=T5_RANK)

        self.conditional_dict = {'prompt_embeds': prompt_embeds}
        return prompt_embeds

    def _run_dit_forward(self, latent, timestep_value, current_start, current_end,
                         updating_cache=False):
        """Run full DiT forward pass (all 30 blocks) for a single timestep."""
        batch_size = latent.shape[0]
        timestep = torch.ones(
            [batch_size, latent.shape[1]], device=self.device,
            dtype=torch.int64) * int(timestep_value)

        return self.generator(
            noisy_image_or_video=latent,
            conditional_dict=self.conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start,
            current_end=current_end,
            updating_cache=updating_cache,
            shared_buffers=self.shared_buffers,
        )

    def _add_noise(self, clean, timestep_value):
        """Add noise to clean prediction at given timestep."""
        noise = torch.randn_like(clean.flatten(0, 1))
        t = timestep_value * torch.ones(
            [clean.shape[0]], device=self.device, dtype=torch.long)
        noisy = self.scheduler.add_noise(clean.flatten(0, 1), noise, t)
        return noisy.unflatten(0, clean.shape[:2])

    def _broadcast_latent(self, latent, src_rank):
        """Broadcast latent from src_rank to all ranks.
        
        Neuron backend supports broadcast but not P2P send/recv.
        All ranks call this; only src_rank provides meaningful data.
        """
        if latent is None:
            latent = torch.zeros(self.latent_shape, dtype=self.dtype, device=self.device)
        latent = latent.contiguous()
        dist.broadcast(latent, src=src_rank)
        return latent

    def prepare_anchor(self, text_prompts, noise, current_start=0, current_end=None):
        """Run anchor block denoising (all steps sequentially on rank 0, broadcast result).
        
        During warmup/anchor, all ranks run sequentially to fill KV caches.
        """
        batch_size = noise.shape[0]
        if current_end is None:
            current_end = self.frame_seq_length

        # Encode prompt
        self.encode_prompt(text_prompts)

        # Initialize caches
        self._initialize_kv_cache(batch_size, self.dtype, self.device)
        self._initialize_crossattn_cache(batch_size, self.dtype, self.device)
        self._initialize_shared_buffers(batch_size, self.dtype, self.device)

        # All ranks run full denoising for anchor (fills KV cache)
        current = noise
        for idx in range(self.num_denoising_steps):
            ts = self.denoising_step_list[idx]
            denoised = self._run_dit_forward(
                current, ts, current_start, current_end, updating_cache=True)

            if idx < self.num_denoising_steps - 1:
                next_ts = self.denoising_step_list[idx + 1]
                current = self._add_noise(denoised, next_ts)
            else:
                current = denoised

        return current  # Clean anchor output

    def inference_pp_step(self, noise, current_start, current_end):
        """Run one PP micro-step using broadcast for inter-rank communication.
        
        Each rank processes its assigned denoising step(s):
        - Rank 0: starts with noise, does step 0, broadcasts result
        - Rank 1: receives broadcast from rank 0, does step 1, broadcasts
        - Rank 2: receives broadcast from rank 1, does step 2, broadcasts
        - Rank 3: receives broadcast from rank 2, does steps 3+4, outputs clean
        
        All ranks participate in every broadcast (required by collective ops).
        Returns: clean latent on VAE_RANK, None on other ranks.
        """
        current_start_int = int(current_start)
        current_end_int = int(current_end)

        # Sequential pipeline: each rank broadcasts its output to all
        current = noise if self.rank == 0 else None

        for stage in range(self.pp_degree):
            if stage > 0:
                # All ranks receive the output from previous stage via broadcast
                current = self._broadcast_latent(current, src_rank=stage - 1)

            if self.rank == stage:
                # This rank runs its denoising step(s)
                for step_idx in self.my_step_indices:
                    ts = self.denoising_step_list[step_idx]
                    denoised = self._run_dit_forward(
                        current, ts, current_start_int, current_end_int)

                    if step_idx < self.num_denoising_steps - 1:
                        next_ts = self.denoising_step_list[step_idx + 1]
                        current = self._add_noise(denoised, next_ts)
                    else:
                        current = denoised

        # Final broadcast from last rank so all ranks have the clean output
        current = self._broadcast_latent(current, src_rank=self.pp_degree - 1)

        if self.rank == VAE_RANK:
            return current
        return None

    def inference_pp_streaming(self, num_blocks, current_start_base):
        """Run PP streaming inference for multiple blocks.
        
        Uses pipeline fill/drain pattern:
        - Fill phase: first pp_degree-1 micro-steps fill the pipeline
        - Steady state: every micro-step produces 1 clean output
        - Drain phase: last pp_degree-1 micro-steps drain remaining
        
        Returns list of (clean_latent, block_index) on VAE_RANK.
        """
        outputs = []
        total_microsteps = num_blocks + self.pp_degree - 1

        # Pipeline buffers: each rank maintains its input queue
        # In steady state, rank gets input from prev rank each micro-step
        for micro in range(total_microsteps):
            block_for_this_rank = micro - self.rank

            if self.rank == 0 and block_for_this_rank >= 0 and block_for_this_rank < num_blocks:
                # Rank 0 creates new noise for this block
                noise = torch.randn(self.latent_shape, dtype=self.dtype, device=self.device)
                current_start = current_start_base + block_for_this_rank * self.frame_seq_length
                current_end = current_start + self.frame_seq_length
            elif self.rank == 0:
                # Pipeline bubble: send zeros
                noise = torch.zeros(self.latent_shape, dtype=self.dtype, device=self.device)
                current_start = current_start_base
                current_end = current_start + self.frame_seq_length
            else:
                noise = None
                current_start = current_start_base + max(0, block_for_this_rank) * self.frame_seq_length
                current_end = current_start + self.frame_seq_length

            result = self.inference_pp_step(
                noise if self.rank == 0 else None,
                current_start, current_end)

            if self.rank == VAE_RANK and result is not None:
                output_block_idx = micro - (self.pp_degree - 1)
                if 0 <= output_block_idx < num_blocks:
                    outputs.append((result, output_block_idx))

        return outputs

    def decode_latents(self, latents):
        """Decode latents to pixel space using VAE (only on VAE_RANK)."""
        if self.rank == VAE_RANK and self.vae is not None:
            return self.vae.decode_to_pixel(latents)
        return None
