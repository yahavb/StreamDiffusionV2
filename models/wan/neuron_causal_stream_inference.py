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

        # In-pod data-parallel: T5_RANK/VAE_RANK are offsets WITHIN a DP group, so
        # each independent stream gets its own T5+VAE. Resolve to absolute ranks via
        # the group base (base=0 for the single-group case -> original behavior).
        from models.wan.tp_utils import get_tp_group_base, get_dp_group_id, get_dp_num_groups
        self.tp_group_base = get_tp_group_base() if dist.is_initialized() else 0
        self.dp_group_id = get_dp_group_id() if dist.is_initialized() else 0
        self.dp_num_groups = get_dp_num_groups() if dist.is_initialized() else 1
        self.t5_rank = self.tp_group_base + T5_RANK   # absolute rank hosting T5 for THIS group
        self.vae_rank = self.tp_group_base + VAE_RANK  # absolute rank hosting VAE for THIS group

        # All ranks need the tokenizer (lightweight, CPU-only)
        wan_base = os.path.join(os.path.dirname(__file__), "wan_base")
        if wan_base not in sys.path:
            sys.path.insert(0, wan_base)
        from modules.tokenizers import HuggingfaceTokenizer
        tokenizer_path = os.path.join(model_path, "google/umt5-xxl/")
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace')

        # DISTILL memory savers: training is 100% in latent space, so the VAE decoder
        # is NEVER called (it only turns latents->pixels for viewing). And T5 runs once
        # per prompt — fine on CPU. Both are dead weight in HBM during training.
        _distill_t5_cpu = os.environ.get("DISTILL_T5_CPU", "").lower() in ("1", "true")
        _distill_no_vae = os.environ.get("DISTILL_NO_VAE", "").lower() in ("1", "true")

        # Initialize T5 (only on this group's T5 rank — CPU during distill to save HBM)
        if self.rank == self.t5_rank:
            t5_dev = "cpu" if _distill_t5_cpu else device
            LOGGER.info(f"Loading T5 text encoder on {t5_dev} (rank {self.rank}, dp_group {self.dp_group_id})...")
            self.text_encoder = NeuronWanTextEncoder(
                model_path=model_path, device=t5_dev)
        else:
            self.text_encoder = None

        # VAE channel tensor-parallel: VAE_TP_DEGREE ranks starting at vae_rank
        # decode TOGETHER (each holds a channel shard of the decoder convs and
        # all-reduces inside them). Default 1 = single-rank (original behavior).
        # The VAE is ~7-8% of the block; VAE-TP=2 roughly halves the conv work.
        self.vae_tp_degree = int(os.environ.get("VAE_TP_DEGREE", "1"))
        self.vae_tp_ranks = [self.tp_group_base + VAE_RANK + i
                             for i in range(self.vae_tp_degree)]
        is_vae_rank = self.rank in self.vae_tp_ranks

        # Initialize VAE (on every VAE-TP rank — with torch.compile).
        # DISTILL_NO_VAE: skip entirely — never called in latent-space training.
        if _distill_no_vae:
            self.vae = None
            LOGGER.info("DISTILL_NO_VAE: skipping VAE build (latent-space training, decoder unused)")
        elif is_vae_rank:
            LOGGER.info(f"Loading VAE decoder on rank {self.rank} (dp_group {self.dp_group_id}, "
                        f"vae_tp={self.vae_tp_degree} ranks={self.vae_tp_ranks})...")
            self.vae = NeuronWanVAEWrapper(
                vae_pth=vae_path, device=device)
            self.vae._ensure_model()
            # Channel-shard the decoder across the VAE-TP ranks (RF VAE variant).
            if self.vae_tp_degree > 1:
                from modules.vae_tp_rf import create_vae_tp_group, shard_vae_model_tp
                create_vae_tp_group(self.vae_tp_ranks)
                vtp_rank = self.vae_tp_ranks.index(self.rank)
                # self.vae._model is the RF WanVAEWrapper; the WanVAE_ (with .decoder)
                # is at ._model.model. shard_vae_model_tp expects the WanVAE_.
                shard_vae_model_tp(self.vae._model.model, vtp_rank, self.vae_tp_degree)
                LOGGER.info(f"VAE channel-TP sharded: vae_tp_rank={vtp_rank}/{self.vae_tp_degree}")
            # Compile VAE for fast decode — gated by USE_VAE_COMPILE (default on).
            # NOTE: VAE-TP shards change channel dims off P=128; keep EAGER when sharded.
            if self.vae_tp_degree == 1 and os.environ.get("USE_VAE_COMPILE", "true").lower() in ("1", "true"):
                self.vae._model = torch.compile(
                    self.vae._model, backend='neuron', dynamic=False)
                LOGGER.info("VAE compiled with torch.compile(backend='neuron')")
            else:
                LOGGER.info("VAE running EAGER (sharded or USE_VAE_COMPILE=0)")
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

        # ── ROLLING-FORCING rolling-window schedule (ported from RF dit_pipeline.py) ──
        # RF's sharpness comes from co-denoising a sliding window of `nds` blocks, each at
        # a DIFFERENT noise level (a diagonal schedule), in one forward. These patterns are
        # the per-frame timesteps for each window phase. Built once here; the loop uses them.
        self.timestep_patterns = self._build_timestep_patterns()   # [num_patterns, max_frames]
        self.sigma_patterns = self._build_sigma_patterns()

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

    def _timestep_to_sigma(self, timestep_val):
        """Map a timestep to its flow-matching sigma (RF dit_pipeline._timestep_to_sigma)."""
        idx = torch.argmin((self.scheduler.timesteps - timestep_val).abs())
        return self.scheduler.sigmas[idx].item()

    def _build_timestep_patterns(self):
        """Per-frame timestep vectors for each rolling-window phase (RF-faithful).
        steady = the diagonal: each nfpb-frame block at a descending noise level, e.g.
        nds=5,nfpb=3 -> [0,0,0, 200,200,200, 400,400,400, 500,500,500, 700,700,700]
        (values are the WARPED timesteps in denoising_step_list). Edge phases (window
        entering/leaving) get prefix/suffix-truncated patterns padded with 0."""
        nds = len(self.denoising_step_list)
        nfpb = self.num_frame_per_block
        max_frames = nds * nfpb
        steady = []
        for ts in reversed(self.denoising_step_list.tolist()):
            steady.extend([float(ts)] * nfpb)
        patterns = [steady]
        for i in range(1, nds):
            cnf = i * nfpb
            patterns.append(steady[-cnf:] + [0.0] * (max_frames - cnf))
        for i in range(1, nds):
            cnf = i * nfpb
            patterns.append(steady[:cnf] + [0.0] * (max_frames - cnf))
        return torch.tensor(patterns, dtype=torch.float32, device=self.device)

    def _build_sigma_patterns(self):
        sp = torch.zeros_like(self.timestep_patterns)
        for i, pattern in enumerate(self.timestep_patterns):
            for j, t in enumerate(pattern.tolist()):
                sp[i, j] = self._timestep_to_sigma(t)
        return sp

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

        # DISTILL bypass: if precomputed prompt-embeds are injected (self._distill_embeds
        # maps prompt-text -> embeds tensor), use them DIRECTLY — NO T5 build, NO T5
        # broadcast collective. This removes the T5-broadcast group (get_tp_group) that
        # collides with the FSDP student's rebuilt process groups (tp_degree=1 + FSDP
        # mesh scramble the global TP group -> all_reduce failure at line 397).
        _pe = getattr(self, "_distill_embeds", None)
        if _pe is not None:
            emb = _pe[text_prompts[0]] if isinstance(_pe, dict) else _pe
            self.conditional_dict = {"prompt_embeds": emb.to(device=device, dtype=dtype)}
        else:
            # Step 1: Tokenize on all ranks (CPU, fast)
            ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device)

            # Step 2: this group's T5 rank encodes with compiled T5 on Neuron
            if self.rank == self.t5_rank:
                seq_len = mask.gt(0).sum(dim=1).long()
                enc_result = self.text_encoder(text_prompts)
                prompt_embeds = enc_result['prompt_embeds'].to(device=device, dtype=dtype)
                prompt_embeds = prompt_embeds.contiguous()
            else:
                prompt_embeds = torch.zeros(
                    batch_size, 512, 4096, dtype=dtype, device=device)

            # Step 3: Broadcast embeddings WITHIN this DP group only.
            if dist.is_initialized():
                from models.wan.tp_utils import get_tp_group
                dist.broadcast(prompt_embeds, src=self.t5_rank, group=get_tp_group())

            self.conditional_dict = {'prompt_embeds': prompt_embeds}

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

        # DISTILL memory: DMD only needs grad through the LAST denoise step. Wrap the
        # earlier steps in no_grad so their activation graph isn't retained for
        # backward (the multi-step x 30-layer rollout graph was the 24GB OOM). Set
        # via env DISTILL_GRAD_LAST_ONLY=1 (inference path keeps full behavior).
        _grad_last = os.environ.get("DISTILL_GRAD_LAST_ONLY", "").lower() in ("1", "true")
        _nsteps = len(self.denoising_step_list)
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=device,
                dtype=torch.int64) * current_timestep

            _is_last = (index == _nsteps - 1)
            import contextlib
            _ctx = torch.no_grad() if (_grad_last and not _is_last) else contextlib.nullcontext()
            with _ctx:
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

        # Build the EFFECTIVE descending schedule for this block (SDEdit strength).
        # The canonical DMD schedule is [700,500,400,200,0] (= noise_scale~0.8, the
        # only point where current_step=int(1000*ns)-100 reproduces it). v2v chunks
        # pass current_step<700; overwriting ONLY index[0] then leaves the tail fixed
        # -> NON-MONOTONIC (e.g. ns=0.5 -> [400,500,400,200,0], which RE-NOISES back
        # up to 500 mid-denoise and corrupts the output). Instead truncate to a proper
        # descending schedule: start at current_step, then all canonical steps below
        # it. Reduces EXACTLY to the canonical list when current_step>=700 (high ns),
        # stays monotonic + uses only distilled timesteps at low ns, and runs fewer
        # steps the lower you go (more input-faithful AND faster) — so a drastic low
        # noise_scale becomes a VALID quality test instead of schedule garbage.
        canonical = [int(s) for s in self.denoising_step_list.tolist()]
        if current_step is None:
            eff_steps = canonical
        else:
            cs = int(current_step)
            eff_steps = [cs] + [s for s in canonical if s < cs]
        num_steps = len(eff_steps)
        for index, current_timestep in enumerate(eff_steps):
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=noise.device,
                dtype=torch.int64) * current_timestep

            # MATCH rolling-forcing pipeline: the expensive KV-cache assembly
            # (pad/concat to max_attention_size=8190 — the 35ms+26ms copy NEFFs)
            # is gated by updating_cache. RF runs the denoise steps WITHOUT it
            # (cheap windowed read) and updates the cache ONCE at the end. SD was
            # passing updating_cache=True on ALL 5 steps -> the big assembly ran
            # 5x/block, the dominant per-block cost. Only update on the last step.
            # A/B TEST (agent finding #2): the is_last_step gating (speed opt) may leave
            # steps 1..N-1 attending a stale cache -> blur. Env UPDATE_CACHE_EVERY_STEP=1
            # updates every step (RF-faithful?) to test. Default keeps the last-only opt.
            is_last_step = (index == num_steps - 1)
            _upd = True if os.environ.get("UPDATE_CACHE_EVERY_STEP", "").lower() in ("1", "true") else is_last_step
            denoised_pred = self.generator(
                noisy_image_or_video=noise,
                conditional_dict=self.conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_int,
                current_end=current_end_int,
                updating_cache=_upd,
                shared_buffers=self.shared_buffers,
            )

            if index < num_steps - 1:
                next_timestep = eff_steps[index + 1]
                noise = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones(
                        [batch_size], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])

        return denoised_pred

    def decode_latents(self, latents):
        """Decode latents to pixel space using VAE.

        Single-rank (vae_tp_degree==1): only vae_rank decodes, returns video there.
        VAE channel-TP (>1): ALL vae_tp_ranks decode together (sharded convs
        all-reduce inside). They need identical latents, so vae_rank broadcasts
        the latents to the VAE-TP group first. Only vae_rank returns the video
        (all ranks compute the full output via all-reduce, but we keep one copy).
        """
        if self.vae_tp_degree <= 1:
            return self.vae.decode_to_pixel(latents) if self.rank == self.vae_rank else None

        # VAE-TP: broadcast latents from vae_rank to the whole VAE-TP group.
        if self.rank not in self.vae_tp_ranks:
            return None
        from modules.vae_tp_rf import get_vae_tp_group
        latents = latents.contiguous()
        dist.broadcast(latents, src=self.vae_rank, group=get_vae_tp_group())
        video = self.vae.decode_to_pixel(latents)
        return video if self.rank == self.vae_rank else None

    def reset_decode_stream(self):
        """Reset the VAE temporal-cache stream — call ONCE per clip before the
        first (anchor) decode so the RF VAE streams its cache across blocks.
        Every VAE-TP rank holds its own cache -> reset on all of them."""
        if self.vae is not None and hasattr(self.vae, "reset_decode_stream"):
            self.vae.reset_decode_stream()

    def encode_video_latents(self, video, num_lat_frames, height, width):
        """v2v: encode pixel video -> latent on VAE_RANK, broadcast to all TP ranks.

        VAE lives only on VAE_RANK, but the DiT (all ranks) needs the latents.
        Mirror the T5 prompt-embed broadcast pattern. `video` is [B,T,C,H,W] in
        [-1,1] on VAE_RANK (None elsewhere). Returns latent [B,num_lat_frames,16,H/8,W/8]
        on every rank.
        """
        lat_shape = (1, num_lat_frames, 16, height // 8, width // 8)
        if self.rank == self.vae_rank and video is not None:
            latent = self.vae.encode_to_latent(video).to(dtype=self.dtype, device=self.device)
            latent = latent.contiguous()
        else:
            latent = torch.zeros(lat_shape, dtype=self.dtype, device=self.device)
        if dist.is_initialized():
            # Broadcast WITHIN this DP group (src = group's VAE rank).
            from models.wan.tp_utils import get_tp_group
            dist.broadcast(latent, src=self.vae_rank, group=get_tp_group())
        return latent
