"""Neuron-compatible wrappers for T5, VAE, and DiT in StreamDiffusionV2.

These wrap the Neuron-ported models so they conform to the StreamDiffusionV2
model interfaces (DiffusionModelInterface, VAEInterface, TextEncoderInterface).
"""
import os
import sys
import json
import math
import time
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange

import logging

from models.model_interface import DiffusionModelInterface, VAEInterface, TextEncoderInterface
from models.wan.flow_match import FlowMatchScheduler
from models.wan.neuron_causal_model import NeuronCausalWanModel

LOGGER = logging.getLogger(__name__)


# ── Text Encoder ────────────────────────────────────────────────────────

class NeuronWanTextEncoder(TextEncoderInterface):
    """T5 text encoder running on Neuron device with torch.compile.
    
    Loaded only on T5_RANK (rank 2). Uses torch.compile(backend='neuron')
    for fast inference (~0.15s vs 30s on CPU).
    Embeddings are broadcast (bf16) to other ranks by the pipeline.
    """

    def __init__(self, model_path="wan_models/Wan2.1-T2V-1.3B", device="neuron"):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device(device)

        # Add the Wan modules path
        wan_base = os.path.join(os.path.dirname(__file__), "wan_base")
        if wan_base not in sys.path:
            sys.path.insert(0, wan_base)

        from modules.tokenizers import HuggingfaceTokenizer
        from modules.t5 import umt5_xxl

        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "google/umt5-xxl/")
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace')

        # Load T5 encoder on CPU first, then move to Neuron
        self.text_encoder = umt5_xxl(
            encoder_only=True, return_tokenizer=False,
            dtype=torch.bfloat16, device=torch.device('cpu')
        ).eval().requires_grad_(False)

        weights_path = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
        self.text_encoder.load_state_dict(
            torch.load(weights_path, map_location='cpu', weights_only=False))

        # Move to device. On Neuron: compile with the neuron backend. On CPU (distill
        # memory saver — T5 off-device to free HBM): keep eager, NO neuron compile.
        self.text_encoder = self.text_encoder.to(device=self.device)
        if self.device.type == "cpu":
            LOGGER.info("T5 on CPU (eager, no neuron compile) — distill HBM saver")
        else:
            self.text_encoder = torch.compile(
                self.text_encoder, backend='neuron', dynamic=False)
            LOGGER.info("T5 compiled with torch.compile(backend='neuron')")

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_len = mask.gt(0).sum(dim=1).long()

        with torch.no_grad():
            context = self.text_encoder(ids, mask)

        # Zero-out padding positions
        context = context.to(torch.bfloat16).contiguous()
        for b in range(context.shape[0]):
            context[b, seq_len[b]:] = 0.0
        return {"prompt_embeds": context}


# ── VAE ─────────────────────────────────────────────────────────────────

class NeuronWanVAEWrapper(VAEInterface):
    """Wan VAE decoder running on Neuron device."""

    def __init__(self, vae_pth="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                 z_dim=16, device="neuron"):
        super().__init__()
        self.z_dim = z_dim
        self.vae_pth = vae_pth
        self.target_device = device
        self._model = None
        # USE_RF_VAE (default off): when on, _ensure_model() builds the ported
        # rolling_forcing per-frame cached decoder (vae_rf.WanVAEWrapper) which
        # decodes one latent frame at a time and avoids the giant aten::cat that
        # crashes the monolithic decode at 480p. When off, the original
        # modules.vae._video_vae path is used unchanged.
        self._use_rf_vae = os.environ.get("USE_RF_VAE", "").lower() in ("1", "true")
        self._decode_chunk_idx = 0  # RF VAE streaming: incremented per decode block

        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.register_buffer('_mean', torch.tensor(mean, dtype=torch.bfloat16))
        self.register_buffer('_std', torch.tensor(std, dtype=torch.bfloat16))

    def _ensure_model(self):
        if self._model is not None:
            return
        wan_base = os.path.join(os.path.dirname(__file__), "wan_base")
        if wan_base not in sys.path:
            sys.path.insert(0, wan_base)
        if self._use_rf_vae:
            # Ported single-rank rolling_forcing per-frame cached decoder.
            from modules.vae_rf import build_vae
            LOGGER.info("Building RF VAE decoder (USE_RF_VAE=1, per-frame cached decode)")
            self._model = build_vae(
                vae_pth=self.vae_pth, dtype=torch.bfloat16,
                device=self.target_device)
        else:
            from modules.vae import _video_vae
            self._model = _video_vae(
                pretrained_path=self.vae_pth, z_dim=self.z_dim
            ).eval().requires_grad_(False).to(dtype=torch.bfloat16, device=self.target_device)

    def reset_decode_stream(self):
        """Start a new clip: clear VAE temporal cache and reset chunk counter.
        Call ONCE per video, before the per-block decode loop. (RF streams the
        cache across chunks via chunk_idx; clearing per-block destroys temporal
        continuity and softens output — that was the quality regression.)"""
        self._decode_chunk_idx = 0
        if self._model is not None and hasattr(self._model, "clear_cache"):
            self._model.clear_cache()

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        self._ensure_model()
        if self._use_rf_vae:
            # Stream the cache across blocks: clear ONLY at chunk 0 (via
            # reset_decode_stream), then pass an incrementing chunk_idx so the
            # CausalConv3d cache + Resample.first_video_frame carry temporal
            # context block-to-block (chunk_idx>0 reuses cache). Matches RF's
            # decode_latents loop.
            idx = getattr(self, "_decode_chunk_idx", 0)
            with torch.no_grad():
                # DECODE EXACTLY LIKE RF: batch_frames=True (RF's default). chunk_idx=0 ->
                # per-frame; chunk_idx>0 -> whole block in ONE decoder call. This yields the
                # SAME, STABLE shapes RF uses (and warms/caches). We previously forced
                # batch_frames=False (always per-frame) to dodge the "cold whole-block shape
                # crashes the service" errno — but that's OBSOLETE now: the VAE warmup +
                # persistent NEFF cache compile that whole-block shape once while the service
                # is fresh. Per-frame was ALSO producing unstable internal temporal-cache cat
                # shapes (1x6/1x9) the warmup never covered -> the crash we kept chasing.
                out = self._model.decode_to_pixel(
                    latent, use_cache=True, chunk_idx=idx, batch_frames=True)
            self._decode_chunk_idx = idx + 1
            return out
        device = latent.device
        scale = [self._mean.to(device), (1.0 / self._std).to(device)]
        latent = rearrange(latent, 'b t c h w -> b c t h w').contiguous()
        # STREAM the temporal cache across blocks (clear only on chunk 0), else each block
        # decode resets the causal Conv3d cache -> wrong frames + blur at boundaries
        # (proven: whole-clip 57 frames vs per-block 45, max_diff 2.15). decode_stream keeps
        # the cache; reset_decode_stream() (once per clip) does the clear.
        idx = getattr(self, "_decode_chunk_idx", 0)
        with torch.no_grad():
            video = self._model.decode_stream(latent, scale, first_chunk=(idx == 0))
        self._decode_chunk_idx = idx + 1
        video = rearrange(video, 'b c t h w -> b t c h w')
        return video

    def encode_to_latent(self, video: torch.Tensor) -> torch.Tensor:
        """Encode pixel video -> latent (v2v input path). Mirrors decode_to_pixel.

        Args:  video [B, T, C, H, W] in [-1, 1] (pixel space, bf16)
        Returns: latent [B, T_lat, z_dim, H/8, W/8]  (matches the noise/denoise layout)
        Encode uses scale=[mean, 1/std] — SAME as the GPU stream_encode. The model's
        encode does mu=(mu-mean)*scale[1], so scale[1] MUST be 1/std (normalize), not
        std. (Earlier bug: passing std mis-scaled latents by std^2 -> noisy v2v output.)
        """
        # The RF VAE (_use_rf_vae) is DECODER-ONLY (encoder weights stripped), so it
        # has no .encode. Lazily build the ORIGINAL full VAE just for encoding; decode
        # still uses the fast RF decoder. Non-RF path uses self._model directly.
        device = video.device
        scale = [self._mean.to(device), (1.0 / self._std).to(device)]
        video = rearrange(video, 'b t c h w -> b c t h w').contiguous()
        if self._use_rf_vae:
            if getattr(self, "_enc_model", None) is None:
                wan_base = os.path.join(os.path.dirname(__file__), "wan_base")
                if wan_base not in sys.path:
                    sys.path.insert(0, wan_base)
                from modules.vae import _video_vae
                self._enc_model = _video_vae(
                    pretrained_path=self.vae_pth, z_dim=self.z_dim
                ).eval().requires_grad_(False).to(dtype=torch.bfloat16, device=self.target_device)
            enc = self._enc_model
        else:
            self._ensure_model()
            enc = self._model
        with torch.no_grad():
            latent = enc.encode(video, scale)
        latent = rearrange(latent, 'b c t h w -> b t c h w')
        return latent


# ── DiT Wrapper ─────────────────────────────────────────────────────────

class NeuronCausalWanDiffusionWrapper(DiffusionModelInterface):
    """Neuron-compatible wrapper for CausalWanModel DiT with TP-4 support."""

    def __init__(self, model_name="Wan2.1-T2V-1.3B", model_path=None,
                 checkpoint_path=None, use_ema=False,
                 denoising_step_list=None, timestep_shift=8.0,
                 num_frame_per_block=1, device="neuron",
                 tp_degree=4):
        super().__init__()
        self.device_name = device
        self.tp_degree = tp_degree

        # Load model config
        if model_path is None:
            model_path = f"wan_models/{model_name}"
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Determine architecture params
        dim = config.get("dim", 1536)
        ffn_dim = config.get("ffn_dim", 8960)
        num_heads = config.get("num_heads", 12)
        num_layers = config.get("num_layers", 30)
        text_dim = config.get("text_dim", 4096)
        freq_dim = config.get("freq_dim", 256)

        self.num_heads = num_heads
        self.num_heads_per_rank = num_heads // tp_degree

        self.model = NeuronCausalWanModel(
            model_type='t2v', patch_size=(1, 2, 2), text_len=512,
            in_dim=16, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim,
            text_dim=text_dim, out_dim=16, num_heads=num_heads,
            num_layers=num_layers, qk_norm=True, cross_attn_norm=True,
        )
        self.model.num_frame_per_block = num_frame_per_block

        # Load base weights (FULL weights - before sharding)
        base_weights = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(base_weights):
            from safetensors.torch import load_file
            state_dict = load_file(base_weights)
            self.model.load_state_dict(state_dict, strict=False)

        # DISTILLATION: the student legitimately starts from BASE weights (we are
        # CREATING the DMD checkpoint, not consuming one). DISTILL_BASE_ONLY=1 skips
        # the DMD-required guard and runs on base weights. Inference path is untouched.
        if os.environ.get("DISTILL_BASE_ONLY", "").lower() in ("1", "true") and not checkpoint_path:
            LOGGER.info("DISTILL_BASE_ONLY: init from base Wan weights, no DMD ckpt (training).")
            checkpoint_path = None
            self._skip_dmd = True
        else:
            self._skip_dmd = False

        # Load fine-tuned (DMD-distilled) checkpoint. REQUIRED for INFERENCE: base Wan2.1
        # weights are not causal/distilled, so without this the model outputs noise.
        if not self._skip_dmd and not checkpoint_path:
            raise ValueError(
                "generator_ckpt is not set — the DMD checkpoint is required (base Wan "
                "weights alone produce noise). Set generator_ckpt in the config, or set "
                "DISTILL_BASE_ONLY=1 for distillation training.")
        if not self._skip_dmd and not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"generator_ckpt not found: {checkpoint_path}. Download the official "
                "StreamDiffusionV2 checkpoint (jerryfeng/StreamDiffusionV2 -> "
                "ckpts/wan_causal_dmd_v2v/model.pt) — do NOT substitute RollingForcing.")

        if self._skip_dmd:
            # base weights already loaded above; no DMD ckpt to merge (training init)
            ckpt = None
        else:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
        if ckpt is None:
            sd = None
        elif use_ema and 'generator_ema' in ckpt:
            sd = ckpt['generator_ema']
        elif 'generator' in ckpt:
            sd = ckpt['generator']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        if sd is not None:
            cleaned = OrderedDict()
            for k, v in sd.items():
                k = k.replace("_fsdp_wrapped_module.", "")
                k = k.replace("model.", "", 1) if k.startswith("model.") else k
                cleaned[k] = v
            # strict=False tolerates extra keys (e.g. logvar/scheduler buffers), but we MUST
            # verify the DiT weights actually landed — a silent mismatch loads base weights
            # and yields noise that looks like a "successful" run.
            result = self.model.load_state_dict(cleaned, strict=False)
            model_keys = set(self.model.state_dict().keys())
            matched = len(model_keys) - len(set(result.missing_keys))
            LOGGER.info(
                "Loaded DMD checkpoint %s: matched %d/%d model params (missing=%d, unexpected=%d)",
                checkpoint_path, matched, len(model_keys),
                len(result.missing_keys), len(result.unexpected_keys))
            if result.missing_keys:
                LOGGER.warning("  first missing keys: %s", result.missing_keys[:8])
            # Guard: if almost nothing matched, the checkpoint is wrong/incompatible.
            if matched < 0.5 * len(model_keys):
                raise RuntimeError(
                    f"DMD checkpoint matched only {matched}/{len(model_keys)} params — "
                    f"wrong checkpoint format or key naming ({checkpoint_path}).")

        # Apply TP-4 sharding BEFORE moving to device (shard on CPU, then move)
        if tp_degree > 1:
            from models.wan.tp_utils import (
                init_tp_group, init_sp_groups, get_tp_rank, shard_model_tp
            )
            # SP mode (rolling-window merged attention): world = tp*sp, set up BOTH TP
            # (contiguous, head shard) and SP (strided, sequence shard) groups. Else the
            # normal per-stream TP groups. sp_degree from env (default 1 = no SP).
            _sp = int(os.environ.get("SP_DEGREE", "1"))
            if _sp > 1:
                init_sp_groups(tp_degree=tp_degree, sp_degree=_sp)
            else:
                init_tp_group(tp_degree)
            tp_rank = get_tp_rank()
            LOGGER.info(f"Applying TP-{tp_degree} sharding (rank {tp_rank}), sp_degree={_sp}")
            shard_model_tp(self.model, tp_rank, tp_degree)

        # Move to Neuron in bfloat16 (Neuron requires matching dtypes for matmul)
        self.model = self.model.to(dtype=torch.bfloat16, device=device).eval()

        # Scheduler (same pattern as WanDiffusionWrapper)
        if denoising_step_list is None:
            denoising_step_list = [700, 500, 400, 200, 0]
        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.post_init()

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        """Convert flow prediction to x0: x0 = xt - sigma_t * flow_pred."""
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.float().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps])
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(self, noisy_image_or_video, conditional_dict, timestep,
                kv_cache=None, crossattn_cache=None,
                current_start=None, current_end=None,
                updating_cache=False, cache_start=None,
                num_valid_frames=None, shared_buffers=None,
                mode="denoise", cache_update_start=None,
                cu_shared_buffers=None, nfpb_cu=None):
        context = conditional_dict["prompt_embeds"]
        x = noisy_image_or_video
        t = timestep

        # DiT expects [B, C, F, H, W], input is [B, F, C, H, W]
        # t must keep [B, num_frames] shape for model's unflatten(dim=0, sizes=t.shape)
        model_out = self.model(
            x.permute(0, 2, 1, 3, 4), t, context,
            updating_cache=updating_cache,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start if current_start is not None else 0,
            cache_start=cache_start,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
            mode=mode, cache_update_start=cache_update_start,
            cu_shared_buffers=cu_shared_buffers, nfpb_cu=nfpb_cu,
        ).permute(0, 2, 1, 3, 4)  # back to [B, F, C, H, W]

        # Convert flow prediction to x0
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=model_out.flatten(0, 1),
            xt=x.flatten(0, 1),
            timestep=t.flatten(0, 1) if t.dim() > 1 else t,
        ).unflatten(0, model_out.shape[:2])

        return pred_x0

    def enable_gradient_checkpointing(self):
        pass  # Not needed for inference
