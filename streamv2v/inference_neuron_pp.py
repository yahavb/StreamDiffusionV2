"""StreamDiffusionV2 Pipeline-Parallel (PP-4) inference entry point.

Launches 4 processes (one per NeuronCore), each holding a full copy of the
Wan2.1-1.3B model. Denoising steps are split across ranks with P2P latent
hand-off. At steady state, 1 clean frame block exits every micro-step.

Usage:
  torchrun --nproc_per_node=4 streamv2v/inference_neuron_pp.py \
    --config streamv2v/configs/wan_causal_pp4_neuron.yaml --benchmark
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
import yaml
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
LOGGER = logging.getLogger("inference_pp")

NEURON_DEVICE = "xla"  # Will be overridden


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A cat walking on the beach at sunset, cinematic lighting")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="output_pp.mp4")
    parser.add_argument("--device", type=str, default="neuron")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup_runs", type=int, default=2)
    parser.add_argument("--benchmark_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(config_path, args):
    """Load YAML config and override with CLI args."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    class Config:
        pass

    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)

    # CLI overrides
    if args.num_frames:
        config.num_frames = args.num_frames
    if args.height:
        config.height = args.height
    if args.width:
        config.width = args.width
    if args.seed is not None:
        config.seed = args.seed

    return config


def save_frames(frames, output_dir="/var/mdl/stream_diffusion_pp_frames"):
    """Save decoded frames as PNGs to persistent storage."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu()
            if frame.dim() == 3:  # [C, H, W]
                frame = frame.permute(1, 2, 0)  # → [H, W, C]
            frame = (frame.clamp(0, 1) * 255).byte().numpy()
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))
    LOGGER.info(f"Saved {len(frames)} frames to {output_dir}")


def main():
    args = parse_args()

    # Initialize distributed — "neuron" backend handles per-rank NeuronCore pinning
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    LOGGER.info(f"[Rank {rank}/{world_size}] PP inference starting")

    # Load config
    config = load_config(args.config, args)

    # Set seed
    seed = getattr(config, "seed", 0)
    torch.manual_seed(seed + rank)

    # Import and build pipeline
    from models.wan.neuron_pp_inference import NeuronPPInferencePipeline
    pipeline = NeuronPPInferencePipeline(config, device=args.device)

    # Calculate blocks needed
    num_frame_per_block = config.num_frame_per_block
    # First block is anchor (4 frames for 1.3B), rest are streaming blocks
    total_frames = config.num_frames
    anchor_frames = num_frame_per_block + 1  # anchor includes 1 extra
    streaming_frames = total_frames - anchor_frames
    num_streaming_blocks = max(1, streaming_frames // num_frame_per_block)

    LOGGER.info(f"[Rank {rank}] Total frames: {total_frames}, "
                f"streaming blocks: {num_streaming_blocks}")

    # Latent shape — VAE compresses 8x spatial
    scale_size = 8
    latent_h = config.height // scale_size   # 60
    latent_w = config.width // scale_size    # 104
    latent_shape = (1, num_frame_per_block, 16, latent_h, latent_w)

    all_frames = []

    def run_generation():
        """Run one full generation pass."""
        nonlocal all_frames
        all_frames = []

        # Step 1: Anchor denoising (all ranks participate)
        anchor_noise = torch.randn(latent_shape, dtype=torch.bfloat16,
                                   device=pipeline.device)
        t_anchor_start = time.time()
        anchor_result = pipeline.prepare_anchor(
            [args.prompt], anchor_noise,
            current_start=0,
            current_end=pipeline.frame_seq_length)
        t_anchor = time.time() - t_anchor_start

        # Decode anchor on VAE rank
        if rank == 3:  # VAE_RANK
            decoded = pipeline.decode_latents(anchor_result)
            if decoded is not None:
                all_frames.extend(_extract_frames(decoded))

        # Step 2: PP streaming inference
        t_stream_start = time.time()
        current_start_base = pipeline.frame_seq_length  # After anchor

        block_times = []
        for block_idx in range(num_streaming_blocks):
            t_block_start = time.time()

            current_start = current_start_base + block_idx * pipeline.frame_seq_length
            current_end = current_start + pipeline.frame_seq_length

            # Generate noise on rank 0
            if rank == 0:
                noise = torch.randn(latent_shape, dtype=torch.bfloat16,
                                    device=pipeline.device)
            else:
                noise = None

            # PP step: each rank does its denoising step(s)
            result = pipeline.inference_pp_step(noise, current_start, current_end)

            t_block = time.time() - t_block_start
            block_times.append(t_block)

            # Decode on last rank
            if rank == 3 and result is not None:
                decoded = pipeline.decode_latents(result)
                if decoded is not None:
                    all_frames.extend(_extract_frames(decoded))

        t_stream = time.time() - t_stream_start
        return t_anchor, t_stream, block_times

    # Warmup
    if args.benchmark:
        LOGGER.info(f"[Rank {rank}] Warming up ({args.warmup_runs} runs)...")
        with torch.no_grad():
            for w in range(args.warmup_runs):
                t0 = time.time()
                run_generation()
                LOGGER.info(f"[Rank {rank}] Warmup {w+1}: {time.time()-t0:.2f}s")

    # Benchmark
    with torch.no_grad():
        t_total_start = time.time()
        t_anchor, t_stream, block_times = run_generation()
        t_total = time.time() - t_total_start

    if args.benchmark and rank == 0:
        # Additional benchmark runs
        all_anchors = [t_anchor]
        all_streams = [t_stream]
        all_block_times = [block_times]

        with torch.no_grad():
            for _ in range(args.benchmark_runs - 1):
                ta, ts, bt = run_generation()
                all_anchors.append(ta)
                all_streams.append(ts)
                all_block_times.append(bt)

        # Compute stats
        avg_anchor = sum(all_anchors) / len(all_anchors)
        avg_stream = sum(all_streams) / len(all_streams)
        avg_block = sum(sum(bt) for bt in all_block_times) / sum(len(bt) for bt in all_block_times)
        total_time = avg_anchor + avg_stream
        total_frames_gen = num_frame_per_block + num_streaming_blocks * num_frame_per_block
        fps = total_frames_gen / total_time if total_time > 0 else 0
        steady_fps = num_frame_per_block / avg_block if avg_block > 0 else 0

        print()
        print("┌─────────────────────────────────────────────────────────┐")
        print("│  PP-4 BENCHMARK RESULTS                                  │")
        print("├─────────────────────────────────────────────────────────┤")
        print(f"│  PP degree:              {pipeline.pp_degree:>6}                      │")
        print(f"│  Denoising steps:        {pipeline.num_denoising_steps:>6}                      │")
        print(f"│  Num frames:             {total_frames_gen:>6}                      │")
        print(f"│  Benchmark runs:         {args.benchmark_runs:>6}                      │")
        print(f"│  T5+anchor time:       {avg_anchor:>8.3f}s                    │")
        print(f"│  DiT/block time (PP):  {avg_block:>8.3f}s                    │")
        print(f"│  Total stream time:    {avg_stream:>8.3f}s                    │")
        print(f"│  Total time:           {total_time:>8.3f}s                    │")
        print("├─────────────────────────────────────────────────────────┤")
        print(f"│  OVERALL FPS:          {fps:>8.2f} frames/sec             │")
        print(f"│  STEADY-STATE FPS:     {steady_fps:>8.2f} frames/sec             │")
        print(f"│  Real-time ratio:      {fps/16:>8.4f}x (vs 16fps)          │")
        print("└─────────────────────────────────────────────────────────┘")
        print()

    # Save frames on VAE rank
    if rank == 3 and all_frames:
        save_frames(all_frames, "/var/mdl/stream_diffusion_pp_frames")

    dist.destroy_process_group()


def _extract_frames(decoded_tensor):
    """Extract individual frames from decoded VAE output."""
    frames = []
    if decoded_tensor is None:
        return frames
    # decoded is [B, C, T, H, W] or [B, T, C, H, W]
    if decoded_tensor.dim() == 5:
        if decoded_tensor.shape[1] == 3:  # [B, C, T, H, W]
            for t in range(decoded_tensor.shape[2]):
                frame = decoded_tensor[0, :, t]  # [C, H, W]
                frames.append(frame)
        else:  # [B, T, C, H, W]
            for t in range(decoded_tensor.shape[1]):
                frame = decoded_tensor[0, t]  # [C, H, W]
                frames.append(frame)
    elif decoded_tensor.dim() == 4:  # [B, C, H, W]
        frames.append(decoded_tensor[0])
    return frames


if __name__ == "__main__":
    main()
