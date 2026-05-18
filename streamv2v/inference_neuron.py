"""Neuron/Trainium inference entry point for StreamDiffusionV2 with TP-4.

Runs the full streaming pipeline on Trainium: T5 encode → DiT denoise → VAE decode.
DiT runs with TP-4 (tensor parallelism across 4 NeuronCores).
Measures and reports FPS for each stage and end-to-end.

Usage:
    torchrun --nproc_per_node=4 streamv2v/inference_neuron.py \
        --config streamv2v/configs/wan_causal_dmd_v2v_neuron.yaml \
        --prompt "A cat walking on the beach" --num_frames 81 --benchmark
"""
import argparse
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from einops import rearrange

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wan.neuron_causal_stream_inference import NeuronCausalStreamInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("inference_neuron")


def init_distributed():
    """Initialize distributed process group for Trainium TP.

    Uses the 'neuron' backend which handles per-rank core assignment.
    torch.neuron.set_device(local_rank) pins each rank to its logical device.
    After set_device, torch.device("neuron") refers to the current rank's core.
    """
    if dist.is_initialized():
        return
    assert "LOCAL_RANK" in os.environ, (
        "inference_neuron.py must be launched via torchrun (LOCAL_RANK not set)")

    dist.init_process_group(backend="neuron")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)

    LOGGER.info(f"Distributed initialized: rank={dist.get_rank()}/{dist.get_world_size()}, "
                f"local_rank={local_rank}")


def parse_args():
    parser = argparse.ArgumentParser(description="StreamDiffusionV2 Neuron Inference")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--prompt", type=str, default="A cat walking on the beach at sunset")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="output_neuron.mp4")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    parser.add_argument("--warmup_runs", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--benchmark_runs", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--device", type=str, default="neuron")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args()


def save_video(video_tensor, output_path, fps=16):
    """Save video tensor [B, T, C, H, W] in [0,1] as PNG frames.
    
    Uses PIL to write individual frames — avoids ffmpeg subprocess fork
    which crashes inside neuron distributed runtime.
    """
    from PIL import Image

    video = video_tensor.cpu().clamp(0, 1)
    video_out = rearrange(video, 'b t c h w -> b t h w c')
    video_out = (255.0 * video_out).to(torch.uint8).numpy()

    # Save as PNG frames in a directory
    frame_dir = output_path.replace(".mp4", "_frames")
    os.makedirs(frame_dir, exist_ok=True)

    for i in range(video_out.shape[0]):
        batch_dir = frame_dir if video_out.shape[0] == 1 else os.path.join(frame_dir, f"batch_{i}")
        os.makedirs(batch_dir, exist_ok=True)
        for t, frame in enumerate(video_out[i]):
            Image.fromarray(frame).save(os.path.join(batch_dir, f"frame_{t:04d}.png"))

    LOGGER.info(f"Saved {video_out.shape[1]} frames to {frame_dir}")


# VAE decode batch = num_frame_per_block (from config).
# With num_frame_per_block=3, each DiT call already produces 3 frames,
# so we decode immediately after each DiT call (no accumulation needed).


def run_inference(pipeline, args, config, verbose=False):
    """Run streaming inference: accumulate 3 DiT blocks → VAE decode batch.
    
    Same as rolling-forcing: accumulate num_frame_per_block=3 latent frames,
    then decode them together through VAE. This gives a fair comparison.
    """
    device = pipeline.device
    dtype = pipeline.dtype
    num_frames = args.num_frames
    num_frame_per_block = pipeline.num_frame_per_block
    frame_seq_length = pipeline.frame_seq_length
    rank = pipeline.rank

    # Generate initial noise for anchor block
    num_anchor_frames = num_frame_per_block
    noise = torch.randn(
        1, num_anchor_frames, 16,
        args.height // 8, args.width // 8,
        dtype=dtype, device=device)

    timings = {
        "t5_encode": 0,      # T5 + anchor block time
        "dit_stream": [],     # per-block DiT time (5 denoising steps)
        "vae_stream": [],     # per-batch VAE decode time (3 frames)
        "block_e2e": [],      # per-batch end-to-end (3×DiT + VAE)
    }

    # Step 1: Encode prompt + anchor denoising (inside prepare)
    t0 = time.perf_counter()
    anchor_pred = pipeline.prepare(
        text_prompts=[args.prompt],
        device=device, dtype=dtype,
        noise=noise,
        current_start=0,
        current_end=frame_seq_length * num_anchor_frames,
        batch_denoise=False,
    )
    if hasattr(torch, 'neuron'):
        torch.neuron.synchronize()
    timings["t5_encode"] = time.perf_counter() - t0

    # VAE decode anchor immediately
    t_vae = time.perf_counter()
    anchor_video = pipeline.decode_latents(anchor_pred)
    if hasattr(torch, 'neuron'):
        torch.neuron.synchronize()
    anchor_vae_time = time.perf_counter() - t_vae
    timings["vae_stream"].append(anchor_vae_time)

    all_videos = [anchor_video] if anchor_video is not None else []

    # Step 2: Stream remaining frames — 1 DiT call (3 frames) + 1 VAE decode per block
    num_remaining = num_frames - num_anchor_frames
    num_stream_blocks = (num_remaining + num_frame_per_block - 1) // num_frame_per_block

    for block_idx in range(num_stream_blocks):
        t_block_start = time.perf_counter()

        current_frame = num_anchor_frames + block_idx * num_frame_per_block
        current_start = current_frame * frame_seq_length
        current_end = current_start + frame_seq_length * num_frame_per_block

        block_noise = torch.randn(
            1, num_frame_per_block, 16,
            args.height // 8, args.width // 8,
            dtype=dtype, device=device)

        # DiT: 5 denoising steps for this block (3 frames)
        t_dit = time.perf_counter()
        pred = pipeline.inference_wo_batch(
            noise=block_noise,
            current_start=current_start,
            current_end=current_end,
            current_step=pipeline.denoising_step_list[0].item(),
        )
        if hasattr(torch, 'neuron'):
            torch.neuron.synchronize()
        dit_time = time.perf_counter() - t_dit
        timings["dit_stream"].append(dit_time)

        # VAE decode immediately (3 frames per call, same as rolling-forcing)
        t_vae = time.perf_counter()
        block_video = pipeline.decode_latents(pred)
        if hasattr(torch, 'neuron'):
            torch.neuron.synchronize()
        vae_time = time.perf_counter() - t_vae
        timings["vae_stream"].append(vae_time)

        block_e2e = time.perf_counter() - t_block_start
        timings["block_e2e"].append(block_e2e)

        if block_video is not None:
            all_videos.append(block_video)

        if verbose and rank == 0 and block_idx < 3:
            LOGGER.info(f"  Block {block_idx}: "
                        f"DiT={dit_time*1000:.0f}ms "
                        f"VAE={vae_time*1000:.0f}ms "
                        f"E2E={block_e2e*1000:.0f}ms "
                        f"({num_frame_per_block} frames)")

    # Concatenate all decoded video blocks
    if all_videos:
        video = torch.cat(all_videos, dim=1)[:, :num_frames]
        video = video * 0.5 + 0.5
    else:
        video = None

    return video, None, timings


def print_benchmark_results(all_timings, warmup_timings, num_frames, num_runs):
    """Print streaming FPS benchmark results.
    
    True streaming: each block = DiT(5 steps) + VAE(1 frame).
    Reports per-block streaming latency and FPS.
    """
    # Warmup/compilation timing (first warmup is compilation)
    w0 = warmup_timings[0]
    compile_time = (w0["t5_encode"] + sum(w0["dit_stream"])
                    + sum(w0["vae_stream"]))

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    # Collect all per-block timings across benchmark runs
    all_dit = []
    all_vae = []
    all_e2e = []
    for t in all_timings:
        all_dit.extend(t["dit_stream"])
        all_vae.extend(t["vae_stream"][1:])  # skip anchor VAE
        all_e2e.extend(t["block_e2e"])

    avg_dit_per_block = avg(all_dit)
    avg_vae_per_block = avg(all_vae)
    avg_e2e_per_block = avg(all_e2e)

    # Encode times (T5 + anchor)
    avg_encode = avg([t["t5_encode"] for t in all_timings])

    # Total time per run = T5+anchor + sum(block_e2e)
    total_times = [t["t5_encode"] + sum(t["block_e2e"]) + t["vae_stream"][0]
                   for t in all_timings]
    avg_total = avg(total_times)

    # num_frame_per_block from config (3 for both SD and RF)
    nfpb = 3  # frames per DiT call / VAE decode

    # Streaming FPS: frames per second during steady-state streaming
    stream_fps = nfpb / avg_e2e_per_block if avg_e2e_per_block > 0 else 0

    # Overall FPS (including T5 encode + anchor)
    e2e_fps = num_frames / avg_total if avg_total > 0 else 0

    # VAE-only FPS (frames decoded per second)
    vae_fps = nfpb / avg_vae_per_block if avg_vae_per_block > 0 else 0

    # Time-to-first-frame
    ttff = avg_encode + avg([t["vae_stream"][0] for t in all_timings])

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK RESULTS (post-compilation, streaming)         │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  Num frames:              {num_frames:>6}                      │")
    print(f"│  Benchmark runs:          {num_runs:>6}                      │")
    print(f"│  Compilation time:     {compile_time:>8.1f}s (warmup run 1)     │")
    print(f"│  T5+anchor time:       {avg_encode:>8.3f}s                    │")
    print(f"│  DiT/block time:       {avg_dit_per_block:>8.3f}s (5 steps)       │")
    print(f"│  VAE/batch time:       {avg_vae_per_block:>8.3f}s ({nfpb} frames)  │")
    print(f"│  Batch E2E time:       {avg_e2e_per_block:>8.3f}s (DiT+VAE, {nfpb}f) │")
    print(f"│  Total time:           {avg_total:>8.3f}s                    │")
    print(f"│  Time-to-first-frame:  {ttff:>8.3f}s                    │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  OVERALL FPS:            {e2e_fps:>8.2f} frames/sec         │")
    print(f"│  STREAMING FPS:          {stream_fps:>8.2f} frames/sec         │")
    print(f"│  VAE decode FPS:         {vae_fps:>8.2f} frames/sec         │")
    print(f"│  Real-time ratio:        {e2e_fps/16:>8.3f}x (vs 16fps)      │")
    print(f"│  Stream real-time ratio: {stream_fps/16:>8.3f}x (vs 16fps)      │")
    print("└─────────────────────────────────────────────────────────┘\n")


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    # Merge config into args
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)

    # Ensure required fields
    if not hasattr(args, 'denoising_step_list'):
        args.denoising_step_list = [700, 500, 400, 200, 0]
    if not hasattr(args, 'model_type'):
        args.model_type = "T2V-1.3B"
    if not hasattr(args, 't2v'):
        args.t2v = True

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # Initialize distributed for TP-4
    init_distributed()

    LOGGER.info("Building Neuron pipeline...")
    pipeline = NeuronCausalStreamInferencePipeline(args, device=args.device)

    if args.benchmark:
        LOGGER.info("=== BENCHMARK MODE ===")
        # Warmup (triggers NEFF compilation on first run)
        warmup_timings = []
        for i in range(args.warmup_runs):
            LOGGER.info(f"Warmup run {i+1}/{args.warmup_runs} "
                        f"{'(NEFF compilation)' if i == 0 else '(cache warm)'}")
            pipeline.kv_cache1 = None
            pipeline.crossattn_cache = None
            pipeline.shared_buffers = None
            video, _, timings = run_inference(pipeline, args, config)
            warmup_timings.append(timings)
            LOGGER.info(f"  warmup t5+anchor={timings['t5_encode']*1000:.0f}ms "
                        f"dit_stream={sum(timings['dit_stream'])*1000:.0f}ms "
                        f"vae_stream={sum(timings['vae_stream'])*1000:.0f}ms")

        # Benchmark (post-compilation — steady-state performance)
        LOGGER.info("=== POST-COMPILATION BENCHMARK ===")
        all_timings = []
        for i in range(args.benchmark_runs):
            LOGGER.info(f"Benchmark run {i+1}/{args.benchmark_runs}")
            pipeline.kv_cache1 = None
            pipeline.crossattn_cache = None
            pipeline.shared_buffers = None
            video, _, timings = run_inference(pipeline, args, config,
                                             verbose=(i == 0))  # detail on first run
            all_timings.append(timings)
            LOGGER.info(f"  t5+anchor={timings['t5_encode']*1000:.0f}ms "
                        f"dit_stream={sum(timings['dit_stream'])*1000:.0f}ms "
                        f"vae_stream={sum(timings['vae_stream'])*1000:.0f}ms "
                        f"blocks={len(timings['block_e2e'])}")

        # Save video from first benchmark run to /var/mdl/ for quality inspection
        if dist.get_rank() == 0 and video is not None:
            output_dir = "/var/mdl"
            os.makedirs(output_dir, exist_ok=True)
            video_path = os.path.join(output_dir, "stream_diffusion_output.mp4")
            save_video(video, video_path, fps=args.fps)
            LOGGER.info(f"Video saved to {video_path}")

        # Only rank 0 prints results (avoid 4× duplicate output)
        if dist.get_rank() == 0:
            print_benchmark_results(all_timings, warmup_timings, args.num_frames, args.benchmark_runs)
    else:
        LOGGER.info("Running inference...")
        video, latents, timings = run_inference(pipeline, args, config)
        LOGGER.info(f"T5+anchor: {timings['t5_encode']*1000:.0f}ms, "
                    f"DiT stream: {sum(timings['dit_stream'])*1000:.0f}ms, "
                    f"VAE stream: {sum(timings['vae_stream'])*1000:.0f}ms")

        # Save (only rank 0 has decoded video)
        if video is not None:
            save_video(video, args.output_path, fps=args.fps)
            LOGGER.info(f"Video saved to {args.output_path}")


if __name__ == "__main__":
    main()
