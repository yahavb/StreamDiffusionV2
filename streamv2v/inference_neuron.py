"""Neuron/Trainium inference entry point for StreamDiffusionV2.

Runs the full streaming pipeline on Trainium: T5 encode → DiT denoise → VAE decode.
Measures and reports FPS for each stage and end-to-end.

Usage:
    python streamv2v/inference_neuron.py --config streamv2v/configs/wan_causal_dmd_v2v_neuron.yaml \
        --prompt "A cat walking on the beach" --num_frames 81 --benchmark
"""
import argparse
import logging
import os
import sys
import time

import torch
from omegaconf import OmegaConf
from einops import rearrange

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wan.neuron_causal_stream_inference import NeuronCausalStreamInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("inference_neuron")


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
    """Save video tensor [B, T, C, H, W] in [0,1] to mp4."""
    video = video_tensor.cpu().clamp(0, 1)
    video_out = rearrange(video, 'b t c h w -> b t h w c')
    video_out = (255.0 * video_out).to(torch.uint8).numpy()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for i in range(video_out.shape[0]):
        path = output_path if video_out.shape[0] == 1 else output_path.replace(".mp4", f"_{i}.mp4")
        try:
            import imageio
            writer = imageio.get_writer(path, fps=fps, codec='libx264')
            for frame in video_out[i]:
                writer.append_data(frame)
            writer.close()
            LOGGER.info(f"Saved video: {path}")
        except ImportError:
            from PIL import Image
            frame_dir = path.replace(".mp4", "_frames")
            os.makedirs(frame_dir, exist_ok=True)
            for t, frame in enumerate(video_out[i]):
                Image.fromarray(frame).save(os.path.join(frame_dir, f"frame_{t:04d}.png"))
            LOGGER.info(f"Saved frames: {frame_dir}")


def run_inference(pipeline, args, config):
    """Run streaming inference and return latents + timing info."""
    device = pipeline.device
    dtype = pipeline.dtype
    num_frames = args.num_frames
    num_frame_per_block = pipeline.num_frame_per_block
    frame_seq_length = pipeline.frame_seq_length

    # Generate initial noise for anchor block
    num_anchor_frames = num_frame_per_block
    noise = torch.randn(
        1, num_anchor_frames, 16,
        args.height // 8, args.width // 8,
        dtype=dtype, device=device)

    timings = {"encode": 0, "dit_anchor": 0, "dit_stream": [], "vae": 0}

    # Step 1: Encode prompt
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
    timings["encode"] = time.perf_counter() - t0
    timings["dit_anchor"] = timings["encode"]  # includes first DiT pass

    # Collect all latent blocks
    all_latents = [anchor_pred]

    # Step 2: Stream remaining frames
    num_remaining = num_frames - num_anchor_frames
    num_stream_blocks = (num_remaining + num_frame_per_block - 1) // num_frame_per_block

    for block_idx in range(num_stream_blocks):
        current_frame = num_anchor_frames + block_idx * num_frame_per_block
        current_start = current_frame * frame_seq_length
        current_end = current_start + frame_seq_length * num_frame_per_block

        block_noise = torch.randn(
            1, num_frame_per_block, 16,
            args.height // 8, args.width // 8,
            dtype=dtype, device=device)

        t_block = time.perf_counter()
        pred = pipeline.inference_wo_batch(
            noise=block_noise,
            current_start=current_start,
            current_end=current_end,
            current_step=pipeline.denoising_step_list[0].item(),
        )
        if hasattr(torch, 'neuron'):
            torch.neuron.synchronize()
        dt = time.perf_counter() - t_block
        timings["dit_stream"].append(dt)
        all_latents.append(pred)

    # Concatenate all latents
    latents = torch.cat(all_latents, dim=1)[:, :num_frames]

    # Step 3: VAE decode
    t_vae = time.perf_counter()
    video = pipeline.decode_latents(latents)
    video = video * 0.5 + 0.5
    if hasattr(torch, 'neuron'):
        torch.neuron.synchronize()
    timings["vae"] = time.perf_counter() - t_vae

    return video, latents, timings


def print_benchmark_results(all_timings, num_frames, num_runs):
    """Print FPS benchmark results in a format matching rolling-forcing."""
    encode_times = [t["encode"] for t in all_timings]
    dit_stream_all = []
    for t in all_timings:
        dit_stream_all.extend(t["dit_stream"])
    vae_times = [t["vae"] for t in all_timings]
    total_times = [t["encode"] + sum(t["dit_stream"]) + t["vae"] for t in all_timings]

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    avg_encode = avg(encode_times)
    avg_dit_per_block = avg(dit_stream_all) if dit_stream_all else 0
    avg_vae = avg(vae_times)
    avg_total = avg(total_times)

    encode_fps = 1.0 / avg_encode if avg_encode > 0 else 0
    dit_fps = num_frames / (avg_encode - avg_encode + sum([avg(t["dit_stream"]) for t in all_timings]) / len(all_timings) + avg_encode) if avg_total > 0 else 0
    vae_fps = num_frames / avg_vae if avg_vae > 0 else 0
    e2e_fps = num_frames / avg_total if avg_total > 0 else 0

    print("\n" + "=" * 70)
    print("StreamDiffusionV2 Neuron Benchmark Results")
    print("=" * 70)
    print(f"  Num frames:          {num_frames}")
    print(f"  Benchmark runs:      {num_runs}")
    print(f"  Avg encode time:     {avg_encode*1000:.1f} ms")
    print(f"  Avg DiT/block time:  {avg_dit_per_block*1000:.1f} ms")
    print(f"  Avg VAE time:        {avg_vae*1000:.1f} ms")
    print(f"  Avg total time:      {avg_total*1000:.1f} ms")
    print(f"  End-to-end FPS:      {e2e_fps:.2f}")
    print(f"  VAE decode FPS:      {vae_fps:.2f}")
    print("=" * 70 + "\n")


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

    LOGGER.info("Building Neuron pipeline...")
    pipeline = NeuronCausalStreamInferencePipeline(args, device=args.device)

    if args.benchmark:
        LOGGER.info("=== BENCHMARK MODE ===")
        # Warmup
        for i in range(args.warmup_runs):
            LOGGER.info(f"Warmup run {i+1}/{args.warmup_runs}")
            video, _, _ = run_inference(pipeline, args, config)
            # Reset pipeline state for next run
            pipeline.kv_cache1 = None
            pipeline.crossattn_cache = None
            pipeline.shared_buffers = None

        # Benchmark
        all_timings = []
        for i in range(args.benchmark_runs):
            LOGGER.info(f"Benchmark run {i+1}/{args.benchmark_runs}")
            pipeline.kv_cache1 = None
            pipeline.crossattn_cache = None
            pipeline.shared_buffers = None
            video, _, timings = run_inference(pipeline, args, config)
            all_timings.append(timings)
            LOGGER.info(f"  encode={timings['encode']*1000:.0f}ms "
                        f"dit_stream={sum(timings['dit_stream'])*1000:.0f}ms "
                        f"vae={timings['vae']*1000:.0f}ms")

        print_benchmark_results(all_timings, args.num_frames, args.benchmark_runs)
    else:
        LOGGER.info("Running inference...")
        video, latents, timings = run_inference(pipeline, args, config)
        LOGGER.info(f"Encode: {timings['encode']*1000:.0f}ms, "
                    f"DiT stream: {sum(timings['dit_stream'])*1000:.0f}ms, "
                    f"VAE: {timings['vae']*1000:.0f}ms")

        # Save
        save_video(video, args.output_path, fps=args.fps)
        LOGGER.info(f"Video saved to {args.output_path}")


if __name__ == "__main__":
    main()
