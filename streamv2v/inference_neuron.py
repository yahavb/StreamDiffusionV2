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
    # Aliases for the neuron-3run-benchmark harness, which appends `--warmup N --iters M`.
    # When given, they override --warmup_runs / --benchmark_runs respectively.
    parser.add_argument("--warmup", type=int, default=None, help="Alias for --warmup_runs (3run harness)")
    parser.add_argument("--iters", type=int, default=None, help="Alias for --benchmark_runs (3run harness)")
    parser.add_argument("--device", type=str, default="neuron")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--fps", type=int, default=16)
    # v2v: input video to restyle (the paper's actual mode). If unset -> t2v from noise.
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.8)
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

    # ── v2v: encode an input video to latents (the mode the paper actually uses) ──
    # The Wan VAE compresses time, so we do NOT assume a frame count — we read the
    # actual latent frame count from the encoder and drive the block loop from it.
    # latent layout matches the noise tensor: [1, T_lat, 16, H/8, W/8].
    video_latents = None
    noise_scale = float(getattr(args, "noise_scale", 0.8))
    if getattr(args, "video_path", None):
        # imageio-based loader (container has imageio/imageio-ffmpeg; avoids torchvision
        # which isn't installed and would risk clobbering the Neuron torch build).
        def _load_video_imageio(path, H, W):
            import imageio.v3 as iio
            import numpy as np
            frames = iio.imread(path, plugin="pyav")  # [T, h, w, C] uint8
            t = torch.from_numpy(np.asarray(frames)).float()        # [T,h,w,C]
            t = t.permute(0, 3, 1, 2)                                # [T,C,h,w]
            t = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
            t = t / 127.5 - 1.0                                      # -> [-1,1]
            return t                                                 # [T,C,H,W]
        # VAE_RANK loads the pixels; encode_video_latents broadcasts latents to all ranks.
        vid = None
        if rank == 0:
            v = _load_video_imageio(args.video_path, args.height, args.width)  # [T,C,H,W] in [-1,1]
            vid = v.unsqueeze(0).to(dtype=dtype, device=device)               # [1,T,C,H,W]
        # encode needs the latent frame count up front for the zero-buffer on non-VAE ranks;
        # compute it from pixel frames via the VAE's temporal rule (causal 4x: (T-1)//4 + 1).
        if rank == 0:
            t_pix = vid.shape[1]
        else:
            t_pix = int(num_frames)
        t_pix_t = torch.tensor([t_pix], device=device)
        if dist.is_initialized():
            dist.broadcast(t_pix_t, src=0)
        t_pix = int(t_pix_t.item())
        t_lat = (t_pix - 1) // 4 + 1
        # round down to a whole number of blocks so the streaming loop is exact
        t_lat = (t_lat // num_frame_per_block) * num_frame_per_block
        video_latents = pipeline.encode_video_latents(vid, t_lat, args.height, args.width)
        num_frames = t_lat
        LOGGER.info(f"[v2v] input={args.video_path} pixel_frames={t_pix} -> latent_frames={t_lat} "
                    f"({num_frames // num_frame_per_block} blocks), noise_scale={noise_scale}")

    def block_input(start_lat, n_lat):
        """Per-block model input: noised input latents (v2v) or pure noise (t2v)."""
        rand = torch.randn(1, n_lat, 16, args.height // 8, args.width // 8,
                           dtype=dtype, device=device)
        if video_latents is None:
            return rand
        chunk = video_latents[:, start_lat:start_lat + n_lat]
        # mix input-video latents with noise (StreamDiffusionV2 v2v: noise*s + latent*(1-s))
        return rand * noise_scale + chunk * (1.0 - noise_scale)

    # Generate initial input for anchor block (v2v latents or t2v noise)
    num_anchor_frames = num_frame_per_block
    noise = block_input(0, num_anchor_frames)

    timings = {
        "t5_encode": 0,      # T5 + anchor block time
        "dit_stream": [],     # per-block DiT time (5 denoising steps)
        "vae_stream": [],     # per-batch VAE decode time
        "block_e2e": [],      # per-batch end-to-end (DiT + VAE)
        "num_frame_per_block": num_frame_per_block,
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

    # New clip: reset the VAE temporal-cache stream so the RF VAE carries cache
    # across blocks (clear once here, NOT per block — per-block clearing softens output).
    pipeline.reset_decode_stream()

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

        # v2v: noised input-video latents for this block; t2v: pure noise
        block_noise = block_input(current_frame, num_frame_per_block)

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

        # Log EVERY block (rank 0) so per-block spread + compile contamination is
        # visible in the log — this is how RF exposes its 8.09 fps steady-state vs
        # the compile-dragged early blocks. block_fps uses block_e2e (DiT+VAE).
        if rank == 0:
            block_fps = num_frame_per_block / block_e2e if block_e2e > 0 else 0
            LOGGER.info(f"  Block {block_idx}: "
                        f"DiT={dit_time*1000:.0f}ms "
                        f"VAE={vae_time*1000:.0f}ms "
                        f"E2E={block_e2e*1000:.0f}ms "
                        f"= {block_fps:.2f} fps "
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

    def median(lst):
        if not lst:
            return 0
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

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

    # MEDIAN per-block = honest steady-state (rejects compile-contaminated blocks).
    # The first blocks of the first run include NEFF compilation (200s+); a mean
    # is dragged down by them, a median rejects them — same methodology as RF's
    # 8.09 fps steady-state figure. This is the apples-to-apples number vs RF.
    med_dit_per_block = median(all_dit)
    med_e2e_per_block = median(all_e2e)

    # Encode times (T5 + anchor)
    avg_encode = avg([t["t5_encode"] for t in all_timings])

    # Total time per run = T5+anchor + sum(block_e2e)
    total_times = [t["t5_encode"] + sum(t["block_e2e"]) + t["vae_stream"][0]
                   for t in all_timings]
    avg_total = avg(total_times)

    # num_frame_per_block from config
    nfpb = all_timings[0].get("num_frame_per_block", 1)  # frames per DiT call / VAE decode

    # Streaming FPS: frames per second during steady-state streaming
    stream_fps = nfpb / avg_e2e_per_block if avg_e2e_per_block > 0 else 0

    # Overall FPS (including T5 encode + anchor)
    e2e_fps = num_frames / avg_total if avg_total > 0 else 0

    # VAE-only FPS (frames decoded per second)
    vae_fps = nfpb / avg_vae_per_block if avg_vae_per_block > 0 else 0

    # STEADY-STATE FPS from per-block MEDIAN (rejects compile blocks) — RF-comparable.
    steady_fps = nfpb / med_e2e_per_block if med_e2e_per_block > 0 else 0

    # Time-to-first-frame
    ttff = avg_encode + avg([t["vae_stream"][0] for t in all_timings])

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  BENCHMARK RESULTS (post-compilation, streaming)         │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  Num frames:              {num_frames:>6}                      │")
    print(f"│  Benchmark runs:          {num_runs:>6}                      │")
    print(f"│  Blocks measured:         {len(all_e2e):>6}                      │")
    print(f"│  Compilation time:     {compile_time:>8.1f}s (warmup run 1)     │")
    print(f"│  T5+anchor time:       {avg_encode:>8.3f}s                    │")
    print(f"│  DiT/block (mean):     {avg_dit_per_block:>8.3f}s (5 steps)       │")
    print(f"│  DiT/block (median):   {med_dit_per_block:>8.3f}s (steady)        │")
    print(f"│  VAE/batch time:       {avg_vae_per_block:>8.3f}s ({nfpb} frames)  │")
    print(f"│  Block E2E (mean):     {avg_e2e_per_block:>8.3f}s (DiT+VAE, {nfpb}f) │")
    print(f"│  Block E2E (median):   {med_e2e_per_block:>8.3f}s (steady)        │")
    print(f"│  Total time:           {avg_total:>8.3f}s                    │")
    print(f"│  Time-to-first-frame:  {ttff:>8.3f}s                    │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  OVERALL FPS:            {e2e_fps:>8.2f} frames/sec         │")
    print(f"│  STREAMING FPS (mean):   {stream_fps:>8.2f} frames/sec         │")
    print(f"│  STEADY-STATE FPS (med): {steady_fps:>8.2f} frames/sec  ← RF-comparable │")
    print(f"│  VAE decode FPS:         {vae_fps:>8.2f} frames/sec         │")
    print(f"│  Real-time ratio:        {steady_fps/16:>8.3f}x (vs 16fps)      │")
    print("└─────────────────────────────────────────────────────────┘\n")
    # Marker for the harness: steady-state per-block median (the RF-comparable fps).
    print(f"steady_state_fps={steady_fps:.3f} block_median_ms={med_e2e_per_block*1000:.1f}",
          flush=True)


def main():
    args = parse_args()

    # Map neuron-3run-benchmark harness aliases onto the native arg names.
    if args.warmup is not None:
        args.warmup_runs = args.warmup
    if args.iters is not None:
        args.benchmark_runs = args.iters

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

    # Print effective config
    print("=" * 60)
    print("  STREAM-DIFFUSION CONFIG (effective)")
    print("=" * 60)
    for k in sorted(vars(args)):
        if k.startswith('_'):
            continue
        print(f"  {k}: {getattr(args, k)}")
    print("=" * 60)

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
            t_warm = time.perf_counter()
            pipeline.kv_cache1 = None
            pipeline.crossattn_cache = None
            pipeline.shared_buffers = None
            video, _, timings = run_inference(pipeline, args, config)
            warmup_timings.append(timings)
            warm_wall = time.perf_counter() - t_warm
            LOGGER.info(f"  warmup t5+anchor={timings['t5_encode']*1000:.0f}ms "
                        f"dit_stream={sum(timings['dit_stream'])*1000:.0f}ms "
                        f"vae_stream={sum(timings['vae_stream'])*1000:.0f}ms")
            # Marker parsed by the neuron-3run-benchmark harness: warmup run 1 wall
            # time is the build/compile cost (cache hit => small; cold => hundreds of s).
            if i == 0 and dist.get_rank() == 0:
                print(f"built={warm_wall:.1f}s", flush=True)

        # Benchmark (post-compilation — steady-state performance)
        LOGGER.info("=== POST-COMPILATION BENCHMARK ===")
        all_timings = []
        total_times = []
        for i in range(args.benchmark_runs):
            LOGGER.info(f"Benchmark run {i+1}/{args.benchmark_runs}")
            pipeline.kv_cache1 = None
            pipeline.crossattn_cache = None
            pipeline.shared_buffers = None
            video, _, timings = run_inference(pipeline, args, config,
                                             verbose=(i == 0))  # detail on first run
            all_timings.append(timings)
            total_times.append(
                timings["t5_encode"] + sum(timings["block_e2e"]) + timings["vae_stream"][0])
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
        if dist.get_rank() == 0 and all_timings:
            print_benchmark_results(all_timings, warmup_timings, args.num_frames, args.benchmark_runs)
            # Markers parsed by the neuron-3run-benchmark harness/gen_report.py.
            # median = full-clip wall time (real latency); throughput = clip frames/s.
            median_total = sorted(total_times)[len(total_times) // 2]
            throughput = args.num_frames / median_total if median_total > 0 else 0.0
            print(f"median={median_total * 1000:.1f}ms throughput={throughput:.3f} frame/s",
                  flush=True)
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
