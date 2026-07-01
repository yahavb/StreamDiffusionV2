"""DMD distillation for StreamDiffusion-v2 — 14B t2v teacher -> few-step 1.3B causal student.

GOAL: produce a Wan2.1-1.3B causal checkpoint that is good at OUR step count (1-2),
on OUR domain (kinetics motion prompts), and correct in OUR streaming pipeline
(schedule + rolling KV cache + sink tokens + block size). That last part is the
thing RollingForcing's checkpoint lacked (technically causal, but noise in our loop).

TRAINING RUNS ON TRAINIUM (torch_neuronx, native-PyTorch-compatible image).
The output checkpoint is drop-in for sd-job.yaml (loads via neuron_wan_wrapper as
{'generator': state_dict}).

Distillation Matching Distillation (DMD2) with Self-Forcing, 3 networks:
  - generator G   : the student (init base 1.3B), TRAINABLE, few-step schedule
  - real_score    : the 14B t2v teacher, FROZEN — score of the target distribution
  - fake_score    : a 1.3B diffusion net, TRAINABLE — score of G's current outputs
G is updated by (real_score - fake_score); fake_score tracks G. Alternate at
dfake_gen_update_ratio.

WHY t2v teacher (not i2v): the student is t2v/v2v (prompt/video -> video, no
conditioning image). An i2v teacher conditions on a start frame the student never
has -> misaligned scores. A t2v-distilled student still runs v2v at inference
(v2v just noises the input latent instead of pure noise — same denoiser).

Usage (single node, torchrun on Trainium):
  torchrun --nproc_per_node=8 distill/distill_sdv2.py \
    --config streamv2v/configs/wan_causal_fps_neuron.yaml \
    --captions /tmp/captions-20260701-062139.jsonl \
    --teacher_ckpt ckpts/wan_causal_dmd_v2v_14b/model.pt \
    --student_base wan_models/Wan2.1-T2V-1.3B \
    --out checkpoints/distilled/model.pt \
    --steps 700,0 --iters 2000
"""
import argparse
import json
import logging
import os
import sys

# Repo root on sys.path — this script lives in distill/, but imports models.wan.*
# and (via wan_base) modules.* which live at the repo root. Mirror how the
# root-level entrypoints resolve imports.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_WAN_BASE = os.path.join(_ROOT, "models", "wan", "wan_base")
if os.path.isdir(_WAN_BASE) and _WAN_BASE not in sys.path:
    sys.path.insert(0, _WAN_BASE)

import torch
import torch.nn.functional as F
import torch.distributed as dist
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
LOGGER = logging.getLogger("distill")


# ── args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="SDV2 neuron config — schedule/KV/sink/block read from here so "
                        "train==inference (no drift). Use the SAME yaml sd-job runs.")
    p.add_argument("--captions", required=True, help="kinetics captions .jsonl (uses 'prompt')")
    p.add_argument("--teacher_ckpt", required=True, help="14B t2v DMD checkpoint (frozen real_score)")
    p.add_argument("--teacher_base", default="wan_models/Wan2.1-T2V-14B")
    p.add_argument("--student_base", default="wan_models/Wan2.1-T2V-1.3B",
                   help="base 1.3B weights to INIT G and fake_score (never random)")
    p.add_argument("--out", default="checkpoints/distilled/model.pt")
    p.add_argument("--steps", default=None,
                   help="student few-step schedule (fps lever), e.g. '700,0' (2-step), "
                        "'0' (1-step). Overrides config denoising_step_list. MUST end in 0.")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-6)          # from configs/wan_causal_dmd_v2v.yaml
    p.add_argument("--dfake_gen_update_ratio", type=int, default=5)  # fake_score updates per G update
    p.add_argument("--warmup_regression", type=int, default=200,
                   help="first N iters also regress G toward teacher trajectory (stabilizes DMD)")
    p.add_argument("--tp_degree", type=int, default=8)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=360)   # not 240 (too small) / not 480 (too heavy)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ── distributed / device ────────────────────────────────────────────────────

def init_dist():
    """Neuron distributed init. torch_neuronx makes 'xla'/neuron device train with
    native autograd; falls back to CPU/GPU for a tiny smoke test off-device."""
    backend = "xla" if os.environ.get("RANK") else None
    device = "cpu"
    try:
        import torch_neuronx  # noqa: F401
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        if os.environ.get("RANK"):
            dist.init_process_group(backend="xla")
        LOGGER.info(f"Neuron/XLA device: {device}")
        return device, xm
    except Exception as e:
        LOGGER.warning(f"torch_neuronx unavailable ({e}); CPU smoke-test mode")
        if os.environ.get("RANK"):
            dist.init_process_group(backend="gloo")
        return torch.device("cpu"), None


# ── data ────────────────────────────────────────────────────────────────────

def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # caption schema: {"prompt": "...", "motion_type", "motion_intensity", ...}
            prompts.append(rec["prompt"])
    LOGGER.info(f"loaded {len(prompts)} prompts from {path}")
    return prompts


# ── model construction — NEURON wrappers/pipeline (train==inference, same as sd-job) ──

def build_student_pipeline(args, device, dtype):
    """Build SDV2's NEURON streaming pipeline for the STUDENT (G). Generate through
    THIS so the training rollout uses the identical schedule + rolling KV cache +
    sink tokens + block size as inference. It builds self.generator (the Neuron
    wrapper) from the config. We flip the DiT to train/grad for distillation —
    the pipeline was inference-only (eager, no grad), so enabling grads on the
    sharded generator + backprop through the eager loop is the Neuron-training work."""
    from models.wan.neuron_causal_stream_inference import NeuronCausalStreamInferencePipeline
    pipe = NeuronCausalStreamInferencePipeline(args, device=device)
    # student DiT must carry grads (pipeline built it eval/frozen for inference)
    pipe.generator.model.train().requires_grad_(True)
    return pipe


def build_neuron_generator(model_path, ckpt_path, args, device, trainable, tp_degree):
    """Build a bare NEURON causal DiT wrapper (teacher real_score / fake_score),
    using the SAME constructor + TP sharding sd-job uses. teacher: frozen."""
    from models.wan.neuron_wan_wrapper import NeuronCausalWanDiffusionWrapper
    g = NeuronCausalWanDiffusionWrapper(
        model_path=model_path,
        checkpoint_path=ckpt_path,
        denoising_step_list=list(args.denoising_step_list),
        timestep_shift=getattr(args, "timestep_shift", 8.0),
        num_frame_per_block=getattr(args, "num_frame_per_block", 3),
        device=device,
        tp_degree=tp_degree,
    )
    if trainable:
        g.model.train().requires_grad_(True)
    else:
        g.model.eval().requires_grad_(False)
    return g


# ── DMD loss ─────────────────────────────────────────────────────────────────

def dmd_generator_loss(x0_student, real_score, fake_score, conditional_dict,
                       scheduler, device, dtype):
    """DMD2 generator gradient: push G's sample toward teacher's score, away from
    fake_score. Sample a random diffusion timestep, noise x0, get both scores,
    backprop the (real - fake) difference into x0_student (hence into G).
    """
    b = x0_student.shape[0]
    # random timestep in the diffusion range
    t = torch.randint(20, 980, (b,), device=device)
    noise = torch.randn_like(x0_student)
    x_t = scheduler.add_noise(x0_student.flatten(0, 1),
                              noise.flatten(0, 1),
                              t).unflatten(0, x0_student.shape[:2])
    tt = t.view(b, 1).expand(b, x0_student.shape[1])
    with torch.no_grad():
        real_pred = real_score(noisy_image_or_video=x_t, conditional_dict=conditional_dict,
                               timestep=tt, kv_cache=None, crossattn_cache=None)
        fake_pred = fake_score(noisy_image_or_video=x_t, conditional_dict=conditional_dict,
                               timestep=tt, kv_cache=None, crossattn_cache=None)
    # DMD gradient surrogate: (fake - real) as the gradient wrt x0 (KL of distributions).
    grad = (fake_pred - real_pred)
    # normalize by magnitude (DMD2 stability trick)
    grad = grad / (grad.abs().mean() + 1e-8)
    # loss whose d/dx0 == grad: 0.5 * (x0 - stopgrad(x0 - grad))^2
    target = (x0_student - grad).detach()
    return 0.5 * F.mse_loss(x0_student, target)


def fake_score_loss(x0_student, fake_score, conditional_dict, scheduler, device):
    """Diffusion (flow) loss training fake_score to model G's current outputs."""
    x0 = x0_student.detach()
    b = x0.shape[0]
    t = torch.randint(20, 980, (b,), device=device)
    noise = torch.randn_like(x0)
    x_t = scheduler.add_noise(x0.flatten(0, 1), noise.flatten(0, 1), t).unflatten(0, x0.shape[:2])
    tt = t.view(b, 1).expand(b, x0.shape[1])
    pred = fake_score(noisy_image_or_video=x_t, conditional_dict=conditional_dict,
                      timestep=tt, kv_cache=None, crossattn_cache=None)
    return F.mse_loss(pred, x0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device, xm = init_dist()
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    # 1) config -> args: schedule/KV/sink/block from OUR inference yaml (train==infer)
    cfg = OmegaConf.load(args.config)
    for k, v in cfg.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    if args.steps:  # few-step schedule = the fps lever
        parsed = [int(s) for s in str(args.steps).split(",") if s.strip() != ""]
        assert parsed[-1] == 0, f"--steps must end in 0, got {parsed}"
        args.denoising_step_list = parsed
    LOGGER.info(f"student schedule (fps lever): {args.denoising_step_list}  "
                f"KV={args.num_kv_cache} sink={args.num_sink_tokens} "
                f"block={args.num_frame_per_block} res={args.height}x{args.width}")

    prompts = load_prompts(args.captions)

    # 2) three networks
    LOGGER.info("building student G (init base 1.3B, trainable)...")
    student = build_student_pipeline(args, device, dtype)      # generation path (== inference)
    G = student.generator                                       # the trainable DiT
    G.requires_grad_(True)

    LOGGER.info("building fake_score (1.3B, trainable)...")
    fake_score = build_neuron_generator(
        args.student_base, None, args, device, trainable=True, tp_degree=args.tp_degree)

    LOGGER.info("building real_score = 14B t2v teacher (FROZEN)...")
    teacher_args = OmegaConf.create(dict(vars(args)))
    teacher_args.model_type = "T2V-14B"
    real_score = build_neuron_generator(
        args.teacher_base, args.teacher_ckpt, teacher_args, device,
        trainable=False, tp_degree=args.tp_degree)

    opt_g = torch.optim.AdamW([p for p in G.model.parameters() if p.requires_grad], lr=args.lr)
    opt_f = torch.optim.AdamW(fake_score.model.parameters(), lr=args.lr)
    scheduler = student.scheduler

    frame_seq = student.frame_seq_length
    npb = args.num_frame_per_block

    # 3) training loop
    for it in range(args.iters):
        prompt = [prompts[it % len(prompts)]]
        # fresh caches per clip (streaming state must not leak across prompts)
        student.kv_cache1 = None
        student.crossattn_cache = None

        # --- Self-Forcing rollout: G generates the anchor block through OUR loop ---
        # (multi-block rollout conditioned on G's OWN kv cache = the streaming signal)
        noise = torch.randn(1, npb, 16, args.height // 8, args.width // 8,
                            dtype=dtype, device=device)
        x0_student = student.prepare(
            text_prompts=prompt, device=device, dtype=dtype,
            noise=noise, current_start=0, current_end=frame_seq * npb,
            batch_denoise=False)  # returns the few-step clean student output, WITH grad

        cond = student.conditional_dict

        update_g = (it % args.dfake_gen_update_ratio == 0)
        if update_g:
            # optional regression warmup toward teacher (stabilize early)
            loss_g = dmd_generator_loss(x0_student, real_score, fake_score, cond,
                                        scheduler, device, dtype)
            if it < args.warmup_regression:
                with torch.no_grad():
                    # teacher's own clean pred at low-noise as a coarse trajectory target
                    t_lo = torch.full((1, x0_student.shape[1]), 50, device=device, dtype=torch.long)
                    tgt = real_score(noisy_image_or_video=x0_student.detach(),
                                     conditional_dict=cond, timestep=t_lo,
                                     kv_cache=None, crossattn_cache=None)
                loss_g = loss_g + F.mse_loss(x0_student, tgt)
            opt_g.zero_grad(); loss_g.backward()
            (xm.optimizer_step(opt_g) if xm else opt_g.step())
            gl = float(loss_g.detach())
        else:
            gl = float("nan")

        # fake_score always tracks G
        loss_f = fake_score_loss(x0_student, fake_score, cond, scheduler, device)
        opt_f.zero_grad(); loss_f.backward()
        (xm.optimizer_step(opt_f) if xm else opt_f.step())

        if it % 20 == 0:
            LOGGER.info(f"it {it}/{args.iters}  loss_G={gl:.4f}  loss_fake={float(loss_f.detach()):.4f}")

        # 4) checkpoint in the {'generator': ...} format sd-job loads (drop-in)
        rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if rank0 and it > 0 and it % args.save_every == 0:
            _save(G, args.out, it)

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        _save(G, args.out, args.iters)
    LOGGER.info("done. VALIDATE: load the checkpoint into sd-job.yaml and WATCH the video "
                "(never trust training loss alone).")


def _save(G, out, it):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # neuron_wan_wrapper strips 'model.' and _fsdp_wrapped_ prefixes; emit plain 'model.' keys.
    sd = {f"model.{k}": v.to(torch.bfloat16).cpu() for k, v in G.model.state_dict().items()}
    torch.save({"generator": sd, "distill_iter": it}, out)
    LOGGER.info(f"[ckpt] wrote {out} (iter {it}) — drop-in for sd-job generator_ckpt")


if __name__ == "__main__":
    main()
