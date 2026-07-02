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
    p.add_argument("--height", type=int, default=352)   # /8=44 even (patchify needs /2); not 240/480
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ── distributed / device ────────────────────────────────────────────────────

def init_dist():
    """Neuron distributed init — EXACTLY the proven inference path (eager torch_neuronx,
    NOT torch_xla): dist backend 'neuron' + torch.neuron.set_device(local_rank), then
    device='neuron'. Training uses native autograd (loss.backward(); optimizer.step()).
    Falls back to CPU only when not launched under torchrun (local dev)."""
    if "LOCAL_RANK" not in os.environ:
        LOGGER.warning("no LOCAL_RANK (not torchrun) -> CPU dev mode")
        return torch.device("cpu")
    import torch_neuronx  # noqa: F401
    dist.init_process_group(backend="neuron")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)
    LOGGER.info(f"Neuron distributed: rank={dist.get_rank()}/{dist.get_world_size()} "
                f"local_rank={local_rank}, device=neuron")
    return torch.device("neuron")


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
    # Match the attention frame_length to OUR resolution (default is 1560 = 480p);
    # the pipeline does this for the student, but teacher/fake built here must too,
    # or attention expects block_length=3*1560 while inputs are 3*frame_seq -> shape err.
    frame_seq = (args.height // 16) * (args.width // 16)
    g.model._update_frame_length(frame_seq,
                                 num_frame_per_block=getattr(args, "num_frame_per_block", 3),
                                 num_kv_cache=getattr(args, "num_kv_cache", 6))
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
    device = init_dist()
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

    # ── TWO-GROUP PLACEMENT (avoids OOM: 3 nets co-located = ~21GB/core -> OOM) ──
    # world=8 on l-trn2, tp_degree=4:
    #   TEACHER group  = ranks [0..3]  -> 14B frozen (7GB/core, no optimizer)
    #   STUDENT group  = ranks [4..7]  -> student + fake (weights+Adam, no 14B sharing)
    # Cross-group transfer via GLOBAL broadcast (Neuron: broadcast yes, P2P no):
    #   student broadcasts x_t (src in student grp) -> teacher scores -> teacher
    #   broadcasts real_score back (src in teacher grp). Every rank calls every
    #   broadcast (collective requirement); only the src group provides real data.
    # ── THREE-GROUP PLACEMENT (2-group student side still OOM'd: student+fake+2xAdam
    # +rollout activations > 24GB). Give EACH net its own TP-4 group = 12 cores:
    #   TEACHER group = ranks [0..3]    14B frozen
    #   STUDENT group = ranks [4..7]    G + its Adam + rollout activations
    #   FAKE    group = ranks [8..11]   fake_score + its Adam
    # Cross-group via global broadcast: student bcasts (x_t,t,embeds); teacher AND
    # fake each score it in their group; each bcasts its pred back to student.
    ws = dist.get_world_size() if dist.is_initialized() else 1
    tp = args.tp_degree
    # need >=3 TP groups (12 cores). world may be larger (e.g. 16 on l-trn2 LNC2 —
    # NPROC MUST equal the full claim for a clean device-barrier topology; ranks
    # beyond 3*tp are IDLE but still join every broadcast in lockstep).
    three_group = ws >= 3 * tp
    teacher_ranks = list(range(0, tp))
    student_ranks = list(range(tp, 2 * tp))
    fake_ranks = list(range(2 * tp, 3 * tp))
    my_rank = dist.get_rank() if dist.is_initialized() else 0
    if three_group:
        in_teacher = my_rank in teacher_ranks
        in_student = my_rank in student_ranks
        in_fake = my_rank in fake_ranks
        # ranks >= 3*tp: idle (no model), but MUST still call every bcast
    else:
        in_teacher = in_student = in_fake = True  # single-proc dev
    tsrc, ssrc, fsrc = teacher_ranks[0], student_ranks[0], fake_ranks[0]
    LOGGER.info(f"placement: world={ws} tp={tp} three_group={three_group} rank={my_rank} "
                f"teacher={in_teacher} student={in_student} fake={in_fake}")

    student = fake_score = real_score = G = None

    # STUDENT (G) on the student group — base 1.3B init (we're CREATING the DMD ckpt)
    if in_student:
        args.generator_ckpt = None
        os.environ["DISTILL_BASE_ONLY"] = "1"
        LOGGER.info("building student G (init base 1.3B, trainable)...")
        student = build_student_pipeline(args, device, dtype)
        G = student.generator
        G.model.requires_grad_(True)
        os.environ["DISTILL_BASE_ONLY"] = ""

    # FAKE_SCORE on its OWN group (base 1.3B init)
    if in_fake:
        os.environ["DISTILL_BASE_ONLY"] = "1"
        LOGGER.info("building fake_score (1.3B, trainable, base init)...")
        fake_score = build_neuron_generator(
            args.student_base, None, args, device, trainable=True, tp_degree=tp)
        os.environ["DISTILL_BASE_ONLY"] = ""

    # TEACHER (14B DMD ckpt) on the teacher group
    if in_teacher:
        LOGGER.info("building real_score = 14B t2v teacher (FROZEN)...")
        teacher_args = OmegaConf.create(dict(vars(args)))
        teacher_args.model_type = "T2V-14B"
        real_score = build_neuron_generator(
            args.teacher_base, args.teacher_ckpt, teacher_args, device,
            trainable=False, tp_degree=tp)

    # G optimizer on student group; fake optimizer on fake group (each net its own group)
    opt_g = torch.optim.AdamW([p for p in G.model.parameters() if p.requires_grad], lr=args.lr) if in_student else None
    opt_f = torch.optim.AdamW(fake_score.model.parameters(), lr=args.lr) if in_fake else None
    scheduler = student.scheduler if in_student else None
    # fake group needs a scheduler too (add_noise for its diffusion loss); the wrapper
    # builds one on .scheduler.
    scheduler_fake = fake_score.scheduler if in_fake else None
    frame_seq = student.frame_seq_length if in_student else 0
    npb = args.num_frame_per_block

    lat_shape = (1, npb, 16, args.height // 8, args.width // 8)
    emb_shape = (1, 512, 4096)

    def bcast(t, src):
        """global broadcast; EVERY rank must call in lockstep. returns src's tensor."""
        t = t.contiguous()
        if dist.is_initialized():
            dist.broadcast(t, src=src)
        return t

    # Teacher/fake forward REQUIRES a kv_cache (model has no cache-free path). For a
    # one-shot score we allocate a FRESH single-block cache each call (no streaming
    # state carried). Sized from the net's own dims (14B vs 1.3B differ).
    def make_score_cache(net):
        m = net.model
        nb = m.num_layers if hasattr(m, "num_layers") else len(m.blocks)
        nh = m.num_heads // tp
        hd = m.dim // m.num_heads
        fseq = (args.height // 8 // 2) * (args.width // 8 // 2)  # patch2 latent tokens
        kvlen = fseq * getattr(args, "num_kv_cache", 6)
        import models.wan.neuron_layers as _nl
        pad = ((21 * fseq + _nl.ATTN_SEQLEN_MULTIPLE - 1) // _nl.ATTN_SEQLEN_MULTIPLE) * _nl.ATTN_SEQLEN_MULTIPLE
        kv = [{"k": torch.zeros(1, kvlen, nh, hd, dtype=dtype, device=device),
               "v": torch.zeros(1, kvlen, nh, hd, dtype=dtype, device=device),
               "global_end_index": 0, "local_end_index": 0} for _ in range(nb)]
        ca = [{"k": torch.zeros(1, 512, nh, hd, dtype=dtype, device=device),
               "v": torch.zeros(1, 512, nh, hd, dtype=dtype, device=device),
               "is_init": False} for _ in range(nb)]
        sb = (torch.zeros(1, pad, nh, hd, dtype=dtype, device=device),
              torch.zeros(1, pad, nh, hd, dtype=dtype, device=device))
        return kv, ca, sb, fseq

    def score(net, x_t, tt, condb, cache):
        kv, ca, sb, fseq = cache
        # reset cache indices each call -> every score is a fresh single-block denoise
        # (no streaming state carried between iters; DMD scores are independent).
        for c in kv:
            c["global_end_index"] = 0; c["local_end_index"] = 0
        for c in ca:
            c["is_init"] = False
        return net(noisy_image_or_video=x_t, conditional_dict=condb, timestep=tt,
                   kv_cache=kv, crossattn_cache=ca, current_start=0,
                   current_end=fseq * npb, updating_cache=True, shared_buffers=sb)

    # allocate scoring caches once (fresh per call would leak; reset indices each iter)
    teacher_cache = make_score_cache(real_score) if in_teacher else None
    fake_cache = make_score_cache(fake_score) if in_fake else None

    shard_dir = os.path.join(os.path.dirname(args.out), "shards")

    def save_ckpt(it):
        """Reassemble FULL weights from TP shards WITHOUT any device/sub-group collective
        (device all_gather hits the compile service errno=2; Neuron barrier only works on
        the DEFAULT group). Phase 1: each student rank writes its CPU shard to a file.
        Phase 2: a GLOBAL barrier (all ranks) so shards are on disk. Phase 3: root concats."""
        from models.wan.tp_utils import ColumnParallelLinear, RowParallelLinear, TPRMSNorm
        # phase 1: student ranks dump shards
        if in_student:
            os.makedirs(shard_dir, exist_ok=True)
            shard = {k: v.detach().to(torch.bfloat16).cpu()
                     for k, v in G.model.state_dict().items()}
            torch.save(shard, os.path.join(shard_dir, f"shard_{my_rank - ssrc}.pt"))
        # phase 2: GLOBAL barrier — EVERY rank must call (default group; sub-group unsupported)
        if dist.is_initialized():
            dist.barrier()
        # phase 3: student root concatenates the shards into the full model
        if my_rank == ssrc or not dist.is_initialized():
            cat_dim = {}
            for mn, m in G.model.named_modules():
                if isinstance(m, ColumnParallelLinear):
                    cat_dim[f"{mn}.weight"] = 0
                    if getattr(m, "bias", None) is not None: cat_dim[f"{mn}.bias"] = 0
                elif isinstance(m, RowParallelLinear):
                    cat_dim[f"{mn}.weight"] = 1
                elif isinstance(m, TPRMSNorm):
                    cat_dim[f"{mn}.weight"] = 0
            shards = [torch.load(os.path.join(shard_dir, f"shard_{r}.pt"), map_location="cpu")
                      for r in range(tp)]
            full = {}
            for k in shards[0]:
                d = cat_dim.get(k, None)
                full[k] = shards[0][k] if (d is None or tp == 1) else torch.cat(
                    [shards[r][k] for r in range(tp)], dim=d)
            # Strip compile/wrapper infixes so keys match the FRESH (uncompiled) inference
            # model at load time: neuron_compile wraps submodules in _ContiguousWrapper
            # (.compiled_module.) and torch.compile adds ._orig_mod. The loader builds a
            # bare model and loads BEFORE compiling, so it expects clean names.
            def _clean(k):
                return (k.replace(".compiled_module.", ".")
                         .replace("._orig_mod.", ".")
                         .replace(".compiled_module", "")
                         .replace("._orig_mod", ""))
            sd = {f"model.{_clean(k)}": v for k, v in full.items()}
            torch.save({"generator": sd, "distill_iter": it}, args.out)
            LOGGER.info(f"[ckpt] wrote {args.out} (iter {it}) full={len(sd)} tensors — drop-in for sd-job")

    def zeros_lat(): return torch.zeros(lat_shape, dtype=dtype, device=device)

    # 3) training loop — ALL groups hit EVERY broadcast in the same order (no deadlock)
    for it in range(args.iters):
        prompt = [prompts[it % len(prompts)]]

        # (a) student generates x0 via Self-Forcing rollout (grad on)
        x0_student = cond = None
        if in_student:
            student.kv_cache1 = None
            student.crossattn_cache = None
            noise = torch.randn(lat_shape, dtype=dtype, device=device)
            x0_student = student.prepare(
                text_prompts=prompt, device=device, dtype=dtype,
                noise=noise, current_start=0, current_end=frame_seq * npb,
                batch_denoise=False)
            cond = student.conditional_dict

        # (b) student builds x_t + timestep + embeds, broadcasts to teacher & fake groups
        if in_student:
            b = x0_student.shape[0]
            t = torch.randint(20, 980, (b,), device=device)
            x_t = scheduler.add_noise(x0_student.flatten(0, 1),
                                      torch.randn_like(x0_student).flatten(0, 1),
                                      t).unflatten(0, x0_student.shape[:2])
            tt = t.view(b, 1).expand(b, x0_student.shape[1])
            embeds = cond["prompt_embeds"]
            # also send x0 (detached) so the fake group can train on it
            x0_send = x0_student.detach()
        else:
            x_t = zeros_lat(); tt = torch.zeros((1, npb), dtype=torch.int64, device=device)
            embeds = torch.zeros(emb_shape, dtype=dtype, device=device); x0_send = zeros_lat()
        x_t = bcast(x_t, ssrc); tt = bcast(tt.to(torch.int64), ssrc)
        embeds = bcast(embeds, ssrc); x0_send = bcast(x0_send, ssrc)
        condb = {"prompt_embeds": embeds}

        # (c) teacher scores x_t (its group), broadcasts real_pred back
        real_pred = zeros_lat()
        if in_teacher:
            with torch.no_grad():
                real_pred = score(real_score, x_t, tt, condb, teacher_cache)
        real_pred = bcast(real_pred, tsrc)

        # (d) fake scores x_t (its group), broadcasts fake_pred back
        fake_pred = zeros_lat()
        if in_fake:
            with torch.no_grad():
                fake_pred = score(fake_score, x_t, tt, condb, fake_cache)
        fake_pred = bcast(fake_pred, fsrc)

        # (e) student DMD update (its group)
        gl = float("nan")
        if in_student and (it % args.dfake_gen_update_ratio == 0):
            grad = (fake_pred - real_pred)
            grad = grad / (grad.abs().mean() + 1e-8)
            target = (x0_student - grad).detach()
            loss_g = 0.5 * F.mse_loss(x0_student, target)
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            gl = float(loss_g.detach())

        # (f) fake trains to track G, on x0_send (its group) — diffusion loss
        lf = float("nan")
        if in_fake:
            bb = x0_send.shape[0]
            tf = torch.randint(20, 980, (bb,), device=device)
            xtf = scheduler_fake.add_noise(x0_send.flatten(0, 1),
                                           torch.randn_like(x0_send).flatten(0, 1),
                                           tf).unflatten(0, x0_send.shape[:2])
            ttf = tf.view(bb, 1).expand(bb, x0_send.shape[1])
            pred_f = score(fake_score, xtf, ttf, condb, fake_cache)
            loss_f = F.mse_loss(pred_f, x0_send)
            opt_f.zero_grad(); loss_f.backward(); opt_f.step()
            lf = float(loss_f.detach())

        if (my_rank == ssrc or not dist.is_initialized()) and it % 20 == 0:
            LOGGER.info(f"it {it}/{args.iters}  loss_G={gl:.4f}  loss_fake={lf:.4f}")
        if it > 0 and it % args.save_every == 0:
            save_ckpt(it)

        # ── FREE the iteration's graph/tensors before the next iter allocates ──
        # OOM at iter 1 start was iter 0's autograd graph + latents lingering. Rebind
        # the big tensors to None (drops refs -> autograd graph freed) and clear the
        # student's KV/cross/shared caches so HBM is reclaimed before the next rollout.
        x0_student = x_t = real_pred = fake_pred = x0_send = None
        if in_student:
            student.kv_cache1 = None
            student.crossattn_cache = None
            student.shared_buffers = None
        import gc as _gc
        _gc.collect()

    save_ckpt(args.iters)
    LOGGER.info("done. VALIDATE: load the checkpoint into sd-job.yaml and WATCH the video "
                "(never trust training loss alone).")


def _gather_full_state_dict(model, tp_group, tp_degree):
    """Reassemble FULL (unsharded) weights from the TP-sharded model.
    Each sharded layer split one dim across ranks; all_gather + concat reverses it:
      ColumnParallelLinear.weight  -> split dim0 (out) -> gather+cat dim0
      RowParallelLinear.weight     -> split dim1 (in)  -> gather+cat dim1
      TPRMSNorm.weight             -> split dim0       -> gather+cat dim0
      everything else (replicated) -> identical on all ranks -> take local
    Returns a full state_dict on CPU (only meaningful on the group root)."""
    from models.wan.tp_utils import ColumnParallelLinear, RowParallelLinear, TPRMSNorm

    # map each parameter's full-name -> concat dim (or None = replicated)
    cat_dim = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, ColumnParallelLinear):
            cat_dim[f"{mod_name}.weight"] = 0
            if getattr(mod, "bias", None) is not None:
                cat_dim[f"{mod_name}.bias"] = 0
        elif isinstance(mod, RowParallelLinear):
            cat_dim[f"{mod_name}.weight"] = 1
            # row bias is full (added after all-reduce) -> replicated
        elif isinstance(mod, TPRMSNorm):
            cat_dim[f"{mod_name}.weight"] = 0

    full = {}
    for name, p in model.state_dict().items():
        d = cat_dim.get(name, None)
        loc = p.detach().to(torch.bfloat16)
        if d is None or tp_degree == 1:
            full[name] = loc.cpu()
            continue
        gathered = [torch.empty_like(loc) for _ in range(tp_degree)]
        if dist.is_initialized():
            dist.all_gather(gathered, loc.contiguous(), group=tp_group)
        else:
            gathered = [loc]
        full[name] = torch.cat(gathered, dim=d).cpu()
    return full




if __name__ == "__main__":
    main()
