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
import gc as _gc
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
    p.add_argument("--lr", type=float, default=2e-5)          # G lr. 2e-6 was ~50x too small
                                                              # (AdamW step~=lr; student couldn't move -> flat gap)
    p.add_argument("--dfake_gen_update_ratio", type=int, default=5)  # fake_score updates per G update (critic leads)
    p.add_argument("--warmup", type=int, default=25,
                   help="freeze G for first N iters so the critic (fake_score) tracks the "
                        "student before we take DMD gradient steps (stabilizes early training)")
    p.add_argument("--tp_degree", type=int, default=8)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=352)   # /8=44 even (patchify needs /2); not 240/480
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--max_prompts", type=int, default=0,
                   help="cap the corpus to the first N prompts (0 = all). For targeted "
                        "small-set distillation (e.g. 10 prompts, memorize them).")
    p.add_argument("--embeds", default=None,
                   help="precomputed prompt->embeds .pt (from precompute_embeds.py). If set, "
                        "T5 is NOT built/broadcast in training (avoids FSDP group collision).")
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
    """Build SDV2's NEURON streaming pipeline for the STUDENT (G), sharded with FSDP2
    (NOT our custom tensor-parallel). FSDP2 shards params+grads+OPTIMIZER STATE across
    the student group and, with per-block activation checkpointing, is the proven
    Neuron path for full-model diffusion training (ref: HunyuanVideo 8.33B DiT on 4
    cores). This fixes the 22.7GB resident-tensor OOM that custom TP (weights-only
    sharding) could not.

    Build the DiT UNSHARDED (tp_degree=1 -> plain nn.Linear blocks), then fully_shard
    each transformer block + the root. Custom TP is bypassed for the student."""
    from models.wan.neuron_causal_stream_inference import NeuronCausalStreamInferencePipeline
    from models.wan.neuron_layers import CausalWanAttentionBlock
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing)
    from models.wan.tp_utils import get_tp_group

    # force UNSHARDED build (FSDP will shard instead of our custom TP)
    saved_tp = getattr(args, "tp_degree", 1)
    args.tp_degree = 1
    pipe = NeuronCausalStreamInferencePipeline(args, device=device)
    args.tp_degree = saved_tp
    m = pipe.generator.model
    m.train().requires_grad_(True)

    # per-block NO_REENTRANT activation checkpointing (mandatory — OOMs without it)
    apply_activation_checkpointing(
        m,
        checkpoint_wrapper_fn=lambda mod: checkpoint_wrapper(mod, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda mod: isinstance(mod, CausalWanAttentionBlock),
    )
    # FSDP2: shard across the STUDENT group ONLY (ranks 4-7), not the whole world.
    # fully_shard needs a DeviceMesh over exactly those ranks. Build a mesh from the
    # student rank list (from tp_utils group base/size).
    # Mesh pattern from private-torch-neuronx run_benchmarks.py: 2D (dp, shard) mesh,
    # slice mesh["shard"] -> each rank's own group submesh. n = GROUP size (the 3-group
    # placement = tp arg), NOT get_tp_world_size() (student built tp=1 so that's 1).
    from torch.distributed.device_mesh import init_device_mesh
    n = int(os.environ.get("TP_DEGREE", "4"))   # ranks per placement group (4)
    mesh = init_device_mesh("neuron", (dist.get_world_size() // n, n),
                            mesh_dim_names=("dp", "shard"))
    local_mesh = mesh["shard"]   # this rank's size-n shard submesh (its group)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in m.blocks:
        fully_shard(blk, mesh=local_mesh, mp_policy=mp, reshard_after_forward=True)
    fully_shard(m, mesh=local_mesh, mp_policy=mp, reshard_after_forward=True)
    LOGGER.info(f"student: FSDP2 per-block+root (group size {n}), "
                f"{len(m.blocks)} blocks, NO_REENTRANT ckpt")
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
    if args.max_prompts and args.max_prompts < len(prompts):
        prompts = prompts[:args.max_prompts]
        LOGGER.info(f"capped to first {len(prompts)} prompts (targeted distill)")

    # precomputed prompt embeds (no T5 in training) — dict prompt->[1,512,4096] on CPU
    embeds_cache = None
    if args.embeds and os.path.exists(args.embeds):
        embeds_cache = torch.load(args.embeds, map_location="cpu")
        LOGGER.info(f"loaded {len(embeds_cache)} precomputed embeds from {args.embeds}")

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
        student = build_student_pipeline(args, device, dtype)  # FSDP2-sharded + ckpt inside
        G = student.generator
        if embeds_cache is not None:
            # inject precomputed embeds -> prepare() skips T5 + its broadcast (no FSDP collision)
            student._distill_embeds = {k: v.to(device) for k, v in embeds_cache.items()}
            LOGGER.info("injected precomputed embeds into student (T5-free rollout)")
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
        """FSDP2 full-state-dict gather (student is FSDP-sharded now, not custom-TP).
        get_model_state_dict(full_state_dict=True) all-gathers the DTensor shards to a
        full CPU state_dict on rank0-of-the-FSDP-group. ALL student ranks must call
        (it's collective). Then strip compile infixes + write {'generator': model.*}."""
        if not in_student:
            # non-student ranks: still hit the global barrier so nobody deadlocks
            if dist.is_initialized():
                dist.barrier()
            return
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict, StateDictOptions)
        full = get_model_state_dict(
            G.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        # DIAG: get_model_state_dict returned 0 tensors in run 41 (empty ckpt). Log the
        # count on EVERY student rank so we see whether the gather lands on ssrc only,
        # all ranks, or nowhere. Cheap; runs once per save.
        LOGGER.info(f"[ckpt-diag] rank {my_rank}: get_model_state_dict -> {len(full)} entries")
        # FALLBACK: if the DCP gather came back empty, gather the raw (possibly DTensor)
        # state_dict and materialize any DTensor shards to full tensors by hand.
        if len(full) == 0:
            from torch.distributed.tensor import DTensor
            raw = G.model.state_dict()
            full = {}
            for k, v in raw.items():
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                full[k] = v.detach().to("cpu")
            LOGGER.info(f"[ckpt-diag] rank {my_rank}: FALLBACK raw state_dict -> {len(full)} entries")
        if dist.is_initialized():
            dist.barrier()
        if my_rank == ssrc:
            def _clean(k):
                return (k.replace(".compiled_module.", ".").replace("._orig_mod.", ".")
                         .replace(".compiled_module", "").replace("._orig_mod", "")
                         .replace("._checkpoint_wrapped_module.", ".")
                         .replace("._checkpoint_wrapped_module", ""))
            sd = {f"model.{_clean(k)}": v.to(torch.bfloat16) for k, v in full.items()}
            assert len(sd) > 0, "EMPTY checkpoint — get_model_state_dict AND fallback both returned 0 tensors"
            torch.save({"generator": sd, "distill_iter": it}, args.out)
            LOGGER.info(f"[ckpt] wrote {args.out} (iter {it}) full={len(sd)} tensors — drop-in for sd-job")

    def zeros_lat(): return torch.zeros(lat_shape, dtype=dtype, device=device)

    # ── ALLOCATION TRACER: report device-tensor count+MB per STAGE, so we see WHICH
    # stage grows and whether cleanup reclaims it. Only rank ssrc logs (avoid spam).
    def _dev_stats():
        n = mb = 0
        for o in _gc.get_objects():
            try:
                if torch.is_tensor(o) and o.device.type == "neuron":
                    n += 1; mb += o.numel() * o.element_size() / 1e6
            except Exception:
                pass
        return n, mb
    _trace_on = os.environ.get("DISTILL_TRACE", "").lower() in ("1", "true")
    _prev = {"n": 0, "mb": 0.0}
    def _stage(it, name):
        if not _trace_on or not (my_rank == ssrc or not dist.is_initialized()):
            return
        n, mb = _dev_stats()
        dn, dmb = n - _prev["n"], mb - _prev["mb"]
        LOGGER.info(f"  [trace it{it}] {name:22s} dev_tensors={n:5d} ({dn:+d})  "
                    f"dev_MB={mb:8.0f} ({dmb:+.0f})")
        _prev["n"], _prev["mb"] = n, mb

    # Fixed DMD timestep buckets (device tensor). Sampling from these instead of
    # arbitrary randint keeps the compiled graph/NEFF set BOUNDED (~8), so NEFFs get
    # reused across iters instead of a new module.neff per iter (the iter-12 OOM).
    _DMD_TIMESTEPS = torch.tensor([100, 250, 400, 500, 600, 700, 800, 900],
                                  dtype=torch.long, device=device)

    # 3) training loop — ALL groups hit EVERY broadcast in the same order (no deadlock)
    _gl_hist = []  # recent loss_G values for the running-average convergence metric
    _gap_hist = []  # recent RAW ||fake-real|| gap = the real convergence signal
    for it in range(args.iters):
        _stage(it, "iter-start")
        prompt = [prompts[it % len(prompts)]]

        # ── STRUCTURAL graph-free DMD (fixes the 22.7GB resident autograd graph) ──
        # OLD: x0_student's full 30-block rollout GRAPH was held alive from (a) through
        # the two broadcasts + teacher/fake scoring to (e) — 22.7GB resident, OOM iter12.
        # NEW: (a) computes x0 under NO_GRAD (detached, cheap, no graph). Teacher/fake
        # score that detached x0. Then (e) RECOMPUTES the student forward WITH grad
        # right before backward, so the graph lives only for the single backward and is
        # freed immediately. Same fixed noise/timestep is reused so recompute matches.

        # (a) student generates x0 — NO GRAD (detached rollout, no retained graph)
        x0_det = cond = None
        _rollout_noise = _rollout_t = None
        if in_student:
            student.kv_cache1 = None
            student.crossattn_cache = None
            _rollout_noise = torch.randn(lat_shape, dtype=dtype, device=device)
            with torch.no_grad():
                x0_det = student.prepare(
                    text_prompts=prompt, device=device, dtype=dtype,
                    noise=_rollout_noise, current_start=0, current_end=frame_seq * npb,
                    batch_denoise=False).detach()
            cond = student.conditional_dict
        _stage(it, "a:student-rollout(no_grad)")

        # (b) build x_t + timestep from the DETACHED x0; broadcast to teacher & fake
        if in_student:
            b = x0_det.shape[0]
            _rollout_t = _DMD_TIMESTEPS[torch.randint(0, len(_DMD_TIMESTEPS), (b,), device=device)]
            _noise_t = torch.randn_like(x0_det)
            x_t = scheduler.add_noise(x0_det.flatten(0, 1), _noise_t.flatten(0, 1),
                                      _rollout_t).unflatten(0, x0_det.shape[:2])
            tt = _rollout_t.view(b, 1).expand(b, x0_det.shape[1])
            embeds = cond["prompt_embeds"]
            x0_send = x0_det  # already detached
        else:
            x_t = zeros_lat(); tt = torch.zeros((1, npb), dtype=torch.int64, device=device)
            embeds = torch.zeros(emb_shape, dtype=dtype, device=device); x0_send = zeros_lat()
        x_t = bcast(x_t, ssrc); tt = bcast(tt.to(torch.int64), ssrc)
        embeds = bcast(embeds, ssrc); x0_send = bcast(x0_send, ssrc)
        condb = {"prompt_embeds": embeds}

        # (c) teacher scores x_t (no student graph alive), broadcasts real_pred back
        real_pred = zeros_lat()
        if in_teacher:
            with torch.no_grad():
                real_pred = score(real_score, x_t, tt, condb, teacher_cache)
        real_pred = bcast(real_pred, tsrc)
        _stage(it, "c:teacher-score")

        # (d) fake scores x_t, broadcasts fake_pred back
        fake_pred = zeros_lat()
        if in_fake:
            with torch.no_grad():
                fake_pred = score(fake_score, x_t, tt, condb, fake_cache)
        fake_pred = bcast(fake_pred, fsrc)
        _stage(it, "d:fake-score")

        # (e) student DMD update — RECOMPUTE the forward WITH grad now (fresh graph,
        # freed right after backward). DMD gradient = (fake-real) applied to x0.
        gl = float("nan")
        dmd_gap = float("nan")
        # WARMUP: for the first args.warmup iters, do NOT step G — let fake_score (the
        # critic) learn to track the still-base student first. A DMD gradient from an
        # untrained critic points nowhere; stepping G on it wastes the early budget.
        # After warmup, G updates once every dfake_gen_update_ratio iters (critic leads).
        _do_g = (it >= args.warmup) and (it % args.dfake_gen_update_ratio == 0)
        # always compute the raw gap (cheap, needs both preds) so we log convergence
        if in_student:
            dmd_gap = float((fake_pred - real_pred).abs().mean().detach())
        if in_student and _do_g:
            student.kv_cache1 = None
            student.crossattn_cache = None
            x0_grad = student.prepare(          # WITH grad this time
                text_prompts=prompt, device=device, dtype=dtype,
                noise=_rollout_noise, current_start=0, current_end=frame_seq * npb,
                batch_denoise=False)
            grad = (fake_pred - real_pred)
            grad = grad / (grad.abs().mean() + 1e-8)
            target = (x0_grad - grad).detach()
            loss_g = 0.5 * F.mse_loss(x0_grad, target)
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
            gl = float(loss_g.detach())
            del grad, target, x0_grad, loss_g   # free the fresh graph immediately
        _stage(it, "e:student-backward")

        # (f) fake trains to track G, on x0_send (its group) — diffusion loss
        lf = float("nan")
        if in_fake:
            bb = x0_send.shape[0]
            tf = _DMD_TIMESTEPS[torch.randint(0, len(_DMD_TIMESTEPS), (bb,), device=device)]
            xtf = scheduler_fake.add_noise(x0_send.flatten(0, 1),
                                           torch.randn_like(x0_send).flatten(0, 1),
                                           tf).unflatten(0, x0_send.shape[:2])
            ttf = tf.view(bb, 1).expand(bb, x0_send.shape[1])
            pred_f = score(fake_score, xtf, ttf, condb, fake_cache)
            loss_f = F.mse_loss(pred_f, x0_send)
            opt_f.zero_grad(set_to_none=True); loss_f.backward(); opt_f.step()
            lf = float(loss_f.detach())
            del pred_f, loss_f, xtf  # drop fake-group autograd graph (was leaking -> OOM)
        _stage(it, "f:fake-backward")

        # CONVERGENCE METRIC = loss_G (DMD generator loss). It should DECREASE and
        # FLATTEN — that's convergence (student distribution -> teacher). Per-iter is
        # noisy (random timestep each step), so also print a running average over the
        # last 50 student updates to see the trend. (loss_fake is the fake-group's
        # metric; on the student-root log it's nan — ignore it here.)
        if in_student and gl == gl:  # gl==gl false when nan (non-G-update iters)
            _gl_hist.append(gl)
            if len(_gl_hist) > 50:
                _gl_hist.pop(0)
        if in_student and dmd_gap == dmd_gap:
            _gap_hist.append(dmd_gap)
            if len(_gap_hist) > 50:
                _gap_hist.pop(0)
        # MEM PROBE: count live device tensors + their MB each iter, so we can SEE
        # what grows (leak vs NEFF-cache) instead of guessing. Cheap gc walk.
        _dev_n = _dev_mb = 0
        for _o in _gc.get_objects() if (it % 2 == 0) else []:
            try:
                if torch.is_tensor(_o) and _o.device.type == "neuron":
                    _dev_n += 1; _dev_mb += _o.numel() * _o.element_size() / 1e6
            except Exception:
                pass
        if my_rank == ssrc or not dist.is_initialized():
            avg = sum(_gl_hist) / len(_gl_hist) if _gl_hist else float("nan")
            gap_avg = sum(_gap_hist) / len(_gap_hist) if _gap_hist else float("nan")
            # magnitudes of each arrow — if real~=fake from iter 0 the setup is degenerate;
            # if real stays fixed while fake drifts, the critic IS tracking the student.
            _rm = float(real_pred.abs().mean()); _fm = float(fake_pred.abs().mean())
            _phase = "warmup" if it < args.warmup else ("G-step" if _do_g else "critic-only")
            LOGGER.info(f"it {it}/{args.iters} [{_phase}] loss_G={gl:.4f} loss_G_avg50={avg:.4f}  "
                        f"DMDgap={dmd_gap:.4f} DMDgap_avg50={gap_avg:.4f}  "
                        f"|real|={_rm:.3f} |fake|={_fm:.3f}  "
                        f"MEM[dev_MB={_dev_mb:.0f}]  (watch DMDgap_avg50 DECREASE)")
        if in_fake and my_rank == fsrc:
            LOGGER.info(f"it {it}/{args.iters}  loss_fake={lf:.4f}  (fake tracks student; should stay low/steady)")
        if it > 0 and it % args.save_every == 0:
            save_ckpt(it)

        # ── FREE the iteration's graph/tensors before the next iter allocates ──
        # OOM at iter 1 start was iter 0's autograd graph + latents lingering. Rebind
        # the big tensors to None (drops refs -> autograd graph freed) and clear the
        # student's KV/cross/shared caches so HBM is reclaimed before the next rollout.
        # Free per-iter tensors (autograd graphs) across ALL groups.
        x0_det = x_t = real_pred = fake_pred = x0_send = None
        if in_student:
            student.kv_cache1 = None
            student.crossattn_cache = None
            student.shared_buffers = None
        _gc.collect()
        # Force Neuron to reclaim HBM now — eager torch_neuronx defers frees, so
        # del/gc alone let device memory creep up across iters (OOM ~iter 12). A
        # synchronize flushes pending frees before the next iter allocates.
        if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
            try:
                torch.neuron.synchronize()
            except Exception:
                pass
        _stage(it, "z:after-cleanup")  # if dev_MB here climbs each iter -> that's the leak

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
