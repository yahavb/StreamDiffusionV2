"""Precompute UMT5 prompt embeddings for distillation — ONCE, on CPU, 1 process.

Removes T5 from the training job entirely: no 9.6GB T5 on device, no T5-broadcast
collective (which collides with the FSDP student's process groups). Training loads
embeds.pt and injects it (pipeline._distill_embeds), skipping T5 in prepare().

Usage:
  python3 distill/precompute_embeds.py \
    --captions /tmp/captions.jsonl --max_prompts 10 \
    --model_path wan_models/Wan2.1-T2V-1.3B --out /tmp/embeds.pt
"""
import argparse, json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "models", "wan", "wan_base"))

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", required=True)
    p.add_argument("--max_prompts", type=int, default=10)
    p.add_argument("--model_path", default="wan_models/Wan2.1-T2V-1.3B")
    p.add_argument("--out", default="/tmp/embeds.pt")
    args = p.parse_args()

    prompts = []
    with open(args.captions) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"precomputing embeds for {len(prompts)} prompts (CPU)")

    # bare UMT5 encoder on CPU (no neuron, no compile, no distributed)
    from modules.tokenizers import HuggingfaceTokenizer
    from modules.t5 import umt5_xxl
    tok = HuggingfaceTokenizer(
        name=os.path.join(args.model_path, "google/umt5-xxl/"), seq_len=512, clean="whitespace")
    enc = umt5_xxl(encoder_only=True, return_tokenizer=False,
                   dtype=torch.bfloat16, device=torch.device("cpu")).eval().requires_grad_(False)
    enc.load_state_dict(torch.load(
        os.path.join(args.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        map_location="cpu", weights_only=False))

    # CFG needs an UNCONDITIONAL (negative-prompt) embedding too. RollingForcing uses a
    # WAN negative prompt at guidance_scale=3.0 to sharpen the teacher's "real" direction
    # (the missing CFG was our blur cause). Encode it under a reserved key "__uncond__".
    RF_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
              "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
              "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
              "杂乱的背景，三条腿，背景人很多，倒着走")

    def _encode(text):
        ids, mask = tok([text], return_mask=True, add_special_tokens=True)
        seq_len = mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            ctx = enc(ids, mask).to(torch.bfloat16).contiguous()
        for b in range(ctx.shape[0]):
            ctx[b, seq_len[b]:] = 0.0
        return ctx[0:1].cpu()   # [1,512,4096]

    embeds = {}
    for i, pr in enumerate(prompts):
        embeds[pr] = _encode(pr)
        print(f"  [{i+1}/{len(prompts)}] {pr[:50]}...")
    embeds["__uncond__"] = _encode(RF_NEG)   # negative-prompt embed for CFG
    print("  [uncond] encoded RF negative prompt for CFG")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(embeds, args.out)
    print(f"wrote {len(embeds)} embeds -> {args.out}")


if __name__ == "__main__":
    main()
