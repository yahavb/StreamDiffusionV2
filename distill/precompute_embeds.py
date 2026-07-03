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

    embeds = {}
    for i, pr in enumerate(prompts):
        ids, mask = tok([pr], return_mask=True, add_special_tokens=True)
        seq_len = mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            ctx = enc(ids, mask).to(torch.bfloat16).contiguous()
        for b in range(ctx.shape[0]):
            ctx[b, seq_len[b]:] = 0.0
        embeds[pr] = ctx[0:1].cpu()   # [1,512,4096]
        print(f"  [{i+1}/{len(prompts)}] {pr[:50]}...")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(embeds, args.out)
    print(f"wrote {len(embeds)} embeds -> {args.out}")


if __name__ == "__main__":
    main()
