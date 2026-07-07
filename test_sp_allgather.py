"""Minimal 16-rank test: dist.all_gather_into_tensor on the STRIDED SP group at our exact
merged-attention shapes. Isolates the 'NRT model scheduling failed, Invalid NEFF' error
from the full pipeline so we can find a tensor layout Neuron accepts — fast iteration.

Launch via torchrun --nproc_per_node=16. TP4xSP4 layout (global = sp_rank*4 + tp_rank):
  TP groups contiguous [0-3],[4-7],...  SP groups strided [0,4,8,12],[1,5,9,13],...
"""
import os
import torch
import torch.distributed as dist


def log(rank, *a):
    if rank in (0, 1, 4):  # a couple ranks in different groups
        print(f"[rank{rank}]", *a, flush=True)


def main():
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()
    tp = int(os.environ.get("TP_DEGREE", "4"))
    sp = world // tp
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    dev = "neuron"

    # DECISIVE TEST: is it the STRIDING? Build BOTH a strided group [tp_i::tp] and a
    # CONTIGUOUS group [sp_i*sp:(sp_i+1)*sp], all_gather on each, compare.
    strided_group = None
    for tp_i in range(tp):
        ranks = list(range(tp_i, world, tp))          # [0,4,8,12] ...
        g = dist.new_group(ranks)
        if rank in ranks:
            strided_group = g
    contig_group = None
    for i in range(tp):
        ranks = list(range(i * sp, (i + 1) * sp))     # [0,1,2,3],[4,5,6,7] ... (size sp)
        g = dist.new_group(ranks)
        if rank in ranks:
            contig_group = g
    log(rank, f"world={world} tp={tp} sp={sp}")

    for gname, grp in [("STRIDED", strided_group), ("CONTIG", contig_group)]:
        try:
            s_local, width = 1260, 384
            inp = torch.randn(s_local, width, dtype=torch.bfloat16, device=dev).contiguous()
            out = torch.empty(sp * s_local, width, dtype=torch.bfloat16, device=dev)
            dist.all_gather_into_tensor(out, inp, group=grp)
            torch.neuron.synchronize()
            log(rank, f"OK   {gname} all_gather")
        except Exception as e:
            log(rank, f"FAIL {gname}: {str(e)[:80]}")
    dist.destroy_process_group()
    return
    sp_group = strided_group; sp_rank = rank // tp  # (unreached; kept for old code below)

    # our merged-attention shapes: frame_seq=1680, window=3 frames (anchor phase) -> s_full
    # = 3*1680=5040, s_local = 5040/sp = 1260. head-sharded n*d: 3 heads * 128 = 384.
    frame_seq = 1680
    for (name, s_full, width) in [
        ("anchor_3f_384", 3 * frame_seq, 384),      # 3 local heads * 128
        ("full_15f_384", 15 * frame_seq, 384),
        ("anchor_3f_1536", 3 * frame_seq, 1536),    # full dim (if not head-sharded)
    ]:
        s_local = s_full // sp
        try:
            inp = torch.randn(s_local, width, dtype=torch.bfloat16, device=dev).contiguous()
            out = torch.empty(s_full, width, dtype=torch.bfloat16, device=dev)
            dist.all_gather_into_tensor(out, inp, group=sp_group)
            torch.neuron.synchronize()
            log(rank, f"OK   {name}: [{s_local},{width}] -> [{s_full},{width}]")
        except Exception as e:
            log(rank, f"FAIL {name}: {type(e).__name__}: {str(e)[:100]}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
