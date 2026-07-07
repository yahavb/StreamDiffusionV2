"""Tensor Parallelism utilities for Wan2.1-T2V-14B on Trainium.

Implements column-parallel and row-parallel linear layers for sharding
the DiT model across 4 NeuronCores. All 4 cores run DiT in TP mode
while TE and VAE are replicated on each rank.

TP Strategy:
  - Q/K/V projections → ColumnParallelLinear (split output dim by heads)
  - O projection → RowParallelLinear (split input dim, all-reduce output)
  - FFN fc1/up → ColumnParallelLinear (split hidden dim)
  - FFN fc2/down → RowParallelLinear (split input dim, all-reduce output)
  - Norms, embeddings, modulation → Replicated (small, needed for correctness)
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Process group management
# ---------------------------------------------------------------------------

_TP_GROUP: Optional[dist.ProcessGroup] = None
_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1
# Data-parallel-within-pod: when world_size > tp_degree, the cores split into
# world_size//tp_degree independent TP groups, each rendering its OWN stream.
_DP_GROUP_ID: int = 0          # which DP group THIS rank is in (0..num-1)
_DP_NUM_GROUPS: int = 1        # world_size // tp_degree
_TP_GROUP_BASE: int = 0        # global rank of this group's local-rank-0

# Sequence parallelism (for RF rolling-window merged attention). TP4xSP4=16 ranks:
#   TP groups CONTIGUOUS (ranks 0-3, 4-7, 8-11, 12-15) — head sharding (existing).
#   SP groups STRIDED stride=tp_degree (ranks 0,4,8,12 / 1,5,9,13 / ...) — sequence
#   sharding. A rank's global id = sp_rank * tp_degree + tp_rank.
_SP_GROUP: Optional[dist.ProcessGroup] = None
_SP_RANK: int = 0
_SP_WORLD_SIZE: int = 1
_WORLD_GROUP: Optional[dist.ProcessGroup] = None   # the full TPxSP world (one stream)


def init_tp_group(tp_degree: int = 4):
    """Initialize tensor parallelism process group(s).

    Must be called after torch.distributed.init_process_group().
    When world_size == tp_degree there is ONE TP group (the original behavior).
    When world_size > tp_degree (in-pod data-parallel), the ranks split into
    world_size//tp_degree contiguous TP groups, each an INDEPENDENT stream:
    e.g. tp=8 on 16 cores -> group 0 = ranks[0..7], group 1 = ranks[8..15].
    All-reduce uses the per-group _TP_GROUP, so each stream's DiT math is
    self-contained. T5/VAE and prompt/latent broadcasts must be done relative
    to _TP_GROUP_BASE (see get_dp_*), not global rank 0/2.

    Args:
        tp_degree: ranks per TP group (4 = one trn2 chip's worth at LNC2).
    """
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE
    global _DP_GROUP_ID, _DP_NUM_GROUPS, _TP_GROUP_BASE

    if not dist.is_initialized():
        # Single-process fallback for testing
        _TP_RANK = 0
        _TP_WORLD_SIZE = 1
        _TP_GROUP = None
        print(f"[TP] Running in single-process mode (no distributed)")
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % tp_degree == 0, (
        f"World size {world_size} must be divisible by tp_degree {tp_degree}")

    # Create TP groups: ranks [0,1,2,3], [4,5,6,7], etc. ALL ranks must call
    # new_group for EVERY group (collective requirement), even groups they're
    # not in. Each rank then keeps the group it belongs to.
    num_groups = world_size // tp_degree
    for i in range(num_groups):
        ranks = list(range(i * tp_degree, (i + 1) * tp_degree))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TP_GROUP = group
            _TP_RANK = rank - i * tp_degree
            _TP_WORLD_SIZE = tp_degree
            _DP_GROUP_ID = i
            _DP_NUM_GROUPS = num_groups
            _TP_GROUP_BASE = i * tp_degree

    print(f"[TP] Initialized: tp_rank={_TP_RANK}/{_TP_WORLD_SIZE}, "
          f"dp_group={_DP_GROUP_ID}/{_DP_NUM_GROUPS}, base={_TP_GROUP_BASE}, "
          f"global_rank={rank}/{world_size}")


def init_sp_groups(tp_degree: int = 4, sp_degree: int = 4):
    """Initialize TP x SP process groups for RF-style merged attention (one stream).

    Layout (RF-faithful, dit_pipeline.init_parallel_groups):
      global_rank = sp_rank * tp_degree + tp_rank
      TP groups CONTIGUOUS: [0..tp-1], [tp..2tp-1], ...  (head sharding)
      SP groups STRIDED stride=tp_degree: [0,tp,2tp,...], [1,...], ...  (seq sharding)
    Requires world_size == tp_degree * sp_degree (single 16-rank stream; no DP here).
    ALL ranks must call new_group for EVERY group (collective requirement).
    """
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE, _TP_GROUP_BASE
    global _SP_GROUP, _SP_RANK, _SP_WORLD_SIZE, _WORLD_GROUP
    global _DP_GROUP_ID, _DP_NUM_GROUPS

    if not dist.is_initialized():
        _TP_RANK = _SP_RANK = 0
        _TP_WORLD_SIZE = _SP_WORLD_SIZE = 1
        _TP_GROUP = _SP_GROUP = _WORLD_GROUP = None
        print("[SP] single-process mode (no distributed)")
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size == tp_degree * sp_degree, (
        f"SP mode needs world_size({world_size}) == tp_degree({tp_degree})*"
        f"sp_degree({sp_degree})")

    _WORLD_GROUP = dist.group.WORLD
    _DP_GROUP_ID, _DP_NUM_GROUPS = 0, 1

    # LAYOUT (Neuron requires CONTIGUOUS collective groups — strided [0,4,8,12] fails ENC
    # 'no_hier no_mesh', proven by isolation test). So:
    #   SP groups CONTIGUOUS: [0,1,2,3],[4,5,6,7],...  (sequence all_gather lives here)
    #   TP groups STRIDED   : [0,4,8,12],[1,5,9,13],...  (head shard / o all-reduce)
    # global_rank = tp_rank * sp_degree + sp_rank.  (was the reverse.)
    # NOTE: TP's all_reduce is also a collective on a strided group — the RowParallel o
    # all-reduce must therefore ALSO work on strided; if ENC rejects it too, TP must stay
    # contiguous and SP strided is impossible -> would need a different SP collective. But
    # all_reduce (vs all_gather) may have hierarchical support; validated at runtime.
    sp_rank = rank % sp_degree
    tp_rank = rank // sp_degree
    _TP_GROUP_BASE = (rank // sp_degree) * sp_degree   # base of this rank's contiguous SP block

    # SP groups: CONTIGUOUS blocks of sp_degree
    for tp_i in range(tp_degree):
        ranks = list(range(tp_i * sp_degree, (tp_i + 1) * sp_degree))
        g = dist.new_group(ranks)
        if rank in ranks:
            _SP_GROUP = g
            _SP_RANK = rank % sp_degree
            _SP_WORLD_SIZE = sp_degree
    # TP groups: STRIDED stride=sp_degree
    for sp_i in range(sp_degree):
        ranks = list(range(sp_i, world_size, sp_degree))
        g = dist.new_group(ranks)
        if rank in ranks:
            _TP_GROUP = g
            _TP_RANK = rank // sp_degree
            _TP_WORLD_SIZE = tp_degree

    print(f"[SP] tp_rank={_TP_RANK}/{_TP_WORLD_SIZE} sp_rank={_SP_RANK}/{_SP_WORLD_SIZE} "
          f"global={rank}/{world_size} (SP contiguous, TP strided)")


def get_sp_group() -> Optional[dist.ProcessGroup]:
    return _SP_GROUP


def get_sp_rank() -> int:
    return _SP_RANK


def get_sp_world_size() -> int:
    return _SP_WORLD_SIZE


def get_world_group() -> Optional[dist.ProcessGroup]:
    return _WORLD_GROUP


def get_tp_group() -> Optional[dist.ProcessGroup]:
    """Get the tensor parallel process group."""
    return _TP_GROUP


def get_tp_rank() -> int:
    """Get the local TP rank (0 to tp_degree-1)."""
    return _TP_RANK


def get_tp_world_size() -> int:
    """Get the TP world size (tp_degree)."""
    return _TP_WORLD_SIZE


def get_dp_group_id() -> int:
    """Which data-parallel group (independent stream) this rank belongs to."""
    return _DP_GROUP_ID


def get_dp_num_groups() -> int:
    """Number of independent DP groups in this pod (world_size // tp_degree)."""
    return _DP_NUM_GROUPS


def get_tp_group_base() -> int:
    """Global rank of this DP group's local-rank-0 (the group's T5/VAE/broadcast root)."""
    return _TP_GROUP_BASE


# ---------------------------------------------------------------------------
# All-reduce communication
# ---------------------------------------------------------------------------

# Maximum bytes per all-reduce call. The Neuron NRT rejects certain large
# payload sizes for multi-rank groups (e.g. 59,904,000 bytes fails for TP=8).
# Chunking to ≤8MB per call avoids hitting unsupported size/topology combos.
_MAX_ALLREDUCE_BYTES = int(os.environ.get("MAX_ALLREDUCE_BYTES", 8 * 1024 * 1024))


@torch.compiler.disable
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """All-reduce (sum) across TP group with chunking for NRT size limits.

    Decorated with @torch.compiler.disable to force a graph break — the
    compiled FFN NEFF handles only matmuls, then all-reduce runs in eager.

    The Neuron NRT rejects certain all-reduce payload sizes for TP=8.
    We chunk large tensors into ≤8MB pieces to stay within limits.
    """
    if _TP_WORLD_SIZE <= 1:
        return x

    elem_size = x.element_size()  # 2 for bf16, 4 for fp32
    total_bytes = x.numel() * elem_size

    if total_bytes <= _MAX_ALLREDUCE_BYTES:
        # Small tensor — single all-reduce
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_TP_GROUP)
    else:
        # Large tensor — chunk to stay within NRT limits
        max_elements = _MAX_ALLREDUCE_BYTES // elem_size
        flat = x.view(-1)
        numel = flat.numel()
        for start in range(0, numel, max_elements):
            end = min(start + max_elements, numel)
            chunk = flat[start:end]
            dist.all_reduce(chunk, op=dist.ReduceOp.SUM, group=_TP_GROUP)

    return x


# ---------------------------------------------------------------------------
# TP-aware RMSNorm (for QK norms that must compute global RMS across ranks)
# ---------------------------------------------------------------------------

class TPRMSNorm(nn.Module):
    """RMSNorm that computes global RMS across all TP ranks.

    Standard RMSNorm computes: x * rsqrt(mean(x², dim=-1) + eps) * weight
    With TP, each rank only sees dim//tp_degree features. Computing mean(x²)
    locally gives a WRONG normalization factor because each rank's heads have
    different magnitudes.

    This version all-reduces sum(x²) across ranks before computing rsqrt,
    giving the mathematically correct global RMS normalization.
    """

    def __init__(self, local_dim: int, global_dim: int, eps: float = 1e-5):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(local_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        # Local sum of squares: [*, 1]
        local_sum_sq = x_float.pow(2).sum(dim=-1, keepdim=True)
        # All-reduce to get global sum of squares across all TP ranks
        global_sum_sq = all_reduce_sum(local_sum_sq.clone())
        # Global RMS: sqrt(sum_sq / global_dim)
        rms_inv = torch.rsqrt(global_sum_sq / self.global_dim + self.eps)
        return (x_float * rms_inv).type_as(x) * self.weight

    def extra_repr(self):
        return (f'local_dim={self.local_dim}, global_dim={self.global_dim}, '
                f'eps={self.eps}')


# ---------------------------------------------------------------------------
# Parallel Linear layers
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension sharded across TP ranks.

    Each rank holds weight of shape [out_features // tp_degree, in_features].
    No communication in forward pass — output is a local shard.

    Used for: Q, K, V projections (split by heads), FFN fc1/up projection.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 tp_degree: int = 4):
        super().__init__()
        assert out_features % tp_degree == 0, (
            f"out_features {out_features} must be divisible by tp_degree {tp_degree}")

        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // tp_degree
        self.tp_degree = tp_degree

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features} '
                f'(local={self.out_features_per_rank}), '
                f'bias={self.bias is not None}, tp={self.tp_degree}')


class RowParallelLinear(nn.Module):
    """Linear layer with input dimension sharded across TP ranks.

    Each rank holds weight of shape [out_features, in_features // tp_degree].
    Forward pass performs matmul then all-reduce to get the full output.

    Used for: O projections, FFN fc2/down projection.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 tp_degree: int = 4):
        super().__init__()
        assert in_features % tp_degree == 0, (
            f"in_features {in_features} must be divisible by tp_degree {tp_degree}")

        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_rank = in_features // tp_degree
        self.tp_degree = tp_degree

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank))
        if bias:
            # Only rank 0 adds bias to avoid double-counting after all-reduce
            # Actually, we add bias on all ranks and scale — simpler: only add
            # bias after all-reduce. Store full bias but only apply on one rank.
            # Simplest correct approach: store bias, add after all-reduce.
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul (no bias yet)
        out = nn.functional.linear(x, self.weight, None)
        # All-reduce across TP ranks
        out = all_reduce_sum(out)
        # Add bias after all-reduce (only one copy of bias needed)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return (f'in_features={self.in_features} '
                f'(local={self.in_features_per_rank}), '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, tp={self.tp_degree}')


# ---------------------------------------------------------------------------
# Weight sharding utility
# ---------------------------------------------------------------------------

def shard_linear_column(linear: nn.Linear, tp_rank: int, tp_degree: int
                        ) -> ColumnParallelLinear:
    """Convert nn.Linear to ColumnParallelLinear by slicing weights.

    Splits output dimension: each rank gets rows [rank*chunk : (rank+1)*chunk].

    Args:
        linear: Original full linear layer.
        tp_rank: This rank's index (0 to tp_degree-1).
        tp_degree: Total number of TP ranks.

    Returns:
        ColumnParallelLinear with sharded weights.
    """
    out_features = linear.out_features
    in_features = linear.in_features
    chunk_size = out_features // tp_degree

    col_linear = ColumnParallelLinear(
        in_features, out_features,
        bias=(linear.bias is not None),
        tp_degree=tp_degree)

    # Shard weight: [out_features, in_features] → [chunk_size, in_features]
    start = tp_rank * chunk_size
    end = start + chunk_size
    col_linear.weight = nn.Parameter(
        linear.weight.data[start:end].contiguous())

    if linear.bias is not None:
        col_linear.bias = nn.Parameter(
            linear.bias.data[start:end].contiguous())

    return col_linear


def shard_linear_row(linear: nn.Linear, tp_rank: int, tp_degree: int
                     ) -> RowParallelLinear:
    """Convert nn.Linear to RowParallelLinear by slicing weights.

    Splits input dimension: each rank gets columns [rank*chunk : (rank+1)*chunk].

    Args:
        linear: Original full linear layer.
        tp_rank: This rank's index (0 to tp_degree-1).
        tp_degree: Total number of TP ranks.

    Returns:
        RowParallelLinear with sharded weights.
    """
    out_features = linear.out_features
    in_features = linear.in_features
    chunk_size = in_features // tp_degree

    row_linear = RowParallelLinear(
        in_features, out_features,
        bias=(linear.bias is not None),
        tp_degree=tp_degree)

    # Shard weight: [out_features, in_features] → [out_features, chunk_size]
    start = tp_rank * chunk_size
    end = start + chunk_size
    row_linear.weight = nn.Parameter(
        linear.weight.data[:, start:end].contiguous())

    if linear.bias is not None:
        # Bias is full-sized, applied after all-reduce
        row_linear.bias = nn.Parameter(linear.bias.data.clone())

    return row_linear


def shard_qkv_norm(norm: nn.Module, tp_rank: int, tp_degree: int) -> nn.Module:
    """Shard RMSNorm weight for Q/K norms using TP-aware global RMS.

    QK norms in Wan have weight shape [dim]. After column-parallel split of Q/K,
    each rank holds [dim // tp_degree] features. The RMS must still be computed
    over the FULL dim features (requiring all-reduce of sum-of-squares) to match
    the non-TP reference model's normalization behavior.

    Uses TPRMSNorm which all-reduces sum(x²) before computing rsqrt.
    """
    if not hasattr(norm, 'weight'):
        return norm

    global_dim = norm.weight.shape[0]
    local_dim = global_dim // tp_degree
    start = tp_rank * local_dim
    end = start + local_dim

    # Create TP-aware norm that computes global RMS via all-reduce
    new_norm = TPRMSNorm(local_dim, global_dim, eps=norm.eps)
    new_norm.weight = nn.Parameter(norm.weight.data[start:end].contiguous())
    return new_norm


def shard_model_tp(model, tp_rank: int, tp_degree: int):
    """Apply tensor parallelism sharding to a CausalWanModel in-place.

    Shards:
      - Self-attention Q/K/V → column-parallel (split heads)
      - Self-attention O → row-parallel
      - Self-attention QK norms → sharded to match local head count
      - Cross-attention Q/K/V → column-parallel (split heads)
      - Cross-attention O → row-parallel
      - Cross-attention QK norms → sharded to match local head count
      - FFN fc1 → column-parallel
      - FFN fc2 → row-parallel

    Args:
        model: CausalWanModel instance with full (unsharded) weights loaded.
        tp_rank: This rank's local TP index (0 to tp_degree-1).
        tp_degree: Number of TP ranks.
    """
    num_heads = model.num_heads
    assert num_heads % tp_degree == 0, (
        f"num_heads {num_heads} must be divisible by tp_degree {tp_degree}")
    heads_per_rank = num_heads // tp_degree

    for block_idx, block in enumerate(model.blocks):
        # --- Self-Attention ---
        self_attn = block.self_attn

        # Q, K, V: column-parallel (split output dim = split heads)
        self_attn.q = shard_linear_column(self_attn.q, tp_rank, tp_degree)
        self_attn.k = shard_linear_column(self_attn.k, tp_rank, tp_degree)
        self_attn.v = shard_linear_column(self_attn.v, tp_rank, tp_degree)

        # O: row-parallel (split input dim = each rank has local heads)
        self_attn.o = shard_linear_row(self_attn.o, tp_rank, tp_degree)

        # QK norms: shard to match local head count
        self_attn.norm_q = shard_qkv_norm(self_attn.norm_q, tp_rank, tp_degree)
        self_attn.norm_k = shard_qkv_norm(self_attn.norm_k, tp_rank, tp_degree)

        # Update num_heads to local count
        self_attn.num_heads = heads_per_rank

        # --- Cross-Attention ---
        cross_attn = block.cross_attn

        cross_attn.q = shard_linear_column(cross_attn.q, tp_rank, tp_degree)
        cross_attn.k = shard_linear_column(cross_attn.k, tp_rank, tp_degree)
        cross_attn.v = shard_linear_column(cross_attn.v, tp_rank, tp_degree)
        cross_attn.o = shard_linear_row(cross_attn.o, tp_rank, tp_degree)

        cross_attn.norm_q = shard_qkv_norm(cross_attn.norm_q, tp_rank, tp_degree)
        cross_attn.norm_k = shard_qkv_norm(cross_attn.norm_k, tp_rank, tp_degree)

        cross_attn.num_heads = heads_per_rank

        # --- FFN ---
        # FFN is nn.Sequential(Linear(dim, ffn_dim), GELU(), Linear(ffn_dim, dim))
        # or WanFFN with .fc1 and .fc2
        ffn = block.ffn
        if hasattr(ffn, 'fc1'):
            # WanFFN class
            ffn.fc1 = shard_linear_column(ffn.fc1, tp_rank, tp_degree)
            ffn.fc2 = shard_linear_row(ffn.fc2, tp_rank, tp_degree)
        elif isinstance(ffn, nn.Sequential):
            # nn.Sequential(Linear, GELU, Linear)
            ffn[0] = shard_linear_column(ffn[0], tp_rank, tp_degree)
            ffn[2] = shard_linear_row(ffn[2], tp_rank, tp_degree)
        else:
            raise ValueError(f"Unknown FFN type: {type(ffn)}")

    # Update model-level num_heads for KV cache sizing
    model.num_heads_per_rank = heads_per_rank
    model.tp_degree = tp_degree
    model.tp_rank = tp_rank

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TP] Sharded model on rank {tp_rank}/{tp_degree}: "
          f"{heads_per_rank} heads/rank, "
          f"{total_params / 1e9:.2f}B params (local)")

    # ── torch.compile submodules AFTER sharding ──────────────────────────
    # Reference: rolling-forcing/app/inference_neuron_tp.py lines 230-237
    # Must happen after TP sharding so isinstance checks see raw nn.Sequential.
    from models.wan.neuron_layers import neuron_compile
    model.patch_embedding = neuron_compile(model.patch_embedding)
    model.text_embedding = neuron_compile(model.text_embedding)
    model.time_embedding = neuron_compile(model.time_embedding)
    model.time_projection = neuron_compile(model.time_projection)
    for block in model.blocks:
        block.ffn = neuron_compile(block.ffn)
    print(f"[TP] torch.compile applied to DiT submodules (backend='neuron')")

    return model
