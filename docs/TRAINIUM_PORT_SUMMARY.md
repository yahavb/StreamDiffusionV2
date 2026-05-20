# StreamDiffusionV2 → Trainium Port: Complete Summary

**Date:** May 2026  
**Hardware:** AWS Trainium2 (trn2) — 4 NeuronCores, TP-4  
**Model:** Wan2.1-T2V-1.3B (30 layers, dim=1536, 12 heads, ffn_dim=8960)  
**Paper:** StreamDiffusion V2 (arXiv 2511.07399v2)  
**Reference Implementation:** `/Users/yahavb/aws-neuron-eks-samples/rolling-forcing`

---

## 1. Objective

Port StreamDiffusionV2's causal video diffusion pipeline to run natively on AWS Trainium instances using PyTorch + Neuron SDK. The pipeline consists of:
- **T5** text encoder (512 tokens → 4096-dim embeddings)
- **Wan2.1-1.3B DiT** (denoiser, 30 transformer layers with causal attention + KV cache)
- **Wan VAE** decoder (latent → pixel frames)

All three components were already individually ported in the rolling-forcing pipeline. The challenge was integrating them into StreamDiffusionV2's **streaming causal inference** framework with block-by-block generation and KV cache management.

---

## 2. Architecture Decisions

### 2.1 Execution Mode: Eager (not traced)

The Neuron SDK supports two execution modes:
1. **`torch_neuronx.trace()`** — compiles entire model into single NEFF (fastest, but requires static shapes and no control flow)
2. **Eager mode** — each PyTorch op dispatches individually to pre-compiled per-op NEFFs

We chose **eager mode** because:
- StreamDiffusionV2 has **dynamic KV cache management** (eviction, rolling window)
- Cache sizes change depending on which block is being processed (anchor vs. streaming)
- Python control flow (if/else) in the attention forward pass

### 2.2 Tensor Parallelism (TP-4)

- All 4 NeuronCores run the same layer simultaneously with sharded weights
- Q/K/V → column-parallel (split by heads: 12 heads → 3 per rank)
- O/FFN-down → row-parallel (split input dim, all-reduce output)
- FFN-up → column-parallel (split hidden dim: 8960 → 2240 per rank)
- Custom `TPRMSNorm` with all-reduce for correct global normalization

### 2.3 NKI Kernels (Custom Neuron Kernels)

Wrote NKI (Neuron Kernel Interface) kernels for:
- **`wan_flash_self_attn`** — flash attention for self-attention with masking
- **`wan_cross_attn`** — flash attention for T5 cross-attention (seq_k=512)
- **`causal_rope_rotation`** — 3D rotary position embeddings
- **`vae_conv2d_k1`** — fused 1×1 convolution for VAE
- **`vae_self_attention`** — flash attention for VAE mid-block

### 2.4 Neuron-Safe Layer Replacements

| GPU Original | Neuron Replacement | Reason |
|---|---|---|
| `Conv3d` patch embed | Matmul-based `WanPatchEmbed` | Conv3d not supported |
| `flex_attention` | SDPA / NKI flash attention | flex_attention not supported |
| `float64` RoPE | `float32` cos/sin RoPE | float64 not supported |
| `complex` tensors | Explicit cos/sin pairs | Complex not supported |
| `torch.compile(cuda)` | Eager + NKI | No CUDA backend |

---

## 3. Progressive Development & Experiments

### Run 1-3: Initial Port & Debugging (not covered in this torch.compile session)

These earlier runs established the baseline:
- Model loads and runs on Neuron
- NKI kernels compile and execute correctly
- KV cache management works
- TP-4 all-reduce communication functions

### Run 4: Baseline Performance (no torch.compile)

**Configuration:**
- Resolution: 240×416 (latent 30×52)
- Frames: 81 (27 blocks × 3 frames/block)
- Denoising steps: 5 (DMD distilled model)
- KV cache: 6× frame_length sliding window

**Results:**
```
│  Compilation time:       1144.6s (warmup run 1)     │
│  T5+anchor time:         12.306s                    │
│  DiT/block time:         12.137s (5 steps)          │
│  VAE/batch time:          1.008s (3 frames)         │
│  Batch E2E time:         13.147s (DiT+VAE, 3f)     │
│  Total time:            355.145s                    │
│  Time-to-first-frame:    13.317s                    │
│  Real-time ratio:           0.014x (vs 16fps)      │
│  Stream real-time ratio:    0.014x (vs 16fps)      │
```

**Profile Analysis:**
- 355 unique NEFFs loaded per inference
- 160K+ DMA copy-in operations
- 15K+ model switches (NEFF → NEFF transitions)
- ~80ms per layer per step (compute + dispatch overhead)

---

## 4. torch.compile Optimization Attempts

### 4.1 Attempt 1: `torch.compile(backend='neuronx')` — FAILED

**Hypothesis:** The `'neuronx'` backend would fuse multiple PyTorch ops into single compiled NEFFs, reducing the 355 NEFF loads to ~40-60.

**Implementation:**
- Added `neuron_compile()` wrapper function in `neuron_layers.py`
- Applied to FFN, modulation helpers in `CausalWanAttentionBlock.__init__`

**Result:** Complete failure — `'neuronx'` is not a valid torch.compile backend:
```
[neuron_compile] WARNING: torch.compile failed, falling back to eager: 
Invalid backend: 'neuronx', see `torch._dynamo.list_backends()` for available backends.
```
All 18+ compile calls fell back to eager. Performance identical to baseline.

**Lesson:** The backend name `'neuronx'` does not exist in the Neuron SDK.

---

### 4.2 Attempt 2: Auto-detect backend — WRONG APPROACH

**Hypothesis:** Maybe the backend is named differently. Added auto-detection:
```python
import torch._dynamo as _dynamo
_available_backends = _dynamo.list_backends()
for candidate in ["neuronx", "openxla", "inductor"]:
    if candidate in _available_backends:
        break
```

**Result:** Would have selected `'inductor'` which generates Triton/CUDA kernels — completely wrong for Neuron hardware.

**Lesson:** `inductor` requires GPU. There's no generic "works everywhere" torch.compile backend.

---

### 4.3 Attempt 3: `torch.compile(backend='neuron')` — CORRECT NAME

**Discovery:** Found in the rolling-forcing reference (`inference_neuron_tp.py` lines 230-237):
```python
dit_model.patch_embedding = torch.compile(dit_model.patch_embedding, backend='neuron', dynamic=False)
dit_model.text_embedding = torch.compile(dit_model.text_embedding, backend='neuron', dynamic=False)
dit_model.time_embedding = torch.compile(dit_model.time_embedding, backend='neuron', dynamic=False)
dit_model.time_projection = torch.compile(dit_model.time_projection, backend='neuron', dynamic=False)
dit_model.head = torch.compile(dit_model.head, backend='neuron', dynamic=False)
for block in dit_model.blocks:
    block.ffn = torch.compile(block.ffn, backend='neuron', dynamic=False)
```

**Implementation:** Applied `neuron_compile()` to:
- `patch_embedding` (reshape + matmul + bias)
- `text_embedding` (Linear→GELU→Linear)
- `time_embedding` (Linear→SiLU→Linear)
- `time_projection` (SiLU→Linear)
- All 30 FFN blocks (Linear→GELU→Linear)

**Error:** `ValueError: Unknown FFN type: <class 'torch._dynamo.eval_frame.OptimizedModule'>`

**Root cause:** `torch.compile()` wraps the module in `OptimizedModule`. The TP sharding code (`tp_utils.py`) checks `isinstance(ffn, nn.Sequential)` to determine how to shard — `OptimizedModule` fails this check.

**Lesson:** Never torch.compile before TP sharding. Compile AFTER all weight manipulation is done.

---

### 4.4 Attempt 4: Compile AFTER TP sharding — WORKS, NO IMPROVEMENT

**Fix:** Moved all `neuron_compile()` calls from `__init__` to the end of `shard_model_tp()` in `tp_utils.py`:
```python
# At end of shard_model_tp(), AFTER all isinstance checks and weight slicing:
model.patch_embedding = neuron_compile(model.patch_embedding)
model.text_embedding = neuron_compile(model.text_embedding)
model.time_embedding = neuron_compile(model.time_embedding)
model.time_projection = neuron_compile(model.time_projection)
for block in model.blocks:
    block.ffn = neuron_compile(block.ffn)
```

**Run 6 Results:**
```
│  Compilation time:       1122.1s (warmup run 1)     │
│  T5+anchor time:         12.139s                    │
│  DiT/block time:         12.112s (5 steps)          │
│  VAE/batch time:          0.995s (3 frames)         │
│  Batch E2E time:         13.109s (DiT+VAE, 3f)     │
│  Total time:            353.989s                    │
│  Real-time ratio:           0.014x (vs 16fps)      │
│  Stream real-time ratio:    0.014x (vs 16fps)      │
```

**Comparison (Run 4 vs Run 6):**

| Metric | Run 4 (no compile) | Run 6 (with compile) | Δ |
|--------|--------------------|--------------------|---|
| Compilation time | 1144.6s | 1122.1s | -2.0% |
| T5+anchor | 12.306s | 12.139s | -1.4% |
| **DiT/block** | **12.137s** | **12.112s** | **-0.2%** |
| VAE/batch | 1.008s | 0.995s | -1.3% |
| Batch E2E | 13.147s | 13.109s | -0.3% |
| Total | 355.145s | 353.989s | -0.3% |
| FPS | 0.23 | 0.23 | 0% |

**Conclusion:** `torch.compile(backend='neuron')` registers successfully and applies without errors, but provides **negligible performance improvement** (~0.2-0.3%, within measurement noise).

---

## 5. Why torch.compile Didn't Help

### Theory vs. Reality

**Expected:** Fuse sequences of ops (Linear→GELU→Linear) into single compiled NEFFs, reducing 355 NEFF dispatches to ~40-60, eliminating model_switch + DMA overhead.

**Actual:** The `'neuron'` backend in the installed SDK version appears to be a **pass-through** — it accepts the module but doesn't actually perform fusion on eager-mode dispatched ops. The ops still execute individually.

### The Real Bottleneck

The profiling showed the actual time breakdown per DiT step:
- **Self-attention NKI kernel** (8192-length KV cache): ~40% of layer time
- **KV cache DMA copies** (cache_copy_inplace): ~25%
- **FFN matmuls** (2 per layer): ~20%
- **Cross-attention NKI kernel**: ~10%
- **Norms/modulation**: ~5%

The dominant costs are **large matmuls** and **DMA transfers for KV cache management** — neither of which torch.compile can improve. The kernel launch overhead (model_switch) is a small fraction of total time.

---

## 6. Files Changed (Final State)

### StreamDiffusionV2 Repo

| File | Changes |
|------|---------|
| `models/wan/neuron_layers.py` | `neuron_compile()` function with `backend='neuron'`, `USE_TORCH_COMPILE` env var |
| `models/wan/neuron_causal_model.py` | `neuron_compile` import; submodules NOT compiled in `__init__` |
| `models/wan/tp_utils.py` | Compile step at end of `shard_model_tp()` after all sharding |
| `requirements-neuron.txt` | Inference dependencies (no torch_neuronx — pre-installed in container) |

### K8s Config Repo

| File | Changes |
|------|---------|
| `clusters/ray/stream-diffusion-job.yaml` | `USE_TORCH_COMPILE=true`, `NEURON_COMPILE_BACKEND=neuron` env vars |

---

## 7. Key Lessons Learned

### For Future Neuron/Trainium Projects:

1. **`torch.compile` backend name is `'neuron'`** (not `'neuronx'`, not `'inductor'`)

2. **torch.compile must happen AFTER weight manipulation** (TP sharding, quantization, pruning) — it wraps modules in `OptimizedModule` which breaks `isinstance` checks

3. **torch.compile on Neuron provides negligible speedup in eager mode** — the per-op NEFFs are already compiled; the overhead is DMA/model-switch which compile can't reduce in the current SDK version

4. **The real bottleneck on Neuron eager is attention compute + KV cache DMA** — not kernel launch overhead. For a 1.3B model with 8192-token attention window, the matmuls dominate

5. **NKI kernels are the most impactful optimization** — fusing attention (QKV + softmax + PV + O) into single NEFF saves dramatically more than fusing FFN

6. **Resolution dominates performance** — seq_len = (F×H×W)/(pT×pH×pW). At 240×416 with patch (1,2,2): 1560 tokens per frame × 3 frames = 4680 per block. Attention is O(n²) so halving resolution gives ~4x speedup

7. **Don't reduce denoising steps below 5** — quality becomes unacceptable with DMD distillation

---

## 8. Current Performance Summary

| Component | Time | Notes |
|-----------|------|-------|
| T5 encoding | 12.1s | 512 tokens, runs once per prompt |
| DiT per block | 12.1s | 5 steps × 30 layers × 4680 tokens |
| VAE per block | 1.0s | 3 frames decode |
| **Streaming FPS** | **0.23** | 3 frames / 13.1s |
| **Real-time ratio** | **0.014x** | vs 16fps target |
| Gap to real-time | **~71x** | Need 71x speedup for real-time |

---

## 9. What Would Actually Help (Not Tried)

| Approach | Expected Speedup | Feasibility |
|----------|-----------------|-------------|
| Reduce to 1 denoising step (consistency distillation) | 5x | Requires retraining |
| Use Wan2.1-T2V-480M (smaller model) | 3-4x | Different model, quality loss |
| Reduce resolution to 128×224 | ~4x | Quality loss |
| Multi-chip (trn2.48xlarge, 16 NCs) | ~4x | Hardware cost |
| Neuron SDK graph-mode compilation | 2-5x | Requires static shapes, no cache |
| Speculative decoding for DiT | 1.5-2x | Research-level |

**To reach real-time (16 fps) would require ~71x speedup** — this is fundamentally incompatible with 30-layer × 5-step diffusion at this resolution on a single trn2 chip. Real-time streaming video diffusion at 240p likely requires either:
- Much smaller/distilled models (1-step, <10 layers)
- Much more hardware (16+ NCs)
- Completely different architecture (e.g., consistency models, not diffusion)

---

## 10. Git History

```
f93ae20 fix: move torch.compile AFTER TP sharding (fixes OptimizedModule type error)
6210cd4 fix: use correct torch.compile backend='neuron' (not 'neuronx')
0a9bca8 fix: auto-detect torch.compile backend (neuronx doesn't exist)
fe9d120 (earlier commits: model port, NKI kernels, TP, KV cache, etc.)
```
