# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

StreamDiffusionV2 is a streaming video-to-video diffusion pipeline (Wan2.1-T2V causal DiT, MLSys 2026). This is a **fork (`yahavb/StreamDiffusionV2`)** whose primary value-add over upstream is a **native AWS Trainium/Neuron port** of the pipeline. Most uncommitted/recent work lives in the Neuron path. When working here, be clear about which target you're touching: **CUDA/GPU (upstream)** or **Neuron/Trainium (the fork's additions)** — they are largely parallel code paths.

## Two parallel pipelines

| Concern | GPU path | Neuron/Trainium path |
|---|---|---|
| Entry scripts | `run_v2v.sh {single,single-wo,pipe}` | `run_v2v_neuron.sh`, `run_pp_neuron.sh` |
| Inference modules | `streamv2v/inference{,_wo_batch,_pipe}.py` | `streamv2v/inference_neuron.py` (TP-4), `streamv2v/inference_neuron_pp.py` (PP-4) |
| Model wrappers | `models/wan/wan_wrapper.py` | `models/wan/neuron_wan_wrapper.py` |
| DiT model | `models/wan/causal_model.py` | `models/wan/neuron_causal_model.py` + `neuron_layers.py` |
| Stream inference loop | `models/wan/causal_stream_inference.py` | `models/wan/neuron_causal_stream_inference.py`, `neuron_pp_inference.py` |
| Configs | `configs/*.yaml`, `streamv2v/configs/*.yaml` | `streamv2v/configs/wan_causal_dmd_v2v_neuron.yaml`, `wan_causal_pp4_neuron.yaml` |
| Dependencies | `pyproject.toml` | `requirements-neuron.txt` (torch_neuronx pre-installed in container) |

The Neuron wrappers are **lazily imported** in `models/__init__.py` (wrapped in `try/except ImportError`) so the GPU path still works in environments without the Neuron SDK. Neuron model classes register under the `neuron_causal_wan` key in the `*_NAME_TO_CLASS` registries.

## Pipeline architecture (shared concept)

The pipeline streams video chunk-by-chunk through three stages, all built on **Wan2.1-T2V-1.3B** (30 transformer layers, dim=1536, 12 heads, ffn_dim=8960):

1. **T5 text encoder** — prompt → embeddings (runs once per prompt)
2. **Causal Wan DiT denoiser** — block-by-block generation with a **rolling/sliding-window KV cache**. Uses DMD distillation (5 denoising steps). This is the dominant cost.
3. **Wan VAE decoder** — latents → pixel frames. Optional lightweight **TAEHV** decoder (`--use_taehv` / `USE_TAEHV=1`) via `models/wan/taehv_wrapper.py`.

The public staged API (`streamdiffusionv2/pipeline.py`, exported from `streamdiffusionv2/__init__.py`) exposes this as `chunk_video → encode_chunk → denoise_chunk → decode_chunk`. The streaming logic (KV cache eviction, sink tokens, anchor vs. streaming blocks) lives in the `*_stream_inference.py` files.

Key streaming config knobs (see `*_neuron.yaml`): `num_frame_per_block`, `num_kv_cache` (sliding window multiplier), `num_sink_tokens`, `denoising_step_list`.

## Running inference

GPU (offline):
```shell
./run_v2v.sh single        # single-GPU streaming
./run_v2v.sh single-wo     # single-GPU, no stream-batch
./run_v2v.sh pipe          # multi-GPU pipeline-parallel
./run_v2v.sh pipe --profile # synchronized throughput measurement only
```
Override via env vars (`CONFIG_PATH`, `CHECKPOINT_FOLDER`, `OUTPUT_FOLDER`, `VIDEO_PATH`, `PROMPT_FILE_PATH`, `HEIGHT`, `WIDTH`, `FPS`, `STEP`, `NPROC_PER_NODE`) or CLI flags after the mode.

Neuron/Trainium (launched via `torchrun`, not a plain python call):
```shell
./run_v2v_neuron.sh            # TP-4: 4 NeuronCores, weights sharded
./run_v2v_neuron.sh --benchmark # adds --warmup_runs 2 --benchmark_runs 5
./run_pp_neuron.sh             # PP-4: 4 processes, each a full model copy
```
Neuron env vars of note: `TP_DEGREE`/`PP_DEGREE` (default 4), `USE_NKI_KERNELS` (default true), `USE_TORCH_COMPILE`, `NEURON_COMPILE_BACKEND` (must be `neuron`, see below).

Web demo (online): see `demo/README.md`; serves on `http://0.0.0.0:7860`. Console scripts `streamdiffusionv2-{single,single-wo,pipe}` are defined in `pyproject.toml`.

## Tests

There is no unified test runner. Tests are standalone scripts run directly:
- `test_vae_kernels.py` (repo root) — validates NKI VAE kernels
- `streamv2v/communication/test_communication.py` — multi-GPU communication
- `demo/test_online_inference.py` — demo backend

Run with `python <path>` (Neuron kernel tests require Trainium hardware + Neuron SDK).

## Neuron/Trainium specifics (read before touching the Neuron path)

`docs/TRAINIUM_PORT_SUMMARY.md` is the authoritative record of the port — read it before any Neuron work. Critical points:

- **Eager mode, not traced.** `torch_neuronx.trace()` is *not* used because the dynamic KV cache and Python control flow in attention require it. Each op dispatches to a pre-compiled per-op NEFF.
- **NKI custom kernels** live in `kernels/` (`self_attention.py`, `cross_attention.py`, `rope.py`) and `models/wan/kernels/` (`vae_attention.py`, `vae_conv2d.py`). These are the highest-impact optimization — fusing attention into single NEFFs. Gated by `USE_NKI_KERNELS`.
- **Tensor parallelism** (`models/wan/tp_utils.py`): QKV/FFN-up are column-parallel, O/FFN-down row-parallel with all-reduce; custom `TPRMSNorm`. `shard_model_tp()` does the weight slicing.
- **Neuron-safe layer swaps**: `Conv3d` patch-embed → matmul-based `WanPatchEmbed`; `flex_attention` → SDPA/NKI; float64/complex RoPE → float32 cos/sin pairs. Don't reintroduce unsupported ops.
- **`torch.compile` gotchas**: the only valid backend is `'neuron'` (not `'neuronx'`, not `'inductor'`). It **must run AFTER TP sharding** — compiling first wraps modules in `OptimizedModule`, which breaks the `isinstance(ffn, nn.Sequential)` checks in `tp_utils.py`. In practice it gives negligible speedup in eager mode (the bottleneck is attention matmuls + KV-cache DMA, not kernel-launch overhead). `neuron_compile()` in `neuron_layers.py` is the wrapper.

## Conventions

- Inference is config-driven (OmegaConf YAML). The model registry in `models/__init__.py` maps config `generator_name`/`model_name` strings to wrapper classes — add new backends there.
- When adding a Neuron variant of a GPU module, mirror the existing `neuron_*` naming and keep the GPU file untouched so the two paths stay independent.
- Checkpoints: Wan base weights go in `wan_models/`, StreamDiffusionV2 DMD checkpoints in `ckpts/` (GPU) or `checkpoints/` (Neuron config default). See README "Download Checkpoints".
