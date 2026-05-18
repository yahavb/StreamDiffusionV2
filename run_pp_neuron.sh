#!/bin/bash
# StreamDiffusionV2 Pipeline-Parallel (PP-4) inference on Trainium
# Usage: ./run_pp_neuron.sh [--benchmark]
set -euxo pipefail

export NEURON_RT_LOG_LEVEL="${NEURON_RT_LOG_LEVEL:-ERROR}"
export NEURON_CC_LOG_LEVEL="${NEURON_CC_LOG_LEVEL:-ERROR}"
export TORCH_NEURONX_LOG_LEVEL="${TORCH_NEURONX_LOG_LEVEL:-ERROR}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---model-type=transformer}"
export USE_NKI_KERNELS="${USE_NKI_KERNELS:-true}"
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-streamv2v/configs/wan_causal_pp4_neuron.yaml}"
PROMPT="${PROMPT:-A cat walking on the beach at sunset, cinematic lighting, high quality}"
NUM_FRAMES="${NUM_FRAMES:-81}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
OUTPUT="${OUTPUT:-output_pp.mp4}"
DEVICE="${DEVICE:-neuron}"

EXTRA_ARGS=""
if [[ "${1:-}" == "--benchmark" ]]; then
    EXTRA_ARGS="--benchmark --warmup_runs 2 --benchmark_runs 5"
fi

# PP-4: 4 processes, each with FULL model (no TP sharding)
PP_DEGREE="${PP_DEGREE:-4}"

torchrun --nproc_per_node="$PP_DEGREE" \
    streamv2v/inference_neuron_pp.py \
    --config "$CONFIG" \
    --prompt "$PROMPT" \
    --num_frames "$NUM_FRAMES" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --output_path "$OUTPUT" \
    --device "$DEVICE" \
    $EXTRA_ARGS 2>&1
