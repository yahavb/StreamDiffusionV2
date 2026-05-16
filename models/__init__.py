"""Inference-facing model registry for StreamDiffusionV2."""

from .wan.wan_wrapper import (
    CausalWanDiffusionWrapper,
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)

# Neuron/Trainium-compatible wrappers (lazy import to avoid breaking GPU-only envs)
try:
    from .wan.neuron_wan_wrapper import (
        NeuronCausalWanDiffusionWrapper,
        NeuronWanTextEncoder,
        NeuronWanVAEWrapper,
    )
    _NEURON_AVAILABLE = True
except ImportError:
    _NEURON_AVAILABLE = False


DIFFUSION_NAME_TO_CLASS = {
    "wan": WanDiffusionWrapper,
    "causal_wan": CausalWanDiffusionWrapper,
}


TEXT_ENCODER_NAME_TO_CLASS = {
    "wan": WanTextEncoder,
    "causal_wan": WanTextEncoder,
}


VAE_NAME_TO_CLASS = {
    "wan": WanVAEWrapper,
    "causal_wan": WanVAEWrapper,
}

if _NEURON_AVAILABLE:
    DIFFUSION_NAME_TO_CLASS["neuron_causal_wan"] = NeuronCausalWanDiffusionWrapper
    TEXT_ENCODER_NAME_TO_CLASS["neuron_causal_wan"] = NeuronWanTextEncoder
    VAE_NAME_TO_CLASS["neuron_causal_wan"] = NeuronWanVAEWrapper


def get_diffusion_wrapper(model_name):
    return DIFFUSION_NAME_TO_CLASS[model_name]


def get_text_encoder_wrapper(model_name):
    return TEXT_ENCODER_NAME_TO_CLASS[model_name]


def get_vae_wrapper(model_name):
    return VAE_NAME_TO_CLASS[model_name]
