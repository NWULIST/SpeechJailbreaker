"""SPIRIT defense for audio LLMs."""
from Defenses.SPIRIT.spirit_wrapper import SPIRITWrapper
from Defenses.SPIRIT.noise_utils import (
    ActivationCapture,
    denoise_audio,
    identify_noisy_neurons,
    save_temp_audio,
)

__all__ = [
    "SPIRITWrapper",
    "ActivationCapture",
    "denoise_audio",
    "identify_noisy_neurons",
    "save_temp_audio",
]
