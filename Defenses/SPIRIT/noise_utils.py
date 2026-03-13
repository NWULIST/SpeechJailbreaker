"""
SPIRIT Defense - Noise Analysis Utilities
==========================================
Provides helpers for:
  1. Generating a *clean* reference audio via ``noisereduce``.
  2. Running a forward pass with hooks to capture intermediate activations
     from both the audio tower and the language-model backbone of
     Qwen2-Audio.
  3. Identifying the *top-k* noisiest neurons by comparing clean vs. noisy
     activations (L2 distance per neuron).

These utilities are consumed by ``spirit_wrapper.py``.

Reference
---------
SPIRIT: Probing LLM Activations to Identify and Mitigate Harmful Noise
in Audio LLMs (2025).
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Clean-reference generation
# ---------------------------------------------------------------------------

def denoise_audio(
    audio_path: str,
    sr: int = 16_000,
    *,
    prop_decrease: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Return a denoised copy of *audio_path* using ``noisereduce``.

    Parameters
    ----------
    audio_path : str
        Path to the (potentially adversarial) audio file.
    sr : int
        Target sample rate.  Qwen2-Audio uses 16 kHz by default.
    prop_decrease : float
        Proportion of noise to remove (1.0 = full stationary removal).

    Returns
    -------
    (denoised_audio, sr) : tuple[np.ndarray, int]
    """
    import noisereduce as nr  # lazy import – not always installed

    y, orig_sr = librosa.load(audio_path, sr=sr)
    denoised = nr.reduce_noise(
        y=y,
        sr=sr,
        prop_decrease=prop_decrease,
        stationary=True,
    )
    return denoised, sr


def save_temp_audio(audio: np.ndarray, sr: int) -> str:
    """Write *audio* to a temporary WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Context manager that hooks into Qwen2-Audio layers and records
    intermediate hidden states.

    Parameters
    ----------
    model : ``Qwen2AudioForConditionalGeneration``
        The loaded Qwen2-Audio model.
    activation_choice : ``"audio"`` | ``"lm"``
        Which tower to hook into.
    mode : ``"all"`` | ``"last"`` | ``"lastN"``
        Which layers to capture.  ``"all"`` captures every layer,
        ``"last"`` captures only the final layer, ``"lastN"`` captures
        the last *N* layers (default N=4).
    last_n : int
        Used when *mode* is ``"lastN"``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        activation_choice: str = "audio",
        mode: str = "all",
        last_n: int = 4,
    ):
        self.model = model
        self.activation_choice = activation_choice
        self.mode = mode
        self.last_n = last_n

        self._hooks: list = []
        self.activations: Dict[str, torch.Tensor] = {}

    # ---- layer discovery ------------------------------------------------

    def _get_layers(self) -> List[Tuple[str, torch.nn.Module]]:
        """Return ``(name, module)`` pairs for the chosen tower."""
        if self.activation_choice == "audio":
            tower = self.model.audio_tower
            layers = list(tower.layers.named_children())
        else:  # "lm"
            lm = self.model.language_model.model
            layers = list(lm.layers.named_children())

        if self.mode == "last":
            layers = layers[-1:]
        elif self.mode == "lastN":
            layers = layers[-self.last_n :]

        return layers

    # ---- hook ------------------------------------------------------------

    def _make_hook(self, name: str):
        def hook_fn(_module, _input, output):
            # Many transformer layers return tuples; take [0].
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach().cpu()

        return hook_fn

    # ---- context manager -------------------------------------------------

    def __enter__(self):
        self.activations.clear()
        for name, module in self._get_layers():
            h = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)
        return self

    def __exit__(self, *_):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Noisy-neuron identification
# ---------------------------------------------------------------------------

def identify_noisy_neurons(
    clean_acts: Dict[str, torch.Tensor],
    noisy_acts: Dict[str, torch.Tensor],
    top_k_percent: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """Compare clean vs. noisy activations per layer and return a boolean
    mask of the top-*k*% most-deviating neurons (hidden-dimension indices).

    Parameters
    ----------
    clean_acts, noisy_acts : dict[str, Tensor]
        Activation dictionaries keyed by layer name.  Tensors are
        ``(batch, seq_len, hidden_dim)`` or ``(seq_len, hidden_dim)``.
    top_k_percent : float
        Percentage of neurons to flag as "noisy" per layer.

    Returns
    -------
    masks : dict[str, Tensor[bool]]
        ``True`` at noisy-neuron positions (shape ``(hidden_dim,)``).
    """
    masks: Dict[str, torch.Tensor] = {}

    for name in clean_acts:
        if name not in noisy_acts:
            continue

        c = clean_acts[name].float()
        n = noisy_acts[name].float()

        # Flatten to (tokens, hidden)
        if c.dim() == 3:
            c = c.view(-1, c.size(-1))
        if n.dim() == 3:
            n = n.view(-1, n.size(-1))

        # Truncate to common length
        min_len = min(c.size(0), n.size(0))
        c = c[:min_len]
        n = n[:min_len]

        # L2 distance per neuron (averaged over tokens)
        diff = (c - n).pow(2).mean(dim=0)  # (hidden_dim,)
        k = max(1, int(diff.numel() * top_k_percent / 100.0))
        threshold = diff.topk(k).values[-1]
        masks[name] = diff >= threshold

    return masks
