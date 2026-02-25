"""
SPIRIT Defense - Inference Wrapper for Qwen2-Audio
====================================================
Drop-in replacement for ``LocalSpeechLLM`` that applies one of the three
SPIRIT mitigation strategies at generation time:

  * **bias**  - shift noisy-neuron activations toward their clean values.
  * **prune** - zero-out noisy-neuron activations.
  * **patch** - replace noisy-neuron activations with their clean counterparts.

Usage in attack scripts::

    from Defenses.SPIRIT.spirit_wrapper import SPIRITWrapper

    base_model = LocalSpeechLLM(args.target_model)
    target_model = SPIRITWrapper(
        base_model,
        method="bias",          # or "prune" / "patch"
    )
    response = target_model.generate(audio_path, text_prompt)

The wrapper is intentionally thin: it intercepts ``generate`` /
``generate_batch``, runs a parallel *clean* forward pass (using
``noisereduce``-denoised audio), identifies the noisiest neurons, installs
corrective hooks, then delegates to the underlying model.

Defaults
--------
All hyper-parameters default to the best values reported in the SPIRIT
paper (Table 4 / §5):

  * ``activation_choice = "audio"``
  * ``mode = "all"``
  * ``top_k_percent = 5.0``
  * ``bias_value = 50.0``

These are *hard-coded* per the user's request.  See the ``__init__``
docstring if you ever need to override them.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import List, Optional

import librosa
import torch

from Defenses.SPIRIT.noise_utils import (
    ActivationCapture,
    denoise_audio,
    identify_noisy_neurons,
    save_temp_audio,
)


class SPIRITWrapper:
    """Wrap a ``LocalSpeechLLM`` with SPIRIT activation-level defense.

    Parameters
    ----------
    base_model : LocalSpeechLLM
        The already-loaded Qwen2-Audio model instance.
    method : str
        One of ``"bias"``, ``"prune"``, or ``"patch"``.
    activation_choice : str
        ``"audio"`` or ``"lm"`` – which tower to defend.
    mode : str
        ``"all"`` / ``"last"`` / ``"lastN"`` – which layers to hook.
    top_k_percent : float
        Percentage of neurons to treat as noisy.
    bias_value : float
        Additive bias applied to noisy neurons (only used when
        *method* is ``"bias"``).
    last_n : int
        Number of layers for ``"lastN"`` mode.
    """

    # SPIRIT methods we recognise
    VALID_METHODS = {"bias", "prune", "patch"}

    def __init__(
        self,
        base_model,
        method: str = "bias",
        # ---- hard-coded paper defaults -----------
        activation_choice: str = "audio",
        mode: str = "all",
        top_k_percent: float = 5.0,
        bias_value: float = 50.0,
        last_n: int = 4,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown SPIRIT method '{method}'. "
                f"Choose from {self.VALID_METHODS}."
            )

        self.base_model = base_model
        self.method = method
        self.activation_choice = activation_choice
        self.mode = mode
        self.top_k_percent = top_k_percent
        self.bias_value = bias_value
        self.last_n = last_n

        # Expose attributes that attack scripts may probe
        self.model = base_model.model
        self.processor = base_model.processor
        self.model_path = base_model.model_path
        self.system_message = base_model.system_message

    # ------------------------------------------------------------------
    # Proxy helpers so the wrapper quacks like LocalSpeechLLM
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return getattr(self.base_model, "tokenizer", None)

    def _sr(self) -> int:
        """Sampling rate expected by the processor."""
        return self.processor.feature_extractor.sampling_rate

    # ------------------------------------------------------------------
    # Internal: obtain clean activations
    # ------------------------------------------------------------------

    def _get_clean_audio_path(self, audio_path: str) -> str:
        """Denoise *audio_path* with ``noisereduce`` and return a temp path."""
        denoised, sr = denoise_audio(audio_path, sr=self._sr())
        return save_temp_audio(denoised, sr)

    def _run_forward_for_activations(
        self, audio_path: str, text: str
    ) -> dict:
        """Execute a forward pass and return captured activations."""
        model = self.base_model.model
        processor = self.base_model.processor

        conversation = [
            {"role": "system", "content": self.system_message or ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "audio_url": audio_path},
                ],
            },
        ]

        chat_text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Load audio waveform
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_bytes = BytesIO(f.read())
        else:
            from urllib.request import urlopen

            audio_bytes = BytesIO(urlopen(audio_path).read())

        waveform = librosa.load(audio_bytes, sr=self._sr())[0]

        inputs = processor(
            text=chat_text, audio=[waveform], return_tensors="pt", padding=True
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with ActivationCapture(
            model,
            activation_choice=self.activation_choice,
            mode=self.mode,
            last_n=self.last_n,
        ) as cap:
            with torch.no_grad():
                model(**inputs)

        return dict(cap.activations)

    # ------------------------------------------------------------------
    # Internal: install corrective hooks
    # ------------------------------------------------------------------

    def _make_correction_hook(self, layer_name, mask, clean_act):
        """Return a forward-hook function that applies the chosen SPIRIT
        method to a single layer during generation."""
        method = self.method
        bias_value = self.bias_value

        def hook_fn(_module, _input, output):
            out = output
            is_tuple = isinstance(out, tuple)
            if is_tuple:
                out = out[0]

            device = out.device
            m = mask.to(device)

            if method == "prune":
                out[:, :, m] = 0.0
            elif method == "bias":
                # Shift noisy neurons toward their clean mean
                c = clean_act.to(device).float()
                if c.dim() == 3:
                    c = c.mean(dim=1)  # average over seq → (1, hidden)
                elif c.dim() == 2:
                    c = c.mean(dim=0, keepdim=True)
                # Broadcast and add bias toward clean direction
                sign = torch.sign(c[:, m] - out[:, :, m].mean(dim=1, keepdim=True))
                out[:, :, m] = out[:, :, m] + sign * bias_value
            elif method == "patch":
                c = clean_act.to(device).float()
                # Align sequence dimension: take the last seq_len tokens
                seq_len = out.size(1)
                if c.dim() == 3:
                    c_seq = c.size(1)
                    if c_seq >= seq_len:
                        c = c[:, :seq_len, :]
                    else:
                        # pad with last value
                        pad = c[:, -1:, :].expand(-1, seq_len - c_seq, -1)
                        c = torch.cat([c, pad], dim=1)
                else:
                    c = c.unsqueeze(0).expand(out.size(0), seq_len, -1)
                out[:, :, m] = c[:, :, m]

            if is_tuple:
                return (out,) + output[1:]
            return out

        return hook_fn

    def _install_hooks(self, masks, clean_acts):
        """Install corrective forward hooks on the model.  Returns a list
        of hook handles (caller must remove them after generation)."""
        model = self.base_model.model

        if self.activation_choice == "audio":
            tower = model.audio_tower
            layers = dict(tower.layers.named_children())
        else:
            lm = model.language_model.model
            layers = dict(lm.layers.named_children())

        handles = []
        for name, m in masks.items():
            if name not in layers:
                continue
            hook = self._make_correction_hook(name, m, clean_acts[name])
            h = layers[name].register_forward_hook(hook)
            handles.append(h)

        return handles

    # ------------------------------------------------------------------
    # Public API — mirrors LocalSpeechLLM
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        text: str,
        temperature: float = 0.01,
        max_tokens: int = 512,
        repetition_penalty: float = 1.0,
    ) -> str:
        # 1. Build clean reference
        clean_path = self._get_clean_audio_path(prompt)

        try:
            # 2. Capture clean & noisy activations
            clean_acts = self._run_forward_for_activations(clean_path, text)
            noisy_acts = self._run_forward_for_activations(prompt, text)

            # 3. Identify noisy neurons
            masks = identify_noisy_neurons(
                clean_acts, noisy_acts, self.top_k_percent
            )

            # 4. Install corrective hooks
            handles = self._install_hooks(masks, clean_acts)

            try:
                # 5. Generate with hooks active
                response = self.base_model.generate(
                    prompt, text, temperature, max_tokens, repetition_penalty
                )
            finally:
                for h in handles:
                    h.remove()
        finally:
            # Clean up temp file
            if os.path.exists(clean_path):
                os.unlink(clean_path)

        return response

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        texts: List[str],
        temperature: float = 0.01,
        max_tokens: int = 512,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """Process each item individually with SPIRIT hooks.

        Batch processing is not trivially compatible with per-sample
        activation comparison, so we iterate.  This is consistent with
        the paper's evaluation methodology.
        """
        results = []
        for prompt, text in zip(prompts, texts):
            r = self.generate(prompt, text, temperature, max_tokens, repetition_penalty)
            results.append(r)
        return results
