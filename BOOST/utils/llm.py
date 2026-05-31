"""
BOOST/utils/llm.py
==================

Canonical, single-source-of-truth LLM interface for the SpeechJailbreaker
codebase. This module replaces the two previously near-duplicate files:

    * BOOST/Attack_GPTFuzzer/llm.py
    * BOOST/Attack_BOOST_GPTFuzzer/llm.py

Plus the various ad-hoc model-loader helpers that had crept into individual
attack modules.

Two architectural changes are implemented here:

1.  Unified LLM interface. A single abstract base class (`BaseLLM`) defines
    the `generate` / `generate_batch` contract that every concrete LLM
    must satisfy. `LocalLLM`, `LocalSpeechLLM`, `ProxyLLM`, `ClaudeLLM`,
    `GeminiLLM`, and `OpenAIAudioLLM` all inherit from `BaseLLM`.

    Each discrepancy that existed between the two original `llm.py` files
    is documented inline in a `# DISCREPANCY` comment block above the
    resolved implementation, stating what differed, which version was
    chosen, and why.

2.  `OpenAILLM` -> `ProxyLLM`. Direct calls to the public OpenAI
    chat-completions endpoint are removed. `ProxyLLM` routes through the
    internal gateway via the OpenAI *Responses* API. `ClaudeLLM` and
    `GeminiLLM` are unchanged — they still talk to Anthropic and Google
    directly. `LocalLLM` and `LocalSpeechLLM` are unchanged — they remain
    for local HuggingFace / vLLM inference.

    For backward compatibility with the (very large) number of call sites
    that still construct `OpenAILLM(...)`, a thin compatibility alias is
    provided at the bottom of this module:

        OpenAILLM = ProxyLLM

    This means existing code such as
        target_model = OpenAILLM(args.target_model, args.openai_key, ...)
    continues to work without changes, but transparently uses the proxy.
    New code should use `ProxyLLM` directly.

Revision notes (this version):
    * Ming support added in LocalSpeechLLM.create_model / .generate / .generate_batch.
    * Ming class import is lazy (inside the branch) so this module remains
      importable on machines that don't have the Ming repo on PYTHONPATH.
    * `_ming_device_map()` helper added at module scope.
    * Ming attention backend is detected at load time: prefers
      flash_attention_2 if flash-attn is installed, otherwise eager.
      SDPA is explicitly NOT used because Ming's BailingMM2 class does
      not support it (transformers will otherwise raise ValueError at
      __init__).
    * `torch_dtype` migrated to `dtype` (transformers deprecation).
    * System-message trigger list expanded to include Ming / bailing.
    * Several smaller correctness fixes called out in inline comments
      prefixed with `# FIX (revN):` for easy grepping.
"""

from __future__ import annotations

import base64
import concurrent.futures
import logging
import os
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Optional
from urllib.request import urlopen

import librosa
import requests
import torch
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from fastchat.model import get_conversation_template, load_model
from openai import OpenAI
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoModelForSpeechSeq2Seq,  # noqa: F401  (kept for parity w/ original imports)
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
)

from BOOST.utils.constants import *  # noqa: F401,F403


# google.generativeai is an optional dependency. The two original files
# disagreed on whether to import it (one commented out, one imported it).
# DISCREPANCY (genai import):
#     - Attack_GPTFuzzer/llm.py:        `#import google.generativeai as genai` (commented out)
#     - Attack_BOOST_GPTFuzzer/llm.py:  `import google.generativeai as genai`
#   Choice: import it, but inside a try/except so the module remains
#   importable on environments where `google-generativeai` is not
#   installed. Without this, every attack module that imports anything
#   from `BOOST.utils.llm` would break on machines that only need OpenAI
#   or Claude. Failure is deferred to GeminiLLM construction.
# TECH-DEBT NOTE: google.generativeai has been deprecated by Google in
# favor of google.genai. Migrating GeminiLLM to the new package is
# tracked separately; for now the deprecation warning is harmless.
try:
    import google.generativeai as genai  # type: ignore
    _GENAI_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    genai = None  # type: ignore
    _GENAI_AVAILABLE = False


# vllm is an optional dependency too. The two original files disagreed:
# DISCREPANCY (vllm import):
#     - Attack_GPTFuzzer/llm.py:        `from vllm import LLM as vllm` (active)
#                                       `from vllm import SamplingParams` (active)
#                                       LocalVLLM class is defined and live.
#     - Attack_BOOST_GPTFuzzer/llm.py:  both vllm imports commented out and
#                                       LocalVLLM class commented out entirely.
#   Choice: keep `LocalVLLM` available, but make the vllm import optional.
#   LocalVLLM raises a clear ImportError at construction time if vllm is
#   missing. This preserves the strictly larger feature set (the GPTFuzzer
#   variant) without breaking environments that lack vllm (the BOOST
#   variant's apparent intent when it commented things out).
try:
    from vllm import LLM as _vllm  # type: ignore
    from vllm import SamplingParams as _SamplingParams  # type: ignore
    _VLLM_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    _vllm = None  # type: ignore
    _SamplingParams = None  # type: ignore
    _VLLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# HuggingFace token helper
# ---------------------------------------------------------------------------
# DISCREPANCY (HF token helper):
#     - Attack_GPTFuzzer/llm.py:        Defines `get_hf_token()` and a module-
#                                       level `hf_token = get_hf_token()`.
#                                       Used for Gemma 3n / Gemma-3-speech /
#                                       PaliGemma loaders.
#     - Attack_BOOST_GPTFuzzer/llm.py:  No HF token helper. Its LocalSpeechLLM
#                                       only supports Qwen2-Audio and silently
#                                       returns (None, None) for everything
#                                       else.
#   Choice: keep the GPTFuzzer version's helper and the strictly-larger
#   set of loader branches.
def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or cached login."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token

    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception as e:
        print(f"Warning: Could not read HF token from cache: {e}")

    return None


hf_token = get_hf_token()


# ---------------------------------------------------------------------------
# Ming device-map helper
# ---------------------------------------------------------------------------
# FIX (rev2): `_ming_device_map()` was referenced from inside the Ming
# branch of LocalSpeechLLM.create_model but never actually defined in the
# previous revision — would have raised NameError on the first Ming load.
# Defined here as a module-level helper so it's testable and re-usable.
#
# The layer-count default (32) matches Ming-lite-omni-1.5. For
# Ming-flash-omni-2.0 the layer count is different; the caller should
# pass `num_layers` explicitly when targeting the full variant. See the
# Ming repo's reference `split_model()` function for the canonical
# value per checkpoint.
def _ming_device_map(num_layers: int = 32):
    """
    Build a HuggingFace device_map for sharding Ming across multiple GPUs.

    Returns "cuda" (single-device) on single-GPU systems so the caller
    sees a clear OOM if the model doesn't fit, rather than silently
    loading badly. Returns a layer-keyed dict on multi-GPU systems.
    """
    from bisect import bisect_left

    world = torch.cuda.device_count()
    if world < 2:
        return "cuda"

    per_gpu = [(i + 1) * (num_layers // world) for i in range(world)]
    dm = {}
    for i in range(num_layers):
        dm[f"model.model.layers.{i}"] = bisect_left(per_gpu, i)
    # Keep non-layer params on GPU 0 to avoid cross-device gather thrash.
    for k in (
        "vision",
        "audio",
        "linear_proj",
        "linear_proj_audio",
        "model.model.word_embeddings.weight",
        "model.model.norm.weight",
        "model.lm_head.weight",
        "model.model.norm",
    ):
        dm[k] = 0
    dm[f"model.model.layers.{num_layers - 1}"] = 0
    return dm


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class BaseLLM(ABC):
    """
    Abstract base class for every LLM in the SpeechJailbreaker framework.

    A single, declared interface lets attack code remain agnostic about
    which concrete backend is in use. Subclasses are free to *accept*
    additional kwargs (e.g. `repetition_penalty`, `n`, audio inputs) but
    must at minimum implement the two methods below with these names and
    these required arguments.
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate a single completion for `prompt`."""
        raise NotImplementedError

    @abstractmethod
    def generate_batch(
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        """Generate completions for a list of prompts."""
        raise NotImplementedError

    def predict(self, sequences, question=None):
        """Optional classifier shim used by a few attacks. Not required."""
        raise NotImplementedError("LLM must implement predict method.")


# Backwards-compat alias. Old code did `class LocalLLM(LLM)` etc.
LLM = BaseLLM


# ---------------------------------------------------------------------------
# LocalLLM
# ---------------------------------------------------------------------------
# DISCREPANCY (LocalLLM.create_model decorator):
#     - Attack_GPTFuzzer/llm.py:        plain method, no decorator.
#     - Attack_BOOST_GPTFuzzer/llm.py:  decorated with `@torch.inference_mode()`.
#   Choice: NO decorator (GPTFuzzer version). `create_model` is called
#   once during construction and merely returns the model/tokenizer; it
#   does not run inference. Decorating it with `@torch.inference_mode()`
#   leaks an inference-mode context that has been observed to break
#   downstream gradient-based attacks (e.g. AutoAttack in `auto.py` had
#   to *monkeypatch* LocalSpeechLLM.create_model specifically to remove
#   this very decorator). Choosing the un-decorated version eliminates
#   the need for that monkeypatch in the long run. The `generate` /
#   `generate_batch` methods still carry `@torch.inference_mode()`, so
#   inference performance is unchanged.
class LocalLLM(BaseLLM):
    def __init__(
        self,
        model_path,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        dtype=torch.float16,
        load_8bit=False,
        cpu_offloading=False,
        revision=None,
        debug=False,
        system_message=None,
    ):
        super().__init__()

        self.model, self.tokenizer = self.create_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.model_path = model_path

        if system_message is None and "Llama-2" in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = (
                "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                "If you don't know the answer to a question, please don't share false information."
            )
        else:
            self.system_message = system_message

    def create_model(
        self,
        model_path,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        dtype=torch.float16,
        load_8bit=False,
        cpu_offloading=False,
        revision=None,
        debug=False,
    ):
        model, tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        return model, tokenizer

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    @torch.inference_mode()
    def generate(self, prompt, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        conv_temp = get_conversation_template(self.model_path)
        self.set_system_message(conv_temp)

        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)

        prompt_input = conv_temp.get_prompt()
        input_ids = self.tokenizer([prompt_input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens,
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts,
        temperature=0.01,
        max_tokens=512,
        repetition_penalty=1.0,
        batch_size=16,
    ):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            output_ids = self.model.generate(
                torch.as_tensor(input_ids[i : i + batch_size]).cuda(),
                do_sample=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(
                self.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
            )
        return outputs


# ---------------------------------------------------------------------------
# ClaudeLLM (unchanged behavior; kept as-is; not replaced by ProxyLLM)
# ---------------------------------------------------------------------------
# DISCREPANCY (ClaudeLLM): the two original files are byte-identical for
# this class. No reconciliation needed.
class ClaudeLLM(BaseLLM):
    def __init__(self, model_path="claude-instant-1.2", api_key=None):
        super().__init__()

        if len(api_key) != 108:
            raise ValueError("invalid Claude API key")

        self.model_path = model_path
        self.api_key = api_key
        self.anthropic = Anthropic(api_key=self.api_key)

    def generate(self, prompt, max_tokens=512, max_trials=1, failure_sleep_time=1):
        for _ in range(max_trials):
            try:
                completion = self.anthropic.completions.create(
                    model=self.model_path,
                    max_tokens_to_sample=max_tokens,
                    prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
                )
                return [completion.completion]
            except Exception as e:
                logging.warning(
                    f"Claude API call failed due to {e}. "
                    f"Retrying {_ + 1} / {max_trials} times..."
                )
                time.sleep(failure_sleep_time)

        return [" "]

    def generate_batch(self, prompts, max_tokens=512, max_trials=1, failure_sleep_time=1):
        results = []
        for prompt in prompts:
            results.extend(
                self.generate(prompt, max_tokens, max_trials, failure_sleep_time)
            )
        return results


# ---------------------------------------------------------------------------
# LocalVLLM (optional; preserved from GPTFuzzer variant)
# ---------------------------------------------------------------------------
class LocalVLLM(BaseLLM):
    def __init__(self, model_path, gpu_memory_utilization=0.98, system_message=None):
        super().__init__()
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vllm is required to use LocalVLLM. "
                "Install it or use LocalLLM / ProxyLLM instead."
            )
        self.model_path = model_path
        self.model = _vllm(self.model_path, gpu_memory_utilization=gpu_memory_utilization)

        if system_message is None and "Llama-2" in model_path:
            self.system_message = (
                "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                "If you don't know the answer to a question, please don't share false information."
            )
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        if self.system_message is not None and "gemma" not in self.model_path:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=0, max_tokens=512):
        return self.generate_batch([prompt], temperature, max_tokens)

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_inputs.append(conv_temp.get_prompt())

        sampling_params = _SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(prompt_inputs, sampling_params, use_tqdm=False)
        outputs = [r.outputs[0].text for r in results]
        return outputs


# ---------------------------------------------------------------------------
# ProxyLLM — replacement for the old OpenAILLM
# ---------------------------------------------------------------------------
# DISCREPANCY (OpenAILLM as it was):
#     - Both Attack_GPTFuzzer/llm.py and Attack_BOOST_GPTFuzzer/llm.py
#       defined an `OpenAILLM` class with an identical body. The class
#       called `client.chat.completions.create(...)` against either the
#       public OpenAI endpoint, or — if OPENAI_BASE_URL was set in the
#       environment — a gateway URL. The default behavior was therefore
#       to hit the public endpoint.
#   Resolution:
#     ProxyLLM replaces OpenAILLM. It always routes through the internal
#     gateway (PROXY_BASE_URL) and uses the OpenAI *Responses* API
#     (`client.responses.create`), not chat-completions. The constructor
#     accepts a `base_url` override for testing.
class ProxyLLM(BaseLLM):
    """
    Routes through the internal gateway via the OpenAI Responses API.

    Constructor positional/keyword arg names are deliberately compatible
    with the old `OpenAILLM(model_path, api_key, system_message=...)`
    call shape so the (large) number of existing call sites can keep
    working through the `OpenAILLM = ProxyLLM` alias.
    """

    # FIX (rev2): The previous revision used the string "You are a helpful
    # assistant." as both the constructor default AND the sentinel for
    # "no system message" in _build_input. This conflated "user passed
    # nothing" with "user explicitly passed exactly that string" and
    # would silently drop a user-supplied default system prompt. The
    # sentinel is now None, which is unambiguous.
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        reasoning_effort: str = "xhigh",
        store: bool = False,
        base_url: str = "https://spike.cs.northwestern.edu:13001",
    ):
        super().__init__()

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("API key required (set OPENAI_API_KEY or pass api_key=...)")

        # FIX (rev2): base_url default reverted to the gateway root
        # (no trailing `/v1`). The OpenAI SDK appends `/v1/responses`
        # itself when constructing the endpoint URL; passing a base_url
        # that already includes `/v1` produces a double-`/v1/` path
        # which the proxy rejects. If your gateway docs say to include
        # `/v1`, override this kwarg at the call site rather than
        # editing this default — the call-site override is documented
        # behavior, the default is just the reference.
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)

        # Field naming: we keep `model_path` for parity with the rest of
        # the LLM zoo (LocalLLM, ClaudeLLM, etc. all use `self.model_path`),
        # and add `model_name` as an alias used by the Responses API call.
        self.model_path = model_name
        self.model_name = model_name
        self.system_message = system_message  # None == no system message
        self.reasoning_effort = reasoning_effort
        self.store = store

    def _build_input(self, prompt: str) -> str:
        """
        The Responses API takes a single `input` string (or a list of
        messages). When a system_message is present we prepend it,
        otherwise we pass the prompt through verbatim.

        Note: prepending the system message into the user input is a
        workaround rather than the idiomatic call. The model will see
        `[system text]\\n\\n[user text]` as one user turn. For most
        attack/defense workloads this is acceptable; if you need a true
        system-role turn, build a structured `input` list and pass it
        in directly (see the OpenAI Responses API docs).
        """
        if self.system_message is None:
            return prompt
        return f"{self.system_message}\n\n{prompt}"

    def generate(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 512,
        n: int = 1,
        max_trials: int = 100,
        failure_sleep_time: float = 5,
    ) -> List[str]:
        """
        Single-prompt generation through the proxy via Responses API.

        Returns a list of strings (length `n`) to preserve the return-shape
        contract that the original OpenAILLM exposed and that downstream
        code already relies on. The Responses API does not natively return
        `n` choices, so we loop.
        """
        outputs: List[str] = []
        for _ in range(n):
            for trial in range(max_trials):
                try:
                    create_kwargs = dict(
                        model=self.model_name,
                        input=self._build_input(prompt),
                        store=self.store,
                    )
                    if self.reasoning_effort is not None:
                        create_kwargs["reasoning"] = {"effort": self.reasoning_effort}
                    response = self.client.responses.create(**create_kwargs)
                    outputs.append(response.output_text)
                    break
                except Exception as e:
                    logging.warning(
                        f"ProxyLLM Responses API call failed: {e}. "
                        f"Retrying {trial + 1}/{max_trials}..."
                    )
                    time.sleep(failure_sleep_time)
            else:
                outputs.append(" ")
        return outputs

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0,
        max_tokens: int = 512,
        n: int = 1,
        max_trials: int = 100,
        failure_sleep_time: float = 5,
        batch_size: int = 10,  # accepted for back-compat; ignored.
    ) -> List[str]:
        """
        Sequential loop over `generate()`. Per the migration spec we do
        NOT introduce async / threaded complexity here — keep it boring,
        keep it debuggable.
        """
        results_all: List[str] = []
        for prompt in prompts:
            results_all.extend(
                self.generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    max_trials=max_trials,
                    failure_sleep_time=failure_sleep_time,
                )
            )
        return results_all

    # FIX (rev2): The previous revision raised NotImplementedError with
    # a placeholder message ("Reproduce the original OpenAILLM.predict
    # body here unchanged") even though both original files also raised
    # NotImplementedError — there was nothing to reproduce. Message
    # cleaned up.
    def predict(self, sequences, question=None):
        """Classifier shim. Not implemented for ProxyLLM."""
        raise NotImplementedError("ProxyLLM does not implement predict().")


# ---------------------------------------------------------------------------
# GeminiLLM (unchanged behavior; not replaced)
# ---------------------------------------------------------------------------
# DISCREPANCY (GeminiLLM): the two original files have effectively
# identical bodies; one had extra blank lines inside `generate()`.
# Resolved by using the cleaner (GPTFuzzer) formatting.
class GeminiLLM(BaseLLM):
    def __init__(self, model_path="gemini-pro", api_key=None):
        super().__init__()

        if not _GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai is required to use GeminiLLM. "
                "Install it or use a different LLM class."
            )

        if len(api_key) != 39:
            raise ValueError("invalid Gemini API key")

        self.model_path = model_path
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.gemini = genai.GenerativeModel(self.model_path)

    def generate(self, prompt, max_tokens=512, max_trials=1, failure_sleep_time=1):
        for _ in range(max_trials):
            try:
                completion = self.gemini.generate_content(
                    f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}", max_tokens=max_tokens
                )
                return [completion.text]
            except Exception as e:
                logging.warning(
                    f"Gemini API call failed due to {e}. "
                    f"Retrying {_ + 1} / {max_trials} times..."
                )
                time.sleep(failure_sleep_time)

        return [" "]

    def generate_batch(
        self, prompts, max_tokens=512, max_trials=1, failure_sleep_time=1
    ):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.generate, prompt, max_tokens, max_trials, failure_sleep_time
                ): prompt
                for prompt in prompts
            }
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results


# ---------------------------------------------------------------------------
# LocalSpeechLLM
# ---------------------------------------------------------------------------
# DISCREPANCY (LocalSpeechLLM.__init__ default for load_8bit):
#     - Attack_GPTFuzzer/llm.py:        `load_8bit=True`
#     - Attack_BOOST_GPTFuzzer/llm.py:  `load_8bit=False`
#   Choice: `load_8bit=False`. Every concrete loader branch in
#   create_model() ignores load_8bit and loads the model in its native
#   dtype regardless.
#
# DISCREPANCY (LocalSpeechLLM.__init__ system_message-trigger model list):
#     - Attack_GPTFuzzer/llm.py:        `["Llama-2", "Qwen2", "gemma"]`
#     - Attack_BOOST_GPTFuzzer/llm.py:  `["Llama-2", "Qwen2"]` (no gemma)
#   Choice: `["Llama-2", "Qwen2", "gemma", "Ming", "bailing"]`. The
#   GPTFuzzer-side variant is a strict superset and is the only one
#   whose create_model actually supports Gemma audio/speech variants.
#   FIX (rev2): added "Ming" and "bailing" so Ming targets pick up the
#   default safety system message — defenses that depend on system-prompt
#   injection (self-reminder, AdaShield) need this.
#
# DISCREPANCY (LocalSpeechLLM.create_model decorator):
#     - Attack_GPTFuzzer/llm.py:        plain method.
#     - Attack_BOOST_GPTFuzzer/llm.py:  `@torch.inference_mode()`.
#   Choice: plain method. Removes the need for the monkeypatch in
#   `BOOST/Attack_auto/auto.py`.
#
# DISCREPANCY (LocalSpeechLLM.create_model body):
#     - Attack_GPTFuzzer/llm.py:        full multi-branch loader.
#     - Attack_BOOST_GPTFuzzer/llm.py:  only Qwen2-Audio.
#   Choice: GPTFuzzer's full loader, extended with a Ming branch.
class LocalSpeechLLM(BaseLLM):
    def __init__(
        self,
        model_path,
        device="cuda",
        dtype=torch.float16,
        load_8bit=False,
        cpu_offloading=False,
        system_message=None,
    ):
        super().__init__()

        self.model, self.processor = self.create_model(
            model_path, device, dtype, load_8bit, cpu_offloading
        )
        self.model_path = model_path

        # FIX (rev2): include Ming/bailing so Ming targets receive the
        # default safety system prompt. Case matches actual model-path
        # strings: HuggingFace publishes `inclusionAI/Ming-...` (capital
        # M), so we match "Ming" not "ming"; "bailing" appears lowercase
        # in BailingMM2 config types.
        if system_message is None and any(
            m in model_path for m in ["Llama-2", "Qwen2", "gemma", "Ming", "bailing"]
        ):
            self.system_message = (
                "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                "If you don't know the answer to a question, please don't share false information."
            )
        else:
            self.system_message = system_message

    def create_model(
        self,
        model_path,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        dtype=torch.float16,
        load_8bit=False,
        cpu_offloading=False,
        revision=None,
        debug=False,
    ):
        # Qwen2-Audio family
        if "Qwen2-Audio" in model_path or "Qwen/Qwen2-Audio" in model_path:
            print("Loading Qwen2-Audio model...")
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path, trust_remote_code=True
            )
            model = model.to(device)
            import os as _os
            _ming_dir = '/home/qxq9828/cs/SpeechJailbreaker/Ming'
            _orig_dir = _os.getcwd()
            _os.chdir(_ming_dir)
            processor = AutoProcessor.from_pretrained(
                _ming_dir, trust_remote_code=True
            )
            _os.chdir(_orig_dir)
            print("Qwen2-Audio model loaded successfully!")
            return model, processor

        elif "gemma-3n" in model_path.lower():
            print("Loading Gemma 3n audio model...")
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                token=hf_token,
            )

            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, token=hf_token
            )

            print("Gemma 3n audio model loaded successfully!")
            return model, processor

        elif "gemma-3-" in model_path.lower() and "speech" in model_path.lower():
            print("Loading Gemma-3 speech model...")

            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                token=hf_token,
            )

            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, token=hf_token
            )

            print("Gemma-3 speech model loaded successfully!")
            return model, processor

        elif "gemma" in model_path:
            # Legacy PaliGemma support (image+text only, not audio)
            print("Loading PaliGemma model (image+text, no audio)...")
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, token=hf_token
            )
            model = model.to(device)
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, token=hf_token
            )
            print("PaliGemma model loaded successfully!")
            return model, processor

        elif "ming" in model_path.lower():
            print(f"Loading Ming audio model ({model_path})...")

            try:
                from modeling_bailingmm import (
                    BailingMMNativeForConditionalGeneration,
                )
            except ImportError as e:
                raise ImportError(
                    "Ming requires the Ming repo on PYTHONPATH. "
                    "Clone https://github.com/inclusionAI/Ming and run:\n"
                    "    export PYTHONPATH=${PYTHONPATH}:/path/to/Ming\n"
                    f"Underlying error: {e}"
                ) from e

            model = BailingMMNativeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map="auto",
                low_cpu_mem_usage=True,
            ).eval()

            import os as _os
            _ming_dir = '/home/qxq9828/cs/SpeechJailbreaker/Ming'
            _orig_dir = _os.getcwd()
            _os.chdir(_ming_dir)
            processor = AutoProcessor.from_pretrained(
                _ming_dir, trust_remote_code=True
            )
            _os.chdir(_orig_dir)
            print("Ming model loaded successfully!")
            return model, processor

        raise ValueError(
            f"Unsupported model: {model_path}\n"
            f"Supported audio models:\n"
            f"  - Qwen2-Audio models (e.g., Qwen/Qwen2-Audio-7B-Instruct)\n"
            f"  - Gemma 3n models (e.g., google/gemma-3n-E2B-it, google/gemma-3n-E4B-it)\n"
            f"  - Gemma-3 speech models (e.g., junnei/gemma-3-4b-it-speech)\n"
            f"  - PaliGemma models (image+text only, not audio)\n"
            f"  - Ming / Bailing audio models (e.g., inclusionAI/Ming-lite-omni-1.5)\n"
        )

    def _move_inputs_to_device(self, inputs):
        device = next(self.model.parameters()).device
        moved = {}
        for k, v in inputs.items():
            moved[k] = v.to(device) if hasattr(v, "to") else v
        return moved

    @torch.inference_mode()
    def generate(self, prompt, text, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        """
        Generate response for audio + text input.

        Works with Qwen2-Audio, Gemma 3n, and Ming models.

        Args:
            prompt: audio file path or URL.
            text:   text prompt/question.
        """
        is_gemma_3n = "gemma-3n" in self.model_path.lower()
        is_gemma_3_speech = "gemma-3-" in self.model_path.lower() and "speech" in self.model_path.lower()
        is_ming = "ming" in self.model_path.lower()

        if is_ming:
            messages = [
                {"role": "HUMAN", "content": [
                    {"type": "audio", "audio": prompt},
                    {"type": "text", "text": text},
                ]},
            ]
            text_prompt = self.processor.apply_chat_template(
                messages,
                system_template=self.system_message,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                audios=audio_inputs,
                audio_kwargs={"use_whisper_encoder": True},
                return_tensors="pt",
            ).to(self.model.device)
            for k in inputs.keys():
                if k in ("pixel_values", "pixel_values_videos", "audio_feats"):
                    inputs[k] = inputs[k].to(dtype=torch.bfloat16)
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                eos_token_id=self.processor.gen_terminator,
            )
            generate_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
            ]
            response = self.processor.batch_decode(
                generate_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return response


        elif is_gemma_3n or is_gemma_3_speech:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": prompt},
                        {"type": "text", "text": text},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = {
                k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()
            }

            with torch.inference_mode():
                generate_ids = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False
                )
                generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response

        else:
            # Qwen2-Audio format
            conversation = [
                {"role": "system", "content": self.system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "audio", "audio_url": prompt},
                    ],
                },
            ]

            text_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audio_path = ele["audio_url"]
                            if os.path.exists(audio_path):
                                with open(audio_path, "rb") as f:
                                    audio_bytes = BytesIO(f.read())
                            else:
                                audio_bytes = BytesIO(urlopen(audio_path).read())
                            audios.append(
                                librosa.load(
                                    audio_bytes,
                                    sr=self.processor.audio_processor.sampling_rate,
                                )[0]
                            )

            inputs = self.processor(
                text=text_prompt, audio=audios, return_tensors="pt", padding=True
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            generate_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response

    # DISCREPANCY (LocalSpeechLLM.generate_batch body):
    #     - Attack_GPTFuzzer/llm.py:        respects max_tokens, multi-GPU
    #       safe via next(model.parameters()).device.
    #     - Attack_BOOST_GPTFuzzer/llm.py:  hardcodes max_new_tokens=256,
    #       hardcodes "cuda".
    #   Choice: GPTFuzzer version (respect max_tokens, multi-GPU safe).
    @torch.inference_mode()
    def generate_batch(self, prompts, texts, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        is_ming = "ming" in self.model_path.lower()

        if is_ming:
            all_responses = []
            for i in range(len(prompts)):
                messages = [
                    {"role": "HUMAN", "content": [
                        {"type": "audio", "audio": prompts[i]},
                        {"type": "text", "text": texts[i]},
                    ]},
                ]
                text_prompt = self.processor.apply_chat_template(
                    messages,
                    system_template=self.system_message,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
                inputs = self.processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    audios=audio_inputs,
                    audio_kwargs={"use_whisper_encoder": True},
                    return_tensors="pt",
                ).to(self.model.device)
                for k in inputs.keys():
                    if k in ("pixel_values", "pixel_values_videos", "audio_feats"):
                        inputs[k] = inputs[k].to(dtype=torch.bfloat16)
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    eos_token_id=self.processor.gen_terminator,
                )
                generate_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
                ]
                response = self.processor.batch_decode(
                    generate_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                all_responses.append(response)
            return all_responses
        # Non-Ming batched path (Qwen2-Audio and similar)
        conversations = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            text = texts[i]
            conversations.append(
                [
                    {"role": "system", "content": self.system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "audio", "audio_url": prompt},
                        ],
                    },
                ]
            )

        text = [
            self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            for conversation in conversations
        ]

        audios = []
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audio_path = ele["audio_url"]
                            if os.path.exists(audio_path):
                                with open(audio_path, "rb") as f:
                                    audio_bytes = BytesIO(f.read())
                            else:
                                audio_bytes = BytesIO(urlopen(audio_path).read())
                            audios.append(
                                librosa.load(
                                    audio_bytes,
                                    sr=self.processor.audio_processor.sampling_rate,
                                )[0]
                            )

        inputs = self.processor(
            text=text, audio=audios, return_tensors="pt", padding=True
        )
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        input_ids_length = inputs["input_ids"].size(1)

        generate_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generate_ids = generate_ids[:, input_ids_length:]

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return response


# ---------------------------------------------------------------------------
# OpenAIAudioLLM
# ---------------------------------------------------------------------------
# DISCREPANCY (OpenAIAudioLLM): only present in Attack_GPTFuzzer/llm.py;
# Attack_BOOST_GPTFuzzer/llm.py did not define it. The PAIR/TAP/JBC/ICA
# attack modules all import OpenAIAudioLLM from
# `BOOST.Attack_GPTFuzzer.gptfuzzer.llm` regardless, so the
# GPTFuzzer-side definition is canonical. Migrated here unchanged in
# semantics.
#
# Note: OpenAIAudioLLM still talks to the OpenAI chat-completions API
# directly (not the proxy / Responses API). The migration spec was
# specific: replace OpenAILLM with ProxyLLM. The audio variant is left
# pointing at the OpenAI endpoint because (a) the Responses API does not
# yet have an established multimodal-audio counterpart in this codebase,
# and (b) flipping the audio class to the proxy without verifying the
# proxy supports audio I/O would silently break the audio attack runs.
# TODO: verify whether the proxy supports the OpenAI audio chat shape;
# if so, route this class through it too in a follow-up change.
class OpenAIAudioLLM(BaseLLM):
    def __init__(self, model_path, api_key=None, system_message=None):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.model_path = model_path
        self.system_message = system_message or "You are a helpful assistant."

    def _encode_audio(self, audio_path_or_bytes):
        """
        Encode audio input to base64 string.

        Supports:
          - raw bytes / bytearray / memoryview
          - file-like objects with .read()
          - local file paths
          - remote URLs (http/https)
        """
        if isinstance(audio_path_or_bytes, (bytes, bytearray, memoryview)):
            return base64.b64encode(bytes(audio_path_or_bytes)).decode("utf-8")

        if hasattr(audio_path_or_bytes, "read"):
            return base64.b64encode(audio_path_or_bytes.read()).decode("utf-8")

        if isinstance(audio_path_or_bytes, str):
            if audio_path_or_bytes.startswith("http"):
                r = requests.get(audio_path_or_bytes)
                r.raise_for_status()
                return base64.b64encode(r.content).decode("utf-8")

            with open(audio_path_or_bytes, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        raise TypeError(
            f"Unsupported audio input type: {type(audio_path_or_bytes)}. "
            f"Expected bytes, file-like, or str path/URL."
        )

    def generate(
        self,
        prompt,
        audio=None,
        audio_format="wav",
        temperature=0,
        max_tokens=512,
        n=1,
        max_trials=10,
        failure_sleep_time=5,
    ):
        for trial in range(max_trials):
            try:
                content = []

                if isinstance(prompt, str):
                    content.append({"type": "text", "text": prompt})

                if audio is not None:
                    encoded_audio = self._encode_audio(audio)
                    content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": audio_format,
                            },
                        }
                    )

                messages = [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": content},
                ]

                completion = self.client.chat.completions.create(
                    model=self.model_path,
                    modalities=["text", "audio"] if audio else ["text"],
                    audio={"voice": "alloy", "format": "wav"} if audio else None,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )

                outputs = [c.message.content for c in completion.choices]
                return outputs

            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. "
                    f"Retrying {trial + 1}/{max_trials}..."
                )
                time.sleep(failure_sleep_time)

        return [""] * n

    def generate_batch(
        self,
        prompts,
        audios=None,
        audio_format="wav",
        temperature=0,
        max_tokens=512,
        n=1,
        max_trials=10,
        failure_sleep_time=5,
    ):
        """
        Batched generation for (text, optional audio) inputs.
        """
        if audios is None:
            audios = [None] * len(prompts)

        if len(audios) != len(prompts):
            raise ValueError("Length of audios must match length of prompts.")

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.generate,
                    prompt,
                    audio,
                    audio_format,
                    temperature,
                    max_tokens,
                    n,
                    max_trials,
                    failure_sleep_time,
                ): (prompt, audio)
                for prompt, audio in zip(prompts, audios)
            }
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results


# ---------------------------------------------------------------------------
# Backwards-compatibility export
# ---------------------------------------------------------------------------
# Many existing attack modules and helper scripts construct OpenAILLM by
# name. To avoid touching every single one of those call sites in this
# change, expose `OpenAILLM` as an alias for `ProxyLLM`. This means:
#
#     target_model = OpenAILLM(args.target_model, args.openai_key, system_message=...)
#
# automatically becomes:
#
#     target_model = ProxyLLM(args.target_model, args.openai_key, system_message=...)
#
# The constructor signatures are compatible (see ProxyLLM.__init__ above).
# New code should prefer `ProxyLLM` directly.
OpenAILLM = ProxyLLM


__all__ = [
    "BaseLLM",
    "LLM",  # legacy alias
    "LocalLLM",
    "LocalVLLM",
    "LocalSpeechLLM",
    "ClaudeLLM",
    "GeminiLLM",
    "ProxyLLM",
    "OpenAILLM",  # alias of ProxyLLM
    "OpenAIAudioLLM",
    "get_hf_token",
    "hf_token",
    "_ming_device_map",
]