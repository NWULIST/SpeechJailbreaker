from BOOST.utils.constants import PROXY_BASE_URL
from openai import OpenAI
import anthropic  # noqa: F401  (kept for compat with original imports)
import os
import time
import torch
import gc
from typing import Dict, List
import urllib3
from copy import deepcopy

from config import LLAMA_API_LINK, VICUNA_API_LINK


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        raise NotImplementedError


class HuggingFace(LanguageModel):
    """Unchanged from original — local HuggingFace inference path."""

    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self, full_prompts_list, max_n_tokens, temperature, top_p=1.0):
        inputs = self.tokenizer(full_prompts_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1,
            )

        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):
        self.eos_token_ids.extend(
            [self.tokenizer.encode("}")[1], 29913, 9092, 16675]
        )


class APIModel(LanguageModel):
    """Unchanged from original — Together-style HTTP API path."""

    API_HOST_LINK = "ADD_LINK"
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    API_TIMEOUT = 100
    MODEL_API_KEY = os.getenv("MODEL_API_KEY")
    API_HOST_LINK = ''

    def generate(self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float):
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                if temperature > 0:
                    payload = {
                        "top_p": top_p,
                        "num_beams": 1,
                        "temperature": temperature,
                        "do_sample": True,
                        "prompt": "",
                        "max_new_tokens": max_n_tokens,
                        "system_prompt": conv,
                    }
                else:
                    payload = {
                        "top_p": 1,
                        "num_beams": 1,
                        "temperature": 1,
                        "do_sample": False,
                        "prompt": "",
                        "max_new_tokens": max_n_tokens,
                        "system_prompt": conv,
                    }
                    if "llama" in self.model_name:
                        payload["extra_eos_tokens"] = 0

                if "llama" in self.model_name:
                    assert payload["prompt"] == ""
                    payload["prompt"] = deepcopy(payload["system_prompt"])
                    del payload["system_prompt"]

                resp = urllib3.request(
                    "POST",
                    self.API_HOST_LINK,
                    headers={"Authorization": f"Api-Key {self.MODEL_API_KEY}"},
                    timeout=urllib3.Timeout(self.API_TIMEOUT),
                    json=payload,
                )
                resp_json = resp.json()

                if "vicuna" in self.model_name:
                    if "error" in resp_json:
                        print(self.API_ERROR_OUTPUT)
                    output = resp_json["output"]
                else:
                    output = resp_json

                if isinstance(output, list):
                    output = output[0]
                break
            except Exception as e:
                print("exception!", type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(self, convs_list, max_n_tokens, temperature, top_p=1.0):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


class APIModelLlama7B(APIModel):
    API_HOST_LINK = LLAMA_API_LINK
    MODEL_API_KEY = os.getenv("LLAMA_API_KEY")


class APIModelVicuna13B(APIModel):
    API_HOST_LINK = VICUNA_API_LINK
    MODEL_API_KEY = os.getenv("VICUNA_API_KEY")


class GPT(LanguageModel):
    """OpenAI / gateway-routed GPT — Path A patched."""

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    API_TIMEOUT = 20

    def __init__(self, model_name, token=None, base_url=None):
        self.model_name = model_name
        self.token = token
        resolved_key = token or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or PROXY_BASE_URL
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base_url)


    def generate(self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float):
        """Unchanged from original — same chat.completions.create call."""
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=self.API_TIMEOUT,
                )
                output = response.choices[0].message.content
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(self, convs_list, max_n_tokens, temperature, top_p=1.0):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


class PaLM:
    """Stub — original implementation unchanged."""
    pass


class GeminiPro:
    """Stub — original implementation unchanged."""
    pass