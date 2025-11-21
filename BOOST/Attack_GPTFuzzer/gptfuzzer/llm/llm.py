import torch
from io import BytesIO
from urllib.request import urlopen
import librosa
import os
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
from vllm import LLM as vllm
from vllm import SamplingParams
from BOOST.utils.constants import *
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import google.generativeai as genai
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModel, Qwen2AudioForConditionalGeneration, AutoModelForImageTextToText
def get_hf_token():
    """Get HuggingFace token from environment or cached login"""
    # Try environment variable first
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if token:
        return token

    # Try to read from huggingface-cli cache
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception as e:
        print(f"Warning: Could not read HF token from cache: {e}")

    return None 
hf_token = get_hf_token()
class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 device='cuda',
                 num_gpus=1,
                 max_gpu_memory=None,
                 dtype=torch.float16,
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 system_message=None
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

        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.float16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
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
            max_new_tokens=max_tokens
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.01, max_tokens=512, repetition_penalty=1.0, batch_size=16):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            output_ids = self.model.generate(
                torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
                do_sample=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
        return outputs


class ClaudeLLM(LLM):
    def __init__(self,
                 model_path='claude-instant-1.2',
                 api_key=None
                ):
        super().__init__()
        
        if len(api_key) != 108:
            raise ValueError('invalid Claude API key')
        
        self.model_path = model_path
        self.api_key = api_key
        self.anthropic = Anthropic(
            api_key=self.api_key
        )

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
                    f"Claude API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" "]
    
    def generate_batch(self, prompts, max_tokens=512, max_trials=1, failure_sleep_time=1):
        results = []
        for prompt in prompts:
            results.extend(self.generate(prompt, max_tokens, max_trials, failure_sleep_time))
        return results

class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.98,
                 system_message=None
                 ):
        super().__init__()
        self.model_path = model_path
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization)
        
        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        if self.system_message is not None and 'gemma' not in self.model_path:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=0, max_tokens=512):
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(
            prompt_inputs, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs

class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                if 'gpt-4o' in self.model_path.lower() or 'gpt-4-turbo' in self.model_path.lower():
                    prompt_text = prompt if isinstance(prompt, str) else str(prompt)
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
                    ]
                else:
                    # For other models, use simple string format
                    messages = [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ]
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    # messages=[
                    #     {"role": "system", "content": self.system_message},
                    #     {"role": "user", "content": prompt},
                    # ],

                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
            
    def predict(self, sequences, question):
        black_list = get_black_list()
        # check if any black list in the sequences
        for black in black_list:
            if black in sequences[0]:
                return [0]
        # verify whether the generation is related with the target
        split = question.split()
        cnt = 0
        for word in split:
            if word in sequences[0]:
                cnt += 1
        if cnt/len(split) < 0.5:
            return [0]
        prediction = self.generate(sequences[0])
        for i in range(len(prediction)):
            try:
                prediction[i] = int(prediction[i])
            except:
                prediction[i] = 0
        # print the sequences if the prediction is harmful
        if prediction[0] == 1:
            print(sequences[0])
        return prediction

class GeminiLLM(LLM):
    def __init__(self,
                 model_path='gemini-pro',
                 api_key=None
                ):
        super().__init__()
        
        if len(api_key) != 39:
            raise ValueError('invalid Gemini API key')
        
        self.model_path = model_path
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.gemini = genai.GenerativeModel(self.model_path)

    def generate(self, prompt, max_tokens=512, max_trials=1, failure_sleep_time=1):
     
   
        for _ in range(max_trials):
            try:
                completion = self.gemini.generate_content(f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}", max_tokens=max_tokens)
                return [completion.text]
            except Exception as e:
                logging.warning(
                    f"Gemini API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" "]
    
    def generate_batch(self, prompts, max_tokens=512, max_trials=1, failure_sleep_time=1):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, max_tokens,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results


class LocalSpeechLLM(LLM):
    def __init__(self,
                 model_path,
                 device='cuda',
                 dtype=torch.float16,
                 load_8bit=True,
                 cpu_offloading=False,
                 system_message=None
                 ):
        super().__init__()

        self.model, self.processor = self.create_model(
            model_path,
            device,
            dtype,
            load_8bit,
            cpu_offloading,
        )
        self.model_path = model_path

        if system_message is None and any(m in model_path for m in ["Llama-2", "Qwen2", "gemma"]):
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.float16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        # Special handling for Qwen2-Audio models which use an Audio processor
        if 'Qwen2-Audio' in model_path or 'Qwen/Qwen2-Audio' in model_path:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path, trust_remote_code=True
            )
            model = model.to(device)
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            return model, processor
        elif 'gemma' in model_path:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, token=hf_token
            )
            model = model.to(device)
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, token=hf_token
            )
            return model, processor
        return None, None


    def _move_inputs_to_device(self, inputs):
        device = next(self.model.parameters()).device
        moved = {}
        for k, v in inputs.items():
            moved[k] = v.to(device) if hasattr(v, 'to') else v
        return moved

    @torch.inference_mode()
    def generate(self, prompt, text, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        conversation = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "audio", "audio_url": prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio_path = ele['audio_url']
                        if os.path.exists(audio_path):  # local file
                            with open(audio_path, 'rb') as f:
                                audio_bytes = BytesIO(f.read())
                        else:  # remote URL
                            audio_bytes = BytesIO(urlopen(audio_path).read())
                        audios.append(librosa.load(
                            audio_bytes, 
                            sr=self.processor.feature_extractor.sampling_rate)[0]
                        )

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
       
        generate_ids = self.model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

    @torch.inference_mode()
    def generate_batch(self, prompts,texts, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        conversations = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            text = texts[i]
            conversations.append([
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "audio_url": prompt},
                ]},
            ])

        text = [self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) for conversation in conversations]

        audios = []

        
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audio_path = ele['audio_url']
                            if os.path.exists(audio_path):  # local file
                                with open(audio_path, 'rb') as f:
                                    audio_bytes = BytesIO(f.read())
                            else:  # remote URL
                                audio_bytes = BytesIO(urlopen(audio_path).read())
                            audios.append(
                                librosa.load(
                                    audio_bytes, 
                                    sr=self.processor.feature_extractor.sampling_rate)[0]
                            )

        # inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        # inputs['input_ids'] = inputs['input_ids'].to("cuda")
        # inputs.input_ids = inputs.input_ids.to("cuda")
        # inputs = {k: v.to(self.model.device) for k, v in inputs.items()}  # ✅ ensure same device


        # generate_ids = self.model.generate(**inputs, max_length=1024)
        # generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        # inputs['input_ids'] = inputs['input_ids'].to("cuda")
        # inputs.input_ids = inputs.input_ids.to("cuda")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_ids_length = inputs['input_ids'].size(1)

        generate_ids = self.model.generate(**inputs, max_length=1024)

        # Remove the input tokens from the generated output
        generate_ids = generate_ids[:, input_ids_length:]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response