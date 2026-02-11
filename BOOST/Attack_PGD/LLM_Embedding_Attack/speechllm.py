
import torch
from io import BytesIO
from urllib.request import urlopen
import librosa
import os
import concurrent.futures
from BOOST.utils.constants import *
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


class LocalSpeechLLM(LLM):
    def __init__(self,
                 model_path,
                 device='cuda',
                 dtype=torch.float16,
                 load_8bit=False,
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

        if system_message is None and any(m in model_path for m in ["Llama-2", "Qwen2"]):
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

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to("cuda")
        inputs.input_ids = inputs.input_ids.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response