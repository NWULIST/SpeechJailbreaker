#importing official defense and input prompt object 
from Defenses.SmoothLLM.smoothllm import SmoothLLM
from Defenses.SmoothLLM.smoothllm import Defense
from Defenses.SmoothLLM.smooth_prompt import smooth_prompt
from BOOST.Attack_GPTFuzzer.gptfuzzer.llm import OpenAIAudioLLM
import torch
import threading

class smoothllmWrapper:
    #contructor that takes in LocalSpeechLLM, perturb method, % of characters to perturb, & number of samples to take 
    def __init__(self, base_model, pert_type="RandomSwapPerturbation", pert_pct=0.15, num_copies=6):
        """
        base_model: represents inputted LocalSpeechLLM
        """
        self.base_model = base_model
        self._current_audio = None
        self._lock = threading.Lock()

        # SmoothLLM expects target_model(batch=..., max_new_tokens=...)
        # wrap base_model into callable format for smoothllm
        self.callable_model = self._build_callable_model()

        #initialize smoothllm defense
        self.smoothllm = SmoothLLM(
            target_model=self.callable_model,
            pert_type=pert_type,
            pert_pct=pert_pct,
            num_copies=num_copies
        )

    def _build_callable_model(self):
        """
        Converts LocalSpeechLLM into a callable model
        that matches SmoothLLM input expectations.

        SmoothLLM calls `self.target_model(batch=batch, max_new_tokens=...)` so smoothLLMWrapper must match
        these parameters
        """

        base_model = self.base_model
        wrapper = self

        class CallableModel:
        
            #make class callable
            def __call__(self, batch, max_new_tokens):

                if wrapper._current_audio is None:
                    raise RuntimeError("Audio prompt not set before SmoothLLM call")

                audio_prompts = [wrapper._current_audio] * len(batch)

                if isinstance(base_model, OpenAIAudioLLM):
                    return base_model.generate_batch(
                        prompts=batch,
                        audios=audio_prompts,
                        max_tokens=max_new_tokens
                    )
                else:
                    # LocalSpeechLLM: positional (audio, text)
                    return base_model.generate_batch(
                        audio_prompts,
                        batch,
                        max_tokens=max_new_tokens
                    )

                #array to collect reponses
                # outputs = []

                #prompt_text = perturbed sample strings
                # for prompt_text in batch:

                #     # Using assumption that audio path is in base model
                #     with torch.no_grad():
                #         response = base_model.generate(
                #             self.question_audio,
                #             prompt_text,
                #             max_tokens=max_new_tokens
                #         )
                #     outputs.append(response)
                # return outputs

        return CallableModel()

    def generate(self, question_audio, prompt_text, max_tokens=512):

        with self._lock:
            self._current_audio = question_audio
            prompt_input = smooth_prompt(prompt_text, max_tokens)
            result = self.smoothllm(prompt_input)
            self._current_audio = None

        return result
      
        #stores audio so callable model can access it
        # self.callable_model.question_audio = question_audio
        # prompt_input = smooth_prompt(prompt_text, max_tokens)

        # torch.cuda.empty_cache() 

        # return self.smoothllm(prompt_input)

    def generate_batch(self, questions, prompts, max_tokens=512):
        """
        Batch interface required by GPTFuzzer.
        questions: list of audio paths
        prompts: list of text prompts
        """
        outputs = []

        for question_audio, prompt_text in zip(questions, prompts):
            response = self.generate(
                question_audio,
                prompt_text,
                max_tokens=max_tokens
            )
            outputs.append(response)

        return outputs
