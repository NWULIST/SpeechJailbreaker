#importing official defense and input prompt object 
from Defenses.SmoothLLM.smoothllm import SmoothLLM
from Defenses.SmoothLLM.smoothllm import Defense
from Defenses.SmoothLLM.smooth_prompt import smooth_prompt

class smoothllmWrapper:
    #contructor that takes in LocalSpeechLLM, perturb method, % of characters to perturb, & number of samples to take 
    def __init__(self, base_model, pert_type="RandomSwapPerturbation", pert_pct=0.1, num_copies=2):
        """
        base_model: represents inputted LocalSpeechLLM
        """
        self.base_model = base_model

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

        class CallableModel:
            #make class callable
            def __call__(self, batch, max_new_tokens):
                #array to collect reponses
                outputs = []

                #prompt_text = perturbed sample strings
                for prompt_text in batch:
            
                    # Using assumption that audio path is in base model
                    response = base_model.generate(
                        self.question_audio,
                        prompt_text,
                        max_tokens=max_new_tokens
                    )
                    outputs.append(response)
                return outputs

        return CallableModel()

    def generate(self, question_audio, prompt_text, max_tokens=512):
      
        #stores audio so callable model can access it
        self.callable_model.question_audio = question_audio

        prompt_input = smooth_prompt(prompt_text, max_tokens)

        return self.smoothllm(prompt_input)
