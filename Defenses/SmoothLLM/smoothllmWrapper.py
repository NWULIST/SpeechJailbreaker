#importing official defense and input prompt object 
from defenses.smoothllm import SmoothLLM
from defenses.smoothllm import Defense
from defenses.smooth_prompt import smooth_prompt

class SmoothLLMWrapper:
    #contructor that takes in LocalSpeechLLM, perturb method, % of characters to perturb, & number of samples to take 
    def __init__(self, base_model, pert_type="RandomSwapPerturbation", pert_pct=0.1, num_copies=5,):
        """
        base_model: represents inputted LocalSpeechLLM
        """
        self.base_model = base_model

        # SmoothLLM expects target_model(batch=..., max_new_tokens=...)
        # wrap base_model into callable format for smoothllm
        self.callable_model = self._build_callable_model()

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
        """

        base_model = self.base_model

        class CallableModel:
            def __call__(self, batch, max_new_tokens):
                outputs = []
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
        """
        This matches your existing attack interface.
        """
        #stores audio so callable model can access it
        self.callable_model.question_audio = question_audio

        smooth_prompt = smooth_prompt(prompt_text, max_tokens)

        return self.smoothllm(smooth_prompt)
