class smooth_prompt:
    def __init__(self, text_prompt, max_new_tokens):
        self.full_prompt = text_prompt
        self.max_new_tokens = max_new_tokens

    def perturb(self, perturbation_fn):
        self.full_prompt = perturbation_fn(self.full_prompt)
