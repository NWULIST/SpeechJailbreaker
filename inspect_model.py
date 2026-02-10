import inspect
from transformers import Qwen2AudioForConditionalGeneration

try:
    print("Source of forward:")
    print(inspect.getsource(Qwen2AudioForConditionalGeneration.forward))
except Exception as e:
    print(f"Could not get source of forward: {e}")

try:
    print("\nSource of prepare_inputs_for_generation:")
    print(inspect.getsource(Qwen2AudioForConditionalGeneration.prepare_inputs_for_generation))
except Exception as e:
    print(f"Could not get source of prepare_inputs_for_generation: {e}")
