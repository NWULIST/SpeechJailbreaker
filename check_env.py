import sys
import torch
import transformers
from transformers import AutoProcessor

print(f"Python version: {sys.version}")
print(f"Transformers version: {transformers.__version__}")

model_path = "Qwen/Qwen2-Audio-7B-Instruct"
try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    target_token_str = "Sure"
    tokens = processor.tokenizer.encode(target_token_str, add_special_tokens=False)
    print(f"Tokens for '{target_token_str}': {tokens}")
    for t in tokens:
        print(f"Token {t}: {processor.tokenizer.decode([t])}")
except Exception as e:
    print(f"Error loading processor: {e}")
