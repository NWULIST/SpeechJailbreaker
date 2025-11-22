from vertexai import init
from vertexai.generative_models import GenerativeModel

# 1) Initialize with your project + region
init(project="speechllm-476905", location="us-central1")

#Model Choice: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
# 2) Load a Gemini model (e.g., gemini-1.5-pro)
model = GenerativeModel("gemini-2.0-flash-001")

# 3) Send a prompt
resp = model.generate_content("Explain test-time scaling laws for reasoning models.")
print(resp.text)