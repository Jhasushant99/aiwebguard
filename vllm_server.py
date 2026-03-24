import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from vllm import LLM, SamplingParams
import base64

# -----------------------------
# MODEL PATH
# -----------------------------
MODEL_PATH = "./model/qwen-vl"

# -----------------------------
# LOAD MODEL
# -----------------------------
llm = LLM(
    model=MODEL_PATH,
    dtype="float16",
    gpu_memory_utilization=0.9
)

# -----------------------------
# IMAGE ENCODE
# -----------------------------
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_base64 = encode_image("test.jpg")

# -----------------------------
# PROMPT
# -----------------------------
prompt = """
Analyze image for phishing/scam.

Return JSON:
{
  "label": "",
  "confidence": 0-1,
  "reason": ""
}
"""

inputs = [{
    "prompt": prompt,
    "multi_modal_data": {
        "image": image_base64
    }
}]

# -----------------------------
# RUN
# -----------------------------
outputs = llm.generate(
    inputs,
    SamplingParams(max_tokens=300, temperature=0)
)

print(outputs[0].outputs[0].text)





🚀 ✅ STEP 4: RUN COMMANDS
▶ Run transformers (main)
CUDA_VISIBLE_DEVICES=3 python main1.py
⚡ Run vLLM (separate)
CUDA_VISIBLE_DEVICES=3 python vllm_server.py
