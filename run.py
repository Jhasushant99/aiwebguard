# -----------------------------
# 0. FORCE GPU 3
# -----------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from vllm import LLM, SamplingParams
import base64
import json

# -----------------------------
# 1. CONFIG
# -----------------------------
MODEL_PATH = "./model/qwen-vl"
IMAGE_PATH = "test.jpg"

# -----------------------------
# 2. LOAD MODEL
# -----------------------------
print("🚀 Loading model...")
llm = LLM(
    model=MODEL_PATH,
    dtype="float16",
    gpu_memory_utilization=0.9
)

# -----------------------------
# 3. IMAGE → BASE64
# -----------------------------
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_base64 = encode_image(IMAGE_PATH)

# -----------------------------
# 4. PROMPT (STRICT JSON)
# -----------------------------
prompt = """
You are a cybersecurity AI.

Analyze the given image for:
- phishing
- scam
- fake login
- suspicious UI

Return ONLY JSON:
{
  "label": "safe/scam/phishing/suspicious",
  "confidence": 0-1,
  "reason": "short explanation"
}
"""

# -----------------------------
# 5. INPUT FORMAT (CRITICAL)
# -----------------------------
inputs = [
    {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image_base64
        }
    }
]

# -----------------------------
# 6. GENERATE
# -----------------------------
print("🧠 Analyzing image...")

outputs = llm.generate(
    inputs,
    SamplingParams(
        max_tokens=300,
        temperature=0
    )
)

result = outputs[0].outputs[0].text

print("\n🤖 Raw Output:\n", result)

# -----------------------------
# 7. PARSE JSON
# -----------------------------
try:
    json_start = result.find("{")
    json_data = json.loads(result[json_start:])
    
    print("\n✅ FINAL RESULT:\n")
    print(json.dumps(json_data, indent=2))

except Exception as e:
    print("\n⚠️ JSON parsing failed:", e)
