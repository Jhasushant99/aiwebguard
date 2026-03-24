import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import easyocr
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import json

# -----------------------------
# PATHS
# -----------------------------
MODEL_PATH = "./model/qwen-vl"
IMAGE_PATH = "test.jpg"

# -----------------------------
# OCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=True)

# -----------------------------
# MODEL
# -----------------------------
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

print("🔥 GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")

# -----------------------------
# OCR
# -----------------------------
ocr_text = " ".join(reader.readtext(IMAGE_PATH, detail=0))

# -----------------------------
# PROMPT
# -----------------------------
prompt = f"""
Extracted text: {ocr_text}

Analyze for phishing/scam.

Return JSON:
{{
  "label": "",
  "confidence": 0-1,
  "reason": ""
}}
"""

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]
}]

# -----------------------------
# RUN
# -----------------------------
inputs = processor(text=messages, images=image, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=300)

result = processor.decode(output[0], skip_special_tokens=True)

print(result)

# SAVE
with open("outputs/result.json", "w") as f:
    f.write(result)
