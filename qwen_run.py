# qwen_run.py
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from pathlib import Path

MODEL_PATH = Path("models/qwen2.5-vl")

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Load model on CPU
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map={"": "cpu"})

def analyze_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    prompt = """
Analyze this webpage screenshot.
Return ONLY JSON:
{
  "has_login_form": true or false,
  "urgency_text": true or false,
  "risk": "LOW" | "MEDIUM" | "HIGH"
}
"""
    # Preprocess inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # Generate output on CPU
    outputs = model.generate(**inputs, max_new_tokens=512)

    # Decode result
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return result