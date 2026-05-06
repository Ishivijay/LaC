#!/usr/bin/env python3
"""Diagnostic script to test Gemma4 image processing.
Runs on a compute node to test both apply_chat_template and legacy patterns.
"""
import os
import sys
import torch
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Setup environment
WORK_DIR = os.environ.get("WORK", os.path.expanduser("~"))
os.environ["HF_HOME"] = os.path.join(WORK_DIR, ".cache", "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_XET"] = "0"

from PIL import Image
import numpy as np

# Monkey-patch for bitsandbytes
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target, module):
        atoms = target.split(".")
        mod = self
        for atom in atoms[:-1]:
            if hasattr(mod, atom):
                mod = getattr(mod, atom)
            else:
                raise AttributeError(f"{mod} has no attribute {atom}")
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule

MODEL_ID = "google/gemma-4-E4B-it"

print("=" * 60)
print("GEMMA4 IMAGE PROCESSING DIAGNOSTIC")
print("=" * 60)

# 1. Load model and processor
print("\n[1] Loading model and processor...")
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

cc = torch.cuda.get_device_capability(0)
dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
print(f"  GPU: {torch.cuda.get_device_name(0)}, dtype={dtype}")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=dtype,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    ),
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"  Model type: {type(model).__name__}")
print(f"  Processor type: {type(processor).__name__}")
print(f"  image_token: {processor.image_token!r}")
print(f"  boi_token: {processor.boi_token!r}")
print(f"  eoi_token: {processor.eoi_token!r}")

# 2. Create a test image (colorful, not gray)
print("\n[2] Creating test image...")
test_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
print(f"  Image: size={test_img.size}, mode={test_img.mode}")

# Load a real image from the dataset
DATA_DIR = os.path.join(WORK_DIR, "mlp_dataset/prospthesisproject-Data/Code/Data")
real_img_path = os.path.join(
    DATA_DIR,
    "lms_kamal_LA_downstairs_Nopeople_1/sharpen_rgb/PNG/image28.png"
)

real_img = None
if os.path.exists(real_img_path):
    real_img = Image.open(real_img_path).convert("RGB")
    print(f"  Real image: {real_img_path}")
    print(f"  Real image: size={real_img.size}, mode={real_img.mode}")
else:
    print(f"  Real image not found: {real_img_path}")

# Use real image if available, otherwise synthetic
test_image = real_img if real_img else test_img
print(f"  Using: {'real' if real_img else 'synthetic'} image")

# 3. Test apply_chat_template(tokenize=True)
print("\n[3] Testing apply_chat_template(tokenize=True)...")
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [
        {"type": "text", "text": "What do you see in this image? Describe briefly."},
        {"type": "image", "image": test_image},
    ]},
]

try:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"  Input keys: {list(inputs.keys())}")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            if 'pixel' in k:
                print(f"      min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
        else:
            print(f"    {k}: {type(v).__name__}")

    # Check for pixel_values
    has_pixel = any(k in inputs for k in ("pixel_values", "pixel_values_images", "pixel_values_videos"))
    print(f"  Has pixel_values: {has_pixel}")

    # Run inference
    print("  Running inference with apply_chat_template inputs...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    response = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"  Response: {response[:300]}")

except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 4. Test legacy pattern
print("\n[4] Testing legacy pattern (processor(text=..., images=...))...")
try:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"  Text preview: {text[:200]}...")
    print(f"  Image token count in text: {text.count(processor.image_token)}")

    inputs_legacy = processor(
        text=[text],
        images=[test_image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"  Legacy input keys: {list(inputs_legacy.keys())}")
    for k, v in inputs_legacy.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            if 'pixel' in k:
                print(f"      min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
        else:
            print(f"    {k}: {type(v).__name__}")

    # Run inference
    print("  Running inference with legacy inputs...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs_legacy,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
        )
    input_len = inputs_legacy["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    response = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"  Response: {response[:300]}")

except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 5. Test with long system prompt (like the actual pipeline)
print("\n[5] Testing with long system prompt (like actual pipeline)...")
try:
    long_system = """You are an expert indoor scene analyst specializing in navigable area detection for prosthesis navigation.
Your task is to analyze the image and identify all free navigable areas where a person can safely walk.

IMPORTANT DEFINITIONS:
- Free navigable area: Any horizontal surface where a person can safely place their foot and walk.
  This includes flat floor, corridors, landings, and stairs (both up and down).
- NOT navigable: Walls, railings, windows, doors, furniture, obstacles.

For each navigable area, provide:
1. A descriptive name
2. A bounding box in percentage coordinates (x1, y1, x2, y2)
3. The type of surface

Respond in JSON format."""

    messages_long = [
        {"role": "system", "content": [{"type": "text", "text": long_system}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze this image and identify all free navigable areas."},
            {"type": "image", "image": test_image},
        ]},
    ]
    inputs_long = processor.apply_chat_template(
        messages_long,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"  Input keys: {list(inputs_long.keys())}")
    for k, v in inputs_long.items():
        if hasattr(v, 'shape') and 'pixel' in k:
            print(f"    {k}: shape={v.shape}, min={v.min().item():.4f}, max={v.max().item():.4f}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs_long,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
        )
    input_len = inputs_long["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    response = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"  Response: {response[:400]}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 6. Test without image (text only)
print("\n[6] Testing text-only (no image)...")
try:
    messages_no_img = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "What is 2+2? Answer briefly."},
        ]},
    ]
    inputs_text = processor.apply_chat_template(
        messages_no_img,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs_text,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
        )
    input_len = inputs_text["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    response = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"  Text-only response: {response[:200]}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
