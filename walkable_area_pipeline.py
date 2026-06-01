#!/usr/bin/env python3
"""Unified Free Ground Space Detection Pipeline
================================================

Supports 4 strategies controlled by --strategy flag:
  1. zero_shot : Single VLM (reasoner prompt) → SAM3
  2. few_shot  : Single VLM (with example images) → SAM3
  3. two_vlm   : Reasoner VLM + Evaluator VLM → SAM3
                 (same or different models)

All strategies use SAM3 for segmentation and support:
  - Input modes: rgb_only, rgb_depth_separate
  - Models: Qwen2.5-VL-7B-Instruct, gemma-4-E4B-it

Usage:
    # Zero-shot
    python lac_pipeline.py --strategy zero_shot --model Qwen2.5-VL-7B-Instruct --input_mode rgb_only

    # Few-shot
    python lac_pipeline.py --strategy few_shot --model Qwen2.5-VL-7B-Instruct \\
        --input_mode rgb_depth_separate --few_shot_dir /path/to/samples

    # Two-VLM (same model)
    python lac_pipeline.py --strategy two_vlm --model Qwen2.5-VL-7B-Instruct --input_mode rgb_depth_separate

    # Two-VLM (different models)
    python lac_pipeline.py --strategy two_vlm \\
        --reasoner_model Qwen2.5-VL-7B-Instruct --evaluator_model gemma-4-E4B-it \\
        --input_mode rgb_depth_separate

Output:
    $WORK/free_ground_results/{strategy}/{model_tag}/{input_mode}_sam3/
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_WORK_DIR = os.environ.get("WORK", str(Path(__file__).parent.parent))
_LOG_MODEL_NAME = "pipeline"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import shared utilities from free_ground_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "free_ground_pipeline"))
from pipeline import (
    MODEL_REGISTRY,
    discover_folders,
    discover_image_pairs,
    filter_image_pairs,
    load_image_for_vlm,
    load_vlm_model,
    run_inference,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHORT_MODEL_NAMES = {
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Qwen3.5-2B": "Qwen3.5-2B",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "gemma-4-E4B-it": "Gemma-4-E4B",
    "ByteDance/Sa2VA-Qwen3-VL-4B": "Sa2VA-Qwen3-VL-4B",
    "Sa2VA-Qwen3-VL-4B": "Sa2VA-Qwen3-VL-4B",
}

STRATEGIES = ["zero_shot", "few_shot", "two_vlm", "sa2va"]
INPUT_MODES = ["rgb_only", "rgb_depth_separate"]


def _model_tag(strategy: str, reasoner_name: str, evaluator_name: str = None) -> str:
    """Generate short model tag for output directory naming."""
    r = SHORT_MODEL_NAMES.get(reasoner_name, reasoner_name)
    if strategy == "two_vlm" and evaluator_name and evaluator_name != reasoner_name:
        e = SHORT_MODEL_NAMES.get(evaluator_name, evaluator_name)
        return f"{r}_{e}"
    return r


# ---------------------------------------------------------------------------
# Prompt Loading
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_DIR = Path(__file__).parent / "prompts_navigable"
_active_prompt_dir = DEFAULT_PROMPT_DIR


def set_prompt_dir(prompt_dir: Path):
    global _active_prompt_dir
    _active_prompt_dir = prompt_dir


def load_prompt(filename: str) -> str:
    path = _active_prompt_dir / filename
    if not path.exists():
        # Fallback to default prompts
        path = DEFAULT_PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Stage 1: Free Ground Reasoner (used by zero_shot, few_shot, two_vlm)
# ---------------------------------------------------------------------------

def run_free_ground_reasoner(
    model, processor, config: Dict, rgb_image: Image.Image,
    depth_image: Optional[Image.Image], model_config: Dict = None,
) -> Dict:
    """Run the reasoner VLM to identify walkable areas safe for traversal."""
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")
    system_prompt = load_prompt("free_ground_reasoner_system_walkable_short.txt")

    if input_mode == "rgb_depth_separate" and depth_image is not None:
        user_prompt = load_prompt("free_ground_reasoner_user_depth_walkable_short.txt")
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": rgb_image},
            {"type": "image", "image": depth_image},
        ]
    else:
        user_prompt = load_prompt("free_ground_reasoner_user_walkable_short.txt")
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": rgb_image},
        ]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    mc = model_config or config["model"]["reasoner"]
    response = run_inference(model, processor, messages, {"model": mc})
    return parse_reasoner_output(response)


def parse_reasoner_output(response: str) -> Dict:
    """Parse the VLM reasoner output into structured data."""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think?>", "", response, flags=re.DOTALL)
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        # Support both new 'walkable_areas' and old 'free_ground_areas' keys
        if isinstance(result, list):
            return {"walkable_areas": result, "navigability_reasoning": "", "obstacles": []}
        
        # Normalize to walkable_areas
        if "walkable_areas" in result:
            areas = result["walkable_areas"]
        elif "free_ground_areas" in result:
            areas = result["free_ground_areas"]
        else:
            areas = []
        
        return {
            "description": result.get("description", ""),
            "walkable_areas": areas,
            "navigability_reasoning": result.get("navigability_reasoning", ""),
            "obstacles": result.get("obstacles", []),
            "raw_response": response,
        }
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, list):
                    return {"walkable_areas": result, "navigability_reasoning": "", "obstacles": []}
                
                # Normalize to walkable_areas
                if "walkable_areas" in result:
                    areas = result["walkable_areas"]
                elif "free_ground_areas" in result:
                    areas = result["free_ground_areas"]
                else:
                    areas = []
                
                return {
                    "description": result.get("description", ""),
                    "walkable_areas": areas,
                    "navigability_reasoning": result.get("navigability_reasoning", ""),
                    "obstacles": result.get("obstacles", []),
                    "raw_response": response,
                }
            except json.JSONDecodeError:
                pass

    return {
        "description": "",
        "walkable_areas": [],
        "navigability_reasoning": cleaned[:500],
        "obstacles": [],
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# SA2VA Strategy Functions (direct segmentation, no SAM3)
# ---------------------------------------------------------------------------

def _local_sa2va_model_path(model_id: str) -> Optional[Path]:
    """Find the local cached path for a SA2VA model."""
    _work_dir = os.environ.get("WORK", str(Path(__file__).parent.parent))
    hf_cache = Path(_work_dir) / ".cache" / "huggingface"
    cache_dir = hf_cache / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not cache_dir.exists():
        return None

    snapshots = sorted([p for p in cache_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        if (snapshot / "config.json").exists():
            return snapshot
    return snapshots[0] if snapshots else None


def load_sa2va_model(model_id: str, device: str):
    """Load SA2VA model and processor."""
    import torch
    from transformers import AutoModel, AutoProcessor
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    local_model_path = _local_sa2va_model_path(model_id)
    model_source = str(local_model_path) if local_model_path is not None else model_id
    model = AutoModel.from_pretrained(
        model_source,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_model_path is not None,
    ).eval()
    if device != "cpu":
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=local_model_path is not None,
    )
    return model, processor


def _run_sa2va_model(model, processor, image: Optional[Image.Image], prompt: str, device: str, video: Optional[List[Image.Image]] = None):
    """Run SA2VA model inference."""
    import torch
    
    input_dict = {
        "image": image,
        "video": list(video) if video is not None else None,
        "text": prompt,
        "past_text": "",
        "mask_prompts": None,
        "processor": processor,
    }
    with torch.no_grad():
        return_dict = model.predict_forward(**input_dict)
    prediction = return_dict.get("prediction", "")
    masks = return_dict.get("prediction_masks", []) or []
    labels = return_dict.get("prediction_labels", []) or []
    return prediction, masks, labels


def _mask_to_bool(mask) -> np.ndarray:
    """Convert mask to boolean array."""
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] > 1:
        arr = arr[0]
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    if arr.dtype != bool:
        arr = arr > 0.5
    return arr.astype(bool)


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize mask to target size."""
    width, height = size
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] > 1:
        mask = mask[0]
    mask = np.squeeze(mask)
    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    if mask.shape == (height, width):
        return mask
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img = mask_img.resize((width, height), Image.NEAREST)
    return np.array(mask_img) > 127


def _mask_bbox(mask: np.ndarray) -> Optional[Dict[str, int]]:
    """Get bounding box of mask in pixels."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {
        "x1": int(xs.min()),
        "y1": int(ys.min()),
        "x2": int(xs.max()) + 1,
        "y2": int(ys.max()) + 1,
    }


def _bbox_percent(bbox: Optional[Dict[str, int]], size: Tuple[int, int]) -> Optional[Dict[str, float]]:
    """Convert pixel bbox to percentage coordinates."""
    if bbox is None:
        return None
    width, height = size
    return {
        "x1": round(bbox["x1"] / width * 100, 2),
        "y1": round(bbox["y1"] / height * 100, 2),
        "x2": round(bbox["x2"] / width * 100, 2),
        "y2": round(bbox["y2"] / height * 100, 2),
    }


def _prepare_sa2va_inputs(rgb_image: Image.Image, depth_image: Optional[Image.Image], input_mode: str, max_side: int = 896) -> Tuple[Optional[Image.Image], Optional[List[Image.Image]]]:
    """Prepare SA2VA model inputs from RGB and depth images."""
    def _resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
        if max_side <= 0:
            return image
        width, height = image.size
        longest_side = max(width, height)
        if longest_side <= max_side:
            return image
        scale = max_side / float(longest_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return image.resize((new_width, new_height), Image.LANCZOS)

    if input_mode == "rgb_only" or depth_image is None:
        return _resize_max_side(rgb_image.convert("RGB"), max_side), None

    if input_mode == "rgb_depth_separate":
        rgb_frame = _resize_max_side(rgb_image.convert("RGB"), max_side)
        depth_frame = _resize_max_side(depth_image.convert("RGB"), max_side)
        if depth_frame.size != rgb_frame.size:
            depth_frame = depth_frame.resize(rgb_frame.size, Image.NEAREST)
        return None, [rgb_frame, depth_frame]

    return None, None


def _load_sa2va_prompt(input_mode: str) -> str:
    """Load SA2VA walkable prompts from prompt files."""
    prompt_dir = Path(__file__).parent / "prompts"
    
    if input_mode == "rgb_only":
        prompt_file = prompt_dir / "sa2va_walkable_short.txt"
    else:
        prompt_file = prompt_dir / "sa2va_walkable_depth_short.txt"
    
    if prompt_file.exists():
        return prompt_file.read_text()
    
    # Fallback to inline prompts if files don't exist
    if input_mode == "rgb_only":
        return "<image>Segment all walkable areas safe for traversal in this indoor scene.\n\nINCLUDE: floors, staircases, ramps, walkways, corridors, and any other surfaces safe for walking.\nEXCLUDE: walls, furniture, obstacles, drop-offs, and unsafe surfaces.\n\nReturn segmentation masks for all walkable areas."
    return (
        "<image>Segment all walkable areas safe for traversal using both RGB and depth images.\n\nINCLUDE: floors, staircases, ramps, walkways, corridors, and any other surfaces safe for walking.\nEXCLUDE: walls, furniture, obstacles, drop-offs, and unsafe surfaces.\n\nDepth cues: \n- Floors: smooth, gradually changing depth colors\n- Stairs: distinct horizontal bands showing step depth progression\n- Ramps: continuous gradient of depth colors\n- Obstacles: sharp depth transitions or no depth data\n\nReturn segmentation masks for all walkable areas in the RGB image."
    )


def process_sa2va_image(
    rgb_path: Path,
    depth_path: Optional[Path],
    image_id: str,
    folder_name: str,
    model,
    processor,
    config: Dict,
    output_dir: Path,
) -> Dict:
    """Process a single image with SA2VA model."""
    import torch
    from PIL import ImageDraw, ImageFont
    from datetime import datetime
    
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")
    device = config["pipeline"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    max_side = config["pipeline"].get("image_max_side", 896)
    
    # Load images
    rgb_image = load_image_for_vlm(rgb_path)
    depth_image = None
    if input_mode != "rgb_only" and depth_path:
        depth_image = load_image_for_vlm(depth_path)
    
    # Prepare model inputs
    model_image, model_video = _prepare_sa2va_inputs(rgb_image, depth_image, input_mode, max_side)
    
    # Load prompt
    prompt = _load_sa2va_prompt(input_mode)
    
    # Run SA2VA model
    prediction, masks, labels = _run_sa2va_model(model, processor, model_image, prompt, device, video=model_video)
    
    # Prepare output directory
    folder_dir = output_dir / folder_name
    mask_dir = folder_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_arr = np.array(rgb_image.convert("RGB"))
    h, w = rgb_arr.shape[:2]
    overlay = rgb_arr.copy()
    colors = [
        (0, 255, 0),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 128, 0),
    ]
    
    areas: List[Dict] = []
    saved_masks: List[np.ndarray] = []
    
    for idx, mask in enumerate(masks):
        mask_bool = _mask_to_bool(mask)
        # Project mask back to RGB size
        mask_bool = _resize_mask(mask_bool, (w, h))
        bbox_px = _mask_bbox(mask_bool)
        if bbox_px is None:
            continue
        
        name = labels[idx] if idx < len(labels) and labels[idx] else f"sa2va_area_{idx}"
        safe_name = re.sub(r"[^\w]", "_", name)[:30]
        base = f"{image_id}_mask_{len(saved_masks)}_{safe_name}"
        
        # Save mask
        mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
        mask_img.save(mask_dir / f"{base}.png")
        np.save(mask_dir / f"{base}.npy", mask_bool)
        
        # Create overlay
        color = colors[len(saved_masks) % len(colors)]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[mask_bool] = color
        overlay[mask_bool] = (overlay[mask_bool] * 0.5 + color_mask[mask_bool] * 0.5).astype(np.uint8)
        
        areas.append({
            "name": name,
            "type": "sa2va_mask",
            "bbox": _bbox_percent(bbox_px, (w, h)),
            "bbox_pixels": bbox_px,
            "area_pixels": int(mask_bool.sum()),
        })
        saved_masks.append(mask_bool)
    
    # Save segmentation overlay
    Image.fromarray(overlay).save(mask_dir / f"{image_id}_segmentation_overlay.png")
    
    # Build consolidated image
    consolidated = _build_sa2va_consolidated_image(rgb_image, depth_image, areas, overlay, input_mode)
    consolidated.save(folder_dir / f"{image_id}_consolidated.png")
    
    # Prepare result
    model_id = config["model"].get("sa2va", {}).get("hf_model_id", "unknown")
    model_tag = _model_tag("sa2va", model_id.split("/")[-1] if "/" in model_id else model_id)
    
    result = {
        "folder": folder_name,
        "image_id": image_id,
        "strategy": "sa2va",
        "model_tag": model_tag,
        "model_id": model_id,
        "input_mode": input_mode,
        "segmentation_method": "sa2va",
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "prediction": prediction,
        "num_masks": len(saved_masks),
        "walkable_areas": areas,
    }
    
    # Save JSON result
    with open(folder_dir / f"{image_id}_lac_analysis.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Save all masks together
    if saved_masks:
        np.save(folder_dir / f"{image_id}_sa2va_masks.npy", np.stack(saved_masks, axis=0))
    
    return result


def _build_sa2va_consolidated_image(
    rgb_image: Image.Image,
    depth_image: Optional[Image.Image],
    areas: List[Dict],
    overlay: np.ndarray,
    input_mode: str,
) -> Image.Image:
    """Build consolidated visualization image for SA2VA results."""
    from PIL import ImageDraw, ImageFont
    
    rgb_arr = np.array(rgb_image.convert("RGB"))
    h, w = rgb_arr.shape[:2]
    label_h = 30
    gap = 4
    
    panels: List[Tuple[str, np.ndarray]] = [("Original RGB", rgb_arr)]
    
    if input_mode != "rgb_only" and depth_image is not None:
        depth_rgb = np.array(depth_image.convert("RGB"))
        if depth_rgb.shape[:2] != (h, w):
            depth_rgb = np.array(depth_image.convert("RGB").resize((w, h), Image.NEAREST))
        panels.append(("Depth Map", depth_rgb))
    
    if areas:
        panels.append(("SA2VA BBox", _draw_sa2va_bbox_panel(rgb_arr, areas)))
    
    panels.append(("Segmentation", overlay))
    
    total_w = sum(panel.shape[1] for _, panel in panels) + gap * (len(panels) - 1)
    total_h = h + label_h
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    
    x_offset = 0
    for label, panel in panels:
        canvas[0:label_h, x_offset:x_offset + panel.shape[1]] = (40, 40, 40)
        label_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((x_offset + 8, 7), label, fill=(255, 255, 255), font=font)
        canvas = np.array(label_img)
        canvas[label_h:, x_offset:x_offset + panel.shape[1]] = panel
        x_offset += panel.shape[1] + gap
    
    return Image.fromarray(canvas)


def _draw_sa2va_bbox_panel(rgb_array: np.ndarray, areas: List[Dict]) -> np.ndarray:
    """Draw bounding boxes on RGB array."""
    panel = rgb_array.copy()
    h, w = panel.shape[:2]
    for idx, area in enumerate(areas):
        bbox = area.get("bbox", {})
        x1 = int(round(bbox["x1"] / 100 * w))
        y1 = int(round(bbox["y1"] / 100 * h))
        x2 = int(round(bbox["x2"] / 100 * w))
        y2 = int(round(bbox["y2"] / 100 * h))
        color = (0, 255, 0)
        try:
            import cv2
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
            cv2.putText(panel, area.get("name", f"mask_{idx}"), (x1 + 3, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except Exception:
            panel[y1:y2, x1:x1 + 2] = color
            panel[y1:y2, max(0, x2 - 2):x2] = color
            panel[y1:y1 + 2, x1:x2] = color
            panel[max(0, y2 - 2):y2, x1:x2] = color
    return panel


# ---------------------------------------------------------------------------
# Stage 2: Traversability Evaluator (used by two_vlm only)
# ---------------------------------------------------------------------------

def run_traversability_evaluator(
    model, processor, config: Dict, reasoner_output: Dict,
    rgb_image: Image.Image, depth_image: Optional[Image.Image],
    model_config: Dict = None,
) -> Dict:
    """Run the evaluator VLM to score traversability of each area."""
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")
    system_template = load_prompt("traversability_evaluator_system.txt")

    free_ground_areas = reasoner_output.get("free_ground_areas", [])
    navigability_reasoning = reasoner_output.get("navigability_reasoning", "None")

    system_prompt = system_template.replace(
        "{free_ground_areas}", json.dumps(free_ground_areas)
    )
    system_prompt = system_prompt.replace(
        "{navigability_reasoning}", str(navigability_reasoning)
    )

    if input_mode == "rgb_depth_separate" and depth_image is not None:
        user_prompt = load_prompt("traversability_evaluator_user_depth.txt")
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": rgb_image},
            {"type": "image", "image": depth_image},
        ]
    else:
        user_prompt = load_prompt("traversability_evaluator_user.txt")
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": rgb_image},
        ]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    mc = model_config or config["model"]["evaluator"]
    response = run_inference(model, processor, messages, {"model": mc})
    return parse_evaluator_output(response)


def parse_evaluator_output(response: str) -> Dict:
    """Parse the traversability evaluator output."""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think?>", "", response, flags=re.DOTALL)
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return {"traversability_reasoning": "", "traversability_score": {},
                    "raw_list_response": result}
        return result
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, list):
                    return {"traversability_reasoning": "", "traversability_score": {},
                            "raw_list_response": result}
                return result
            except json.JSONDecodeError:
                pass

    return {"traversability_reasoning": {}, "traversability_score": {},
            "raw_response": response}


# ---------------------------------------------------------------------------
# Few-Shot Support
# ---------------------------------------------------------------------------

def _analyze_mask_for_few_shot(mask_path: Path) -> Optional[Dict]:
    """Analyze a binary mask to extract bbox and area info for few-shot examples."""
    mask = np.array(Image.open(mask_path).convert("L"))
    h, w = mask.shape
    white = np.where(mask > 128)

    if len(white[0]) == 0:
        return None

    y_min, y_max = white[0].min(), white[0].max()
    x_min, x_max = white[1].min(), white[1].max()

    bbox = {
        "x1": round(x_min / w * 100, 1),
        "y1": round(y_min / h * 100, 1),
        "x2": round(x_max / w * 100, 1),
        "y2": round(y_max / h * 100, 1),
    }

    cx = (x_min + x_max) / 2 / w * 100
    cy = (y_min + y_max) / 2 / h * 100
    if cy > 66:
        v_pos = "bottom"
    elif cy > 33:
        v_pos = "center"
    else:
        v_pos = "top"
    if cx > 66:
        h_pos = "right"
    elif cx > 33:
        h_pos = "center"
    else:
        h_pos = "left"
    position = f"{v_pos}-{h_pos}"

    area_pct = mask.sum() / 255 / (h * w) * 100
    if area_pct > 30:
        area_size = "large"
    elif area_pct > 10:
        area_size = "medium"
    else:
        area_size = "small"

    return {"bbox": bbox, "position": position, "area_size": area_size}


def _generate_few_shot_expected_output(analysis: Dict) -> Optional[str]:
    """Generate expected JSON output for a few-shot example."""
    bbox = analysis.get("bbox", {})
    if not bbox:
        return None
    return json.dumps({
        "free_ground_areas": [{
            "name": "walkable floor area",
            "type": "floor",
            "bbox": bbox,
            "reasoning": f"Flat navigable surface in {analysis.get('position', 'center')} of the image"
        }],
        "navigability_reasoning": "Clear walkable surface identified",
        "obstacles": []
    }, indent=2)


def load_few_shot_samples(
    few_shot_dir: str, data_dir: str, rgb_subfolder: str, depth_subfolder: str,
    num_examples: int = 3,
) -> Tuple[List[Dict], set]:
    """Load few-shot sample images (RGB + depth + mask).

    Discovers GT masks from few_shot_dir, then finds corresponding
    RGB and depth images from data_dir.

    Returns:
        Tuple of (samples, used_keys) where used_keys is a set of
        (folder_name, image_id) tuples that should be EXCLUDED from
        the test set to prevent data leakage.

    Args:
        few_shot_dir: Directory with GT masks (subfolder format).
        data_dir: Base data directory with RGB/depth images.
        rgb_subfolder: Subfolder path for RGB images.
        depth_subfolder: Subfolder path for depth images.
        num_examples: Max number of examples to load.
    """
    gt_path = Path(few_shot_dir)
    data_path = Path(data_dir)
    samples = []
    used_keys = set()  # Track (folder, image_id) used as examples

    # Discover masks from subfolders — pick from DIFFERENT folders for diversity
    for subfolder in sorted(gt_path.iterdir()):
        if not subfolder.is_dir():
            continue
        if len(samples) >= num_examples:
            break

        folder_name = subfolder.name
        for mask_file in sorted(subfolder.glob("*_mask.png")):
            if len(samples) >= num_examples:
                break

            image_id = mask_file.stem.replace("_mask", "")
            analysis = _analyze_mask_for_few_shot(mask_file)
            if not analysis:
                continue

            expected_output = _generate_few_shot_expected_output(analysis)
            if not expected_output:
                continue

            # Find RGB image
            rgb_path = data_path / folder_name / rgb_subfolder / "PNG" / f"{image_id}.png"
            if not rgb_path.exists():
                rgb_path = data_path / folder_name / rgb_subfolder / f"{image_id}.png"
            if not rgb_path.exists():
                continue

            # Find depth image
            depth_path = data_path / folder_name / depth_subfolder / f"{image_id}_depth_colored.png"
            if not depth_path.exists():
                depth_path = data_path / folder_name / depth_subfolder / f"{image_id}.png"
            depth_img = None
            if depth_path.exists():
                depth_img = Image.open(depth_path).convert("RGB")

            # Track which images are used as examples (to exclude from test set)
            used_keys.add((folder_name, image_id))

            samples.append({
                "name": f"{folder_name}/{image_id}",
                "rgb_image": Image.open(rgb_path).convert("RGB"),
                "depth_image": depth_img,
                "expected_output": expected_output,
            })

    logger.info(f"Loaded {len(samples)} few-shot examples from {few_shot_dir}")
    logger.info(f"  Example images (excluded from test): {used_keys}")
    return samples, used_keys


def build_few_shot_messages(
    config: Dict, samples: List[Dict],
    rgb_image: Image.Image, depth_image: Optional[Image.Image],
) -> List[Dict]:
    """Build chat messages with few-shot examples + query image.

    Uses the reasoner system prompt with examples prepended.
    The model is instructed to output JSON with free_ground_areas.
    """
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")

    # System prompt (reasoner prompt + example instruction)
    system_prompt = load_prompt("free_ground_reasoner_system.txt")
    system_prompt += (
        "\n\nYou will see EXAMPLE images with their correct analysis, "
        "then a NEW image to analyze. "
        "Output the same JSON format for the new image."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    ]

    # Add few-shot examples
    for i, sample in enumerate(samples):
        # User message with example images
        example_content = [
            {"type": "text", "text": f"EXAMPLE {i+1} — analyze this scene:"},
        ]
        if input_mode == "rgb_depth_separate" and sample.get("depth_image") is not None:
            example_content.extend([
                {"type": "image", "image": sample["rgb_image"]},
                {"type": "image", "image": sample["depth_image"]},
            ])
        else:
            example_content.append({"type": "image", "image": sample["rgb_image"]})

        messages.append({"role": "user", "content": example_content})

        # Assistant message with expected output
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["expected_output"]}],
        })

    # Query message
    if input_mode == "rgb_depth_separate" and depth_image is not None:
        user_prompt = load_prompt("free_ground_reasoner_user_depth.txt")
        user_content = [
            {"type": "text", "text": "Now analyze this NEW scene and identify all free navigable areas."},
            {"type": "image", "image": rgb_image},
            {"type": "image", "image": depth_image},
        ]
    else:
        user_prompt = load_prompt("free_ground_reasoner_user.txt")
        user_content = [
            {"type": "text", "text": "Now analyze this NEW scene and identify all free navigable areas."},
            {"type": "image", "image": rgb_image},
        ]

    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Stage 3: SAM3 Segmentation
# ---------------------------------------------------------------------------

_sam3_cache = {"model": None, "processor": None, "loaded": False, "failed": False}


def _load_sam3_model(config: Dict):
    """Load SAM3 model once and cache it globally."""
    if _sam3_cache["loaded"] or _sam3_cache["failed"]:
        return

    import torch
    from transformers import Sam3Model, Sam3Processor

    seg_config = config["model"].get("segmentation", {})
    sam3_model_id = seg_config.get("sam3_model_id", "facebook/sam3")
    device = seg_config.get("device", "cuda")

    try:
        logger.info(f"Loading SAM3 model: {sam3_model_id} (0.8B params)")
        _sam3_cache["processor"] = Sam3Processor.from_pretrained(sam3_model_id)
        _sam3_cache["model"] = Sam3Model.from_pretrained(
            sam3_model_id, torch_dtype=torch.float16
        ).to(device)
        _sam3_cache["loaded"] = True
        logger.info("SAM3 model loaded successfully")
    except Exception as e:
        logger.error(f"SAM3 model load failed ({e}) — segmentation will be skipped")
        _sam3_cache["failed"] = True


def _segment_with_sam3(
    rgb_image: Image.Image, vlm_areas: List[Dict], config: Dict,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """SAM3 segmentation with configurable input mode.

    sam3_input_mode controls what VLM output is fed to SAM3:
      - "text_only":     VLM text prompt only (default)
      - "bbox_only":     VLM bounding box only (no text)
      - "text_and_bbox": Both text prompt and bounding box

    If SAM3 produces no valid masks, falls back to VLM bbox → binary mask.
    """
    import torch
    import torch.nn.functional as F

    w, h = rgb_image.size
    seg_config = config["model"].get("segmentation", {})
    sam3_mode = seg_config.get("sam3_input_mode", "text_only")

    if not _sam3_cache["loaded"] and not _sam3_cache["failed"]:
        _load_sam3_model(config)

    if _sam3_cache["failed"] or not _sam3_cache["loaded"]:
        logger.warning("    SAM3 model unavailable — no segmentation possible")
        return [], []

    sam3_model = _sam3_cache["model"]
    sam3_processor = _sam3_cache["processor"]
    device = next(sam3_model.parameters()).device
    mask_threshold = seg_config.get("sam3_mask_threshold", 0.5)

    logger.info(f"    SAM3 input mode: {sam3_mode}")

    masks = []
    bboxes = []

    for area in vlm_areas:
        name = area.get("name", "")
        area_type = area.get("type", "")
        text_prompt = name or area_type.replace("_", " ")

        vlm_bbox = area.get("bbox", {})
        has_vlm_bbox = vlm_bbox and all(k in vlm_bbox for k in ["x1", "y1", "x2", "y2"])

        # Determine what inputs to use based on mode
        use_text = sam3_mode in ("text_only", "text_and_bbox")
        use_bbox = sam3_mode in ("bbox_only", "text_and_bbox")

        # Skip if required input is missing
        if use_text and not text_prompt:
            if use_bbox and has_vlm_bbox:
                use_text = False  # Fall back to bbox-only for this area
            else:
                logger.warning("      Skipping area with no name/type from VLM")
                continue

        if use_bbox and not has_vlm_bbox:
            if use_text and text_prompt:
                use_bbox = False  # Fall back to text-only for this area
            else:
                logger.warning(f"      Skipping '{name}': no bbox from VLM")
                continue

        # Build log label
        parts = []
        if use_text:
            parts.append(f"text='{text_prompt}'")
        if use_bbox:
            parts.append(f"bbox({vlm_bbox['x1']},{vlm_bbox['y1']},{vlm_bbox['x2']},{vlm_bbox['y2']})")
        logger.info(f"      SAM3 input: {' + '.join(parts)}")

        try:
            # ── Build processor inputs based on mode ──────────────────────
            processor_kwargs = {
                "images": rgb_image,
                "return_tensors": "pt",
            }

            if use_text:
                processor_kwargs["text"] = text_prompt

            if use_bbox:
                # Convert percentage bbox to pixel coordinates
                box_xyxy = [
                    vlm_bbox["x1"] / 100 * w,
                    vlm_bbox["y1"] / 100 * h,
                    vlm_bbox["x2"] / 100 * w,
                    vlm_bbox["y2"] / 100 * h,
                ]
                processor_kwargs["input_boxes"] = [[box_xyxy]]
                processor_kwargs["input_boxes_labels"] = [[1]]

            inputs = sam3_processor(**processor_kwargs).to(device, dtype=torch.float16)

            with torch.no_grad():
                outputs = sam3_model(**inputs)

            # ── Post-process: use official API for bbox-only, manual for text ─
            if use_bbox and not use_text:
                # bbox-only: use post_process_instance_segmentation
                # (matches test_sam3_bbox.py working implementation)
                results = sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                sam3_masks = results.get("masks", [])
                sam3_scores = results.get("scores", [])

                # Use len() only — `not tensor` raises
                # "Boolean value of Tensor with more than one value is ambiguous"
                if len(sam3_masks) == 0:
                    logger.info(f"      SAM3 bbox-only: no masks for '{name}'")
                    continue

                # Take best mask by score
                best_idx = 0
                if len(sam3_scores) > 0:
                    scores_t = sam3_scores if torch.is_tensor(sam3_scores) else torch.tensor(sam3_scores)
                    best_idx = int(scores_t.argmax().item())

                best_mask = sam3_masks[best_idx]
                if hasattr(best_mask, "cpu"):
                    best_mask = best_mask.cpu().numpy()
                best_mask = best_mask.astype(bool)
                best_score = float(sam3_scores[best_idx]) if best_idx < len(sam3_scores) else 0.0

            else:
                # text-only or text+bbox: use manual post-processing
                scores = (
                    outputs.pred_logits.sigmoid().squeeze(0)
                    * outputs.presence_logits.sigmoid().squeeze(0)
                )
                best_idx = scores.argmax().item()
                best_score = scores[best_idx].item()

                best_mask_logits = outputs.pred_masks[0, best_idx].unsqueeze(0).unsqueeze(0).float()
                best_mask_probs = torch.sigmoid(best_mask_logits)
                best_mask_resized = F.interpolate(
                    best_mask_probs, size=(h, w), mode="bilinear", align_corners=False,
                )
                best_mask = best_mask_resized.squeeze().cpu().numpy() > mask_threshold

            if best_mask.sum() < 100:
                logger.info(f"      SAM3 mask for '{name}' too small, skipping")
                continue

            masks.append(best_mask)
            ys, xs = np.where(best_mask)
            bboxes.append({
                "x1": round(int(xs.min()) / w * 100, 2),
                "y1": round(int(ys.min()) / h * 100, 2),
                "x2": round(int(xs.max()) / w * 100, 2),
                "y2": round(int(ys.max()) / h * 100, 2),
                "score": round(best_score, 4),
                "source": "sam3",
                "name": name or text_prompt,
                "sam3_mode": sam3_mode,
            })

            coverage = best_mask.sum() / (h * w) * 100
            logger.info(f"      SAM3: '{name}' score={best_score:.3f} coverage={coverage:.1f}%")

        except Exception as e:
            logger.warning(f"      SAM3 failed for '{name}': {e}")
            continue

    if not masks:
        logger.warning("    SAM3 produced no valid masks — no segmentation for this image")
        return [], []

    return masks, bboxes


def _fallback_bbox_masks(
    vlm_areas: List[Dict], image_size: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Convert VLM bounding boxes directly to binary masks (fallback).

    Used when SAM3 produces no valid masks or is unavailable.
    """
    w, h = image_size
    masks = []
    bboxes = []

    for area in vlm_areas:
        vlm_bbox = area.get("bbox", {})
        if not vlm_bbox or not all(k in vlm_bbox for k in ["x1", "y1", "x2", "y2"]):
            continue

        name = area.get("name", "")
        x1 = int(vlm_bbox["x1"] / 100 * w)
        y1 = int(vlm_bbox["y1"] / 100 * h)
        x2 = int(vlm_bbox["x2"] / 100 * w)
        y2 = int(vlm_bbox["y2"] / 100 * h)

        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True

        if mask.sum() < 100:
            continue

        masks.append(mask)
        bboxes.append({
            "x1": vlm_bbox["x1"],
            "y1": vlm_bbox["y1"],
            "x2": vlm_bbox["x2"],
            "y2": vlm_bbox["y2"],
            "score": None,
            "source": "vlm_bbox",
            "name": name,
        })
        coverage = mask.sum() / (h * w) * 100
        logger.info(f"      VLM bbox fallback: '{name}' bbox=({vlm_bbox['x1']},{vlm_bbox['y1']},"
                     f"{vlm_bbox['x2']},{vlm_bbox['y2']}) coverage={coverage:.1f}%")

    if not masks:
        logger.warning("    No valid VLM bboxes for fallback either")

    return masks, bboxes


def _bbox_to_masks(bboxes: List[Dict], image_size: Tuple[int, int]) -> List[np.ndarray]:
    """Convert bounding box percentages to binary masks (fallback)."""
    w, h = image_size
    masks = []
    for bbox in bboxes:
        mask = np.zeros((h, w), dtype=bool)
        x1 = int(bbox.get("x1", 0) / 100 * w)
        y1 = int(bbox.get("y1", 0) / 100 * h)
        x2 = int(bbox.get("x2", 100) / 100 * w)
        y2 = int(bbox.get("y2", 100) / 100 * h)
        mask[y1:y2, x1:x2] = True
        masks.append(mask)
    return masks


# ---------------------------------------------------------------------------
# Mask Saving
# ---------------------------------------------------------------------------

def _draw_vlm_bboxes(rgb_array: np.ndarray, vlm_areas: List[Dict]) -> np.ndarray:
    """Draw VLM bounding boxes on RGB image (percentage-based coords)."""
    import cv2
    img = rgb_array.copy()
    h, w = img.shape[:2]
    bbox_colors = [
        (0, 255, 0), (0, 128, 255), (255, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 128, 0),
    ]
    for i, area in enumerate(vlm_areas):
        bbox = area.get("bbox", {})
        if not bbox or not all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            continue
        x1 = int(bbox["x1"] / 100 * w)
        y1 = int(bbox["y1"] / 100 * h)
        x2 = int(bbox["x2"] / 100 * w)
        y2 = int(bbox["y2"] / 100 * h)
        color = bbox_colors[i % len(bbox_colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = area.get("name", f"area_{i}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, 0.5, (255, 255, 255), 1)
    return img


def save_segmentation_masks(
    masks: List[np.ndarray], area_names: List[str],
    rgb_image: Image.Image, bboxes: List[Dict],
    output_dir: Path, image_id: str,
    depth_image: Image.Image = None,
    input_mode: str = "rgb_only",
    vlm_areas: List[Dict] = None,
):
    """Save segmentation masks as PNGs + overlay + consolidated visualization.

    Consolidated image layout:
      rgb_only:         [Original RGB] | [VLM BBox] | [Segmentation Overlay]
      rgb_depth_separate: [Original RGB] | [Depth Map] | [VLM BBox] | [Segmentation Overlay]
    """
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    rgb_array = np.array(rgb_image.convert("RGB"))
    overlay = rgb_array.copy()

    colors = [
        (0, 255, 0), (0, 128, 255), (255, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 128, 0),
    ]

    for i, (mask, name) in enumerate(zip(masks, area_names)):
        safe_name = re.sub(r'[^\w]', '_', name)[:30]
        base = f"{image_id}_mask_{i}_{safe_name}"

        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(mask_dir / f"{base}.png")
        np.save(mask_dir / f"{base}.npy", mask)

        color = colors[i % len(colors)]
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mask[mask] = color
        overlay[mask] = (overlay[mask] * 0.5 + color_mask[mask] * 0.5).astype(np.uint8)

    Image.fromarray(overlay).save(mask_dir / f"{image_id}_segmentation_overlay.png")

    # ── Consolidated visualization ─────────────────────────────────────────
    h, w = rgb_array.shape[:2]
    label_h = 30  # height for text label above each panel
    gap = 4       # pixel gap between panels

    panels = []

    # Panel 1: Original RGB
    panels.append(("Original RGB", rgb_array))

    # Panel 2: Depth map (if available)
    if input_mode != "rgb_only" and depth_image is not None:
        depth_rgb = np.array(depth_image.convert("RGB"))
        # Resize to match RGB if needed
        if depth_rgb.shape[:2] != (h, w):
            depth_rgb = np.array(depth_image.convert("RGB").resize((w, h), Image.NEAREST))
        panels.append(("Depth Map", depth_rgb))

    # Panel: VLM BBox visualization
    if vlm_areas:
        vlm_bbox_img = _draw_vlm_bboxes(rgb_array, vlm_areas)
        panels.append(("VLM BBox", vlm_bbox_img))

    # Panel: Segmentation overlay
    panels.append(("Segmentation", overlay))

    # Build consolidated image with labels
    total_w = sum(panel.shape[1] for _, panel in panels) + gap * (len(panels) - 1)
    total_h = h + label_h
    consolidated = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    x_offset = 0
    for label, panel in panels:
        # Draw label background
        consolidated[0:label_h, x_offset:x_offset + panel.shape[1]] = (40, 40, 40)
        # Draw label text using PIL
        label_img = Image.fromarray(consolidated)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        text_x = x_offset + (panel.shape[1] - draw.textlength(label, font=font)) // 2
        draw.text((text_x, 5), label, fill=(255, 255, 255), font=font)
        consolidated = np.array(label_img)

        # Paste panel below label
        consolidated[label_h:label_h + h, x_offset:x_offset + panel.shape[1]] = panel
        x_offset += panel.shape[1] + gap

    consolidated_path = output_dir / f"{image_id}_consolidated.png"
    Image.fromarray(consolidated).save(consolidated_path)
    logger.debug(f"    Saved consolidated visualization: {consolidated_path.name}")


# ---------------------------------------------------------------------------
# Single Image Processing
# ---------------------------------------------------------------------------

def process_single_image(
    rgb_path: Path, depth_path: Path, image_id: str, folder_name: str,
    models: Dict, config: Dict, output_dir: Path,
    few_shot_samples: List[Dict] = None,
) -> Dict:
    """Process a single image through the pipeline.

    Dispatches to strategy-specific VLM processing, then runs SAM3.
    """
    strategy = config["pipeline"]["strategy"]
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")
    logger.info(f"Processing {folder_name}/{image_id} [strategy={strategy}, mode={input_mode}]")

    # Load images
    rgb_image = load_image_for_vlm(rgb_path)
    depth_image = None
    if input_mode != "rgb_only":
        depth_image = load_image_for_vlm(depth_path)

    result = {
        "folder": folder_name,
        "image_id": image_id,
        "strategy": strategy,
        "input_mode": input_mode,
        "segmentation_method": "sam3",
        "timestamp": datetime.now().isoformat(),
    }

    # ── Stage 1: VLM processing (strategy-dependent) ──────────────────────
    areas = []
    evaluator_output = None

    if strategy == "zero_shot":
        # Single VLM with reasoner prompt
        logger.info("  Stage 1: Zero-shot Reasoner...")
        t0 = time.time()
        reasoner_output = run_free_ground_reasoner(
            models["reasoner"][0], models["reasoner"][1],
            config, rgb_image, depth_image,
        )
        t1 = time.time()
        areas = reasoner_output.get("free_ground_areas", [])
        logger.info(f"    Found {len(areas)} areas ({t1-t0:.1f}s)")
        result["reasoner"] = {
            "output": reasoner_output,
            "inference_time": round(t1 - t0, 2),
            "num_areas": len(areas),
        }

    elif strategy == "few_shot":
        # Single VLM with few-shot examples
        logger.info(f"  Stage 1: Few-shot Reasoner ({len(few_shot_samples)} examples)...")
        t0 = time.time()
        messages = build_few_shot_messages(
            config, few_shot_samples, rgb_image, depth_image,
        )
        response = run_inference(
            models["reasoner"][0], models["reasoner"][1],
            messages, {"model": config["model"]["reasoner"]},
        )
        reasoner_output = parse_reasoner_output(response)
        t1 = time.time()
        areas = reasoner_output.get("free_ground_areas", [])
        logger.info(f"    Found {len(areas)} areas ({t1-t0:.1f}s)")
        result["reasoner"] = {
            "output": reasoner_output,
            "inference_time": round(t1 - t0, 2),
            "num_areas": len(areas),
            "num_examples": len(few_shot_samples),
        }

    elif strategy == "two_vlm":
        # Reasoner VLM
        logger.info("  Stage 1: Reasoner VLM...")
        t0 = time.time()
        reasoner_output = run_free_ground_reasoner(
            models["reasoner"][0], models["reasoner"][1],
            config, rgb_image, depth_image,
        )
        t1 = time.time()
        areas = reasoner_output.get("free_ground_areas", [])
        logger.info(f"    Found {len(areas)} areas ({t1-t0:.1f}s)")
        result["reasoner"] = {
            "output": reasoner_output,
            "inference_time": round(t1 - t0, 2),
            "num_areas": len(areas),
        }

        # Evaluator VLM
        if areas:
            logger.info("  Stage 2: Evaluator VLM...")
            t0 = time.time()
            evaluator_output = run_traversability_evaluator(
                models["evaluator"][0], models["evaluator"][1],
                config, reasoner_output, rgb_image, depth_image,
            )
            t1 = time.time()
            scores = evaluator_output.get("traversability_score", {})
            logger.info(f"    Scores: {scores} ({t1-t0:.1f}s)")
            result["evaluator"] = {
                "output": evaluator_output,
                "inference_time": round(t1 - t0, 2),
                "areas_with_scores": [
                    {
                        "name": a.get("name", ""),
                        "type": a.get("type", ""),
                        "bbox": a.get("bbox", {}),
                        "traversability_score": scores.get(a.get("name", ""), None),
                    }
                    for a in areas
                ],
            }

            # Filter score=0 areas
            filtered = []
            for area in areas:
                name = area.get("name", "")
                score = scores.get(name, None)
                if score == 0:
                    logger.info(f"    Filtering '{name}' (score=0)")
                    continue
                filtered.append(area)
            if len(filtered) < len(areas):
                logger.info(f"    Kept {len(filtered)}/{len(areas)} after score filtering")
            areas = filtered
        else:
            result["evaluator"] = {"output": {}, "skipped": True}

    # Store VLM areas with bboxes in result for evaluation
    result["vlm_areas"] = [
        {
            "name": a.get("name", ""),
            "type": a.get("type", ""),
            "bbox": a.get("bbox", {}),
        }
        for a in areas
    ]

    # ── Stage 3: SAM3 Segmentation ────────────────────────────────────────
    logger.info(f"  Stage 3: SAM3 Segmentation ({len(areas)} regions)...")
    t0 = time.time()
    masks, bboxes_used = _segment_with_sam3(rgb_image, areas, config)
    t1 = time.time()
    logger.info(f"    Generated {len(masks)} masks ({t1-t0:.1f}s)")

    area_names = [b.get("name", f"sam3_area_{i}") for i, b in enumerate(bboxes_used)]

    result["segmentation"] = {
        "method": "sam3",
        "num_masks": len(masks),
        "inference_time": round(t1 - t0, 2),
    }

    # ── Depth-based flatness validation ───────────────────────────────────
    if masks and depth_image is not None:
        depth_array = np.array(depth_image.convert("L")).astype(np.float32)
        validated_masks, validated_names, validated_bboxes = [], [], []
        for i, (mask, name) in enumerate(zip(masks, area_names)):
            pixels = depth_array[mask > 0]
            if len(pixels) < 100:
                continue
            flatness_thresh = config["pipeline"].get("depth_flatness_threshold", 80)
            if np.std(pixels) > flatness_thresh:
                continue
            if np.max(pixels) - np.min(pixels) > flatness_thresh * 3:
                continue
            validated_masks.append(mask)
            validated_names.append(name)
            if i < len(bboxes_used):
                validated_bboxes.append(bboxes_used[i])

        if len(validated_masks) < len(masks):
            logger.info(f"    Depth validation: kept {len(validated_masks)}/{len(masks)}")
            masks = validated_masks
            area_names = validated_names
            bboxes_used = validated_bboxes

    # ── Save outputs ──────────────────────────────────────────────────────
    if masks and config["output"].get("save_visualizations", True):
        save_segmentation_masks(
            masks, area_names, rgb_image, bboxes_used,
            output_dir / folder_name, image_id,
            depth_image=depth_image, input_mode=input_mode,
            vlm_areas=result.get("vlm_areas"),
        )

    if config["output"].get("save_individual_json", True):
        json_dir = output_dir / folder_name
        json_dir.mkdir(parents=True, exist_ok=True)
        with open(json_dir / f"{image_id}_lac_analysis.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

def run_pipeline(config: Dict):
    """Run the unified walkable area detection pipeline."""
    start_time = time.time()
    strategy = config["pipeline"]["strategy"]
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")

    # Output directory: {strategy}/{model_tag}/{input_mode}_{method}/
    if strategy == "sa2va":
        sa2va_name = config["model"]["sa2va"]["name"]
        mtag = SHORT_MODEL_NAMES.get(sa2va_name, sa2va_name)
        method = "sa2va"
    else:
        r_name = config["model"]["reasoner"]["name"]
        e_name = config["model"].get("evaluator", {}).get("name", r_name)
        mtag = _model_tag(strategy, r_name, e_name if strategy == "two_vlm" else None)
        method = "sam3"
    
    suffix = config["pipeline"].get("output_suffix", "")

    output_dir = Path(config["output"]["dir"]) / strategy / mtag / f"{input_mode}_{method}"
    if suffix:
        output_dir = Path(config["output"]["dir"]) / f"{strategy}{suffix}" / mtag / f"{input_mode}_{method}"

    # Close any file handlers pointing into output_dir BEFORE cleaning
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            try:
                if hasattr(h, "baseFilename") and output_dir.resolve() in Path(h.baseFilename).resolve().parents:
                    h.close()
                    root_logger.removeHandler(h)
            except (OSError, ValueError):
                pass

    if config["pipeline"].get("clean_output", False) and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging inside the run output directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Discover folders
    folders = discover_folders(config)
    if not folders:
        logger.error("No valid folders found.")
        return

    # ── Load models ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Loading models for strategy: {strategy}")
    logger.info("=" * 60)

    models = {}
    few_shot_samples = None
    few_shot_excluded = set()  # (folder, image_id) to exclude from test set

    if strategy == "sa2va":
        # Load SA2VA model
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sa2va_config = config["model"]["sa2va"]
        logger.info(f"Loading SA2VA model: {sa2va_config['hf_model_id']}")
        sa2va_model, sa2va_proc = load_sa2va_model(sa2va_config["hf_model_id"], device)
        models["sa2va"] = (sa2va_model, sa2va_proc)
        config["pipeline"]["device"] = device
    else:
        # VLM strategies
        # Reasoner model (always needed)
        reasoner_config = {"model": config["model"]["reasoner"]}
        reasoner_model, reasoner_proc = load_vlm_model(reasoner_config)
        models["reasoner"] = (reasoner_model, reasoner_proc)

        # Evaluator model (only for two_vlm)
        if strategy == "two_vlm":
            evaluator_config = {"model": config["model"]["evaluator"]}
            if config["model"]["evaluator"]["hf_model_id"] == config["model"]["reasoner"]["hf_model_id"]:
                logger.info("Reusing reasoner model for evaluator (same model)")
                models["evaluator"] = (reasoner_model, reasoner_proc)
            else:
                logger.info(f"Loading separate evaluator model: {config['model']['evaluator']['name']}")
                evaluator_model, evaluator_proc = load_vlm_model(evaluator_config)
                models["evaluator"] = (evaluator_model, evaluator_proc)

        # Few-shot samples (only for few_shot)
        if strategy == "few_shot":
            few_shot_dir = config["pipeline"].get("few_shot_dir")
            num_examples = config["pipeline"].get("num_examples", 3)
            if few_shot_dir:
                few_shot_samples, few_shot_excluded = load_few_shot_samples(
                    few_shot_dir,
                    config["data"]["base_dir"],
                    config["data"]["rgb_subfolder"],
                    config["data"]["depth_subfolder"],
                    num_examples,
                )
                if not few_shot_samples:
                    logger.error("No few-shot samples loaded! Check --few_shot_dir path.")
                    return
                logger.info(f"  Excluding {len(few_shot_excluded)} example images from test set")
            else:
                logger.error("--few_shot_dir is required for few_shot strategy")
                return

        # Pre-load SAM3
        logger.info("Pre-loading SAM3 model...")
        _load_sam3_model(config)

    # ── Process images ────────────────────────────────────────────────────
    all_results = []

    for folder in folders:
        logger.info("=" * 60)
        logger.info(f"Processing folder: {folder.name}")
        logger.info("=" * 60)

        pairs = discover_image_pairs(folder, config)
        if not pairs:
            logger.warning(f"No image pairs found in {folder.name}")
            continue

        pairs = filter_image_pairs(pairs, config, folder_name=folder.name)

        for rgb_path, depth_path, image_id in pairs:
            # Skip images used as few-shot examples (prevent data leakage)
            if (folder.name, str(image_id)) in few_shot_excluded:
                logger.info(f"  {folder.name}/{image_id} [SKIPPED — used as few-shot example]")
                continue

            existing_json = output_dir / folder.name / f"{image_id}_lac_analysis.json"
            if existing_json.exists() and not config["pipeline"].get("clean_output", False):
                logger.info(f"  {folder.name}/{image_id} [SKIPPED — already processed]")
                with open(existing_json) as f:
                    all_results.append(json.load(f))
                continue

            try:
                if strategy == "sa2va":
                    # Use SA2VA processing
                    sa2va_model, sa2va_proc = models["sa2va"]
                    result = process_sa2va_image(
                        rgb_path, depth_path, str(image_id), folder.name,
                        sa2va_model, sa2va_proc, config, output_dir,
                    )
                else:
                    # Use VLM processing
                    result = process_single_image(
                        rgb_path, depth_path, str(image_id), folder.name,
                        models, config, output_dir, few_shot_samples,
                    )
                all_results.append(result)
            except Exception as e:
                logger.error(f"FAILED {folder.name}/{image_id}: {e}")
                import traceback
                traceback.print_exc()

    # Save combined CSV
    if all_results:
        csv_path = output_dir / "all_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["folder", "image_id", "strategy", "input_mode"],
            )
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k, "") for k in ["folder", "image_id", "strategy", "input_mode"]})

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE — {len(all_results)} images in {elapsed:.1f}s")
    logger.info(f"{'='*60}")


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def _resolve_model_id(name_or_id: str) -> Tuple[str, str]:
    """Resolve a model name or HF ID to (name, hf_model_id)."""
    if name_or_id in MODEL_REGISTRY:
        return name_or_id, MODEL_REGISTRY[name_or_id]
    if "/" in name_or_id:
        return name_or_id.split("/")[-1], name_or_id
    return name_or_id, name_or_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Walkable Area Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot
  python lac_pipeline.py --strategy zero_shot --model Qwen2.5-VL-7B-Instruct --input_mode rgb_only

  # Few-shot
  python lac_pipeline.py --strategy few_shot --model Qwen2.5-VL-7B-Instruct \\
      --input_mode rgb_depth_separate --few_shot_dir /path/to/samples

  # Two-VLM (same model)
  python lac_pipeline.py --strategy two_vlm --model Qwen2.5-VL-7B-Instruct --input_mode rgb_depth_separate

  # Two-VLM (different models)
  python lac_pipeline.py --strategy two_vlm \\
      --reasoner_model Qwen2.5-VL-7B-Instruct --evaluator_model gemma-4-E4B-it \\
      --input_mode rgb_depth_separate

  # SA2VA (direct segmentation)
  python lac_pipeline.py --strategy sa2va --model ByteDance/Sa2VA-Qwen3-VL-4B --input_mode rgb_depth_separate
        """,
    )
    # Strategy
    parser.add_argument("--strategy", type=str, default="two_vlm",
                        choices=STRATEGIES,
                        help="Pipeline strategy: zero_shot, few_shot, two_vlm, sa2va (default: two_vlm)")
    parser.add_argument("--config", type=str, default="lac_config.yaml",
                        help="Path to YAML config file")

    # Model selection (VLM strategies)
    parser.add_argument("--model", type=str, default=None,
                        help="Model for all VLM stages (name or HF ID), or SA2VA model for sa2va strategy")
    parser.add_argument("--reasoner_model", type=str, default=None,
                        help="Reasoner model (overrides --model for reasoner, VLM strategies only)")
    parser.add_argument("--evaluator_model", type=str, default=None,
                        help="Evaluator model (overrides --model for evaluator, two_vlm only)")
    parser.add_argument("--hf_model_id", type=str, default=None,
                        help="Override full HuggingFace model ID")

    # SA2VA-specific
    parser.add_argument("--image_max_side", type=int, default=896,
                        help="Resize SA2VA input so longest side does not exceed this value (0 = keep original, default: 896)")

    # Input mode
    parser.add_argument("--input_mode", type=str, default=None,
                        choices=INPUT_MODES,
                        help="Input mode: rgb_only or rgb_depth_separate")

    # Few-shot
    parser.add_argument("--few_shot_dir", type=str, default=None,
                        help="Directory with few-shot sample images (required for few_shot strategy)")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of few-shot examples to use (default: 3)")

    # Data filtering
    parser.add_argument("--specific_images", nargs="+", default=None,
                        help="Specific image IDs to process")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Path to annotated ground truth directory")
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Specific folder names to process")
    parser.add_argument("--quick_test", action="store_true",
                        help="Quick test: 3 images per folder")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of images per folder to process (overrides quick_test)")

    # SAM3 segmentation mode
    parser.add_argument("--sam3_input_mode", type=str, default=None,
                        choices=["text_only", "bbox_only", "text_and_bbox"],
                        help="SAM3 input mode: text_only (VLM text), bbox_only (VLM bbox), "
                             "text_and_bbox (both). Default: text_only")

    # Output
    parser.add_argument("--prompt_dir", type=str, default=None,
                        help="Path to custom prompt directory")
    parser.add_argument("--output_suffix", type=str, default=None,
                        help="Suffix appended to output folder name")
    parser.add_argument("--clean", action="store_true",
                        help="Remove output directory before running")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    return parser.parse_args()


def merge_args(config: Dict, args: argparse.Namespace) -> Dict:
    """Merge CLI arguments into config. CLI takes precedence."""
    strategy = getattr(args, "strategy", None) or config["pipeline"].get("strategy", "two_vlm")
    config["pipeline"]["strategy"] = strategy

    # Model: --model sets both reasoner and evaluator
    if getattr(args, "model", None):
        name, hf_id = _resolve_model_id(args.model)
        for stage in ["reasoner", "evaluator"]:
            config["model"][stage]["name"] = name
            config["model"][stage]["hf_model_id"] = hf_id

    if getattr(args, "hf_model_id", None):
        name = args.hf_model_id.split("/")[-1]
        for stage in ["reasoner", "evaluator"]:
            config["model"][stage]["name"] = name
            config["model"][stage]["hf_model_id"] = args.hf_model_id

    # Reasoner model override
    if getattr(args, "reasoner_model", None):
        name, hf_id = _resolve_model_id(args.reasoner_model)
        config["model"]["reasoner"]["name"] = name
        config["model"]["reasoner"]["hf_model_id"] = hf_id
        # If no evaluator specified, use same as reasoner
        if not getattr(args, "evaluator_model", None) and not getattr(args, "model", None):
            config["model"]["evaluator"]["name"] = name
            config["model"]["evaluator"]["hf_model_id"] = hf_id

    # Evaluator model override
    if getattr(args, "evaluator_model", None):
        name, hf_id = _resolve_model_id(args.evaluator_model)
        config["model"]["evaluator"]["name"] = name
        config["model"]["evaluator"]["hf_model_id"] = hf_id

    if getattr(args, "input_mode", None):
        config["pipeline"]["input_mode"] = args.input_mode

    if getattr(args, "specific_images", None):
        ids = []
        for img_id in args.specific_images:
            if img_id.startswith("image"):
                ids.append(int(img_id.replace("image", "")))
            else:
                ids.append(int(img_id))
        config["pipeline"]["specific_images"] = ids

    if getattr(args, "gt_dir", None):
        from pipeline import read_gt_directory
        gt_mapping = read_gt_directory(args.gt_dir)
        config["pipeline"]["gt_folder_images"] = gt_mapping
        if gt_mapping:
            config["data"]["folders"] = list(gt_mapping.keys())

    if getattr(args, "folders", None):
        config["data"]["folders"] = args.folders

    if getattr(args, "quick_test", False):
        config["pipeline"]["num_images_per_folder"] = 3

    if getattr(args, "num_images", None):
        config["pipeline"]["num_images_per_folder"] = args.num_images

    if getattr(args, "few_shot_dir", None):
        config["pipeline"]["few_shot_dir"] = args.few_shot_dir
    elif config["pipeline"].get("strategy") == "few_shot" and getattr(args, "gt_dir", None):
        # Default: use GT annotations as few-shot examples
        config["pipeline"]["few_shot_dir"] = args.gt_dir
        logger.info(f"Few-shot: using GT directory as few-shot examples: {args.gt_dir}")

    if getattr(args, "num_examples", None):
        config["pipeline"]["num_examples"] = args.num_examples

    if getattr(args, "sam3_input_mode", None):
        config["model"]["segmentation"]["sam3_input_mode"] = args.sam3_input_mode

    if getattr(args, "image_max_side", None):
        config["pipeline"]["image_max_side"] = args.image_max_side

    if getattr(args, "clean", False):
        config["pipeline"]["clean_output"] = True

    if getattr(args, "output_suffix", None):
        config["pipeline"]["output_suffix"] = args.output_suffix

    # Handle SA2VA model configuration
    strategy = config["pipeline"].get("strategy", "two_vlm")
    if strategy == "sa2va":
        if getattr(args, "model", None):
            name, hf_id = _resolve_model_id(args.model)
            config["model"]["sa2va"] = {
                "name": name,
                "hf_model_id": hf_id,
            }
        elif getattr(args, "hf_model_id", None):
            name = args.hf_model_id.split("/")[-1]
            config["model"]["sa2va"] = {
                "name": name,
                "hf_model_id": args.hf_model_id,
            }
        # Set default SA2VA model if not specified
        if "sa2va" not in config["model"]:
            config["model"]["sa2va"] = {
                "name": "Sa2VA-Qwen3-VL-4B",
                "hf_model_id": "ByteDance/Sa2VA-Qwen3-VL-4B",
            }

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Use $WORK with hardcoded fallback to the woody directory (where models are cached)
    _work_dir = os.environ.get("WORK", "/home/woody/iwnt/iwnt164h")
    os.environ["HF_HOME"] = os.path.join(_work_dir, ".cache", "huggingface")

    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config = merge_args(config, args)

    # Set prompt directory
    if args.prompt_dir:
        set_prompt_dir(Path(args.prompt_dir))

    # Print config summary
    strategy = config["pipeline"]["strategy"]
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")
    r_name = config["model"]["reasoner"]["name"]
    e_name = config["model"].get("evaluator", {}).get("name", r_name)
    mtag = _model_tag(strategy, r_name, e_name if strategy == "two_vlm" else None)
    global _LOG_MODEL_NAME
    _LOG_MODEL_NAME = mtag
    logger.info("=" * 60)
    logger.info("Unified Walkable Area Detection Pipeline")
    logger.info("=" * 60)
    logger.info(f"Strategy:     {strategy}")
    logger.info(f"Input mode:   {input_mode}")
    
    if strategy == "sa2va":
        sa2va_model_id = config["model"].get("sa2va", {}).get("hf_model_id", "unknown")
        logger.info(f"SA2VA Model:  {sa2va_model_id}")
        logger.info(f"Max side:     {config['pipeline'].get('image_max_side', 896)}")
    else:
        logger.info(f"Reasoner:     {config['model']['reasoner']['name']}")
        if strategy == "two_vlm":
            logger.info(f"Evaluator:    {config['model']['evaluator']['name']}")
            same = config["model"]["evaluator"]["hf_model_id"] == config["model"]["reasoner"]["hf_model_id"]
            logger.info(f"Same model:   {same}")
        if strategy == "few_shot":
            logger.info(f"Few-shot dir: {config['pipeline'].get('few_shot_dir', 'NOT SET')}")
            logger.info(f"Num examples: {config['pipeline'].get('num_examples', 3)}")
        logger.info(f"Segmentation: SAM3 ({config['model']['segmentation'].get('sam3_model_id', 'facebook/sam3')})")
        logger.info(f"SAM3 mode:    {config['model']['segmentation'].get('sam3_input_mode', 'text_only')}")

    run_pipeline(config)


if __name__ == "__main__":
    main()
