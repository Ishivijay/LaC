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
sys.path.insert(0, str(Path(__file__).parent.parent / "free_ground_pipeline"))
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
    "Qwen2.5-VL-7B-Instruct": "Qwen",
    "gemma-4-E4B-it": "Gemma",
}

STRATEGIES = ["zero_shot", "few_shot", "two_vlm"]
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
    """Run the reasoner VLM to identify free ground areas."""
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")
    system_prompt = load_prompt("free_ground_reasoner_system.txt")

    if input_mode == "rgb_depth_separate" and depth_image is not None:
        user_prompt = load_prompt("free_ground_reasoner_user_depth.txt")
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": rgb_image},
            {"type": "image", "image": depth_image},
        ]
    else:
        user_prompt = load_prompt("free_ground_reasoner_user.txt")
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
        if isinstance(result, list):
            return {"free_ground_areas": result, "navigability_reasoning": "", "obstacles": []}
        return result
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, list):
                    return {"free_ground_areas": result, "navigability_reasoning": "", "obstacles": []}
                return result
            except json.JSONDecodeError:
                pass

    return {
        "description": "",
        "free_ground_areas": [],
        "navigability_reasoning": cleaned[:500],
        "obstacles": [],
        "raw_response": response,
    }


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
        logger.warning(f"SAM3 model load failed ({e}), will use bbox masks as fallback")
        _sam3_cache["failed"] = True


def _segment_with_sam3(
    rgb_image: Image.Image, vlm_areas: List[Dict], config: Dict,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """SAM3 text-prompted segmentation from VLM area descriptions."""
    import torch
    import torch.nn.functional as F

    w, h = rgb_image.size

    if not _sam3_cache["loaded"] and not _sam3_cache["failed"]:
        _load_sam3_model(config)

    if _sam3_cache["failed"] or not _sam3_cache["loaded"]:
        logger.warning("    SAM3 unavailable, falling back to bbox masks")
        vlm_bboxes = [a.get("bbox", {}) for a in vlm_areas if a.get("bbox")]
        return _bbox_to_masks(vlm_bboxes, rgb_image.size), vlm_bboxes

    sam3_model = _sam3_cache["model"]
    sam3_processor = _sam3_cache["processor"]
    device = next(sam3_model.parameters()).device
    mask_threshold = config["model"].get("segmentation", {}).get("sam3_mask_threshold", 0.5)

    masks = []
    bboxes = []

    for area in vlm_areas:
        name = area.get("name", "")
        area_type = area.get("type", "")
        text_prompt = name or area_type.replace("_", " ") or "floor"

        logger.info(f"      SAM3 prompt: '{text_prompt}'")

        try:
            inputs = sam3_processor(
                images=rgb_image, text=text_prompt, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = sam3_model(**inputs)

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
                logger.info(f"      SAM3 mask for '{text_prompt}' too small, skipping")
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
                "name": text_prompt,
            })

            coverage = best_mask.sum() / (h * w) * 100
            logger.info(f"      SAM3: '{text_prompt}' score={best_score:.3f} coverage={coverage:.1f}%")

        except Exception as e:
            logger.warning(f"      SAM3 failed for '{text_prompt}': {e}")
            bbox = area.get("bbox", {})
            if bbox and all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                mask = np.zeros((h, w), dtype=bool)
                x1 = int(bbox["x1"] / 100 * w)
                y1 = int(bbox["y1"] / 100 * h)
                x2 = int(bbox["x2"] / 100 * w)
                y2 = int(bbox["y2"] / 100 * h)
                mask[y1:y2, x1:x2] = True
                masks.append(mask)
                bboxes.append({**bbox, "source": "sam3_fallback", "name": text_prompt})

    if not masks:
        logger.warning("    SAM3 produced no valid masks, falling back to bbox masks")
        vlm_bboxes = [a.get("bbox", {}) for a in vlm_areas if a.get("bbox")]
        return _bbox_to_masks(vlm_bboxes, rgb_image.size), vlm_bboxes

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

def save_segmentation_masks(
    masks: List[np.ndarray], area_names: List[str],
    rgb_image: Image.Image, bboxes: List[Dict],
    output_dir: Path, image_id: str,
    depth_image: Image.Image = None,
    input_mode: str = "rgb_only",
):
    """Save segmentation masks as PNGs + overlay + consolidated visualization.

    Consolidated image layout:
      rgb_only:         [Original RGB] | [Segmentation Overlay]
      rgb_depth_separate: [Original RGB] | [Depth Map] | [Segmentation Overlay]
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

    # Panel 3: Segmentation overlay
    panels.append(("Segmentation", overlay))

    # Build consolidated image with labels
    total_w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
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
    """Run the unified free ground detection pipeline."""
    start_time = time.time()
    strategy = config["pipeline"]["strategy"]
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")

    # Output directory: {strategy}/{model_tag}/{input_mode}_sam3/
    r_name = config["model"]["reasoner"]["name"]
    e_name = config["model"].get("evaluator", {}).get("name", r_name)
    mtag = _model_tag(strategy, r_name, e_name if strategy == "two_vlm" else None)
    suffix = config["pipeline"].get("output_suffix", "")

    output_dir = Path(config["output"]["dir"]) / strategy / mtag / f"{input_mode}_sam3"
    if suffix:
        output_dir = Path(config["output"]["dir"]) / f"{strategy}{suffix}" / mtag / f"{input_mode}_sam3"

    if config["pipeline"].get("clean_output", False) and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    # Reasoner model (always needed)
    reasoner_config = {"model": config["model"]["reasoner"]}
    reasoner_model, reasoner_proc = load_vlm_model(reasoner_config)
    models = {"reasoner": (reasoner_model, reasoner_proc)}

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
    few_shot_samples = None
    few_shot_excluded = set()  # (folder, image_id) to exclude from test set
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
        description="Unified Free Ground Space Detection Pipeline",
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
        """,
    )
    # Strategy
    parser.add_argument("--strategy", type=str, default="two_vlm",
                        choices=STRATEGIES,
                        help="Pipeline strategy (default: two_vlm)")
    parser.add_argument("--config", type=str, default="lac_config.yaml",
                        help="Path to YAML config file")

    # Model selection
    parser.add_argument("--model", type=str, default=None,
                        help="Model for all VLM stages (name or HF ID)")
    parser.add_argument("--reasoner_model", type=str, default=None,
                        help="Reasoner model (overrides --model for reasoner)")
    parser.add_argument("--evaluator_model", type=str, default=None,
                        help="Evaluator model (overrides --model for evaluator, two_vlm only)")
    parser.add_argument("--hf_model_id", type=str, default=None,
                        help="Override full HuggingFace model ID")

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

    if getattr(args, "few_shot_dir", None):
        config["pipeline"]["few_shot_dir"] = args.few_shot_dir

    if getattr(args, "num_examples", None):
        config["pipeline"]["num_examples"] = args.num_examples

    if getattr(args, "clean", False):
        config["pipeline"]["clean_output"] = True

    if getattr(args, "output_suffix", None):
        config["pipeline"]["output_suffix"] = args.output_suffix

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _work_dir = os.environ.get("WORK", str(Path(__file__).parent.parent))
    os.environ.setdefault("HF_HOME", os.path.join(_work_dir, ".cache", "huggingface"))

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

    # Setup logging to file
    strategy = config["pipeline"]["strategy"]
    r_name = config["model"]["reasoner"]["name"]
    e_name = config["model"].get("evaluator", {}).get("name", r_name)
    mtag = _model_tag(strategy, r_name, e_name if strategy == "two_vlm" else None)
    global _LOG_MODEL_NAME
    _LOG_MODEL_NAME = mtag

    log_dir = Path(_WORK_DIR) / "free_ground_results" / f"{strategy}_{mtag}" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    # Print config summary
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")
    logger.info("=" * 60)
    logger.info("Unified Free Ground Space Detection Pipeline")
    logger.info("=" * 60)
    logger.info(f"Strategy:     {strategy}")
    logger.info(f"Input mode:   {input_mode}")
    logger.info(f"Reasoner:     {config['model']['reasoner']['name']}")
    if strategy == "two_vlm":
        logger.info(f"Evaluator:    {config['model']['evaluator']['name']}")
        same = config["model"]["evaluator"]["hf_model_id"] == config["model"]["reasoner"]["hf_model_id"]
        logger.info(f"Same model:   {same}")
    if strategy == "few_shot":
        logger.info(f"Few-shot dir: {config['pipeline'].get('few_shot_dir', 'NOT SET')}")
        logger.info(f"Num examples: {config['pipeline'].get('num_examples', 3)}")
    logger.info(f"Segmentation: SAM3 ({config['model']['segmentation'].get('sam3_model_id', 'facebook/sam3')})")

    run_pipeline(config)


if __name__ == "__main__":
    main()
