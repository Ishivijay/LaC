#!/usr/bin/env python3
"""LaC-adapted Free Ground Space Detection Pipeline.

Based on "Language as Cost: Proactive Hazard Mapping using VLM for Robot Navigation"
(Oh et al., IROS 2025). Adapted for free ground detection using local VLMs.

Pipeline stages:
  1. Free Ground Reasoner (VLM) — identifies flat floor areas with bounding boxes
  2. Traversability Evaluator (VLM) — scores each area's walkability (1-3)
  3. Segmentation — creates precise masks via SAM, Grounding-DINO+SAM, or VLM bbox
  4. Gaussian Cost Map — builds traversability map from masks + depth

Segmentation methods (config: model.segmentation.method):
  - "sam"            — VLM bbox percentages → SAM mask
  - "grounding_dino" — text prompt → Grounding-DINO detection → SAM mask
  - "vlm_only"       — VLM bbox percentages → rectangular mask (no extra model)

Usage:
    python3 lac_pipeline.py --config lac_config.yaml
    python3 lac_pipeline.py --quick_test
    python3 lac_pipeline.py --specific_images image28 image36 image188
    python3 lac_pipeline.py --segmentation_method grounding_dino
"""

import argparse
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
# Logging setup — dynamic model name resolved after config is loaded
# ---------------------------------------------------------------------------

_WORK_DIR = os.environ.get("WORK", str(Path(__file__).parent.parent))
_LOG_MODEL_NAME = "LaC"  # overwritten in main() once config is known


def _make_log_dir(model_name: str, suffix: str = "") -> Path:
    """Create log directory under $WORK/free_ground_results/<model>_LaC<suffix>/logs/."""
    lac_folder = f"{model_name}_LaC{suffix}" if suffix else f"{model_name}_LaC"
    d = Path(_WORK_DIR) / "free_ground_results" / lac_folder / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Placeholder — real handlers are installed in main() after config is parsed
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add parent pipeline directory to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "free_ground_pipeline"))
from pipeline import (
    MODEL_REGISTRY,
    create_overlay_image,
    discover_folders,
    discover_image_pairs,
    filter_image_pairs,
    load_image_for_vlm,
    load_vlm_model,
    run_inference,
)


# ---------------------------------------------------------------------------
# Prompt Loading
# ---------------------------------------------------------------------------

# Default prompt directory — can be overridden via --prompt_dir
DEFAULT_PROMPT_DIR = Path(__file__).parent / "prompts"
_active_prompt_dir = DEFAULT_PROMPT_DIR


def set_prompt_dir(prompt_dir: Path):
    """Set the active prompt directory (called from main() after arg parsing)."""
    global _active_prompt_dir
    _active_prompt_dir = prompt_dir


def load_prompt(filename: str) -> str:
    """Load a prompt template from the active prompts directory."""
    path = _active_prompt_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Stage 1: Free Ground Reasoner
# ---------------------------------------------------------------------------

def run_free_ground_reasoner(
    model, processor, config: Dict, rgb_image: Image.Image, depth_image: Optional[Image.Image]
) -> Dict:
    """Stage 1: Identify free ground areas with bounding boxes.

    Uses the VLM to analyze the scene and output structured JSON
    with free ground areas, bounding boxes, and reasoning.
    """
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_overlay")

    # Load prompts
    system_prompt = load_prompt("free_ground_reasoner_system.txt")
    user_prompt = load_prompt("free_ground_reasoner_user.txt")

    # Prepare image
    if input_mode == "rgb_depth_overlay" and depth_image is not None:
        alpha = config["pipeline"].get("overlay_alpha", 0.4)
        image = create_overlay_image(rgb_image, depth_image, alpha)
    else:
        image = rgb_image

    # Build messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    # Run inference — wrap in {"model": ...} since run_inference expects config["model"]
    response = run_inference(model, processor, messages, {"model": config["model"]["reasoner"]})
    return parse_reasoner_output(response)


def parse_reasoner_output(response: str) -> Dict:
    """Parse the VLM reasoner output into structured data."""
    # Clean response
    cleaned = re.sub(r"<think\b[^>]*>.*?</think?>", "", response, flags=re.DOTALL)
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    # Try to parse as JSON
    try:
        result = json.loads(cleaned)
        # VLM may return a list instead of dict — wrap it
        if isinstance(result, list):
            logger.warning(f"Reasoner returned a list ({len(result)} items), wrapping as dict")
            return {
                "free_ground_areas": result,
                "navigability_reasoning": "",
                "obstacles": [],
            }
        return result
    except json.JSONDecodeError:
        # Try to find JSON in the response
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, list):
                    return {
                        "free_ground_areas": result,
                        "navigability_reasoning": "",
                        "obstacles": [],
                    }
                return result
            except json.JSONDecodeError:
                pass

    # Fallback: return raw response
    return {
        "description": "",
        "free_ground_areas": [],
        "navigability_reasoning": cleaned[:500],
        "obstacles": [],
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# Stage 2: Traversability Evaluator
# ---------------------------------------------------------------------------

def run_traversability_evaluator(
    model, processor, config: Dict, reasoner_output: Dict,
    rgb_image: Image.Image, depth_image: Optional[Image.Image]
) -> Dict:
    """Stage 2: Evaluate traversability scores for each free ground area.

    Adapted from LaC's Emotion Evaluator. Instead of anxiety scores,
    assigns traversability scores (1-3) to each identified free ground area.
    """
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_overlay")

    # Load prompt template
    system_template = load_prompt("traversability_evaluator_system.txt")
    user_prompt = load_prompt("traversability_evaluator_user.txt")

    # Fill in template
    free_ground_areas = reasoner_output.get("free_ground_areas", [])
    navigability_reasoning = reasoner_output.get("navigability_reasoning", "None")

    system_prompt = system_template.replace(
        "{free_ground_areas}", json.dumps(free_ground_areas)
    )
    system_prompt = system_prompt.replace(
        "{navigability_reasoning}", str(navigability_reasoning)
    )

    # Prepare image
    if input_mode == "rgb_depth_overlay" and depth_image is not None:
        alpha = config["pipeline"].get("overlay_alpha", 0.4)
        image = create_overlay_image(rgb_image, depth_image, alpha)
    else:
        image = rgb_image

    # Build messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    # Run inference — wrap in {"model": ...} since run_inference expects config["model"]
    response = run_inference(model, processor, messages, {"model": config["model"]["evaluator"]})
    return parse_evaluator_output(response)


def parse_evaluator_output(response: str) -> Dict:
    """Parse the traversability evaluator output."""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think?>", "", response, flags=re.DOTALL)
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        # VLM may return a list instead of dict — wrap it
        if isinstance(result, list):
            logger.warning(f"Evaluator returned a list ({len(result)} items), wrapping as dict")
            return {
                "traversability_reasoning": "",
                "traversability_score": {},
                "raw_list_response": result,
            }
        return result
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, list):
                    return {
                        "traversability_reasoning": "",
                        "traversability_score": {},
                        "raw_list_response": result,
                    }
                return result
            except json.JSONDecodeError:
                pass

    return {
        "traversability_reasoning": {},
        "traversability_score": {},
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# Stage 3: Segmentation (SAM / Grounding-DINO+SAM / VLM-only)
# ---------------------------------------------------------------------------

# Global model caches (loaded once, reused across images)
_sam_cache = {"model": None, "processor": None, "loaded": False, "failed": False}
_gdino_cache = {"model": None, "processor": None, "loaded": False, "failed": False}


def _load_sam_model(config: Dict):
    """Load SAM model once and cache it globally."""
    if _sam_cache["loaded"] or _sam_cache["failed"]:
        return

    seg_config = config["model"].get("segmentation", {})
    method = seg_config.get("method", "vlm_only")

    if method == "vlm_only":
        _sam_cache["failed"] = True
        return

    try:
        import torch
        from transformers import SamModel, SamProcessor

        device = seg_config.get("device", "cuda")
        sam_model_id = seg_config.get("sam_model_id", "facebook/sam-vit-base")

        logger.info(f"Loading SAM model: {sam_model_id} (one-time load)")
        _sam_cache["model"] = SamModel.from_pretrained(sam_model_id).to(device)
        _sam_cache["processor"] = SamProcessor.from_pretrained(sam_model_id)
        _sam_cache["loaded"] = True
        logger.info("SAM model loaded successfully")
    except Exception as e:
        logger.warning(f"SAM model load failed ({e}), will use bbox masks for all images")
        _sam_cache["failed"] = True


def _load_grounding_dino_model(config: Dict):
    """Load Grounding-DINO model once and cache it globally."""
    if _gdino_cache["loaded"] or _gdino_cache["failed"]:
        return

    seg_config = config["model"].get("segmentation", {})
    gdino_model_id = seg_config.get(
        "grounding_dino_model_id", "IDEA-Research/grounding-dino-tiny"
    )

    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = seg_config.get("device", "cuda")

        logger.info(f"Loading Grounding-DINO model: {gdino_model_id} (one-time load)")
        _gdino_cache["processor"] = AutoProcessor.from_pretrained(gdino_model_id)
        _gdino_cache["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_model_id
        ).to(device)
        _gdino_cache["loaded"] = True
        logger.info("Grounding-DINO model loaded successfully")
    except Exception as e:
        logger.warning(f"Grounding-DINO model load failed ({e}), falling back to SAM-only")
        _gdino_cache["failed"] = True


def _run_grounding_dino_detection(
    rgb_image: Image.Image,
    config: Dict,
) -> List[Dict]:
    """Run Grounding-DINO on the image to detect floor/ground regions.

    Returns a list of bbox dicts with pixel coordinates:
        [{"x1": int, "y1": int, "x2": int, "y2": int, "score": float}, ...]
    """
    seg_config = config["model"].get("segmentation", {})
    text_prompt = seg_config.get("grounding_dino_text_prompt", "flat floor . floor . ground .")
    box_threshold = seg_config.get("grounding_dino_box_threshold", 0.3)
    text_threshold = seg_config.get("grounding_dino_text_threshold", 0.25)

    # Ensure model is loaded
    if not _gdino_cache["loaded"] and not _gdino_cache["failed"]:
        _load_grounding_dino_model(config)

    if _gdino_cache["failed"] or not _gdino_cache["loaded"]:
        return []

    try:
        import torch

        gdino_model = _gdino_cache["model"]
        gdino_processor = _gdino_cache["processor"]
        device = next(gdino_model.parameters()).device

        # Prepare inputs
        inputs = gdino_processor(
            images=rgb_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = gdino_model(**inputs)

        # Post-process — get bounding boxes in pixel coordinates
        w, h = rgb_image.size
        target_sizes = torch.tensor([[h, w]])
        try:
            # Try new API (transformers >= 5.x — no threshold kwargs)
            results = gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            # Fallback: old API (transformers < 5.x — with threshold kwargs)
            results = gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=target_sizes,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )[0]

        # Extract boxes and scores
        boxes = results["boxes"].cpu().numpy()  # (N, 4) — [x1, y1, x2, y2] pixel coords
        scores = results["scores"].cpu().numpy()

        # Apply thresholds manually if using new API (filter by score)
        if scores.ndim > 0 and len(scores) > 0:
            keep = scores >= box_threshold
            boxes = boxes[keep]
            scores = scores[keep]
        labels = results.get("text_labels", results.get("labels", []))

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            # Clamp to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": float(scores[i]),
                "label": labels[i] if i < len(labels) else "floor",
            })

        logger.info(f"    Grounding-DINO detected {len(detections)} regions "
                     f"(prompt: '{text_prompt}', box_thresh={box_threshold})")
        return detections

    except Exception as e:
        logger.warning(f"Grounding-DINO detection failed ({e})")
        return []


def run_segmentation(
    rgb_image: Image.Image,
    bboxes: List[Dict],
    config: Dict,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Stage 3: Segment free ground areas.

    Dispatches to the configured segmentation method:
      - "sam"            — VLM bbox → SAM mask
      - "grounding_dino" — text prompt → Grounding-DINO bbox → SAM mask
      - "vlm_only"       — VLM bbox → rectangular mask

    Args:
        rgb_image: Input RGB image.
        bboxes: List of bbox dicts from VLM reasoner with keys: x1, y1, x2, y2 (%).
        config: Pipeline config.

    Returns:
        Tuple of (masks, bboxes_used) where:
          - masks: list of binary numpy arrays
          - bboxes_used: list of bbox dicts (pixel coords for grounding_dino,
                         percentage coords for sam/vlm_only)
    """
    seg_config = config["model"].get("segmentation", {})
    method = seg_config.get("method", "vlm_only")

    # ----- Grounding-DINO method -----
    if method == "grounding_dino":
        return _segment_with_grounding_dino(rgb_image, bboxes, config)

    # ----- SAM method (VLM bbox → SAM) -----
    if method == "sam":
        masks = _segment_with_sam(rgb_image, bboxes, config)
        return masks, bboxes

    # ----- VLM-only fallback -----
    return bbox_to_masks(bboxes, rgb_image.size), bboxes


def _segment_with_grounding_dino(
    rgb_image: Image.Image,
    vlm_bboxes: List[Dict],
    config: Dict,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Grounding-DINO detection → SAM segmentation.

    Uses Grounding-DINO to get better bounding boxes from text prompts,
    then feeds those pixel-accurate boxes to SAM for mask generation.
    Falls back to VLM bboxes → SAM if Grounding-DINO fails.
    """
    w, h = rgb_image.size

    # Run Grounding-DINO detection
    gdino_detections = _run_grounding_dino_detection(rgb_image, config)

    if not gdino_detections:
        logger.info("    Grounding-DINO found no detections, falling back to VLM bboxes → SAM")
        masks = _segment_with_sam(rgb_image, vlm_bboxes, config)
        return masks, vlm_bboxes

    # Convert pixel-coordinate detections to percentage-based bboxes
    # (so downstream code that expects percentages still works)
    gdino_bboxes_pct = []
    for det in gdino_detections:
        gdino_bboxes_pct.append({
            "x1": round(det["x1"] / w * 100, 2),
            "y1": round(det["y1"] / h * 100, 2),
            "x2": round(det["x2"] / w * 100, 2),
            "y2": round(det["y2"] / h * 100, 2),
            "score": det["score"],
            "label": det.get("label", "floor"),
        })

    # Feed Grounding-DINO pixel bboxes to SAM for precise masks
    # SAM expects pixel coords, so use the original detections
    sam_bboxes = []
    for det in gdino_detections:
        sam_bboxes.append({
            "x1": det["x1"],
            "y1": det["y1"],
            "x2": det["x2"],
            "y2": det["y2"],
        })

    # Ensure SAM is loaded
    if not _sam_cache["loaded"] and not _sam_cache["failed"]:
        _load_sam_model(config)

    if _sam_cache["loaded"]:
        masks = _run_sam_with_pixel_bboxes(rgb_image, sam_bboxes)
        if masks:
            return masks, gdino_bboxes_pct

    # SAM failed — use pixel bboxes as rectangular masks
    logger.info("    SAM unavailable, using Grounding-DINO bboxes as rectangular masks")
    masks = []
    for det in gdino_detections:
        mask = np.zeros((h, w), dtype=bool)
        mask[det["y1"]:det["y2"], det["x1"]:det["x2"]] = True
        masks.append(mask)
    return masks, gdino_bboxes_pct


def _segment_with_sam(
    rgb_image: Image.Image,
    bboxes: List[Dict],
    config: Dict,
) -> List[np.ndarray]:
    """SAM segmentation from percentage-based VLM bboxes."""
    if not bboxes:
        return []

    # Try to use cached SAM model
    if _sam_cache["failed"]:
        return bbox_to_masks(bboxes, rgb_image.size)

    if not _sam_cache["loaded"]:
        _load_sam_model(config)

    if _sam_cache["failed"] or not _sam_cache["loaded"]:
        return bbox_to_masks(bboxes, rgb_image.size)

    try:
        import torch

        sam_model = _sam_cache["model"]
        sam_processor = _sam_cache["processor"]
        device = next(sam_model.parameters()).device

        w, h = rgb_image.size
        masks = []

        for bbox in bboxes:
            # Convert percentage bbox to pixel coordinates
            x1 = int(bbox["x1"] / 100 * w)
            y1 = int(bbox["y1"] / 100 * h)
            x2 = int(bbox["x2"] / 100 * w)
            y2 = int(bbox["y2"] / 100 * h)

            # SAM expects [x1, y1, x2, y2] box prompt
            inputs = sam_processor(
                rgb_image,
                input_boxes=[[[x1, y1, x2, y2]]],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = sam_model(**inputs, multimask_output=False)

            mask = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0][0][0].numpy()

            masks.append(mask > 0)

        return masks

    except Exception as e:
        logger.warning(f"SAM segmentation failed ({e}), falling back to bbox masks")
        return bbox_to_masks(bboxes, rgb_image.size)


def _run_sam_with_pixel_bboxes(
    rgb_image: Image.Image,
    pixel_bboxes: List[Dict],
) -> List[np.ndarray]:
    """Run SAM with pixel-coordinate bounding boxes (from Grounding-DINO)."""
    if not _sam_cache["loaded"]:
        return []

    try:
        import torch

        sam_model = _sam_cache["model"]
        sam_processor = _sam_cache["processor"]
        device = next(sam_model.parameters()).device

        masks = []
        for bbox in pixel_bboxes:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

            inputs = sam_processor(
                rgb_image,
                input_boxes=[[[x1, y1, x2, y2]]],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = sam_model(**inputs, multimask_output=False)

            mask = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0][0][0].numpy()

            masks.append(mask > 0)

        return masks

    except Exception as e:
        logger.warning(f"SAM segmentation with pixel bboxes failed ({e})")
        return []


def bbox_to_masks(bboxes: List[Dict], image_size: Tuple[int, int]) -> List[np.ndarray]:
    """Convert bounding box percentages to binary masks."""
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
# Stage 4: Gaussian Cost Map
# ---------------------------------------------------------------------------

def build_cost_map(
    masks: List[np.ndarray],
    scores: Dict[str, int],
    area_names: List[str],
    image_size: Tuple[int, int],
    config: Dict,
) -> np.ndarray:
    """Build a Gaussian traversability cost map from segmented areas.

    Adapted from LaC's Gaussian cost map. Instead of anxiety-based hazard costs,
    this creates a traversability map where:
    - Free ground areas get LOW cost (safe to traverse)
    - Obstacles/unknown areas get HIGH cost (avoid)
    - Gaussian smoothing creates natural gradients

    Args:
        masks: List of binary masks for each free ground area.
        scores: Dict mapping area_name → traversability score (1-3).
        area_names: List of area names corresponding to masks.
        image_size: (width, height) of the image.
        config: Pipeline config.

    Returns:
        Cost map as numpy array (H, W) with values in [0, 1].
        0 = safe (free ground), 1 = obstacle/unknown.
    """
    cm_config = config.get("cost_map", {})
    sigma_base = cm_config.get("sigma_base", 0.3)

    w, h = image_size
    cost_map = np.ones((h, w), dtype=np.float32)  # Start with all obstacles

    from scipy.ndimage import gaussian_filter

    for i, (mask, name) in enumerate(zip(masks, area_names)):
        score = scores.get(name, 2)  # Default moderate traversability

        # Score 0 = NOT traversable (stairs, inclines) — treat as obstacle
        if score == 0:
            logger.info(f"    Area '{name}' scored 0 (not traversable), treating as obstacle")
            continue

        # Higher score = more traversable = lower cost
        # Score 3 → cost 0.1, Score 2 → cost 0.3, Score 1 → cost 0.6
        area_cost = max(0.0, 1.0 - (score / 3.0))

        # Apply mask
        cost_map[mask] = area_cost

        # Gaussian smoothing around the mask boundary
        # Higher traversability → wider safe zone (larger sigma)
        sigma = sigma_base * score
        smoothed = gaussian_filter(cost_map, sigma=sigma)

        # Only apply smoothing where it reduces cost (don't spread obstacles)
        cost_map = np.minimum(cost_map, smoothed)

    return np.clip(cost_map, 0, 1)


def save_segmentation_masks(
    masks: List[np.ndarray],
    area_names: List[str],
    rgb_image: Image.Image,
    bboxes: List[Dict],
    output_dir: Path,
    image_id: str,
):
    """Save segmentation masks as individual PNGs, NPYs, and an overlay visualization.

    Saves:
      - masks/{image_id}_mask_{i}_{name}.png  — binary mask
      - masks/{image_id}_mask_{i}_{name}.npy  — raw numpy array
      - masks/{image_id}_segmentation_overlay.png — all masks overlaid on RGB
    """
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    rgb_array = np.array(rgb_image.convert("RGB"))
    overlay = rgb_array.copy()

    # Distinct colors for different areas
    colors = [
        (0, 255, 0),    # green
        (0, 128, 255),  # orange-blue
        (255, 0, 255),  # magenta
        (255, 255, 0),  # yellow
        (0, 255, 255),  # cyan
        (255, 128, 0),  # orange
    ]

    for i, (mask, name) in enumerate(zip(masks, area_names)):
        safe_name = re.sub(r'[^\w]', '_', name)[:30]
        base = f"{image_id}_mask_{i}_{safe_name}"

        # Save binary mask as PNG
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(mask_dir / f"{base}.png")

        # Save raw numpy array
        np.save(mask_dir / f"{base}.npy", mask)

        # Add colored overlay
        color = colors[i % len(colors)]
        colored = np.zeros_like(rgb_array)
        for c in range(3):
            colored[:, :, c] = mask * color[c]
        overlay = np.where(mask[:, :, np.newaxis] > 0,
                           (overlay * 0.5 + colored * 0.5).astype(np.uint8),
                           overlay)

        # Draw bbox rectangle
        if i < len(bboxes):
            bbox = bboxes[i]
            x1 = int(bbox.get("x1", 0) * rgb_image.width / 100
                     if bbox.get("x1", 0) <= 100 else bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0) * rgb_image.height / 100
                     if bbox.get("y1", 0) <= 100 else bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 100) * rgb_image.width / 100
                     if bbox.get("x2", 100) <= 100 else bbox.get("x2", 100))
            y2 = int(bbox.get("y2", 100) * rgb_image.height / 100
                     if bbox.get("y2", 100) <= 100 else bbox.get("y2", 100))
            overlay[y1:y1+2, x1:x2] = color
            overlay[y2:y2+2, x1:x2] = color
            overlay[y1:y2, x1:x1+2] = color
            overlay[y1:y2, x2:x2+2] = color

    # Save overlay
    Image.fromarray(overlay).save(mask_dir / f"{image_id}_segmentation_overlay.png")
    logger.info(f"    Saved {len(masks)} masks + overlay to masks/")


def visualize_cost_map(cost_map: np.ndarray, rgb_image: Image.Image, output_path: Path):
    """Create a visualization of the cost map overlaid on the RGB image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(np.array(rgb_image))
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # Cost map
    im = axes[1].imshow(cost_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title("Traversability Cost Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Cost (0=safe, 1=obstacle)")

    # Overlay
    overlay = np.array(rgb_image).copy()
    cost_colored = plt.cm.RdYlGn_r(cost_map)[:, :, :3]
    overlay = (overlay * 0.6 + cost_colored * 255 * 0.4).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("Cost Map Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def process_single_image_lac(
    rgb_path: Path,
    depth_path: Path,
    image_id: str,
    folder_name: str,
    reasoner_model, reasoner_processor,
    evaluator_model, evaluator_processor,
    config: Dict,
    output_dir: Path,
) -> Dict:
    """Process a single image through the full LaC pipeline."""
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_overlay")
    seg_method = config["model"].get("segmentation", {}).get("method", "vlm_only")
    logger.info(f"Processing {folder_name}/{image_id} [LaC pipeline, mode={input_mode}, seg={seg_method}]")

    # Load images
    rgb_image = load_image_for_vlm(rgb_path)
    depth_image = None
    if input_mode != "rgb_only":
        depth_image = load_image_for_vlm(depth_path)

    result = {
        "folder": folder_name,
        "image_id": image_id,
        "pipeline": "LaC",
        "input_mode": input_mode,
        "segmentation_method": seg_method,
        "timestamp": datetime.now().isoformat(),
    }

    # ---- Stage 1: Free Ground Reasoner ----
    if config["pipeline"].get("run_reasoner", True):
        logger.info("  Stage 1: Free Ground Reasoner...")
        t0 = time.time()
        reasoner_output = run_free_ground_reasoner(
            reasoner_model, reasoner_processor, config, rgb_image, depth_image
        )
        t1 = time.time()
        areas = reasoner_output.get("free_ground_areas", [])
        logger.info(f"    Found {len(areas)} free ground areas ({t1-t0:.1f}s)")
        result["reasoner"] = {
            "output": reasoner_output,
            "inference_time": round(t1 - t0, 2),
            "num_areas": len(areas),
        }
    else:
        reasoner_output = {"free_ground_areas": [], "navigability_reasoning": ""}
        result["reasoner"] = {"output": reasoner_output, "skipped": True}

    # ---- Stage 2: Traversability Evaluator ----
    areas = reasoner_output.get("free_ground_areas", [])
    if config["pipeline"].get("run_evaluator", True) and areas:
        logger.info("  Stage 2: Traversability Evaluator...")
        t0 = time.time()
        evaluator_output = run_traversability_evaluator(
            evaluator_model, evaluator_processor, config,
            reasoner_output, rgb_image, depth_image
        )
        t1 = time.time()
        scores = evaluator_output.get("traversability_score", {})
        logger.info(f"    Scores: {scores} ({t1-t0:.1f}s)")
        result["evaluator"] = {
            "output": evaluator_output,
            "inference_time": round(t1 - t0, 2),
        }
    else:
        evaluator_output = {"traversability_score": {}}
        result["evaluator"] = {"output": evaluator_output, "skipped": True}

    # ---- Stage 2.5: Filter score=0 areas (non-traversable) ----
    scores = evaluator_output.get("traversability_score", {})
    filtered_areas = []
    for area in areas:
        name = area.get("name", "")
        score = scores.get(name, None)
        if score == 0:
            logger.info(f"    Filtering out '{name}' (score=0, non-traversable)")
            continue
        filtered_areas.append(area)

    if len(filtered_areas) < len(areas):
        logger.info(f"    Kept {len(filtered_areas)}/{len(areas)} areas after score filtering")

    # ---- Stage 3: Segmentation ----
    # Collect VLM bboxes (percentage-based) for the filtered areas
    vlm_bboxes = []
    area_names = []
    for area in filtered_areas:
        bbox = area.get("bbox", {})
        if bbox and all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
            vlm_bboxes.append(bbox)
            area_names.append(area.get("name", f"area_{len(vlm_bboxes)}"))

    if config["pipeline"].get("run_segmentation", True):
        logger.info(f"  Stage 3: Segmentation — method={seg_method} ({len(vlm_bboxes)} VLM regions)...")
        t0 = time.time()
        masks, bboxes_used = run_segmentation(rgb_image, vlm_bboxes, config)
        t1 = time.time()
        logger.info(f"    Generated {len(masks)} masks ({t1-t0:.1f}s)")

        # Generate area names for Grounding-DINO detections if they produced more masks
        if len(masks) > len(area_names):
            for i in range(len(area_names), len(masks)):
                area_names.append(f"gdino_region_{i}")

        result["segmentation"] = {
            "method": seg_method,
            "num_masks": len(masks),
            "num_filtered": len(areas) - len(filtered_areas),
            "inference_time": round(t1 - t0, 2),
        }
        bboxes = bboxes_used
    else:
        masks = bbox_to_masks(vlm_bboxes, rgb_image.size)
        bboxes = vlm_bboxes
        result["segmentation"] = {"method": "vlm_only", "num_masks": len(masks), "skipped": True}

    # ---- Stage 3.5: Depth-based flatness validation ----
    if masks and depth_image is not None:
        depth_array = np.array(depth_image.convert("L")).astype(np.float32)
        validated_masks = []
        validated_names = []
        validated_bboxes = []
        for i, (mask, name) in enumerate(zip(masks, area_names)):
            mask_pixels = depth_array[mask > 0]
            if len(mask_pixels) < 100:
                logger.info(f"    Skipping '{name}' — too few pixels ({len(mask_pixels)})")
                continue
            # Compute depth statistics within the mask
            depth_std = np.std(mask_pixels)
            depth_range = np.max(mask_pixels) - np.min(mask_pixels)
            # Flat surfaces should have low depth variance
            # High std or range suggests stairs/inclines
            flatness_threshold = config["pipeline"].get("depth_flatness_threshold", 80)
            if depth_std > flatness_threshold:
                logger.info(f"    Filtering '{name}' — depth std={depth_std:.1f} > {flatness_threshold} (likely stairs/incline)")
                continue
            if depth_range > flatness_threshold * 3:
                logger.info(f"    Filtering '{name}' — depth range={depth_range:.1f} (likely stairs/incline)")
                continue
            validated_masks.append(mask)
            validated_names.append(name)
            if i < len(bboxes):
                validated_bboxes.append(bboxes[i])

        if len(validated_masks) < len(masks):
            logger.info(f"    Depth validation: kept {len(validated_masks)}/{len(masks)} masks")
            masks = validated_masks
            area_names = validated_names
            bboxes = validated_bboxes
            result["segmentation"]["depth_filtered"] = len(masks) - len(validated_masks)

    # Save segmentation masks (if any were generated)
    if masks and config["output"].get("save_visualizations", True):
        save_segmentation_masks(
            masks, area_names, rgb_image, bboxes,
            output_dir / folder_name, image_id,
        )

    # ---- Stage 4: Cost Map ----
    scores = evaluator_output.get("traversability_score", {})
    if config["pipeline"].get("run_cost_map", True) and masks:
        logger.info("  Stage 4: Building cost map...")
        t0 = time.time()
        cost_map = build_cost_map(masks, scores, area_names, rgb_image.size, config)
        t1 = time.time()
        logger.info(f"    Cost map shape: {cost_map.shape} ({t1-t0:.1f}s)")
        result["cost_map"] = {
            "shape": list(cost_map.shape),
            "min_cost": float(cost_map.min()),
            "max_cost": float(cost_map.max()),
            "mean_cost": float(cost_map.mean()),
            "inference_time": round(t1 - t0, 2),
        }

        # Save visualizations
        if config["output"].get("save_cost_maps", True):
            vis_dir = output_dir / folder_name / "cost_maps"
            vis_dir.mkdir(parents=True, exist_ok=True)
            visualize_cost_map(cost_map, rgb_image, vis_dir / f"{image_id}_costmap.png")

            # Also save raw cost map
            np.save(vis_dir / f"{image_id}_costmap.npy", cost_map)

    # Save individual result
    if config["output"].get("save_individual_json", True):
        json_dir = output_dir / folder_name
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"{image_id}_lac_analysis.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


def run_lac_pipeline(config: Dict):
    """Run the full LaC-adapted free ground detection pipeline."""
    start_time = time.time()

    # Setup output directory
    model_name = config["model"]["reasoner"]["name"]
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_overlay")
    seg_method = config["model"].get("segmentation", {}).get("method", "vlm_only")
    lac_suffix = config["pipeline"].get("output_suffix", "")
    lac_folder = f"{model_name}_LaC{lac_suffix}" if lac_suffix else f"{model_name}_LaC"
    output_dir = Path(config["output"]["dir"]) / lac_folder / f"{input_mode}_{seg_method}"
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

    # Load VLM models
    logger.info("=" * 60)
    logger.info("Loading VLM Models...")
    logger.info("=" * 60)

    # Stage 1 model (Reasoner)
    reasoner_config = {"model": config["model"]["reasoner"]}
    reasoner_model, reasoner_processor = load_vlm_model(reasoner_config)

    # Stage 2 model (Evaluator) — can reuse same model
    evaluator_config = {"model": config["model"]["evaluator"]}
    if config["model"]["evaluator"]["hf_model_id"] == config["model"]["reasoner"]["hf_model_id"]:
        logger.info("Reusing reasoner model for evaluator (same model)")
        evaluator_model, evaluator_processor = reasoner_model, reasoner_processor
    else:
        evaluator_model, evaluator_processor = load_vlm_model(evaluator_config)

    # Pre-load segmentation models (SAM / Grounding-DINO) — done once
    seg_method = config["model"].get("segmentation", {}).get("method", "vlm_only")
    if seg_method == "grounding_dino":
        logger.info("Pre-loading Grounding-DINO + SAM models...")
        _load_grounding_dino_model(config)
        _load_sam_model(config)
    elif seg_method == "sam":
        logger.info("Pre-loading SAM model...")
        _load_sam_model(config)

    all_results = []

    # Process each folder
    for folder in folders:
        logger.info("=" * 60)
        logger.info(f"Processing folder: {folder.name}")
        logger.info("=" * 60)

        pairs = discover_image_pairs(folder, config)
        if not pairs:
            logger.warning(f"No image pairs found in {folder.name}")
            continue

        pairs = filter_image_pairs(pairs, config, folder_name=folder.name)

        for i, (rgb_path, depth_path, image_id) in enumerate(pairs):
            try:
                result = process_single_image_lac(
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    image_id=image_id,
                    folder_name=folder.name,
                    reasoner_model=reasoner_model,
                    reasoner_processor=reasoner_processor,
                    evaluator_model=evaluator_model,
                    evaluator_processor=evaluator_processor,
                    config=config,
                    output_dir=output_dir,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {folder.name}/{image_id}: {e}")
                all_results.append({
                    "folder": folder.name,
                    "image_id": image_id,
                    "error": str(e),
                })

    # Save combined results
    if all_results:
        csv_path = output_dir / "lac_results_summary.json"
        with open(csv_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Saved results to {csv_path}")

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed:.1f}s ({len(all_results)} images)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LaC-adapted Free Ground Space Detection Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="lac_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Override reasoner+evaluator model name. Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}",
    )
    parser.add_argument(
        "--hf_model_id", type=str, default=None,
        help="Override full HuggingFace model ID for reasoner+evaluator",
    )
    parser.add_argument(
        "--reasoner_model", type=str, default=None,
        help="Override reasoner model only (name or HF ID)",
    )
    parser.add_argument(
        "--evaluator_model", type=str, default=None,
        help="Override evaluator model only (name or HF ID)",
    )
    parser.add_argument(
        "--segmentation_method", type=str, default=None,
        choices=["sam", "grounding_dino", "vlm_only"],
        help="Override segmentation method",
    )
    parser.add_argument(
        "--grounding_dino_text_prompt", type=str, default=None,
        help="Override Grounding-DINO text prompt (e.g. 'flat floor . floor . ground .')",
    )
    parser.add_argument(
        "--grounding_dino_box_threshold", type=float, default=None,
        help="Override Grounding-DINO box threshold (default: 0.3)",
    )
    parser.add_argument(
        "--input_mode", type=str, default=None,
        choices=["rgb_only", "rgb_depth_overlay"],
        help="Input mode for VLM",
    )
    parser.add_argument(
        "--specific_images", nargs="+", default=None,
        help="Specific image IDs to process (e.g., image28 image86)",
    )
    parser.add_argument(
        "--gt_dir", type=str, default=None,
        help="Path to annotated ground truth directory. Reads exact folder→image "
             "mapping so only annotated images are processed in their specific folders.",
    )
    parser.add_argument(
        "--folders", nargs="+", default=None,
        help="Specific folder names to process",
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Quick test: 3 images per folder",
    )
    parser.add_argument(
        "--stages", type=str, default=None,
        help="Comma-separated stages to run: reasoner,evaluator,segmentation,costmap",
    )
    parser.add_argument(
        "--prompt_dir", type=str, default=None,
        help="Path to custom prompt directory (default: lac_free_ground/prompts/). "
             "Use 'prompts_navigable' for the navigable-area variant.",
    )
    parser.add_argument(
        "--output_suffix", type=str, default=None,
        help="Suffix appended to output folder name (e.g., 'navigable' → {model}_LaC_navigable/...). "
             "Prevents overwriting previous results.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove output directory before running (avoids stale files)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def _resolve_model_id(name_or_id: str) -> Tuple[str, str]:
    """Resolve a model name or HF ID to (name, hf_model_id).

    Checks MODEL_REGISTRY first, then treats as a full HF ID if it
    contains '/', otherwise uses as-is for both name and ID.
    """
    if name_or_id in MODEL_REGISTRY:
        return name_or_id, MODEL_REGISTRY[name_or_id]
    if "/" in name_or_id:
        return name_or_id.split("/")[-1], name_or_id
    return name_or_id, name_or_id


def merge_args(config: Dict, args: argparse.Namespace) -> Dict:
    """Merge CLI arguments into config. CLI takes precedence."""
    # --model overrides both reasoner and evaluator
    if getattr(args, "model", None):
        model_name, hf_id = _resolve_model_id(args.model)
        for stage in ["reasoner", "evaluator"]:
            config["model"][stage]["name"] = model_name
            config["model"][stage]["hf_model_id"] = hf_id

    if getattr(args, "hf_model_id", None):
        model_name = args.hf_model_id.split("/")[-1]
        for stage in ["reasoner", "evaluator"]:
            config["model"][stage]["name"] = model_name
            config["model"][stage]["hf_model_id"] = args.hf_model_id

    # --reasoner_model / --evaluator_model override individual stages
    # When --reasoner_model is set without --evaluator_model, reuse same model for both
    if getattr(args, "reasoner_model", None):
        name, hf_id = _resolve_model_id(args.reasoner_model)
        config["model"]["reasoner"]["name"] = name
        config["model"]["reasoner"]["hf_model_id"] = hf_id
        # Also set evaluator to same model unless explicitly overridden
        if not getattr(args, "evaluator_model", None):
            config["model"]["evaluator"]["name"] = name
            config["model"]["evaluator"]["hf_model_id"] = hf_id

    if getattr(args, "evaluator_model", None):
        name, hf_id = _resolve_model_id(args.evaluator_model)
        config["model"]["evaluator"]["name"] = name
        config["model"]["evaluator"]["hf_model_id"] = hf_id

    # Segmentation overrides
    if getattr(args, "segmentation_method", None):
        config["model"]["segmentation"]["method"] = args.segmentation_method

    if getattr(args, "grounding_dino_text_prompt", None):
        config["model"]["segmentation"]["grounding_dino_text_prompt"] = (
            args.grounding_dino_text_prompt
        )

    if getattr(args, "grounding_dino_box_threshold", None) is not None:
        config["model"]["segmentation"]["grounding_dino_box_threshold"] = (
            args.grounding_dino_box_threshold
        )

    # Pipeline overrides
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

    if getattr(args, "stages", None):
        stages = [s.strip() for s in args.stages.split(",")]
        config["pipeline"]["run_reasoner"] = "reasoner" in stages
        config["pipeline"]["run_evaluator"] = "evaluator" in stages
        config["pipeline"]["run_segmentation"] = "segmentation" in stages
        config["pipeline"]["run_cost_map"] = "costmap" in stages

    if getattr(args, "clean", False):
        config["pipeline"]["clean_output"] = True

    if getattr(args, "output_suffix", None):
        config["pipeline"]["output_suffix"] = args.output_suffix

    return config


def main():
    # ── Ensure all HuggingFace downloads/cache use $WORK ──────────────
    _work_dir = os.environ.get("WORK", str(Path(__file__).parent.parent))
    os.environ.setdefault("HF_HOME", os.path.join(_work_dir, ".cache", "huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info(f"HuggingFace cache (HF_HOME): {os.environ['HF_HOME']}")

    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set custom prompt directory if provided
    if getattr(args, "prompt_dir", None):
        prompt_path = Path(args.prompt_dir)
        if not prompt_path.is_absolute():
            prompt_path = Path(__file__).parent / prompt_path
        set_prompt_dir(prompt_path)
        logger.info(f"Using custom prompt directory: {prompt_path}")

    # Load config
    config_path = args.config
    if not Path(config_path).exists():
        # Try relative to script directory
        config_path = Path(__file__).parent / config_path
    config = yaml.safe_load(open(config_path))
    config = merge_args(config, args)

    # Reconfigure logging with model-specific directory
    model_name = config["model"]["reasoner"]["name"]
    output_suffix = config["pipeline"].get("output_suffix", "")
    log_dir = _make_log_dir(model_name, suffix=output_suffix)
    log_file = log_dir / f"lac_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    root_logger = logging.getLogger()
    # Remove existing handlers and add new ones with model-specific path
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(logging.StreamHandler())
    root_logger.addHandler(logging.FileHandler(log_file))

    seg_method = config["model"].get("segmentation", {}).get("method", "vlm_only")

    # Print summary
    logger.info("=" * 60)
    logger.info("LaC FREE GROUND DETECTION PIPELINE")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    logger.info(f"Reasoner model: {config['model']['reasoner']['name']} "
                f"({config['model']['reasoner']['hf_model_id']})")
    logger.info(f"Evaluator model: {config['model']['evaluator']['name']} "
                f"({config['model']['evaluator']['hf_model_id']})")
    logger.info(f"Segmentation method: {seg_method}")
    if seg_method == "grounding_dino":
        gdino_id = config["model"]["segmentation"].get("grounding_dino_model_id", "N/A")
        gdino_prompt = config["model"]["segmentation"].get("grounding_dino_text_prompt", "N/A")
        gdino_box_th = config["model"]["segmentation"].get("grounding_dino_box_threshold", 0.3)
        logger.info(f"  Grounding-DINO model: {gdino_id}")
        logger.info(f"  Text prompt: '{gdino_prompt}'")
        logger.info(f"  Box threshold: {gdino_box_th}")
    elif seg_method == "sam":
        sam_id = config["model"]["segmentation"].get("sam_model_id", "N/A")
        logger.info(f"  SAM model: {sam_id}")
    logger.info(f"Input mode: {config['pipeline'].get('input_mode', 'rgb_depth_overlay')}")
    logger.info(f"Stages: reasoner={config['pipeline'].get('run_reasoner', True)}, "
                f"evaluator={config['pipeline'].get('run_evaluator', True)}, "
                f"segmentation={config['pipeline'].get('run_segmentation', True)}, "
                f"costmap={config['pipeline'].get('run_cost_map', True)}")
    logger.info(f"Output: {config['output']['dir']}")
    logger.info("=" * 60)

    run_lac_pipeline(config)


if __name__ == "__main__":
    main()
