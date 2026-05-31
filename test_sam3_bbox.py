#!/usr/bin/env python3
"""Test SAM3 with bbox-only input on 5 images from VLM output.

Reads VLM bboxes from pipeline JSON output, runs SAM3 with bbox-only
prompt (no text), and saves segmentation masks side-by-side with VLM bbox.

Uses the official SAM3 bbox API from:
https://huggingface.co/facebook/sam3

Usage:
    python3 test_sam3_bbox.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR = Path("/home/woody/iwnt/iwnt164h")
RUN_DIR = WORK_DIR / "free_ground_results" / "two_vlm" / "Gemma" / "rgb_depth_separate"
DATA_BASE = WORK_DIR / "mlp_dataset/prospthesisproject-Data/Code/Data"
RGB_SUBFOLDER = "sharpen_rgb/PNG"
OUTPUT_DIR = RUN_DIR / "sam3_bbox_test"
SAM3_MODEL_ID = "facebook/sam3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The 5 images from vlm_bbox_viz
TEST_IMAGES = [
    ("lms_kamal_LA_downstairs_Nopeople_1", "116"),
    ("lms_kamal_LA_downstairs_Nopeople_1", "124"),
    ("lms_kamal_LA_downstairs_Nopeople_1", "405"),
    ("lms_kamal_RA_downstairs_Nopeople_1", "162"),
    ("lms_kamal_RA_downstairs_Nopeople_1", "191"),
]


def load_sam3():
    """Load SAM3 model and processor."""
    from transformers import Sam3Model, Sam3Processor

    print(f"Loading SAM3: {SAM3_MODEL_ID} ...")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    print("SAM3 loaded.")
    return model, processor


def segment_with_bbox(model, processor, image: Image.Image, bbox_xyxy: list):
    """Run SAM3 with bbox-only prompt (no text).

    Args:
        image: PIL RGB image
        bbox_xyxy: [x1, y1, x2, y2] in pixel coordinates

    Returns:
        dict with masks, scores, and labels
    """
    # Bbox-only input per SAM3 docs
    input_boxes = [[bbox_xyxy]]       # [batch, num_boxes, 4]
    input_boxes_labels = [[1]]         # 1 = positive box

    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt",
    ).to(DEVICE, dtype=torch.float16)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process using SAM3's official method
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    return results


def draw_comparison(rgb_image, vlm_bbox, sam3_masks, area_name, output_path):
    """Save side-by-side: Original | VLM bbox | SAM3 segmentation."""
    rgb_array = np.array(rgb_image.convert("RGB"))
    h, w = rgb_array.shape[:2]

    # Panel 1: Original RGB
    panel_orig = rgb_array.copy()

    # Panel 2: VLM bbox overlay
    panel_bbox = rgb_array.copy()
    x1 = int(vlm_bbox["x1"] / 100 * w)
    y1 = int(vlm_bbox["y1"] / 100 * h)
    x2 = int(vlm_bbox["x2"] / 100 * w)
    y2 = int(vlm_bbox["y2"] / 100 * h)
    cv2.rectangle(panel_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(panel_bbox, f"VLM: {area_name}", (x1 + 3, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Panel 3: SAM3 segmentation overlay
    panel_sam3 = rgb_array.copy()
    if sam3_masks is not None and len(sam3_masks) > 0:
        # Combine all SAM3 masks
        combined = np.zeros((h, w), dtype=bool)
        for mask in sam3_masks:
            if mask.shape != (h, w):
                # Resize if needed
                mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((w, h), Image.NEAREST)
                mask = np.array(mask_pil) > 127
            combined |= mask

        # Green overlay
        color_mask = np.zeros_like(panel_sam3)
        color_mask[combined] = (0, 255, 0)
        panel_sam3 = (panel_sam3 * 0.6 + color_mask * 0.4).astype(np.uint8)
        coverage = combined.sum() / (h * w) * 100
        cv2.putText(panel_sam3, f"SAM3 bbox-only: {coverage:.1f}%", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(panel_sam3, "SAM3: NO MASK", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Stack panels horizontally
    gap = np.full((h, 4, 3), 255, dtype=np.uint8)
    combined_img = np.hstack([panel_orig, gap, panel_bbox, gap, panel_sam3])

    # Add labels
    label_h = 30
    label_bar = np.full((label_h, combined_img.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(label_bar, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(label_bar, "VLM Bbox", (w + 14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(label_bar, "SAM3 (bbox-only)", (2 * w + 22, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    final = np.vstack([label_bar, combined_img])
    cv2.imwrite(str(output_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load SAM3
    model, processor = load_sam3()

    for folder, image_id in TEST_IMAGES:
        print(f"\n{'='*60}")
        print(f"Processing: {folder}/image{image_id}")

        # Read JSON
        json_path = RUN_DIR / folder / f"image{image_id}_lac_analysis.json"
        if not json_path.exists():
            print(f"  JSON not found: {json_path}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        vlm_areas = data.get("vlm_areas", [])
        if not vlm_areas:
            print(f"  No VLM areas found")
            continue

        # Load RGB image
        rgb_path = DATA_BASE / folder / RGB_SUBFOLDER / f"image{image_id}.png"
        if not rgb_path.exists():
            print(f"  RGB not found: {rgb_path}")
            continue

        rgb_image = Image.open(rgb_path).convert("RGB")
        w, h = rgb_image.size
        print(f"  Image size: {w}x{h}")
        print(f"  VLM areas: {len(vlm_areas)}")

        # Process each VLM area with SAM3 bbox-only
        all_sam3_masks = []
        for i, area in enumerate(vlm_areas):
            bbox = area.get("bbox", {})
            name = area.get("name", f"area_{i}")

            if not bbox or not all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                print(f"  Area '{name}': no valid bbox, skipping")
                continue

            # Convert percentage bbox to pixel coordinates
            box_xyxy = [
                int(bbox["x1"] / 100 * w),
                int(bbox["y1"] / 100 * h),
                int(bbox["x2"] / 100 * w),
                int(bbox["y2"] / 100 * h),
            ]
            print(f"  Area '{name}': bbox=({bbox['x1']},{bbox['y1']},{bbox['x2']},{bbox['y2']}) "
                  f"→ pixels=({box_xyxy[0]},{box_xyxy[1]},{box_xyxy[2]},{box_xyxy[3]})")

            try:
                results = segment_with_bbox(model, processor, rgb_image, box_xyxy)

                # Extract masks from results
                masks = results.get("masks", [])
                scores = results.get("scores", [])
                labels = results.get("labels", [])

                print(f"    SAM3 results: {len(masks)} masks, scores={scores}")

                if len(masks) > 0:
                    # Convert tensor masks to numpy
                    for m in masks:
                        if hasattr(m, "cpu"):
                            m = m.cpu().numpy()
                        all_sam3_masks.append(m.astype(bool))
                else:
                    print(f"    SAM3 produced no masks for this bbox")

            except Exception as e:
                print(f"    SAM3 failed: {e}")
                import traceback
                traceback.print_exc()

        # Save comparison visualization
        # Use first area's bbox for the VLM bbox panel
        first_bbox = vlm_areas[0].get("bbox", {})
        first_name = vlm_areas[0].get("name", "unknown")
        out_path = OUTPUT_DIR / f"{folder}_image{image_id}_sam3_bbox.png"
        draw_comparison(rgb_image, first_bbox, all_sam3_masks, first_name, out_path)

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
