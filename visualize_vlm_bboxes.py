#!/usr/bin/env python3
"""Visualize VLM bounding boxes or polygons on images.

Usage:
    python3 visualize_vlm_bboxes.py --run_dir /path/to/two_vlm/Qwen/rgb_only --num_images 5
    python3 visualize_vlm_bboxes.py --run_dir /path/to/two_vlm/Qwen/rgb_only --image_ids 105 110 200
    python3 visualize_vlm_bboxes.py --image_path /path/to/image.png --annotations_file /path/to/annotations.json
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Colors for different areas
COLORS = [
    (0, 255, 0),    # green
    (0, 128, 255),  # orange
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 128, 0),  # blue
]

DATA_BASE = Path("/home/woody/iwnt/iwnt164h/mlp_dataset/prospthesisproject-Data/Code/Data")
RGB_SUBFOLDER = "sharpen_rgb/PNG"


def find_rgb_image(folder_name: str, image_id: str) -> Path:
    """Find RGB image path."""
    path = DATA_BASE / folder_name / RGB_SUBFOLDER / f"{image_id}.png"
    if path.exists():
        return path
    return None


def draw_bbox(image: np.ndarray, bbox: dict, label: str, color: tuple, thickness=2):
    """Draw a bounding box with label on image."""
    h, w = image.shape[:2]
    x1 = int(bbox["x1"] / 100 * w)
    y1 = int(bbox["y1"] / 100 * h)
    x2 = int(bbox["x2"] / 100 * w)
    y2 = int(bbox["y2"] / 100 * h)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(image, label, (x1 + 3, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

    return image


def draw_bbox_pixels(image: np.ndarray, bbox: dict, label: str, color: tuple, thickness=2):
    """Draw a bounding box given pixel coordinates."""
    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    x2 = int(bbox["x2"])
    y2 = int(bbox["y2"])

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y1 = max(0, y1 - text_h - 10)
        label_y2 = max(0, y1)
        cv2.rectangle(image, (x1, label_y1), (x1 + text_w + 6, label_y2), color, -1)
        cv2.putText(image, label, (x1 + 3, max(0, y1 - 5)), font, font_scale, (255, 255, 255), font_thickness)

    return image


def draw_polygon(image: np.ndarray, polygon: list, label: str, color: tuple, thickness=2, fill_alpha=0.2):
    """Draw a polygon with label on image."""
    if not polygon:
        return image

    points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0, image)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

    if label:
        x, y = points[:, 0, 0].min(), points[:, 0, 1].min()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y1 = max(0, y - text_h - 10)
        label_y2 = max(0, y)
        cv2.rectangle(image, (x, label_y1), (x + text_w + 6, label_y2), color, -1)
        cv2.putText(image, label, (x + 3, max(0, y - 5)), font, font_scale, (255, 255, 255), font_thickness)

    return image


def load_annotations(path: Path | None, raw_json: str | None):
    if path:
        with open(path) as f:
            return json.load(f)
    if raw_json:
        return json.loads(raw_json)
    return None


def load_bbox(raw_json: str | None):
    if not raw_json:
        return None
    return json.loads(raw_json)


def main():
    parser = argparse.ArgumentParser(description="Visualize VLM bounding boxes")
    parser.add_argument("--image_path", type=Path, default=None, help="Visualize shapes on a single image")
    parser.add_argument("--annotations_file", type=Path, default=None, help="JSON file with polygon annotations")
    parser.add_argument("--annotations_json", type=str, default=None, help="JSON string with polygon annotations")
    parser.add_argument("--bbox_json", type=str, default=None, help="JSON string with a single bbox object")
    parser.add_argument("--bbox_pixels", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"), help="Single bbox in pixel coordinates")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"), help="Single bbox in percent coordinates")
    parser.add_argument("--label", type=str, default="", help="Label for single bbox mode")
    parser.add_argument("--output_path", type=Path, default=None, help="Output image path for single-image mode")
    parser.add_argument("--run_dir", type=Path, default=None, help="Path to run output dir (e.g., two_vlm/Qwen/rgb_only)")
    parser.add_argument("--num_images", type=int, default=5, help="Number of random images to visualize")
    parser.add_argument("--image_ids", nargs="+", help="Specific image IDs to visualize (e.g., 105 110 200)")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory (default: run_dir/vlm_bbox_viz)")
    parser.add_argument("--show_scores", action="store_true", help="Show evaluator scores if available")
    args = parser.parse_args()

    if args.image_path:
        annotations = load_annotations(args.annotations_file, args.annotations_json)
        bbox_json = load_bbox(args.bbox_json)

        image = np.array(Image.open(args.image_path).convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if bbox_json:
            color = COLORS[0]
            if all(k in bbox_json for k in ["x1", "y1", "x2", "y2"]):
                draw_bbox(image, bbox_json, args.label, color)
            else:
                print("--bbox_json must contain x1, y1, x2, and y2")
                return
        elif args.bbox_pixels or args.bbox:
            color = COLORS[0]
            if args.bbox_pixels:
                bbox = {"x1": args.bbox_pixels[0], "y1": args.bbox_pixels[1], "x2": args.bbox_pixels[2], "y2": args.bbox_pixels[3]}
                draw_bbox_pixels(image, bbox, args.label, color)
            else:
                bbox = {"x1": args.bbox[0], "y1": args.bbox[1], "x2": args.bbox[2], "y2": args.bbox[3]}
                draw_bbox(image, bbox, args.label, color)
        elif not annotations:
            print("Provide --annotations_file, --annotations_json, --bbox_pixels, or --bbox in single-image mode")
            return
        else:
            for i, annotation in enumerate(annotations):
                label = annotation.get("label", f"shape_{i}")
                color = COLORS[i % len(COLORS)]

                polygon = annotation.get("polygon")
                if polygon:
                    draw_polygon(image, polygon, label, color)
                    continue

                bbox = annotation.get("bbox_pixels") or annotation.get("bbox")
                if bbox and all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                    if annotation.get("bbox_pixels"):
                        draw_bbox_pixels(image, bbox, label, color)
                    else:
                        draw_bbox(image, bbox, label, color)

        output_path = args.output_path or args.image_path.with_name(f"{args.image_path.stem}_viz.png")
        cv2.imwrite(str(output_path), image)
        print(f"Saved: {output_path}")
        return

    if not args.run_dir:
        print("Provide --run_dir for bbox mode or --image_path for polygon mode")
        return

    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "vlm_bbox_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(run_dir.glob("*/*_lac_analysis.json"))
    if not json_files:
        print(f"No JSON files found in {run_dir}")
        return

    # Filter to images that have VLM areas with bboxes
    valid_jsons = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        vlm_areas = data.get("vlm_areas", [])
        if vlm_areas and any(a.get("bbox") for a in vlm_areas):
            valid_jsons.append((jf, data))

    print(f"Found {len(valid_jsons)} images with VLM bboxes (out of {len(json_files)} total)")

    if not valid_jsons:
        print("No images with VLM bboxes found!")
        return

    # Select images
    if args.image_ids:
        selected = [(jf, d) for jf, d in valid_jsons
                     if d.get("image_id", "") in args.image_ids
                     or str(d.get("image_id", "")) in args.image_ids]
        if not selected:
            print(f"No images found with IDs: {args.image_ids}")
            return
    else:
        n = min(args.num_images, len(valid_jsons))
        selected = random.sample(valid_jsons, n)

    for jf, data in selected:
        folder = data.get("folder", jf.parent.name)
        image_id = str(data.get("image_id", jf.stem.replace("_lac_analysis", "")))
        vlm_areas = data.get("vlm_areas", [])
        evaluator = data.get("evaluator", {})
        scores = evaluator.get("output", {}).get("traversability_score", {}) if evaluator else {}

        # Find RGB image
        rgb_path = find_rgb_image(folder, image_id)
        if not rgb_path:
            print(f"  Skipping {folder}/{image_id} — RGB not found")
            continue

        image = np.array(Image.open(rgb_path).convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw each VLM area bbox
        for i, area in enumerate(vlm_areas):
            bbox = area.get("bbox", {})
            if not bbox or not all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                continue

            name = area.get("name", f"area_{i}")
            color = COLORS[i % len(COLORS)]

            label = name
            if args.show_scores and name in scores:
                label += f" (score={scores[name]})"

            draw_bbox(image, bbox, label, color)
            print(f"  {folder}/{image_id}: '{name}' bbox=({bbox['x1']},{bbox['y1']},{bbox['x2']},{bbox['y2']})"
                  + (f" score={scores.get(name, 'N/A')}" if args.show_scores else ""))

        out_path = output_dir / f"{folder}_{image_id}_vlm_bbox.png"
        cv2.imwrite(str(out_path), image)
        print(f"  Saved: {out_path}")

    print(f"\nDone! {len(selected)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
