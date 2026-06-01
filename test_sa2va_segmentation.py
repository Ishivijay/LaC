#!/usr/bin/env python3
"""Run Sa2VA on images and save segmentation mask overlays.

This is a standalone test path for ByteDance/Sa2VA-Qwen3-VL-4B.
It does not use the LaC SAM3 pipeline because Sa2VA can return masks
directly from `predict_forward`.

Example:
    export HF_HOME=$WORK/.cache/huggingface
    python3 test_sa2va_segmentation.py \
        --input-dir /path/to/images \
        --output-dir /path/to/output \
        --prompt "<image>Please segment the floor and free ground in this image."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_MODEL_ID = "ByteDance/Sa2VA-Qwen3-VL-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sa2VA segmentation on images.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save overlays")
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>Please segment the free walkable areas in this image.",
        help="Text prompt sent to Sa2VA",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional limit on number of images to process (0 = all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]


def load_model(model_id: str, device: str):
    from transformers import AutoModel, AutoProcessor

    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    if device != "cpu":
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    return model, processor


def mask_to_bool(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.dtype != bool:
        arr = arr > 0.5
    return arr


def mask_bbox(mask: np.ndarray) -> dict[str, int] | None:
    mask_bool = mask_to_bool(mask)
    ys, xs = np.where(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return None
    return {
        "x1": int(xs.min()),
        "y1": int(ys.min()),
        "x2": int(xs.max()),
        "y2": int(ys.max()),
    }


def combine_masks(masks: Sequence[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    combined = np.zeros((height, width), dtype=bool)
    for mask in masks:
        mask_bool = mask_to_bool(mask)
        if mask_bool.shape != (height, width):
            mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
            mask_img = mask_img.resize((width, height), Image.NEAREST)
            mask_bool = np.array(mask_img) > 127
        combined |= mask_bool
    return combined


def overlay_masks(image: Image.Image, masks: Sequence[np.ndarray], labels: Sequence[str] | None = None) -> Image.Image:
    base = image.convert("RGB")
    width, height = base.size
    rgb = np.array(base).copy()
    combined = combine_masks(masks, (width, height)) if masks else np.zeros((height, width), dtype=bool)

    if combined.any():
        color = np.zeros_like(rgb)
        color[combined] = (0, 255, 0)
        rgb = (rgb * 0.6 + color * 0.4).astype(np.uint8)

    out = Image.fromarray(rgb)
    draw = ImageDraw.Draw(out)
    if labels:
        draw.text((10, 10), f"Masks: {len(masks)}", fill=(255, 255, 255))
        draw.text((10, 30), ", ".join(labels[:3]), fill=(255, 255, 255))
    else:
        draw.text((10, 10), f"Masks: {len(masks)}", fill=(255, 255, 255))
    return out


def save_mask_images(output_dir: Path, stem: str, masks: Sequence[np.ndarray], size: tuple[int, int]) -> None:
    width, height = size
    combined = combine_masks(masks, (width, height)) if masks else np.zeros((height, width), dtype=bool)

    if combined.any():
        combined_img = Image.fromarray((combined.astype(np.uint8) * 255))
        combined_img.save(output_dir / f"{stem}_sa2va_mask_combined.png")

    for index, mask in enumerate(masks):
        mask_bool = mask_to_bool(mask)
        if mask_bool.shape != (height, width):
            mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
            mask_img = mask_img.resize((width, height), Image.NEAREST)
            mask_bool = np.array(mask_img) > 127
        mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
        mask_img.save(output_dir / f"{stem}_sa2va_mask_{index:02d}.png")


def save_prediction_json(
    output_dir: Path,
    stem: str,
    prediction: str,
    masks: Sequence[np.ndarray],
    labels: Sequence[str],
) -> None:
    payload = {
        "prediction": prediction,
        "labels": list(labels),
        "mask_bboxes": [],
    }
    for index, mask in enumerate(masks):
        payload["mask_bboxes"].append(
            {
                "index": index,
                "label": labels[index] if index < len(labels) else None,
                "bbox": mask_bbox(mask),
            }
        )
    json_path = output_dir / f"{stem}_sa2va_result.json"
    json_path.write_text(json.dumps(payload, indent=2))


def run_one(model, processor, image: Image.Image, prompt: str):
    input_dict = {
        "image": image,
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


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.input_dir)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    if not images:
        print(f"No images found in: {args.input_dir}")
        return 1

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id, args.device)
    print("Model loaded.")

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        print(f"Processing {image_path.name} ...")
        prediction, masks, labels = run_one(model, processor, image, args.prompt)

        overlay = overlay_masks(image, masks, labels)
        overlay_path = args.output_dir / f"{image_path.stem}_sa2va_overlay.png"
        overlay.save(overlay_path)

        save_mask_images(args.output_dir, image_path.stem, masks, image.size)

        txt_path = args.output_dir / f"{image_path.stem}_sa2va_prediction.txt"
        txt_path.write_text(prediction)

        save_prediction_json(args.output_dir, image_path.stem, prediction, masks, labels)

        if masks:
            npy_path = args.output_dir / f"{image_path.stem}_sa2va_masks.npy"
            np.save(npy_path, np.array([mask_to_bool(m).astype(np.uint8) for m in masks], dtype=object), allow_pickle=True)

        print(f"  saved overlay: {overlay_path}")
        print(f"  saved prediction: {txt_path}")
        print(f"  saved result json: {args.output_dir / f'{image_path.stem}_sa2va_result.json'}")
        print(f"  masks returned: {len(masks)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())