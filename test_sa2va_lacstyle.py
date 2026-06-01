#!/usr/bin/env python3
"""Run Sa2VA on LaC-style RGB/depth pairs and save LaC-like outputs.

This script mirrors the LaC pipeline's path handling and output layout:
    $WORK/free_ground_results/{strategy}/{model_tag}/{input_mode}_sa2va/{folder}/
        - masks/
        - {image_id}_lac_analysis.json
        - {image_id}_consolidated.png

Input handling:
- `rgb_only`: use only the RGB image.
- `rgb_depth_separate`: pass RGB and depth as separate frames (two-frame input)
    so the model can use depth explicitly when available.

Note: earlier iterations used a side-by-side RGB|Depth composite image for
`rgb_depth_separate`. The current implementation sends the two frames
separately so the model can better utilize depth information.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "free_ground_pipeline"))
from pipeline import discover_folders, discover_image_pairs, filter_image_pairs, load_image_for_vlm

DEFAULT_BASE_DIR = "/home/woody/iwnt/iwnt164h/mlp_dataset/prospthesisproject-Data/Code/Data"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("WORK", "/home/woody/iwnt/iwnt164h")) / "free_ground_results"
DEFAULT_MODEL_ID = "ByteDance/Sa2VA-Qwen3-VL-4B"
DEFAULT_MODEL_TAG = "Sa2VA-Qwen3-VL-4B"
DEFAULT_HF_CACHE = Path(os.environ.get("HF_HOME", str(Path(os.environ.get("WORK", "/home/woody/iwnt/iwnt164h")) / ".cache" / "huggingface")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sa2VA on LaC-style RGB/depth folders.")
    parser.add_argument("--base-dir", type=Path, default=Path(DEFAULT_BASE_DIR), help="Base data directory")
    parser.add_argument("--folders", nargs="*", default=None, help="Optional list of lms_kamal_* folders")
    parser.add_argument("--rgb-subfolder", type=str, default="sharpen_rgb/PNG", help="RGB subfolder name")
    parser.add_argument(
        "--depth-subfolder",
        type=str,
        default="marigold_zero_shot/depth_colored",
        help="Colored depth subfolder name",
    )
    parser.add_argument(
        "--depth-bw-subfolder",
        type=str,
        default="marigold_zero_shot/depth_bw",
        help="Black-and-white depth subfolder name",
    )
    parser.add_argument(
        "--use-colored-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use colored depth maps instead of BW depth maps",
    )
    parser.add_argument(
        "--input-mode",
        choices=["rgb_only", "rgb_depth_separate"],
        default="rgb_depth_separate",
        help="How to present RGB+depth to Sa2VA",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    parser.add_argument("--strategy", type=str, default="sa2va", help="Top-level strategy folder")
    parser.add_argument("--model-tag", type=str, default=DEFAULT_MODEL_TAG, help="Output model tag")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="HF model ID")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional custom prompt. If empty, a mode-specific default is used.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Optional ground-truth annotation directory to restrict processed images",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=0,
        help="Maximum images per folder to process (0 = all)",
    )
    parser.add_argument(
        "--specific-images",
        nargs="*",
        default=None,
        help="Optional image indices like 116 124 405 (or image116, ...)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for the generated response",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=896,
        help="Resize the model input so the longest side does not exceed this value (0 = keep original)",
    )
    return parser.parse_args()


def parse_specific_images(values: Optional[Sequence[str]]) -> Optional[List[int]]:
    if not values:
        return None
    image_ids = []
    for value in values:
        token = value.strip()
        if token.startswith("image"):
            token = token.replace("image", "")
        image_ids.append(int(token))
    return image_ids


def build_config(args: argparse.Namespace) -> Dict:
    return {
        "data": {
            "base_dir": str(args.base_dir),
            "folders": args.folders,
            "rgb_subfolder": args.rgb_subfolder,
            "depth_subfolder": args.depth_subfolder,
            "depth_bw_subfolder": args.depth_bw_subfolder,
        },
        "pipeline": {
            "num_images_per_folder": args.num_images if args.num_images > 0 else None,
            "specific_images": parse_specific_images(args.specific_images),
            "use_colored_depth": args.use_colored_depth,
            "input_mode": args.input_mode,
        },
    }


def list_depth_folders(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob("lms_kamal_*") if p.is_dir()])


def _local_model_path(model_id: str) -> Optional[Path]:
    cache_dir = DEFAULT_HF_CACHE / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not cache_dir.exists():
        return None

    snapshots = sorted([p for p in cache_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        if (snapshot / "config.json").exists():
            return snapshot
    return snapshots[0] if snapshots else None


def load_model(model_id: str, device: str):
    from transformers import AutoModel, AutoProcessor
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    local_model_path = _local_model_path(model_id)
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


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w]", "_", name)[:30]


def _mask_to_bool(mask) -> np.ndarray:
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
    if bbox is None:
        return None
    width, height = size
    return {
        "x1": round(bbox["x1"] / width * 100, 2),
        "y1": round(bbox["y1"] / height * 100, 2),
        "x2": round(bbox["x2"] / width * 100, 2),
        "y2": round(bbox["y2"] / height * 100, 2),
    }


def _make_composite(rgb_image: Image.Image, depth_image: Image.Image) -> Image.Image:
    rgb_image = rgb_image.convert("RGB")
    depth_image = depth_image.convert("RGB")
    if depth_image.size != rgb_image.size:
        depth_image = depth_image.resize(rgb_image.size, Image.NEAREST)
    width, height = rgb_image.size
    composite = Image.new("RGB", (width * 2 + 10, height), (0, 0, 0))
    composite.paste(rgb_image, (0, 0))
    composite.paste(depth_image, (width + 10, 0))
    try:
        draw = ImageDraw.Draw(composite)
        draw.text((10, 10), "RGB Image", fill=(255, 255, 255))
        draw.text((width + 20, 10), "Depth Map", fill=(255, 255, 255))
    except Exception:
        pass
    return composite


def _make_overlay(rgb_image: Image.Image, depth_image: Image.Image, alpha: float = 0.4) -> Image.Image:
    rgb_image = rgb_image.convert("RGB")
    depth_image = depth_image.convert("RGB")
    if depth_image.size != rgb_image.size:
        depth_image = depth_image.resize(rgb_image.size, Image.NEAREST)
    rgb_arr = np.array(rgb_image).astype(np.float32)
    depth_arr = np.array(depth_image).astype(np.float32)
    blended = np.clip(rgb_arr * (1.0 - alpha) + depth_arr * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def _prepare_model_inputs(
    rgb_image: Image.Image,
    depth_image: Optional[Image.Image],
    input_mode: str,
) -> Tuple[Optional[Image.Image], Optional[List[Image.Image]]]:
    if input_mode == "rgb_only" or depth_image is None:
        return rgb_image.convert("RGB"), None

    if input_mode == "rgb_depth_separate":
        rgb_frame = rgb_image.convert("RGB")
        depth_frame = depth_image.convert("RGB")
        if depth_frame.size != rgb_frame.size:
            depth_frame = depth_frame.resize(rgb_frame.size, Image.NEAREST)
        return None, [rgb_frame, depth_frame]

    return _make_overlay(rgb_image, depth_image), None


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


def _default_prompt(input_mode: str) -> str:
    """Load the unified walkable area prompts from prompt files."""
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


def _project_mask_to_rgb(mask: np.ndarray, rgb_size: Tuple[int, int]) -> np.ndarray:
    """Map a model mask back into the RGB image frame.

    When rgb_depth_separate is passed as a two-frame video input, the returned
    mask already corresponds to the RGB frame.
    """
    rgb_width, rgb_height = rgb_size
    return _resize_mask(mask, (rgb_width, rgb_height))


def _run_model(
    model,
    processor,
    image: Optional[Image.Image],
    prompt: str,
    device: str,
    video: Optional[Sequence[Image.Image]] = None,
):
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


def _draw_bbox_panel(rgb_array: np.ndarray, areas: List[Dict]) -> np.ndarray:
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


def _build_consolidated_image(
    rgb_image: Image.Image,
    depth_image: Optional[Image.Image],
    areas: List[Dict],
    overlay: np.ndarray,
    input_mode: str,
) -> Image.Image:
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
        panels.append(("Sa2VA BBox", _draw_bbox_panel(rgb_arr, areas)))

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


def _save_outputs(
    output_dir: Path,
    image_id: str,
    rgb_image: Image.Image,
    depth_image: Optional[Image.Image],
    masks: Sequence[np.ndarray],
    input_mode: str,
    prediction: str,
    prompt: str,
    folder_name: str,
    strategy: str,
    model_tag: str,
    model_id: str,
    labels: Sequence[str],
) -> Dict:
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
        mask_bool = _project_mask_to_rgb(mask_bool, (w, h))
        bbox_px = _mask_bbox(mask_bool)
        if bbox_px is None:
            continue

        name = labels[idx] if idx < len(labels) and labels[idx] else f"sa2va_area_{idx}"
        safe_name = _safe_name(name)
        base = f"{image_id}_mask_{len(saved_masks)}_{safe_name}"

        mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
        mask_img.save(mask_dir / f"{base}.png")
        np.save(mask_dir / f"{base}.npy", mask_bool)

        color = colors[len(saved_masks) % len(colors)]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[mask_bool] = color
        overlay[mask_bool] = (overlay[mask_bool] * 0.5 + color_mask[mask_bool] * 0.5).astype(np.uint8)

        areas.append(
            {
                "name": name,
                "type": "sa2va_mask",
                "bbox": _bbox_percent(bbox_px, (w, h)),
                "bbox_pixels": bbox_px,
                "area_pixels": int(mask_bool.sum()),
            }
        )
        saved_masks.append(mask_bool)

    Image.fromarray(overlay).save(mask_dir / f"{image_id}_segmentation_overlay.png")
    consolidated = _build_consolidated_image(rgb_image, depth_image, areas, overlay, input_mode)
    consolidated.save(folder_dir / f"{image_id}_consolidated.png")

    result = {
        "folder": folder_name,
        "image_id": image_id,
        "strategy": strategy,
        "model_tag": model_tag,
        "model_id": model_id,
        "input_mode": input_mode,
        "segmentation_method": "sa2va",
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "prediction": prediction,
        "num_masks": len(saved_masks),
        "vlm_areas": areas,
    }

    with open(folder_dir / f"{image_id}_lac_analysis.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    if saved_masks:
        np.save(folder_dir / f"{image_id}_sa2va_masks.npy", np.stack(saved_masks, axis=0))

    return result


def main() -> int:
    args = parse_args()
    config = build_config(args)
    # If a GT directory is provided, read mapping and restrict folders/images
    if getattr(args, "gt_dir", None):
        try:
            from pipeline import read_gt_directory

            gt_mapping = read_gt_directory(args.gt_dir)
            if gt_mapping:
                config["pipeline"]["gt_folder_images"] = gt_mapping
                config["data"]["folders"] = list(gt_mapping.keys())
        except Exception:
            print(f"Warning: failed to read gt_dir={args.gt_dir}; continuing without GT filter")
    output_root = args.output_root / args.strategy / args.model_tag / f"{args.input_mode}_sa2va"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id, args.device)
    print("Model loaded.")

    folders = discover_folders(config)
    if not folders:
        print("No valid folders found.")
        return 1

    for folder in folders:
        pairs = discover_image_pairs(folder, config)
        pairs = filter_image_pairs(pairs, config, folder.name)
        if not pairs:
            print(f"Skipping {folder.name}: no paired RGB/depth images")
            continue

        print(f"Processing folder {folder.name} with {len(pairs)} image pairs")
        for rgb_path, depth_path, image_id in pairs:
            rgb_image = load_image_for_vlm(rgb_path)
            depth_image = None
            if args.input_mode != "rgb_only":
                depth_image = load_image_for_vlm(depth_path)

            model_image, model_video = _prepare_model_inputs(rgb_image, depth_image, args.input_mode)
            if model_image is not None:
                model_image = _resize_max_side(model_image, args.image_max_side)
            if model_video is not None:
                model_video = [_resize_max_side(frame, args.image_max_side) for frame in model_video]
            prompt = args.prompt.strip() if args.prompt.strip() else _default_prompt(args.input_mode)

            print(f"  {folder.name}/{image_id}")
            prediction, masks, labels = _run_model(model, processor, model_image, prompt, args.device, video=model_video)
            result = _save_outputs(
                output_root,
                image_id,
                rgb_image,
                depth_image,
                masks,
                args.input_mode,
                prediction,
                prompt,
                folder.name,
                args.strategy,
                args.model_tag,
                args.model_id,
                labels,
            )
            print(f"    saved {result['num_masks']} masks -> {output_root / folder.name}")

    print(f"Done. Results saved under: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
