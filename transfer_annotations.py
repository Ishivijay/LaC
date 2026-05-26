#!/usr/bin/env python3
"""
Transfer GT annotations using SAM3 with visual exemplars.

For each "improper" image (failed SAM3 annotation), finds the closest "proper" image
(by temporal proximity) and uses its mask as a visual exemplar for SAM3 to segment
the same region in the target image.

SAM3 supports:
  - segmentation_maps: visual exemplar mask from a reference image
  - input_boxes: bounding box prompt derived from the exemplar mask
  - text: text prompt (e.g., "floor")

Usage:
    # Dry run (preview only):
    python3 transfer_annotations.py --dry-run

    # Transfer all:
    python3 transfer_annotations.py

    # With distance threshold:
    python3 transfer_annotations.py --max_distance 10

    # Specific folders only:
    python3 transfer_annotations.py --folders lms_kamal_LA_downstairs_Nopeople_1
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

WORK_DIR = Path(os.environ.get("WORK", "/home/woody/iwnt/iwnt164h"))
PROPER_DIR = WORK_DIR / "mlp_ground_truth" / "proper_annotations"
IMPROPER_DIR = WORK_DIR / "mlp_ground_truth" / "improper_annotations"
DEFAULT_OUTPUT_DIR = WORK_DIR / "mlp_ground_truth" / "transferred_annotations"
DATA_DIR = WORK_DIR / "mlp_dataset" / "prospthesisproject-Data" / "Code" / "Data"

# SAM3 model cache
_sam3_cache = {"model": None, "processor": None, "loaded": False}


# ──────────────────────────────────────────────────────────────────────────────
# SAM3 model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_sam3():
    """Load SAM3 model and processor."""
    if _sam3_cache["loaded"]:
        return _sam3_cache["model"], _sam3_cache["processor"]

    import torch
    from transformers import Sam3Model, Sam3Processor

    os.environ.setdefault("HF_HOME", str(WORK_DIR / ".cache" / "huggingface"))

    logger.info("Loading SAM3 model: facebook/sam3 (0.8B params)...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float16).to("cuda")
    logger.info("SAM3 model loaded successfully")

    _sam3_cache["model"] = model
    _sam3_cache["processor"] = processor
    _sam3_cache["loaded"] = True
    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation with visual exemplar
# ──────────────────────────────────────────────────────────────────────────────

def segment_with_exemplar(
    target_image: Image.Image,
    exemplar_image: Image.Image,
    exemplar_mask: Image.Image,
    text_prompt: str = "floor",
    mask_threshold: float = 0.3,
) -> Optional[np.ndarray]:
    """Segment target image using SAM3 with visual exemplar guidance.

    Tries three approaches in order:
      1. Visual exemplar (segmentation_maps from exemplar)
      2. Box prompt (bbox from exemplar mask)
      3. Text-only fallback
    """
    import torch
    import torch.nn.functional as F

    model, processor = load_sam3()
    device = next(model.parameters()).device
    w, h = target_image.size

    # ── Approach 1: Visual exemplar with segmentation_maps ────────────────
    try:
        # Resize exemplar mask to match exemplar image if needed
        ex_w, ex_h = exemplar_image.size
        if exemplar_mask.size != (ex_w, ex_h):
            exemplar_mask = exemplar_mask.resize((ex_w, ex_h), Image.NEAREST)

        inputs = processor(
            images=target_image,
            text=text_prompt,
            segmentation_maps=exemplar_mask.convert("L"),
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        scores = (
            outputs.pred_logits.sigmoid().squeeze(0)
            * outputs.presence_logits.sigmoid().squeeze(0)
        )
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()

        if best_score > 0.1:
            best_mask_logits = outputs.pred_masks[0, best_idx].unsqueeze(0).unsqueeze(0).float()
            best_mask_probs = torch.sigmoid(best_mask_logits)
            best_mask_resized = F.interpolate(
                best_mask_probs, size=(h, w), mode="bilinear", align_corners=False,
            )
            mask = best_mask_resized.squeeze().cpu().numpy() > mask_threshold

            if mask.sum() > 100:
                coverage = mask.sum() / (h * w) * 100
                logger.info(f"    Visual exemplar: score={best_score:.3f}, coverage={coverage:.1f}%")
                return mask
    except Exception as e:
        logger.debug(f"    Visual exemplar failed: {e}")

    # ── Approach 2: Box prompt from exemplar mask ─────────────────────────
    try:
        ex_mask_arr = np.array(exemplar_mask.convert("L"))
        if ex_mask_arr.max() > 0:
            ys, xs = np.where(ex_mask_arr > 127)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())

            # Scale box to target image size
            ex_w, ex_h = exemplar_image.size
            scale_x, scale_y = w / ex_w, h / ex_h
            box = [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]]

            inputs = processor(
                images=target_image,
                text=text_prompt,
                input_boxes=[box],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            scores = (
                outputs.pred_logits.sigmoid().squeeze(0)
                * outputs.presence_logits.sigmoid().squeeze(0)
            )
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()

            if best_score > 0.1:
                best_mask_logits = outputs.pred_masks[0, best_idx].unsqueeze(0).unsqueeze(0).float()
                best_mask_probs = torch.sigmoid(best_mask_logits)
                best_mask_resized = F.interpolate(
                    best_mask_probs, size=(h, w), mode="bilinear", align_corners=False,
                )
                mask = best_mask_resized.squeeze().cpu().numpy() > mask_threshold

                if mask.sum() > 100:
                    coverage = mask.sum() / (h * w) * 100
                    logger.info(f"    Box prompt: score={best_score:.3f}, coverage={coverage:.1f}%")
                    return mask
    except Exception as e:
        logger.debug(f"    Box prompt failed: {e}")

    # ── Approach 3: Text-only fallback ────────────────────────────────────
    try:
        inputs = processor(
            images=target_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

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
        mask = best_mask_resized.squeeze().cpu().numpy() > mask_threshold

        if mask.sum() > 100:
            coverage = mask.sum() / (h * w) * 100
            logger.info(f"    Text-only fallback: score={best_score:.3f}, coverage={coverage:.1f}%")
            return mask
    except Exception as e:
        logger.debug(f"    Text-only failed: {e}")

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Core transfer logic
# ──────────────────────────────────────────────────────────────────────────────

def get_image_ids(annot_dir: Path, folder: str) -> List[int]:
    """Get sorted list of image IDs that have masks in a folder."""
    folder_dir = annot_dir / folder
    if not folder_dir.exists():
        return []
    ids = []
    for f in folder_dir.glob("*_mask.png"):
        try:
            img_id = int(f.stem.replace("_mask", "").replace("image", ""))
            ids.append(img_id)
        except ValueError:
            pass
    return sorted(ids)


def find_nearest_proper(improper_id: int, proper_ids: List[int]) -> Optional[Tuple[int, int]]:
    """Find nearest proper image ID. Returns (id, distance)."""
    if not proper_ids:
        return None
    best_id, best_dist = None, float("inf")
    for pid in proper_ids:
        dist = abs(pid - improper_id)
        if dist < best_dist:
            best_dist = dist
            best_id = pid
    return (best_id, best_dist)


def find_rgb_path(folder: str, image_id: int) -> Optional[Path]:
    """Find RGB image path for a given folder/image_id."""
    for subdir in ["sharpen_rgb/PNG", "rgb/PNG"]:
        candidate = DATA_DIR / folder / subdir / f"image{image_id}.png"
        if candidate.exists():
            return candidate
    return None


def create_annotated_overlay(rgb_path: Path, mask: np.ndarray, output_path: Path):
    """Create annotated overlay (RGB with green mask)."""
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    if mask.shape[:2] != rgb.shape[:2]:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
        mask = np.array(mask_img) > 127

    overlay = rgb.copy()
    overlay_color = np.zeros_like(rgb)
    overlay_color[mask] = [0, 180, 160]  # teal/cyan to match proper annotations
    overlay[mask] = (overlay[mask] * 0.5 + overlay_color[mask] * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


def transfer_folder(
    folder: str,
    output_dir: Path,
    max_distance: Optional[int],
    dry_run: bool,
) -> Dict:
    """Transfer annotations for a single folder using SAM3 visual exemplars."""
    import shutil

    proper_ids = get_image_ids(PROPER_DIR, folder)
    improper_ids = get_image_ids(IMPROPER_DIR, folder)

    if not proper_ids:
        print(f"  ⚠ No proper annotations for {folder}")
        return {"folder": folder, "transferred": 0, "failed": len(improper_ids)}

    out_folder = output_dir / folder
    out_folder.mkdir(parents=True, exist_ok=True)

    if not improper_ids:
        print(f"  ✓ No improper images to fix")
        return {"folder": folder, "transferred": 0, "failed": 0}

    print(f"  Improper to fix: {len(improper_ids)}")

    transferred = 0
    failed = 0
    flagged = 0
    distances = []

    for i, imp_id in enumerate(improper_ids):
        result = find_nearest_proper(imp_id, proper_ids)
        if result is None:
            failed += 1
            continue

        near_id, dist = result
        distances.append(dist)

        # Check distance threshold
        if max_distance is not None and dist > max_distance:
            flagged += 1
            continue

        if dry_run:
            print(f"    [DRY-RUN] image{imp_id} ← exemplar image{near_id} (dist={dist})")
            transferred += 1
            continue

        # Load exemplar (proper) image + mask
        exemplar_rgb_path = find_rgb_path(folder, near_id)
        exemplar_mask_path = PROPER_DIR / folder / f"image{near_id}_mask.png"
        target_rgb_path = find_rgb_path(folder, imp_id)

        if not all(p and p.exists() for p in [exemplar_rgb_path, exemplar_mask_path, target_rgb_path]):
            logger.warning(f"    Missing files for image{imp_id}, skipping")
            failed += 1
            continue

        exemplar_image = Image.open(exemplar_rgb_path).convert("RGB")
        exemplar_mask = Image.open(exemplar_mask_path).convert("L")
        target_image = Image.open(target_rgb_path).convert("RGB")

        # Run SAM3 with visual exemplar
        t0 = time.time()
        mask = segment_with_exemplar(
            target_image, exemplar_image, exemplar_mask,
            text_prompt="floor walkable ground area",
        )
        t1 = time.time()

        if mask is not None:
            # Save mask
            mask_uint8 = (mask.astype(np.uint8)) * 255
            Image.fromarray(mask_uint8).save(out_folder / f"image{imp_id}_mask.png")

            # Save annotated overlay
            create_annotated_overlay(
                target_rgb_path, mask,
                out_folder / f"image{imp_id}_annotated.png",
            )
            transferred += 1
            coverage = mask.sum() / mask.size * 100
            logger.info(f"    ✓ image{imp_id} ← image{near_id} (dist={dist}, {coverage:.1f}%, {t1-t0:.1f}s)")
        else:
            failed += 1
            logger.warning(f"    ✗ image{imp_id} ← image{near_id} (dist={dist}): SAM3 failed")

        # Progress
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(improper_ids)} ({transferred} ok, {failed} failed)")

    stats = {
        "folder": folder,
        "transferred": transferred,
        "failed": failed,
        "flagged": flagged,
        "mean_distance": round(float(np.mean(distances)), 1) if distances else 0,
    }

    print(f"  Result: {transferred} transferred, {failed} failed, {flagged} flagged")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Transfer GT masks using SAM3 visual exemplars"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--max_distance", type=int, default=None,
                        help="Max frame distance for exemplar matching")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Only process these folders")
    args = parser.parse_args()

    print("=" * 60)
    print("SAM3 Visual Exemplar Annotation Transfer")
    print("=" * 60)
    print(f"Proper dir:   {PROPER_DIR}")
    print(f"Improper dir: {IMPROPER_DIR}")
    print(f"Output dir:   {args.output_dir}")
    if args.dry_run:
        print("Mode: DRY-RUN")
    if args.max_distance is not None:
        print(f"Max distance: {args.max_distance} frames")
    print()

    # Discover folders
    folders = args.folders or sorted(d.name for d in IMPROPER_DIR.iterdir() if d.is_dir())
    print(f"Folders: {len(folders)}")
    for f in folders:
        print(f"  {f}")

    # Load SAM3 (unless dry run)
    if not args.dry_run:
        load_sam3()

    all_stats = []
    for folder in folders:
        print(f"\n{'─' * 60}")
        print(f"Processing: {folder}")
        print(f"{'─' * 60}")
        stats = transfer_folder(folder, args.output_dir, args.max_distance, args.dry_run)
        all_stats.append(stats)

    # Summary
    total_transferred = sum(s["transferred"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    total_flagged = sum(s["flagged"] for s in all_stats)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Transferred:      {total_transferred}")
    print(f"Failed:           {total_failed}")
    print(f"Flagged:          {total_flagged}")
    print(f"Output:           {args.output_dir}")

    if args.dry_run:
        print("\nDRY-RUN — re-run without --dry-run to apply.")
    else:
        print(f"\nDone! {total_transferred} fixed annotations in {args.output_dir}")
        if total_failed:
            print(f"⚠ {total_failed} images failed SAM3 segmentation (may need manual annotation).")


if __name__ == "__main__":
    main()
