#!/usr/bin/env python3
"""
Generate consolidated visualization images for already-completed pipeline runs.

Reads existing masks + RGB images and creates side-by-side visualizations:
  rgb_only:         [Original RGB] | [Segmentation Overlay]
  rgb_depth_separate: [Original RGB] | [Depth Map] | [Segmentation Overlay]

Usage:
    # Generate for all runs:
    python3 generate_consolidated.py

    # Specific run:
    python3 generate_consolidated.py --run few_shot/Qwen/rgb_only_sam3

    # Dry run:
    python3 generate_consolidated.py --dry-run
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORK_DIR = Path(os.environ.get("WORK", "/home/woody/iwnt/iwnt164h"))
RESULTS_DIR = WORK_DIR / "free_ground_results"
DATA_DIR = WORK_DIR / "mlp_dataset" / "prospthesisproject-Data" / "Code" / "Data"


def find_runs(results_dir: Path) -> List[Path]:
    """Find all pipeline run directories."""
    runs = []
    for strategy_dir in sorted(results_dir.iterdir()):
        if not strategy_dir.is_dir() or strategy_dir.name in ("slurm_logs", "evaluation", "comparison"):
            continue
        for model_dir in sorted(strategy_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for mode_dir in sorted(model_dir.iterdir()):
                if not mode_dir.is_dir() or not mode_dir.name.endswith("_sam3"):
                    continue
                runs.append(mode_dir)
    return runs


def get_input_mode(dir_name: str) -> str:
    """Extract input_mode from directory name like 'rgb_only_sam3'."""
    if dir_name.startswith("rgb_depth_separate"):
        return "rgb_depth_separate"
    return "rgb_only"


def generate_for_run(run_dir: Path, dry_run: bool) -> dict:
    """Generate consolidated images for a single run."""
    input_mode = get_input_mode(run_dir.name)
    folder_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name != "masks"])

    generated = 0
    skipped = 0
    errors = 0

    for folder_dir in folder_dirs:
        folder_name = folder_dir.name
        mask_dir = folder_dir / "masks"

        if not mask_dir.exists():
            continue

        # Find all image IDs that have masks
        mask_files = sorted(mask_dir.glob("*_mask_*_*.png"))
        mask_files = [f for f in mask_files if "overlay" not in f.name and "consolidated" not in f.name]

        # Get unique image IDs
        image_ids = sorted(set(
            f.name.split("_mask_")[0] for f in mask_files
        ))

        for image_id in image_ids:
            consolidated_path = folder_dir / f"{image_id}_consolidated.png"

            if dry_run and not consolidated_path.exists():
                generated += 1
                continue

            # Load RGB image
            rgb_path = None
            for subdir in ["sharpen_rgb/PNG", "rgb/PNG"]:
                candidate = DATA_DIR / folder_name / subdir / f"{image_id}.png"
                if candidate.exists():
                    rgb_path = candidate
                    break

            if rgb_path is None:
                errors += 1
                continue

            rgb_image = Image.open(rgb_path).convert("RGB")
            rgb_array = np.array(rgb_image)
            h, w = rgb_array.shape[:2]

            # Load segmentation overlay
            overlay_path = mask_dir / f"{image_id}_segmentation_overlay.png"
            if overlay_path.exists():
                overlay = np.array(Image.open(overlay_path).convert("RGB"))
                if overlay.shape[:2] != (h, w):
                    overlay = np.array(Image.open(overlay_path).convert("RGB").resize((w, h), Image.NEAREST))
            else:
                # Reconstruct overlay from individual masks
                overlay = rgb_array.copy()
                colors = [(0, 255, 0), (0, 128, 255), (255, 0, 255),
                          (255, 255, 0), (0, 255, 255), (255, 128, 0)]
                individual_masks = sorted(mask_dir.glob(f"{image_id}_mask_*_*.png"))
                for j, mf in enumerate(individual_masks):
                    if "overlay" in mf.name:
                        continue
                    mask = np.array(Image.open(mf).convert("L"))
                    if mask.shape[:2] != (h, w):
                        mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))
                    binary = mask > 127
                    color = colors[j % len(colors)]
                    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                    color_mask[binary] = color
                    overlay[binary] = (overlay[binary] * 0.5 + color_mask[binary] * 0.5).astype(np.uint8)

            # Build panels
            label_h = 30
            gap = 4
            panels = [("Original RGB", rgb_array)]

            if input_mode != "rgb_only":
                depth_path = None
                for subdir in [
                    "marigold_zero_shot/depth_colored",
                    "colored_depth/PNG",
                    "sharpen_depth/PNG",
                    "depth/PNG",
                ]:
                    for naming in [
                        f"{image_id}_depth_colored.png",
                        f"{image_id}.png",
                    ]:
                        candidate = DATA_DIR / folder_name / subdir / naming
                        if candidate.exists():
                            depth_path = candidate
                            break
                    if depth_path:
                        break
                if depth_path:
                    depth_rgb = np.array(Image.open(depth_path).convert("RGB").resize((w, h), Image.NEAREST))
                    panels.append(("Depth Map", depth_rgb))

            panels.append(("Segmentation", overlay))

            # Create consolidated image
            total_w = sum(panel.shape[1] for _, panel in panels) + gap * (len(panels) - 1)
            total_h = h + label_h
            consolidated = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

            x_offset = 0
            for label, panel in panels:
                consolidated[0:label_h, x_offset:x_offset + panel.shape[1]] = (40, 40, 40)
                label_img = Image.fromarray(consolidated)
                draw = ImageDraw.Draw(label_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except (OSError, IOError):
                    font = ImageFont.load_default()
                text_x = x_offset + (panel.shape[1] - draw.textlength(label, font=font)) // 2
                draw.text((text_x, 5), label, fill=(255, 255, 255), font=font)
                consolidated = np.array(label_img)
                consolidated[label_h:label_h + h, x_offset:x_offset + panel.shape[1]] = panel
                x_offset += panel.shape[1] + gap

            Image.fromarray(consolidated).save(consolidated_path)
            generated += 1

    return {"run": run_dir.relative_to(RESULTS_DIR), "generated": generated, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated visualizations for completed runs")
    parser.add_argument("--run", type=str, default=None,
                        help="Specific run path (e.g., few_shot/Qwen/rgb_only_sam3)")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no files written")
    args = parser.parse_args()

    print("=" * 60)
    print("Generate Consolidated Visualizations")
    print("=" * 60)

    if args.run:
        run_dirs = [RESULTS_DIR / args.run]
    else:
        run_dirs = find_runs(RESULTS_DIR)

    print(f"Runs found: {len(run_dirs)}")
    for rd in run_dirs:
        print(f"  {rd.relative_to(RESULTS_DIR)}")

    total_generated = 0
    total_skipped = 0
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"\n⚠ Not found: {run_dir}")
            continue
        print(f"\nProcessing: {run_dir.relative_to(RESULTS_DIR)}")
        result = generate_for_run(run_dir, args.dry_run)
        print(f"  Generated: {result['generated']}, Skipped (existing): {result['skipped']}, Errors: {result['errors']}")
        total_generated += result["generated"]
        total_skipped += result["skipped"]

    print(f"\n{'=' * 60}")
    print(f"Total generated: {total_generated}")
    print(f"Total skipped (already exist): {total_skipped}")
    if args.dry_run:
        print("DRY-RUN — re-run without --dry-run to generate files.")


if __name__ == "__main__":
    main()
