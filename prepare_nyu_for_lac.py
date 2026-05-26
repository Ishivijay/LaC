#!/usr/bin/env python3
"""
Prepare NYU Depth V2 Test Set for LAC Pipeline
================================================

Converts the NYU Depth V2 test set into the folder structure expected by the
original lac_pipeline.py, so you can run your existing pipeline directly:

    python lac_pipeline.py --config nyu_lac_config.yaml

What this script does:
1. Creates an lms_kamal_nyu_testset/ folder with the expected subfolder structure
2. Symlinks RGB images from nyu_testset/rgb/ → sharpen_rgb/PNG/
3. Converts raw depth (uint16 mm) to colored depth maps → marigold_zero_shot/depth_colored/
4. Generates a config YAML (nyu_lac_config.yaml) pointing to the new data

Usage:
    python prepare_nyu_for_lac.py
    python prepare_nyu_for_lac.py --num_images 20  # Prepare only 20 images (testing)
    python prepare_nyu_for_lac.py --force           # Overwrite existing symlinks/copies

After running this:
    cd /home/hpc/iwnt/iwnt164h/lac_free_ground
    python lac_pipeline.py --config nyu_lac_config.yaml --strategy two_vlm --input_mode rgb_depth_separate
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def nyu_depth_to_colored(depth_raw: np.ndarray) -> np.ndarray:
    """Convert raw NYU depth (uint16 mm) to a colored depth map (TURBO colormap).

    Args:
        depth_raw: Raw depth array (uint16, values in mm).

    Returns:
        BGR colored depth map (uint8, same HxW).
    """
    if depth_raw.dtype == np.uint16:
        depth_meters = depth_raw.astype(np.float32) / 1000.0
    else:
        depth_meters = depth_raw.astype(np.float32)

    # Normalize to 0-255
    valid = depth_meters[depth_meters > 0]
    if len(valid) > 0:
        vmin, vmax = valid.min(), valid.max()
        if vmax > vmin:
            normalized = ((depth_meters - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth_meters, dtype=np.uint8)
        normalized[depth_meters == 0] = 0
    else:
        normalized = np.zeros_like(depth_meters, dtype=np.uint8)

    # Apply TURBO colormap
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    # Black out invalid pixels
    colored[depth_meters == 0] = [0, 0, 0]
    return colored


def prepare_nyu_dataset(
    nyu_dir: Path,
    output_base: Path,
    num_images: int = None,
    force: bool = False,
):
    """Prepare NYU test set in LAC pipeline format.

    Creates:
        {output_base}/lms_kamal_nyu_testset/
            sharpen_rgb/PNG/
                image000.png ... image653.png  (symlinks to NYU RGB)
            marigold_zero_shot/depth_colored/
                image000_depth_colored.png ...  (converted from NYU depth)
    """
    rgb_dir = nyu_dir / "rgb"
    depth_dir = nyu_dir / "depth"

    if not rgb_dir.exists() or not depth_dir.exists():
        logger.error(f"NYU directories not found: {rgb_dir}, {depth_dir}")
        return False

    # Create output structure
    nyu_folder = output_base / "lms_kamal_nyu_testset"
    out_rgb = nyu_folder / "sharpen_rgb" / "PNG"
    out_depth = nyu_folder / "marigold_zero_shot" / "depth_colored"

    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)

    # Discover NYU images
    rgb_files = sorted(rgb_dir.glob("nyu_*.png"))
    if not rgb_files:
        rgb_files = sorted(rgb_dir.glob("*.png"))

    if num_images:
        rgb_files = rgb_files[:num_images]

    logger.info(f"Preparing {len(rgb_files)} NYU images")
    logger.info(f"  RGB → {out_rgb}")
    logger.info(f"  Depth → {out_depth}")

    created_rgb = 0
    created_depth = 0

    for i, rgb_path in enumerate(rgb_files):
        nyu_id = rgb_path.stem  # e.g., "nyu_0000"
        # LAC pipeline expects imageXXX naming
        image_id = f"image{i:03d}"  # e.g., "image000"

        # ── Symlink RGB ─────────────────────────────────────────────
        rgb_link = out_rgb / f"{image_id}.png"
        if force and rgb_link.exists():
            rgb_link.unlink()
        if not rgb_link.exists():
            rgb_link.symlink_to(rgb_path.resolve())
        created_rgb += 1

        # ── Convert depth ───────────────────────────────────────────
        depth_src = depth_dir / f"{nyu_id}.png"
        depth_dst = out_depth / f"{image_id}_depth_colored.png"

        if depth_src.exists() and (force or not depth_dst.exists()):
            # Read raw depth
            depth_raw = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                depth_pil = Image.open(depth_src)
                depth_raw = np.array(depth_pil)

            if depth_raw is not None:
                colored = nyu_depth_to_colored(depth_raw)
                cv2.imwrite(str(depth_dst), colored)
                created_depth += 1
            else:
                logger.warning(f"Failed to read depth: {depth_src}")

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(rgb_files)}")

    logger.info(f"Created {created_rgb} RGB symlinks, {created_depth} colored depth maps")
    logger.info(f"NYU folder: {nyu_folder}")

    return True


def generate_config(
    output_base: Path,
    config_path: Path,
):
    """Generate a nyu_lac_config.yaml for the original lac_pipeline.py."""
    nyu_folder = output_base / "lms_kamal_nyu_testset"

    config = f"""# Auto-generated config for running LAC pipeline on NYU Depth V2 test set
# Generated by prepare_nyu_for_lac.py
#
# Usage:
#   python lac_pipeline.py --config nyu_lac_config.yaml --strategy two_vlm --input_mode rgb_depth_separate
#   python lac_pipeline.py --config nyu_lac_config.yaml --strategy zero_shot --input_mode rgb_only --quick_test

# Data paths — NYU test set prepared in LAC format
data:
  base_dir: "{output_base}"
  folders:
    - "lms_kamal_nyu_testset"
  rgb_subfolder: "sharpen_rgb/PNG"
  depth_subfolder: "marigold_zero_shot/depth_colored"

# Model configuration
model:
  reasoner:
    name: "Qwen2.5-VL-7B-Instruct"
    hf_model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
    quantization: "4bit"
    dtype: "auto"
    max_new_tokens: 1024
    temperature: 0.1

  evaluator:
    name: "Qwen2.5-VL-7B-Instruct"
    hf_model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
    quantization: "4bit"
    dtype: "auto"
    max_new_tokens: 512
    temperature: 0.1

  # SAM3 segmentation
  segmentation:
    method: "sam3"
    sam3_model_id: "facebook/sam3"
    sam3_mask_threshold: 0.5
    device: "cuda"

# Pipeline settings
pipeline:
  strategy: "two_vlm"
  input_mode: "rgb_depth_separate"
  depth_flatness_threshold: 80
  few_shot_dir: null
  num_examples: 3

# Output settings
output:
  dir: "{output_base}/nyu_lac_results"
  save_individual_json: true
  save_visualizations: true

# Evaluation settings (used by evaluate_lac_nyu.py)
evaluation:
  gt_mask_dir: "{output_base}/nyu_testset_gt/floor_masks"
  metrics: ["iou", "precision", "recall", "f1", "dice"]
  min_floor_coverage: 1.0
"""

    with open(config_path, "w") as f:
        f.write(config)

    logger.info(f"Config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare NYU Depth V2 test set for the original LAC pipeline",
    )
    parser.add_argument(
        "--nyu_dir", type=str,
        default="/home/woody/iwnt/iwnt164h/nyu_testset",
        help="Path to NYU test set directory (contains rgb/ and depth/)",
    )
    parser.add_argument(
        "--output_base", type=str,
        default=None,
        help="Base output directory (default: $WORK)",
    )
    parser.add_argument(
        "--num_images", type=int, default=None,
        help="Prepare only N images (for testing)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files/symlinks",
    )
    args = parser.parse_args()

    work_dir = args.output_base or os.environ.get("WORK", "/home/woody/iwnt/iwnt164h")
    output_base = Path(work_dir)
    nyu_dir = Path(args.nyu_dir)

    logger.info("=" * 60)
    logger.info("Preparing NYU Depth V2 for LAC Pipeline")
    logger.info("=" * 60)
    logger.info(f"NYU dir:      {nyu_dir}")
    logger.info(f"Output base:  {output_base}")

    # Step 1: Prepare dataset
    success = prepare_nyu_dataset(nyu_dir, output_base, args.num_images, args.force)

    if not success:
        logger.error("Failed to prepare dataset")
        sys.exit(1)

    # Step 2: Generate config
    config_path = Path(__file__).parent / "nyu_lac_config.yaml"
    generate_config(output_base, config_path)

    logger.info("=" * 60)
    logger.info("Preparation complete! Now run:")
    logger.info(f"  python lac_pipeline.py --config nyu_lac_config.yaml --strategy two_vlm --input_mode rgb_depth_separate")
    logger.info(f"  # Or quick test:")
    logger.info(f"  python lac_pipeline.py --config nyu_lac_config.yaml --quick_test")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
