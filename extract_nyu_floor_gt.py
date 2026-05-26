#!/usr/bin/env python3
"""
Extract Floor Ground Truth Masks from NYU Depth V2 Labeled Dataset
===================================================================

Extracts binary floor segmentation masks from the NYU Depth V2 labeled MAT file.
Floor class ID = 11 (1-indexed) in the NYU label taxonomy.

The script reads the test split file to identify which of the 1449 labeled images
belong to the test set, then extracts and saves binary floor masks.

Usage:
    python extract_nyu_floor_gt.py
    python extract_nyu_floor_gt.py --mat_file /path/to/nyu_depth_v2_labeled.mat
    python extract_nyu_floor_gt.py --num_samples 10  # Quick test with 10 samples

Output:
    {output_dir}/floor_masks/nyu_XXXX_floor.png  — Binary floor masks (0/255)
    {output_dir}/floor_masks/nyu_XXXX_floor_vis.png  — Visualizations (optional)
    {output_dir}/floor_stats.json  — Statistics about floor coverage
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# NYU Depth V2 class IDs relevant for free ground detection
# From the 894-class label taxonomy:
#   11 = floor
#   4  = ceiling (useful for exclusion)
FLOOR_CLASS_ID = 11

# Additional classes that could be considered "walkable" surfaces
WALKABLE_CLASSES = {
    11: "floor",
    # 3: "cabinet",  # Not walkable, but low surface
}

# Standard NYU 40-class mapping for reference
NYU40_FLOOR = 4  # floor in the 40-class mapping


def extract_frame_indices(split_file: Path) -> list:
    """Extract 0-based frame indices from the NYU test split file.
    
    The split file has lines like:
        test/kitchen_0004/rgb_0001.png test/kitchen_0004/depth_0001.png ...
    
    The frame number in the filename (e.g., 0001) is 1-based and corresponds
    to the index in the MAT file's arrays (1-based → 0-based for Python).
    """
    lines = split_file.read_text().splitlines()
    indices = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        filename = parts[0]  # e.g., 'test/kitchen_0004/rgb_0001.png'
        basename = filename.split("/")[-1]  # e.g., 'rgb_0001.png'
        number_part = basename.split("_")[-1].split(".")[0]  # e.g., '0001'
        frame_idx = int(number_part) - 1  # Convert to 0-based
        indices.append(frame_idx)
    return indices


def extract_floor_masks(
    mat_file: Path,
    split_file: Path,
    output_dir: Path,
    num_samples: int = None,
    save_visualizations: bool = True,
):
    """Extract binary floor masks from the NYU labeled dataset.
    
    Args:
        mat_file: Path to nyu_depth_v2_labeled.mat
        split_file: Path to the test split file
        output_dir: Output directory for masks
        num_samples: Limit to N samples (for testing)
        save_visualizations: Whether to save RGB+mask overlay visualizations
    """
    logger.info(f"Loading MAT file: {mat_file}")
    
    # Get test set frame indices
    test_indices = extract_frame_indices(split_file)
    logger.info(f"Found {len(test_indices)} test images in split file")
    
    if num_samples:
        test_indices = test_indices[:num_samples]
        logger.info(f"Limited to {num_samples} samples")
    
    # Create output directories
    mask_dir = output_dir / "floor_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    if save_visualizations:
        vis_dir = output_dir / "floor_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the MAT file and extract masks
    stats = {
        "total_images": len(test_indices),
        "images_with_floor": 0,
        "floor_coverage": [],
        "scene_types": {},
    }
    
    with h5py.File(mat_file, "r") as f:
        labels = f["labels"]    # (1449, 640, 480) uint16
        images = f["images"]    # (1449, 3, 640, 480) uint8
        scenes = f["scenes"]    # (1, 1449) object refs
        
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Images shape: {images.shape}")
        
        for new_idx, old_idx in enumerate(test_indices):
            # Extract label map
            label_map = labels[old_idx]  # (640, 480) uint16
            label_map = np.array(label_map).T  # Transpose to (480, 640)
            
            # Create binary floor mask
            floor_mask = (label_map == FLOOR_CLASS_ID).astype(np.uint8) * 255
            
            # Calculate floor coverage
            floor_pixels = (floor_mask > 0).sum()
            total_pixels = floor_mask.size
            coverage_pct = floor_pixels / total_pixels * 100
            
            # Get scene type
            try:
                scene_ref = scenes[0, old_idx]
                scene_name = "".join(chr(c[0]) for c in f[scene_ref])
            except Exception:
                scene_name = "unknown"
            
            # Track statistics
            if floor_pixels > 0:
                stats["images_with_floor"] += 1
                stats["floor_coverage"].append(round(coverage_pct, 2))
            
            if scene_name not in stats["scene_types"]:
                stats["scene_types"][scene_name] = {"total": 0, "with_floor": 0}
            stats["scene_types"][scene_name]["total"] += 1
            if floor_pixels > 0:
                stats["scene_types"][scene_name]["with_floor"] += 1
            
            # Save binary mask
            mask_path = mask_dir / f"nyu_{new_idx:04d}_floor.png"
            cv2.imwrite(str(mask_path), floor_mask)
            
            # Save visualization
            if save_visualizations:
                # Extract RGB image
                img = images[old_idx]  # (3, 640, 480) uint8
                img = np.transpose(img, (1, 2, 0))  # (640, 480, 3)
                img = np.ascontiguousarray(img)  # Ensure contiguous memory
                img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR
                
                # Create overlay
                overlay = img_rgb.copy()
                green_mask = np.zeros_like(img_rgb)
                green_mask[floor_mask > 0] = [0, 255, 0]
                overlay = cv2.addWeighted(img_rgb, 0.7, green_mask, 0.3, 0)
                
                # Add text
                cv2.putText(
                    overlay, f"Floor: {coverage_pct:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.putText(
                    overlay, f"Scene: {scene_name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                )
                
                vis_path = vis_dir / f"nyu_{new_idx:04d}_floor_vis.png"
                cv2.imwrite(str(vis_path), overlay)
            
            if (new_idx + 1) % 50 == 0 or new_idx == 0:
                logger.info(
                    f"  Processed {new_idx + 1}/{len(test_indices)} "
                    f"(floor: {coverage_pct:.1f}%, scene: {scene_name})"
                )
    
    # Compute summary statistics
    stats["floor_coverage_mean"] = (
        round(np.mean(stats["floor_coverage"]), 2)
        if stats["floor_coverage"] else 0
    )
    stats["floor_coverage_median"] = (
        round(np.median(stats["floor_coverage"]), 2)
        if stats["floor_coverage"] else 0
    )
    stats["floor_coverage_min"] = (
        round(np.min(stats["floor_coverage"]), 2)
        if stats["floor_coverage"] else 0
    )
    stats["floor_coverage_max"] = (
        round(np.max(stats["floor_coverage"]), 2)
        if stats["floor_coverage"] else 0
    )
    
    # Save statistics
    stats_path = output_dir / "floor_stats.json"
    with open(stats_path, "w") as fp:
        json.dump(stats, fp, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info(f"Floor GT Extraction Complete")
    logger.info(f"  Total images:  {stats['total_images']}")
    logger.info(f"  With floor:    {stats['images_with_floor']}")
    logger.info(f"  Coverage mean: {stats['floor_coverage_mean']}%")
    logger.info(f"  Coverage med:  {stats['floor_coverage_median']}%")
    logger.info(f"  Masks saved to: {mask_dir}")
    logger.info(f"  Stats saved to: {stats_path}")
    logger.info("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract floor ground truth masks from NYU Depth V2"
    )
    parser.add_argument(
        "--mat_file", type=str,
        default="/home/woody/iwnt/iwnt164h/nyuv2/nyu_depth_v2_labeled.mat",
        help="Path to nyu_depth_v2_labeled.mat",
    )
    parser.add_argument(
        "--split_file", type=str,
        default="/home/hpc/iwnt/iwnt164h/Marigold/data_split/nyu_depth/labeled/filename_list_test.txt",
        help="Path to test split file",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=None,
        help="Output directory (default: $WORK/nyu_testset_gt)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Limit to N samples (for testing)",
    )
    parser.add_argument(
        "--no_vis", action="store_true",
        help="Skip visualization generation",
    )
    args = parser.parse_args()
    
    work_dir = os.environ.get("WORK", "/home/woody/iwnt/iwnt164h")
    output_dir = Path(args.output_dir) if args.output_dir else Path(work_dir) / "nyu_testset_gt"
    
    extract_floor_masks(
        mat_file=Path(args.mat_file),
        split_file=Path(args.split_file),
        output_dir=output_dir,
        num_samples=args.num_samples,
        save_visualizations=not args.no_vis,
    )


if __name__ == "__main__":
    main()
