#!/usr/bin/env python3
"""
Evaluate LAC Pipeline Results on NYU Depth V2 Test Set
======================================================

Compares the original LAC pipeline's predicted free-ground masks against the
NYU Depth V2 floor ground truth masks.

The original pipeline stores results in:
    {results_dir}/lms_kamal_nyu_testset/
        masks/
            image000_mask_0_floor.png
            image000_mask_0_floor.npy
            image000_segmentation_overlay.png
        image000_lac_analysis.json
        image000_consolidated.png

Metrics:
    - IoU (Intersection over Union)
    - Precision (what % of predicted floor is actually floor)
    - Recall (what % of GT floor was detected)
    - F1 Score (harmonic mean of precision and recall)
    - Dice Coefficient (equivalent to F1 for binary masks)

Usage:
    # Evaluate a specific run
    python evaluate_lac_nyu.py --results_dir /path/to/two_vlm/Qwen/rgb_depth_separate_sam3

    # With custom GT mask dir
    python evaluate_lac_nyu.py --results_dir ... --gt_mask_dir /path/to/floor_masks

Output:
    {output_dir}/nyu_evaluation.json    — Per-image and aggregate metrics
    {output_dir}/nyu_evaluation.csv     — Per-image metrics as CSV
    {output_dir}/evaluation_summary.txt — Human-readable summary
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Compute binary segmentation metrics between prediction and ground truth."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    # IoU
    iou = float(intersection / union) if union > 0 else 0.0

    # Precision: TP / (TP + FP)
    precision = float(intersection / pred_sum) if pred_sum > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = float(intersection / gt_sum) if gt_sum > 0 else 0.0

    # F1 Score
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Dice Coefficient
    dice = float(2 * intersection / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else 0.0

    # Pixel Accuracy
    total_pixels = pred.size
    correct = (pred == gt).sum()
    pixel_accuracy = float(correct / total_pixels) if total_pixels > 0 else 0.0

    tp = int(intersection)
    fp = int(pred_sum - intersection)
    fn = int(gt_sum - intersection)
    tn = int(total_pixels - tp - fp - fn)

    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "dice": round(dice, 4),
        "pixel_accuracy": round(pixel_accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_coverage": round(float(pred_sum / total_pixels * 100), 2) if total_pixels > 0 else 0.0,
        "gt_coverage": round(float(gt_sum / total_pixels * 100), 2) if total_pixels > 0 else 0.0,
    }


def load_mask(mask_path: Path, target_size: Tuple[int, int] = None) -> np.ndarray:
    """Load a binary mask from a PNG file."""
    mask = Image.open(mask_path).convert("L")
    if target_size:
        mask = mask.resize(target_size, Image.NEAREST)
    return np.array(mask) > 128


def find_predicted_masks(results_dir: Path, image_id: str) -> Optional[np.ndarray]:
    """Find and load predicted masks for a given image from the original pipeline output.

    The original pipeline stores masks in:
        {results_dir}/lms_kamal_nyu_testset/masks/{image_id}_mask_*.png
    or:
        {results_dir}/{any_folder}/masks/{image_id}_mask_*.png

    Returns combined union mask, or None if no masks found.
    """
    # Search for masks directory — could be under a folder subdirectory
    mask_dirs = []
    
    # Check direct masks/ dir
    direct_masks = results_dir / "masks"
    if direct_masks.exists():
        mask_dirs.append(direct_masks)
    
    # Check under folder subdirectories (original pipeline structure)
    for subdir in results_dir.iterdir():
        if subdir.is_dir() and subdir.name != "evaluation":
            candidate = subdir / "masks"
            if candidate.exists():
                mask_dirs.append(candidate)

    all_masks = []
    for mask_dir in mask_dirs:
        # Try .npy files first (more reliable)
        npy_files = sorted(mask_dir.glob(f"{image_id}_mask_*_*.npy"))
        for npy_file in npy_files:
            mask = np.load(str(npy_file))
            all_masks.append(mask)

        if not all_masks:
            # Try PNG files
            png_files = sorted(mask_dir.glob(f"{image_id}_mask_*_*.png"))
            # Exclude the segmentation overlay
            png_files = [f for f in png_files if "segmentation_overlay" not in f.name]
            for png_file in png_files:
                mask = load_mask(png_file)
                all_masks.append(mask)

    if not all_masks:
        return None

    # Combine all masks into a single binary mask (union)
    combined = all_masks[0].copy()
    for m in all_masks[1:]:
        combined = combined | m

    return combined


def nyu_image_id_to_gt_id(image_id: str) -> str:
    """Convert pipeline image_id to GT mask filename.

    Pipeline uses: image000, image001, ...
    GT uses: nyu_0000_floor.png, nyu_0001_floor.png, ...

    So image000 → nyu_0000, image001 → nyu_0001, etc.
    """
    if image_id.startswith("image"):
        num = int(image_id.replace("image", ""))
        return f"nyu_{num:04d}"
    return image_id


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_nyu_results(
    results_dir: Path,
    gt_mask_dir: Path,
    output_dir: Path = None,
    min_gt_coverage: float = 0.0,
    summary_only: bool = False,
) -> Dict:
    """Evaluate LAC pipeline results against NYU floor ground truth.

    Args:
        results_dir: Directory with pipeline results.
        gt_mask_dir: Directory with GT floor masks (nyu_XXXX_floor.png).
        output_dir: Directory to save evaluation results.
        min_gt_coverage: Skip images with less than this % floor in GT.
        summary_only: Only print summary, don't save per-image details.
    """
    if output_dir is None:
        output_dir = results_dir / "evaluation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results dir: {results_dir}")
    logger.info(f"GT mask dir:  {gt_mask_dir}")
    logger.info(f"Output dir:   {output_dir}")

    if not gt_mask_dir.exists():
        logger.error(f"GT mask directory not found: {gt_mask_dir}")
        logger.info("Run extract_nyu_floor_gt.py first to generate floor GT masks.")
        return {}

    # Discover GT masks
    gt_masks = sorted(gt_mask_dir.glob("nyu_*_floor.png"))
    logger.info(f"Found {len(gt_masks)} GT floor masks")

    if not gt_masks:
        logger.error("No GT masks found!")
        return {}

    # Load analysis JSON files to get list of processed images
    json_files = []
    # Check direct directory
    json_files.extend(sorted(results_dir.glob("*_lac_analysis.json")))
    # Check under folder subdirectories
    for subdir in results_dir.iterdir():
        if subdir.is_dir() and subdir.name not in ("evaluation", "masks", "logs"):
            json_files.extend(sorted(subdir.glob("*_lac_analysis.json")))
    json_files = sorted(set(json_files))

    processed_ids = []
    for jf in json_files:
        image_id = jf.stem.replace("_lac_analysis", "")
        processed_ids.append(image_id)

    logger.info(f"Found {len(processed_ids)} processed image results")

    # Evaluate each image
    per_image_metrics = []
    skipped_no_gt = 0
    skipped_no_pred = 0
    skipped_low_coverage = 0

    for image_id in processed_ids:
        # Find GT mask — convert pipeline ID to GT ID
        gt_id = nyu_image_id_to_gt_id(image_id)
        gt_path = gt_mask_dir / f"{gt_id}_floor.png"

        if not gt_path.exists():
            skipped_no_gt += 1
            continue

        # Load GT mask
        gt_mask = load_mask(gt_path)
        gt_coverage = gt_mask.sum() / gt_mask.size * 100

        # Skip images with very little floor in GT
        if gt_coverage < min_gt_coverage:
            skipped_low_coverage += 1
            continue

        # Find predicted mask
        pred_mask = find_predicted_masks(results_dir, image_id)
        if pred_mask is None:
            # No prediction — treat as empty mask (all non-floor)
            pred_mask = np.zeros_like(gt_mask, dtype=bool)
            skipped_no_pred += 1

        # Ensure same size
        if pred_mask.shape != gt_mask.shape:
            pred_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
            pred_pil = pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
            pred_mask = np.array(pred_pil) > 128

        # Compute metrics
        metrics = compute_binary_metrics(pred_mask, gt_mask)
        metrics["image_id"] = image_id
        metrics["gt_id"] = gt_id
        metrics["gt_coverage"] = round(gt_coverage, 2)
        per_image_metrics.append(metrics)

    if not per_image_metrics:
        logger.error("No images could be evaluated!")
        return {}

    # Compute aggregate statistics
    aggregate = _compute_aggregate(per_image_metrics)

    # Build evaluation results
    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(results_dir),
        "gt_mask_dir": str(gt_mask_dir),
        "total_gt_masks": len(gt_masks),
        "total_processed": len(processed_ids),
        "evaluated": len(per_image_metrics),
        "skipped_no_gt": skipped_no_gt,
        "skipped_no_pred": skipped_no_pred,
        "skipped_low_coverage": skipped_low_coverage,
        "min_gt_coverage_threshold": min_gt_coverage,
        "aggregate": aggregate,
        "per_image": per_image_metrics,
    }

    # Save results
    _save_evaluation(evaluation, output_dir, summary_only)

    return evaluation


def _compute_aggregate(per_image: List[Dict]) -> Dict:
    """Compute aggregate statistics from per-image metrics."""
    metrics_keys = ["iou", "precision", "recall", "f1", "dice", "pixel_accuracy"]

    aggregate = {}
    for key in metrics_keys:
        values = [m[key] for m in per_image if key in m]
        if values:
            aggregate[f"{key}_mean"] = round(float(np.mean(values)), 4)
            aggregate[f"{key}_median"] = round(float(np.median(values)), 4)
            aggregate[f"{key}_std"] = round(float(np.std(values)), 4)
            aggregate[f"{key}_min"] = round(float(np.min(values)), 4)
            aggregate[f"{key}_max"] = round(float(np.max(values)), 4)
            aggregate[f"{key}_p25"] = round(float(np.percentile(values, 25)), 4)
            aggregate[f"{key}_p75"] = round(float(np.percentile(values, 75)), 4)

    # Total pixel counts
    total_tp = sum(m.get("tp", 0) for m in per_image)
    total_fp = sum(m.get("fp", 0) for m in per_image)
    total_fn = sum(m.get("fn", 0) for m in per_image)
    total_tn = sum(m.get("tn", 0) for m in per_image)

    aggregate["total_tp"] = total_tp
    aggregate["total_fp"] = total_fp
    aggregate["total_fn"] = total_fn
    aggregate["total_tn"] = total_tn

    # Micro-averaged metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0
    )
    micro_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    aggregate["micro_precision"] = round(micro_precision, 4)
    aggregate["micro_recall"] = round(micro_recall, 4)
    aggregate["micro_f1"] = round(micro_f1, 4)
    aggregate["micro_iou"] = round(micro_iou, 4)

    # Detection rate
    detected = sum(1 for m in per_image if m.get("pred_coverage", 0) > 0)
    has_gt_floor = sum(1 for m in per_image if m.get("gt_coverage", 0) > 0)
    aggregate["detection_rate"] = round(detected / len(per_image) * 100, 2) if per_image else 0
    aggregate["images_with_gt_floor"] = has_gt_floor
    aggregate["images_with_detection"] = detected

    pred_coverages = [m.get("pred_coverage", 0) for m in per_image]
    gt_coverages = [m.get("gt_coverage", 0) for m in per_image]
    aggregate["pred_coverage_mean"] = round(float(np.mean(pred_coverages)), 2)
    aggregate["gt_coverage_mean"] = round(float(np.mean(gt_coverages)), 2)

    return aggregate


def _save_evaluation(evaluation: Dict, output_dir: Path, summary_only: bool = False):
    """Save evaluation results to files."""
    # Save full JSON
    json_path = output_dir / "nyu_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(evaluation, f, indent=2, default=str)
    logger.info(f"Saved evaluation JSON: {json_path}")

    # Save per-image CSV
    if not summary_only and evaluation.get("per_image"):
        csv_path = output_dir / "nyu_evaluation.csv"
        fieldnames = ["image_id", "gt_id", "iou", "precision", "recall", "f1", "dice",
                      "pixel_accuracy", "pred_coverage", "gt_coverage",
                      "tp", "fp", "fn", "tn"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in evaluation["per_image"]:
                writer.writerow(row)
        logger.info(f"Saved per-image CSV: {csv_path}")

    # Save human-readable summary
    agg = evaluation.get("aggregate", {})
    summary_lines = [
        "=" * 70,
        "LAC Pipeline — NYU Depth V2 Evaluation Summary",
        "=" * 70,
        f"Timestamp:     {evaluation.get('timestamp', 'N/A')}",
        f"Results dir:   {evaluation.get('results_dir', 'N/A')}",
        f"GT masks:      {evaluation.get('total_gt_masks', 0)}",
        f"Evaluated:     {evaluation.get('evaluated', 0)} images",
        f"Skipped (no GT):    {evaluation.get('skipped_no_gt', 0)}",
        f"Skipped (no pred):  {evaluation.get('skipped_no_pred', 0)}",
        f"Skipped (low cov):  {evaluation.get('skipped_low_coverage', 0)}",
        "",
        "-" * 70,
        "Macro-Averaged Metrics (mean over images)",
        "-" * 70,
        f"  IoU:            {agg.get('iou_mean', 0):.4f} ± {agg.get('iou_std', 0):.4f}  "
        f"(median: {agg.get('iou_median', 0):.4f})",
        f"  Precision:      {agg.get('precision_mean', 0):.4f} ± {agg.get('precision_std', 0):.4f}  "
        f"(median: {agg.get('precision_median', 0):.4f})",
        f"  Recall:         {agg.get('recall_mean', 0):.4f} ± {agg.get('recall_std', 0):.4f}  "
        f"(median: {agg.get('recall_median', 0):.4f})",
        f"  F1 Score:       {agg.get('f1_mean', 0):.4f} ± {agg.get('f1_std', 0):.4f}  "
        f"(median: {agg.get('f1_median', 0):.4f})",
        f"  Dice:           {agg.get('dice_mean', 0):.4f} ± {agg.get('dice_std', 0):.4f}",
        f"  Pixel Accuracy: {agg.get('pixel_accuracy_mean', 0):.4f} ± {agg.get('pixel_accuracy_std', 0):.4f}",
        "",
        "-" * 70,
        "Micro-Averaged Metrics (pooled pixel counts)",
        "-" * 70,
        f"  Micro IoU:       {agg.get('micro_iou', 0):.4f}",
        f"  Micro Precision: {agg.get('micro_precision', 0):.4f}",
        f"  Micro Recall:    {agg.get('micro_recall', 0):.4f}",
        f"  Micro F1:        {agg.get('micro_f1', 0):.4f}",
        "",
        "-" * 70,
        "Detection Statistics",
        "-" * 70,
        f"  Images with GT floor:    {agg.get('images_with_gt_floor', 0)}",
        f"  Images with detection:   {agg.get('images_with_detection', 0)}",
        f"  Detection rate:          {agg.get('detection_rate', 0):.1f}%",
        f"  Avg predicted coverage:  {agg.get('pred_coverage_mean', 0):.2f}%",
        f"  Avg GT coverage:         {agg.get('gt_coverage_mean', 0):.2f}%",
        "",
        "-" * 70,
        "IoU Distribution",
        "-" * 70,
        f"  Min:  {agg.get('iou_min', 0):.4f}",
        f"  P25:  {agg.get('iou_p25', 0):.4f}",
        f"  P50:  {agg.get('iou_median', 0):.4f}",
        f"  P75:  {agg.get('iou_p75', 0):.4f}",
        f"  Max:  {agg.get('iou_max', 0):.4f}",
        "=" * 70,
    ]

    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LAC pipeline results on NYU Depth V2 test set",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Directory with pipeline results (contains folder/masks/ and JSON files)",
    )
    parser.add_argument(
        "--gt_mask_dir", type=str,
        default="/home/woody/iwnt/iwnt164h/nyu_testset_gt/floor_masks",
        help="Directory with GT floor masks (nyu_XXXX_floor.png)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for evaluation results (default: {results_dir}/evaluation)",
    )
    parser.add_argument(
        "--min_gt_coverage", type=float, default=1.0,
        help="Skip images with less than this %% floor coverage in GT (default: 1.0)",
    )
    parser.add_argument(
        "--summary_only", action="store_true",
        help="Only print summary, don't save per-image CSV",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    gt_mask_dir = Path(args.gt_mask_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    evaluate_nyu_results(
        results_dir=results_dir,
        gt_mask_dir=gt_mask_dir,
        output_dir=output_dir,
        min_gt_coverage=args.min_gt_coverage,
        summary_only=args.summary_only,
    )


if __name__ == "__main__":
    main()
