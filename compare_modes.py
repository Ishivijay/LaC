#!/usr/bin/env python3
"""Compare LaC pipeline results across input modes (rgb_only vs rgb_depth_overlay)
against ground truth binary masks.

Usage:
    # First run both modes:
    python3 lac_pipeline.py --config lac_config.yaml --input_mode rgb_only \
        --specific_images image28 image36 image86 image95 image188 --clean
    python3 lac_pipeline.py --config lac_config.yaml --input_mode rgb_depth_overlay \
        --specific_images image28 image36 image86 image95 image188 --clean

    # Then compare:
    python3 compare_modes.py
    python3 compare_modes.py --modes rgb_only rgb_depth_overlay
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Ground truth mapping: gt_name → (folder_name, image_id)
GT_ANNOTATIONS = {
    "LA_Downstairs_image28": ("lms_kamal_LA_downstairs_Nopeople_1", "image28"),
    "LA_Upstairs_image188":  ("lms_kamal_LA_upstairs_Nopeople_1", "image188"),
    "LA_Upstairs_image86":   ("lms_kamal_LA_upstairs_Nopeople_1", "image86"),
    "RA_Downstairs_image36": ("lms_kamal_RA_downstairs_Nopeople_1", "image36"),
    "RA_Upstairs_image28":   ("lms_kamal_RA_upstairs_Nopeople_1", "image28"),
    "RB_Downstairs_image95": ("lms_kamal_RB_downstairs_Nopeople_1", "image95"),
}

# Default paths
DEFAULT_GT_DIR = Path(
    "/home/woody/iwnt/iwnt164h/mlp_dataset/prospthesisproject-Data"
    "/Code/annotated_samples/image_ann_binary_mask"
)
DEFAULT_RESULTS_DIR = Path("/home/woody/iwnt/iwnt164h/free_ground_results")
DEFAULT_MODEL = "Qwen2.5-VL-7B-Instruct_LaC"


def load_gt_mask(gt_path: Path) -> np.ndarray:
    """Load ground truth binary mask (white = free ground)."""
    mask = np.array(Image.open(gt_path).convert("L"))
    return (mask > 127).astype(np.uint8)


def load_predicted_mask(mask_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """Load a predicted mask and resize to match target."""
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape != (target_size[0], target_size[1]):
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize((target_size[1], target_size[0]), Image.NEAREST)
        mask = np.array(mask_img)
    return (mask > 127).astype(np.uint8)


def combine_predicted_masks(mask_dir: Path, target_h: int, target_w: int) -> np.ndarray:
    """Combine all individual mask PNGs into a single binary mask."""
    combined = np.zeros((target_h, target_w), dtype=np.uint8)
    mask_files = list(mask_dir.glob("*.png"))
    mask_files = [f for f in mask_files if "overlay" not in f.name]

    for mf in mask_files:
        mask = load_predicted_mask(mf, (target_h, target_w))
        combined = np.maximum(combined, mask)

    return combined


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict:
    """Compute IoU, precision, recall, F1, accuracy."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = intersection
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    iou = intersection / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Also compute "stair safety" — fraction of predicted mask that overlaps with GT
    # This tells us if we're detecting stairs as free ground
    pred_coverage = pred.sum() / pred.size * 100  # % of image predicted as free ground
    gt_coverage = gt.sum() / gt.size * 100  # % of image that IS free ground

    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "pred_coverage_pct": round(pred_coverage, 1),
        "gt_coverage_pct": round(gt_coverage, 1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_mode(
    mode: str,
    results_dir: Path,
    gt_dir: Path,
) -> List[Dict]:
    """Evaluate a single mode against all ground truth annotations."""
    mode_dir = results_dir / DEFAULT_MODEL / mode
    if not mode_dir.exists():
        print(f"  ⚠️  No results found at {mode_dir}")
        return []

    results = []
    for gt_name, (folder, image_id) in GT_ANNOTATIONS.items():
        gt_path = gt_dir / f"{gt_name}_mask.png"
        if not gt_path.exists():
            print(f"  ⚠️  GT mask not found: {gt_path}")
            continue

        gt_mask = load_gt_mask(gt_path)
        h, w = gt_mask.shape

        # Check for segmentation masks
        mask_dir = mode_dir / folder / "masks"
        json_path = mode_dir / folder / f"{image_id}_lac_analysis.json"

        # Load JSON for metadata
        metadata = {}
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                metadata = {
                    "num_areas": data.get("reasoner", {}).get("num_areas", 0),
                    "description": data.get("reasoner", {}).get("output", {}).get("description", ""),
                    "areas": [
                        area.get("name", "")
                        for area in data.get("reasoner", {}).get("output", {}).get("free_ground_areas", [])
                    ],
                    "obstacles": data.get("reasoner", {}).get("output", {}).get("obstacles", []),
                    "scores": data.get("evaluator", {}).get("output", {}).get("traversability_score", {}),
                }

        # Load predicted mask
        if mask_dir.exists():
            pred_mask = combine_predicted_masks(mask_dir, h, w)
        else:
            # No masks generated — all zeros
            pred_mask = np.zeros((h, w), dtype=np.uint8)

        metrics = compute_metrics(pred_mask, gt_mask)
        metrics["gt_name"] = gt_name
        metrics["folder"] = folder
        metrics["image_id"] = image_id
        metrics["mode"] = mode
        metrics["has_masks"] = mask_dir.exists()
        metrics.update(metadata)

        results.append(metrics)

    return results


def print_comparison_table(all_results: Dict[str, List[Dict]]):
    """Print a formatted comparison table."""
    modes = list(all_results.keys())
    gt_names = list(GT_ANNOTATIONS.keys())

    print("\n" + "=" * 120)
    print("COMPARISON: rgb_only vs rgb_depth_overlay")
    print("=" * 120)

    # Per-image comparison
    header = f"{'Image':<30}"
    for mode in modes:
        header += f" │ {mode:^20}"
    header += " │ Winner"
    print(header)
    print("-" * 120)

    mode_ious = {mode: [] for mode in modes}

    for gt_name in gt_names:
        row = f"{gt_name:<30}"
        best_iou = -1
        best_mode = ""
        ious = {}

        for mode in modes:
            result = next((r for r in all_results[mode] if r["gt_name"] == gt_name), None)
            if result:
                iou = result["iou"]
                ious[mode] = iou
                mode_ious[mode].append(iou)
                prec = result["precision"]
                rec = result["recall"]
                row += f" │ IoU={iou:.3f} P={prec:.2f} R={rec:.2f}"
                if iou > best_iou:
                    best_iou = iou
                    best_mode = mode
            else:
                row += f" │ {'N/A':^20}"

        if best_mode:
            row += f" │ {best_mode}"
        print(row)

    print("-" * 120)

    # Summary
    row = f"{'MEAN IoU':<30}"
    best_mean = -1
    best_mode = ""
    for mode in modes:
        ious = mode_ious[mode]
        mean_iou = np.mean(ious) if ious else 0
        row += f" │ {mean_iou:^20.4f}"
        if mean_iou > best_mean:
            best_mean = mean_iou
            best_mode = mode
    row += f" │ {best_mode}"
    print(row)

    # Additional metrics
    print("\n" + "=" * 120)
    print("DETAILED METRICS PER MODE")
    print("=" * 120)

    for mode in modes:
        results = all_results[mode]
        if not results:
            continue

        print(f"\n--- {mode} ---")
        print(f"  {'Image':<30} {'IoU':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Areas':>5} {'Pred%':>6} {'GT%':>6} {'Obstacles'}")
        print(f"  {'-'*95}")

        for r in results:
            obs_str = ", ".join(r.get("obstacles", [])) if r.get("obstacles") else "none"
            areas_str = str(r.get("num_areas", 0))
            print(f"  {r['gt_name']:<30} {r['iou']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} "
                  f"{r['f1']:>6.3f} {areas_str:>5} {r['pred_coverage_pct']:>5.1f}% {r['gt_coverage_pct']:>5.1f}% {obs_str}")

        # Mean
        mean_iou = np.mean([r["iou"] for r in results])
        mean_prec = np.mean([r["precision"] for r in results])
        mean_rec = np.mean([r["recall"] for r in results])
        mean_f1 = np.mean([r["f1"] for r in results])
        print(f"  {'-'*95}")
        print(f"  {'MEAN':<30} {mean_iou:>6.3f} {mean_prec:>6.3f} {mean_rec:>6.3f} {mean_f1:>6.3f}")

    # Stair detection analysis
    print("\n" + "=" * 120)
    print("STAIR DETECTION ANALYSIS")
    print("=" * 120)
    print("  Checking if stairs are incorrectly identified as free ground...")
    print()

    for mode in modes:
        results = all_results[mode]
        stair_issues = 0
        for r in results:
            # High predicted coverage but low IoU suggests stairs are being detected as floor
            if r["pred_coverage_pct"] > r["gt_coverage_pct"] * 1.5 and r["iou"] < 0.3:
                stair_issues += 1
                print(f"  ⚠️  [{mode}] {r['gt_name']}: predicted {r['pred_coverage_pct']}% vs GT {r['gt_coverage_pct']}% "
                      f"(IoU={r['iou']:.3f}) — likely detecting stairs as floor")

        if stair_issues == 0:
            print(f"  ✅ [{mode}] No obvious stair misclassification detected")


def generate_visualizations(
    all_results: Dict[str, List[Dict]],
    results_dir: Path,
    gt_dir: Path,
    output_dir: Path,
):
    """Generate side-by-side comparison images for each annotated image."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    modes = list(all_results.keys())
    n_cols = 2 + len(modes)  # RGB + GT + one per mode

    for gt_name, (folder, image_id) in GT_ANNOTATIONS.items():
        # Load RGB image
        rgb_path = None
        base_dir = Path(
            "/home/woody/iwnt/iwnt164h/mlp_dataset/prospthesisproject-Data/Code/Data"
        )
        for suffix in ["_Nopeople_1", "_Nopeople_2"]:
            candidate = base_dir / f"{folder}{suffix}" if suffix == "_Nopeople_1" else base_dir / folder.replace("_1", "_2")
            # Try the original folder name
            candidate = base_dir / folder / "sharpen_rgb" / "PNG" / f"{image_id}.png"
            if candidate.exists():
                rgb_path = candidate
                break

        if rgb_path is None or not rgb_path.exists():
            # Try alternative path construction
            parts = folder.split("_")
            # e.g., lms_kamal_LA_downstairs_Nopeople_1
            rgb_path = base_dir / folder / "sharpen_rgb" / "PNG" / f"{image_id}.png"

        rgb_image = None
        if rgb_path.exists():
            rgb_image = np.array(Image.open(rgb_path).convert("RGB"))
        else:
            print(f"  ⚠️  RGB not found for {gt_name}: {rgb_path}")
            continue

        h, w = rgb_image.shape[:2]

        # Load GT mask
        gt_path = gt_dir / f"{gt_name}_mask.png"
        gt_mask = load_gt_mask(gt_path) if gt_path.exists() else np.zeros((h, w), dtype=np.uint8)

        # Create figure
        fig = plt.figure(figsize=(5 * n_cols, 5))
        gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.05)

        # Panel 1: Original RGB
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(rgb_image)
        ax.set_title("Original RGB", fontsize=11, fontweight="bold")
        ax.axis("off")

        # Panel 2: Ground Truth
        ax = fig.add_subplot(gs[0, 1])
        gt_overlay = rgb_image.copy()
        gt_colored = np.zeros_like(rgb_image)
        gt_colored[:, :, 1] = gt_mask * 255  # Green for GT
        gt_overlay = np.where(
            gt_mask[:, :, np.newaxis] > 0,
            (gt_overlay * 0.5 + gt_colored * 0.5).astype(np.uint8),
            gt_overlay,
        )
        ax.imshow(gt_overlay)
        coverage = gt_mask.sum() / gt_mask.size * 100
        ax.set_title(f"Ground Truth ({coverage:.1f}%)", fontsize=11, fontweight="bold", color="green")
        ax.axis("off")

        # Panels 3+: Each mode's prediction
        for idx, mode in enumerate(modes):
            ax = fig.add_subplot(gs[0, 2 + idx])
            mask_dir = results_dir / DEFAULT_MODEL / mode / folder / "masks"

            if mask_dir.exists():
                pred_mask = combine_predicted_masks(mask_dir, h, w)
            else:
                pred_mask = np.zeros((h, w), dtype=np.uint8)

            # Create colored overlay: TP=green, FP=red, FN=blue
            comparison = np.zeros_like(rgb_image)
            tp = np.logical_and(pred_mask, gt_mask)
            fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
            fn = np.logical_and(np.logical_not(pred_mask), gt_mask)

            comparison[tp] = [0, 255, 0]    # Green = correct
            comparison[fp] = [255, 0, 0]    # Red = false positive (predicted stairs as floor)
            comparison[fn] = [0, 0, 255]    # Blue = false negative (missed floor)

            overlay = rgb_image.copy()
            has_mask = np.logical_or(pred_mask, gt_mask)
            overlay = np.where(
                has_mask[:, :, np.newaxis] > 0,
                (overlay * 0.4 + comparison * 0.6).astype(np.uint8),
                overlay,
            )

            ax.imshow(overlay)

            # Get metrics for title
            result = next((r for r in all_results[mode] if r["gt_name"] == gt_name), None)
            if result:
                iou = result["iou"]
                prec = result["precision"]
                rec = result["recall"]
                pred_cov = result["pred_coverage_pct"]
                color = "green" if iou > 0.4 else ("orange" if iou > 0.2 else "red")
                ax.set_title(
                    f"{mode}\nIoU={iou:.3f} P={prec:.2f} R={rec:.2f} ({pred_cov:.1f}%)",
                    fontsize=10, fontweight="bold", color=color,
                )
            else:
                ax.set_title(f"{mode}\nNo results", fontsize=10, color="gray")

            ax.axis("off")

        plt.suptitle(gt_name.replace("_", " "), fontsize=14, fontweight="bold", y=1.02)
        # Add color legend
        legend_text = "🟢 Green = Correct (TP)    🔴 Red = Wrong detection (FP, stairs→floor)    🔵 Blue = Missed floor (FN)"
        fig.text(0.5, 0.01, legend_text, ha="center", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])

        save_path = vis_dir / f"{gt_name}_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # Generate summary image
    print("\n  Generating summary grid...")
    n_images = len(GT_ANNOTATIONS)
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4 * n_cols, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for row, (gt_name, (folder, image_id)) in enumerate(GT_ANNOTATIONS.items()):
        # Load RGB
        rgb_path = Path(
            "/home/woody/iwnt/iwnt164h/mlp_dataset/prospthesisproject-Data"
            f"/Code/Data/{folder}/sharpen_rgb/PNG/{image_id}.png"
        )
        if not rgb_path.exists():
            continue
        rgb_image = np.array(Image.open(rgb_path).convert("RGB"))
        h, w = rgb_image.shape[:2]

        # GT mask
        gt_path = gt_dir / f"{gt_name}_mask.png"
        gt_mask = load_gt_mask(gt_path) if gt_path.exists() else np.zeros((h, w), dtype=np.uint8)

        # RGB
        axes[row, 0].imshow(rgb_image)
        axes[row, 0].set_ylabel(gt_name.replace("_", "\n"), fontsize=8, fontweight="bold")
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("RGB", fontsize=10, fontweight="bold")

        # GT
        gt_overlay = rgb_image.copy()
        gt_green = np.zeros_like(rgb_image)
        gt_green[:, :, 1] = gt_mask * 255
        gt_overlay = np.where(gt_mask[:, :, np.newaxis] > 0,
                              (gt_overlay * 0.5 + gt_green * 0.5).astype(np.uint8), gt_overlay)
        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title("Ground Truth", fontsize=10, fontweight="bold", color="green")

        # Each mode
        for idx, mode in enumerate(modes):
            mask_dir = results_dir / DEFAULT_MODEL / mode / folder / "masks"
            pred_mask = combine_predicted_masks(mask_dir, h, w) if mask_dir.exists() else np.zeros((h, w), dtype=np.uint8)

            # Color overlay: TP=green, FP=red, FN=blue
            comp = np.zeros_like(rgb_image)
            tp = np.logical_and(pred_mask, gt_mask)
            fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
            fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
            comp[tp] = [0, 255, 0]
            comp[fp] = [255, 0, 0]
            comp[fn] = [0, 0, 255]

            overlay = rgb_image.copy()
            has_mask = np.logical_or(pred_mask, gt_mask)
            overlay = np.where(has_mask[:, :, np.newaxis] > 0,
                               (overlay * 0.4 + comp * 0.6).astype(np.uint8), overlay)
            axes[row, 2 + idx].imshow(overlay)

            result = next((r for r in all_results[mode] if r["gt_name"] == gt_name), None)
            iou = result["iou"] if result else 0
            color = "green" if iou > 0.4 else ("orange" if iou > 0.2 else "red")
            axes[row, 2 + idx].axis("off")
            if row == 0:
                axes[row, 2 + idx].set_title(f"{mode}\nIoU={iou:.3f}", fontsize=10, fontweight="bold", color=color)
            else:
                axes[row, 2 + idx].set_title(f"IoU={iou:.3f}", fontsize=9, color=color)

    # Legend
    legend_text = "🟢 Green = True Positive  🔴 Red = False Positive (stairs as floor)  🔵 Blue = False Negative (missed floor)"
    fig.text(0.5, 0.01, legend_text, ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    summary_path = vis_dir / "summary_comparison_grid.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary grid: {summary_path}")


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)

    def flush(self):
        for f in self.files:
            f.flush()


def generate_metric_charts(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
):
    """Generate evaluation metric charts (bar charts, comparison plots)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    modes = list(all_results.keys())
    gt_names = list(GT_ANNOTATIONS.keys())
    mode_colors = {"rgb_only": "#2196F3", "rgb_depth_overlay": "#FF9800"}
    default_colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    # ---- Chart 1: IoU per image (grouped bar chart) ----
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(gt_names))
    width = 0.35
    for i, mode in enumerate(modes):
        results_dict = {r["gt_name"]: r for r in all_results[mode]}
        ious = [results_dict.get(name, {}).get("iou", 0) for name in gt_names]
        color = mode_colors.get(mode, default_colors[i % len(default_colors)])
        bars = ax.bar(x + i * width, ious, width, label=mode, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, ious):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlabel("Annotated Image", fontsize=12)
    ax.set_ylabel("IoU Score", fontsize=12)
    ax.set_title("IoU per Annotated Image — Mode Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels([name.replace("_", "\n") for name in gt_names], fontsize=8, rotation=15)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(charts_dir / "iou_per_image.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {charts_dir / 'iou_per_image.png'}")

    # ---- Chart 2: Mean metrics comparison (grouped bar) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = ["IoU", "Precision", "Recall", "F1"]
    metric_keys = ["iou", "precision", "recall", "f1"]
    x = np.arange(len(metric_names))
    for i, mode in enumerate(modes):
        means = [np.mean([r[k] for r in all_results[mode]]) for k in metric_keys]
        color = mode_colors.get(mode, default_colors[i % len(default_colors)])
        bars = ax.bar(x + i * width, means, width, label=mode, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Mean Metrics — Mode Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(charts_dir / "mean_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {charts_dir / 'mean_metrics.png'}")

    # ---- Chart 3: Coverage comparison (predicted vs GT) ----
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(gt_names))
    # Plot GT coverage as a horizontal reference
    gt_coverages = []
    for name in gt_names:
        results_dict = {r["gt_name"]: r for r in all_results[modes[0]]}
        gt_coverages.append(results_dict.get(name, {}).get("gt_coverage_pct", 0))
    ax.bar(x - width * len(modes) / 2, gt_coverages, width * 0.8,
           label="Ground Truth", color="#4CAF50", alpha=0.6, edgecolor="white")

    for i, mode in enumerate(modes):
        results_dict = {r["gt_name"]: r for r in all_results[mode]}
        pred_coverages = [results_dict.get(name, {}).get("pred_coverage_pct", 0) for name in gt_names]
        color = mode_colors.get(mode, default_colors[i % len(default_colors)])
        ax.bar(x + (i + 0.5) * width, pred_coverages, width,
               label=f"{mode} (predicted)", color=color, alpha=0.85, edgecolor="white")

    ax.set_xlabel("Annotated Image", fontsize=12)
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title("Free Ground Coverage — Predicted vs Ground Truth", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in gt_names], fontsize=8, rotation=15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(charts_dir / "coverage_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {charts_dir / 'coverage_comparison.png'}")

    # ---- Chart 4: Per-image detailed metrics (radar/spider chart per mode) ----
    for mode in modes:
        results_dict = {r["gt_name"]: r for r in all_results[mode]}
        valid_results = [(name, results_dict[name]) for name in gt_names if name in results_dict]
        if not valid_results:
            continue

        categories = ["IoU", "Precision", "Recall", "F1", "Accuracy"]
        cat_keys = ["iou", "precision", "recall", "f1", "accuracy"]

        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        colors_list = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]
        for idx, (name, r) in enumerate(valid_results):
            values = [r.get(k, 0) for k in cat_keys]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=name.replace("_", " "),
                    color=colors_list[idx % len(colors_list)], alpha=0.7)
            ax.fill(angles, values, alpha=0.1, color=colors_list[idx % len(colors_list)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f"{mode} — Per-Image Metrics", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        safe_mode = mode.replace("/", "_")
        fig.savefig(charts_dir / f"radar_{safe_mode}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {charts_dir / f'radar_{safe_mode}.png'}")

    # ---- Chart 5: Error analysis (FP vs FN per image per mode) ----
    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 6))
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        results_dict = {r["gt_name"]: r for r in all_results[mode]}
        tp_vals = [results_dict.get(n, {}).get("tp", 0) for n in gt_names]
        fp_vals = [results_dict.get(n, {}).get("fp", 0) for n in gt_names]
        fn_vals = [results_dict.get(n, {}).get("fn", 0) for n in gt_names]

        x = np.arange(len(gt_names))
        ax.bar(x, tp_vals, label="True Positive", color="#4CAF50", alpha=0.85)
        ax.bar(x, fp_vals, bottom=tp_vals, label="False Positive (stairs→floor)", color="#F44336", alpha=0.85)
        ax.bar(x, fn_vals, bottom=[t + f for t, f in zip(tp_vals, fp_vals)],
               label="False Negative (missed floor)", color="#2196F3", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([n.split("_")[-1] for n in gt_names], fontsize=9, rotation=30)
        ax.set_ylabel("Pixels", fontsize=11)
        ax.set_title(f"{mode}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Error Analysis — TP / FP / FN per Image", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(charts_dir / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {charts_dir / 'error_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="Compare LaC pipeline modes against ground truth")
    parser.add_argument(
        "--modes", nargs="+", default=["rgb_only", "rgb_depth_overlay"],
        help="Input modes to compare",
    )
    parser.add_argument(
        "--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Base results directory",
    )
    parser.add_argument(
        "--gt_dir", type=str, default=str(DEFAULT_GT_DIR),
        help="Ground truth masks directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save comparison results (default: $WORK/free_ground_results/Qwen2.5-VL-7B-Instruct_LaC/comparison/)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    gt_dir = Path(args.gt_dir)

    # Setup output directory — default to $WORK/free_ground_results/Qwen2.5-VL-7B-Instruct_LaC/comparison/
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        work_dir = os.environ.get("WORK", str(Path(__file__).parent.parent))
        output_dir = Path(work_dir) / "free_ground_results" / DEFAULT_MODEL / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comparison_{timestamp}.txt"
    json_path = output_dir / f"comparison_{timestamp}.json"

    # Tee output to both console and file
    report_file = open(report_path, "w")
    tee = Tee(sys.stdout, report_file)
    sys.stdout = tee

    print(f"Results directory: {results_dir}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Annotated images: {len(GT_ANNOTATIONS)}")
    print(f"Modes to compare: {args.modes}")
    print(f"Report saved to: {report_path}")
    print(f"JSON saved to: {json_path}")

    all_results = {}
    for mode in args.modes:
        print(f"\nEvaluating mode: {mode}...")
        all_results[mode] = evaluate_mode(mode, results_dir, gt_dir)

    print_comparison_table(all_results)

    # Generate visualizations
    print("\n" + "=" * 120)
    print("GENERATING VISUALIZATIONS")
    print("=" * 120)
    print("\nSide-by-side comparison images:")
    generate_visualizations(all_results, results_dir, gt_dir, output_dir)
    print("\nMetric charts:")
    generate_metric_charts(all_results, output_dir)

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)

    # Save JSON results
    json_output = {
        "timestamp": timestamp,
        "modes": args.modes,
        "results_dir": str(results_dir),
        "gt_dir": str(gt_dir),
        "annotated_images": list(GT_ANNOTATIONS.keys()),
        "per_mode": {},
        "summary": {},
    }

    for mode in args.modes:
        json_output["per_mode"][mode] = all_results[mode]
        ious = [r["iou"] for r in all_results[mode]]
        json_output["summary"][mode] = {
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0,
            "mean_precision": round(float(np.mean([r["precision"] for r in all_results[mode]])), 4),
            "mean_recall": round(float(np.mean([r["recall"] for r in all_results[mode]])), 4),
            "mean_f1": round(float(np.mean([r["f1"] for r in all_results[mode]])), 4),
            "num_images": len(all_results[mode]),
        }

    # Determine winner
    mode_means = {
        mode: json_output["summary"][mode]["mean_iou"]
        for mode in args.modes
    }
    winner = max(mode_means, key=mode_means.get)
    json_output["winner"] = winner
    json_output["winner_mean_iou"] = mode_means[winner]

    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)

    # Restore stdout and close file
    sys.stdout = sys.__stdout__
    report_file.close()

    print(f"\nReport saved to: {report_path}")
    print(f"JSON results saved to: {json_path}")
    print(f"Winner: {winner} (mean IoU: {mode_means[winner]:.4f})")


if __name__ == "__main__":
    main()
