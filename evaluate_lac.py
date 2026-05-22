#!/usr/bin/env python3
"""
Evaluate LaC navigable-area pipeline runs against ground truth masks.

Evaluates each of the 4 LaC navigable runs (2 models × 2 input modes):
  - Qwen2.5-VL-7B-Instruct: rgb_only_grounding_dino, rgb_depth_overlay_grounding_dino
  - gemma-4-E4B-it:         rgb_only_grounding_dino, rgb_depth_overlay_grounding_dino

For each run, computes per-image and aggregate metrics (IoU, Precision, Recall,
F1, Dice, Accuracy), saves JSON + CSV, and generates visualizations.

Ground truth masks: sam3_output_v7/*_mask.png

Usage:
    python3 evaluate_lac.py
    python3 evaluate_lac.py --results_dir /path/to/free_ground_results
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

WORK_DIR = Path(os.environ.get("WORK", "/home/woody/iwnt/iwnt164h"))
DEFAULT_RESULTS_DIR = WORK_DIR / "free_ground_results"
DEFAULT_GT_MASK_DIR = WORK_DIR / "free_ground_results" / "Annotated_Ground_Truth"
DEFAULT_DATA_DIR = (
    WORK_DIR / "mlp_dataset" / "prospthesisproject-Data" / "Code" / "Data"
)
DEFAULT_OUTPUT_DIR = WORK_DIR / "free_ground_results" / "lac_navigable_evaluation"


def discover_gt_annotations(gt_mask_dir: Path) -> Dict[str, tuple]:
    """Auto-discover GT masks from directory structure.

    Supports two formats:
      1. Subfolder format: {gt_dir}/{folder_name}/{image_id}_mask.png
      2. Flat format: {gt_dir}/{prefix}_{image_id}_mask.png  (sam3_output_v7 style)

    Returns dict: gt_key → (folder_name, image_id)
    """
    annotations = {}

    # Check for subfolder format: {gt_dir}/{folder}/image*_mask.png
    subfolders = [d for d in gt_mask_dir.iterdir() if d.is_dir()]
    if subfolders:
        for subfolder in sorted(subfolders):
            folder_name = subfolder.name
            for mask_file in sorted(subfolder.glob("*_mask.png")):
                # Extract image_id from {image_id}_mask.png
                image_id = mask_file.stem.replace("_mask", "")
                gt_key = f"{folder_name}/{image_id}"
                annotations[gt_key] = (folder_name, image_id)
        if annotations:
            return annotations

    # Fallback: flat format with {prefix}_{image_id}_mask.png
    for mask_file in sorted(gt_mask_dir.glob("*_mask.png")):
        stem = mask_file.stem.replace("_mask", "")
        # Try to parse: LA_Downstairs_image28 → folder + image_id
        # Use hardcoded mapping for known sam3_output_v7 format
        SAM3_MAPPING = {
            "LA_Downstairs_image28": ("lms_kamal_LA_downstairs_Nopeople_1", "image28"),
            "LA_Upstairs_image188":  ("lms_kamal_LA_upstairs_Nopeople_1", "image188"),
            "LA_Upstairs_image86":   ("lms_kamal_LA_upstairs_Nopeople_1", "image86"),
            "LB_Upstairs_image147":  ("lms_kamal_LB_upstairs_Nopeople_2", "image147"),
            "RA_Downstairs_image36": ("lms_kamal_RA_downstairs_Nopeople_1", "image36"),
            "RA_Upstairs_image28":   ("lms_kamal_RA_upstairs_Nopeople_1", "image28"),
            "RB_Downstairs_image95": ("lms_kamal_RB_downstairs_Nopeople_1", "image95"),
        }
        if stem in SAM3_MAPPING:
            annotations[stem] = SAM3_MAPPING[stem]

    return annotations

MODELS = ["Qwen2.5-VL-7B-Instruct", "gemma-4-E4B-it"]


def _build_lac_runs(seg_methods=None):
    """Build LaC navigable pipeline run definitions.

    Args:
        seg_methods: List of segmentation methods to include.
                     Defaults to ["grounding_dino", "sam3"].
                     Pass ["sam3"] for SAM3-only evaluation.
    """
    if seg_methods is None:
        seg_methods = ["grounding_dino", "sam3"]
    _mode_labels = {
        "rgb_only": "RGB only",
        "rgb_depth_overlay": "RGB+D overlay",
        "rgb_depth_separate": "RGB+D separate",
    }
    runs = []
    for model in MODELS:
        for mode in ["rgb_only", "rgb_depth_separate"]:
            mode_label = _mode_labels.get(mode, mode)
            for seg_method in seg_methods:
                seg_label = "G-DINO+SAM" if seg_method == "grounding_dino" else "SAM3"
                runs.append({
                    "label": f"{model} — LaC Navigable ({mode_label}, {seg_label})",
                    "short": f"{model}_lac_navigable_{mode}_{seg_method}",
                    "model": model,
                    "mode": mode,
                    "results_subdir": f"{model}_LaC_navigable/{mode}_{seg_method}",
                    "result_type": "mask",
                    "json_pattern": "*_lac_analysis.json",
                })
    return runs


LAC_RUNS = _build_lac_runs()


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def load_gt_mask(mask_path: Path) -> np.ndarray:
    """Load ground truth binary mask (white = free ground)."""
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 127).astype(np.uint8)


def load_predicted_mask(mask_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """Load a predicted mask and resize to match target."""
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape != target_size:
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize((target_size[1], target_size[0]), Image.NEAREST)
        mask = np.array(mask_img)
    return (mask > 127).astype(np.uint8)


def combine_predicted_masks(mask_dir: Path, target_h: int, target_w: int,
                            image_id: str = None) -> np.ndarray:
    """Combine individual mask PNGs into a single binary mask.

    Args:
        mask_dir: Directory containing mask PNG files.
        target_h: Target height.
        target_w: Target width.
        image_id: If provided, only combine masks for this image
                  (filenames starting with ``{image_id}_mask_``).
    """
    combined = np.zeros((target_h, target_w), dtype=np.uint8)
    mask_files = [f for f in mask_dir.glob("*.png") if "overlay" not in f.name]
    if image_id is not None:
        prefix = f"{image_id}_mask_"
        mask_files = [f for f in mask_files if f.name.startswith(prefix)]
    for mf in mask_files:
        mask = load_predicted_mask(mf, (target_h, target_w))
        combined = np.maximum(combined, mask)
    return combined


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict:
    """Compute comprehensive metrics between predicted and GT masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = int(intersection)
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_and(np.logical_not(pred), np.logical_not(gt)), pred | gt | ~pred | ~gt).sum())
    # Simpler TN:
    tn = int((pred.size) - tp - fp - fn)

    iou = intersection / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    dice = 2 * intersection / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0

    pred_coverage = pred.sum() / pred.size * 100
    gt_coverage = gt.sum() / gt.size * 100

    return {
        "iou": round(float(iou), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "dice": round(float(dice), 4),
        "accuracy": round(float(accuracy), 4),
        "pred_coverage_pct": round(float(pred_coverage), 1),
        "gt_coverage_pct": round(float(gt_coverage), 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def load_rgb_image(folder: str, image_id: str) -> Optional[np.ndarray]:
    """Load the original RGB image for a given folder/image_id."""
    rgb_path = DEFAULT_DATA_DIR / folder / "sharpen_rgb" / "PNG" / f"{image_id}.png"
    if rgb_path.exists():
        return np.array(Image.open(rgb_path).convert("RGB"))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single_image(
    run: Dict,
    gt_name: str,
    folder: str,
    image_id: str,
    gt_mask: np.ndarray,
    results_dir: Path,
) -> Dict:
    """Evaluate a single image for a given LaC pipeline run."""
    h, w = gt_mask.shape
    run_dir = results_dir / run["results_subdir"]

    # Load prediction masks
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    metadata = {}
    prediction_found = False
    inference_time = None
    num_vlm_areas = 0

    # Load LaC analysis JSON for metadata/timing
    json_path = run_dir / folder / f"{image_id}_lac_analysis.json"
    if json_path.exists():
        with open(json_path) as f:
            analysis = json.load(f)
        reasoner = analysis.get("reasoner", {})
        num_vlm_areas = reasoner.get("num_areas", 0)
        inference_time = reasoner.get("inference_time", None)

        # Also get segmentation and cost_map timing
        seg_time = analysis.get("segmentation", {}).get("inference_time", None)
        costmap_time = analysis.get("cost_map", {}).get("inference_time", None)
        total_time = None
        if inference_time is not None:
            total_time = inference_time
            if seg_time is not None:
                total_time += seg_time
            if costmap_time is not None:
                total_time += costmap_time

        metadata = {
            "num_vlm_areas": num_vlm_areas,
            "reasoner_time": inference_time,
            "seg_time": seg_time,
            "costmap_time": costmap_time,
            "total_time": round(total_time, 2) if total_time else None,
        }

    # Load mask files
    mask_dir = run_dir / folder / "masks"
    if mask_dir.exists():
        mask_pngs = sorted([f for f in mask_dir.glob(f"{image_id}_mask_*.png")
                           if "overlay" not in f.name])
        if mask_pngs:
            pred_mask = combine_predicted_masks(mask_dir, h, w, image_id=image_id)
            prediction_found = True

    # Compute metrics
    metrics = compute_metrics(pred_mask, gt_mask) if prediction_found else {
        "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "dice": 0.0, "accuracy": 0.0,
        "pred_coverage_pct": 0.0, "gt_coverage_pct": round(float(gt_mask.sum() / gt_mask.size * 100), 1),
        "tp": 0, "fp": 0, "fn": int(gt_mask.sum()), "tn": int(gt_mask.size - gt_mask.sum()),
    }

    return {
        "gt_name": gt_name,
        "folder": folder,
        "image_id": image_id,
        "prediction_found": prediction_found,
        "num_masks": len(mask_pngs) if prediction_found else 0,
        "num_vlm_areas": num_vlm_areas,
        **metrics,
        **metadata,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-strategy evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(run: Dict, results_dir: Path, gt_mask_dir: Path, output_base: Path, gt_annotations: Dict) -> Dict:
    """Evaluate a single LaC pipeline run against all GT annotations."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {run['label']}")
    print(f"  subdir: {run['results_subdir']}")
    print(f"{'='*60}")

    image_results = []
    for gt_name, (folder, image_id) in gt_annotations.items():
        # Support both subfolder format ({gt_dir}/{folder}/{image}_mask.png)
        # and flat format ({gt_dir}/{gt_name}_mask.png)
        gt_mask_path = gt_mask_dir / folder / f"{image_id}_mask.png"
        if not gt_mask_path.exists():
            gt_mask_path = gt_mask_dir / f"{gt_name}_mask.png"
        if not gt_mask_path.exists():
            print(f"  ⚠ GT mask not found for {gt_name}")
            continue

        gt_mask = load_gt_mask(gt_mask_path)
        result = evaluate_single_image(run, gt_name, folder, image_id, gt_mask, results_dir)
        image_results.append(result)

        iou = result["iou"]
        f1 = result["f1"]
        areas = result["num_vlm_areas"]
        masks = result["num_masks"]
        t = result.get("total_time", "?")
        status = "✓" if result["prediction_found"] else "✗"
        print(f"  {status} {gt_name}: IoU={iou:.3f}, F1={f1:.3f}, areas={areas}, masks={masks}, time={t}s")

    if not image_results:
        print("  No images evaluated!")
        return {"label": run["label"], "short": run["short"], "error": "no_images"}

    # Aggregate metrics
    ious = [r["iou"] for r in image_results]
    precisions = [r["precision"] for r in image_results]
    recalls = [r["recall"] for r in image_results]
    f1s = [r["f1"] for r in image_results]
    dices = [r["dice"] for r in image_results]
    accuracies = [r["accuracy"] for r in image_results]

    # Timing stats
    total_times = [r["total_time"] for r in image_results if r.get("total_time") is not None]
    reasoner_times = [r["reasoner_time"] for r in image_results if r.get("reasoner_time") is not None]

    agg = {
        "label": run["label"],
        "short": run["short"],
        "model": run["model"],
        "mode": run["mode"],
        "results_subdir": run["results_subdir"],
        "num_images": len(image_results),
        "predictions_found": sum(1 for r in image_results if r["prediction_found"]),
        "mean_iou": round(float(np.mean(ious)), 4),
        "median_iou": round(float(np.median(ious)), 4),
        "std_iou": round(float(np.std(ious)), 4),
        "min_iou": round(float(np.min(ious)), 4),
        "max_iou": round(float(np.max(ious)), 4),
        "mean_precision": round(float(np.mean(precisions)), 4),
        "mean_recall": round(float(np.mean(recalls)), 4),
        "mean_f1": round(float(np.mean(f1s)), 4),
        "mean_dice": round(float(np.mean(dices)), 4),
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "per_image": image_results,
    }

    if total_times:
        agg["timing"] = {
            "mean_total_time": round(float(np.mean(total_times)), 2),
            "median_total_time": round(float(np.median(total_times)), 2),
            "std_total_time": round(float(np.std(total_times)), 2),
            "min_total_time": round(float(np.min(total_times)), 2),
            "max_total_time": round(float(np.max(total_times)), 2),
        }
    if reasoner_times:
        agg["timing"]["mean_reasoner_time"] = round(float(np.mean(reasoner_times)), 2)

    # Save per-strategy results
    strategy_dir = output_base / run["short"]
    strategy_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(strategy_dir / "evaluation.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)

    # CSV
    csv_path = strategy_dir / "per_image_metrics.csv"
    csv_fields = ["gt_name", "folder", "image_id", "iou", "precision", "recall",
                  "f1", "dice", "accuracy", "pred_coverage_pct", "gt_coverage_pct",
                  "num_masks", "num_vlm_areas", "total_time", "reasoner_time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in image_results:
            writer.writerow(r)

    print(f"\n  Summary: mean_IoU={agg['mean_iou']:.4f}, mean_F1={agg['mean_f1']:.4f}, "
          f"found={agg['predictions_found']}/{agg['num_images']}")
    if total_times:
        print(f"  Timing: mean={agg['timing']['mean_total_time']:.1f}s, "
              f"median={agg['timing']['median_total_time']:.1f}s")

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────────────────────────────────────

def generate_visualizations(all_results: List[Dict], output_base: Path, results_dir: Path,
                            gt_annotations: Dict):
    """Generate evaluation visualizations for LaC runs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    vis_dir = output_base / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. IoU comparison bar chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    shorts = [r["short"].replace("_", "\n") for r in all_results]
    mean_ious = [r["mean_iou"] for r in all_results]
    colors = ["#2196F3" if "Qwen" in r["model"] else "#FF5722" for r in all_results]
    bars = ax.bar(range(len(shorts)), mean_ious, color=colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(range(len(shorts)))
    ax.set_xticklabels(shorts, fontsize=8, ha="center")
    ax.set_ylabel("Mean IoU")
    ax.set_title("LaC Navigable — Mean IoU by Model & Mode")
    ax.set_ylim(0, max(mean_ious + [0.5]) * 1.2)
    for bar, val in zip(bars, mean_ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="#2196F3", alpha=0.8, label="Qwen"),
        plt.Rectangle((0, 0), 1, 1, fc="#FF5722", alpha=0.8, label="Gemma"),
    ])
    plt.tight_layout()
    plt.savefig(vis_dir / "iou_comparison.png", dpi=150)
    plt.close()

    # ── 2. Per-image IoU heatmap ──
    gt_names = list(gt_annotations.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    data = []
    labels = []
    for r in all_results:
        labels.append(r["short"].replace("lac_navigable_", ""))
        row = []
        for gt_name in gt_names:
            per = next((p for p in r["per_image"] if p["gt_name"] == gt_name), None)
            row.append(per["iou"] if per else 0.0)
        data.append(row)
    data = np.array(data)
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=max(0.5, data.max()))
    ax.set_xticks(range(len(gt_names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in gt_names], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(gt_names)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=7,
                    color="black" if data[i,j] > 0.5 else "gray")
    plt.colorbar(im, ax=ax, label="IoU")
    ax.set_title("LaC Navigable — Per-Image IoU")
    plt.tight_layout()
    plt.savefig(vis_dir / "per_image_iou_heatmap.png", dpi=150)
    plt.close()

    # ── 3. Metric radar chart ──
    metric_names = ["IoU", "Precision", "Recall", "F1", "Dice"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    for r in all_results:
        values = [r[f"mean_{m.lower()}"] for m in metric_names]
        values += values[:1]
        label = r["short"].replace("lac_navigable_", "")
        ax.plot(angles, values, "o-", linewidth=1.5, label=label, markersize=4)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title("LaC Navigable — Metric Comparison", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    plt.tight_layout()
    plt.savefig(vis_dir / "metric_radar.png", dpi=150)
    plt.close()

    # ── 4. Timing comparison ──
    timed = [r for r in all_results if "timing" in r]
    if timed:
        fig, ax = plt.subplots(figsize=(10, 5))
        t_labels = [r["short"].replace("lac_navigable_", "") for r in timed]
        t_means = [r["timing"]["mean_total_time"] for r in timed]
        t_stds = [r["timing"].get("std_total_time", 0) for r in timed]
        colors_t = ["#2196F3" if "Qwen" in r["model"] else "#FF5722" for r in timed]
        bars = ax.barh(range(len(t_labels)), t_means, xerr=t_stds, color=colors_t,
                       alpha=0.8, edgecolor="black", capsize=3)
        ax.set_yticks(range(len(t_labels)))
        ax.set_yticklabels(t_labels, fontsize=9)
        ax.set_xlabel("Mean Total Time (seconds)")
        ax.set_title("LaC Navigable — Mean Inference Time per Image")
        for bar, val in zip(bars, t_means):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}s", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(vis_dir / "timing_comparison.png", dpi=150)
        plt.close()

    # ── 5. Per-image overlay visualizations ──
    overlay_dir = vis_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        for per in r["per_image"]:
            gt_name = per["gt_name"]
            folder = per["folder"]
            image_id = per["image_id"]

            rgb = load_rgb_image(folder, image_id)
            if rgb is None:
                continue

            # Support both subfolder and flat GT mask formats
            gt_mask_path = DEFAULT_GT_MASK_DIR / folder / f"{image_id}_mask.png"
            if not gt_mask_path.exists():
                gt_mask_path = DEFAULT_GT_MASK_DIR / f"{gt_name}_mask.png"
            gt_mask = load_gt_mask(gt_mask_path)

            # Load prediction mask
            run_dir = results_dir / r["results_subdir"]
            mask_dir = run_dir / folder / "masks"
            if mask_dir.exists():
                pred_mask = combine_predicted_masks(mask_dir, rgb.shape[0], rgb.shape[1],
                                                    image_id=image_id)
            else:
                pred_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

            # Create overlay
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[0].axis("off")

            axes[1].imshow(rgb)
            gt_overlay = np.zeros_like(rgb)
            gt_overlay[gt_mask > 0] = [0, 255, 0]
            axes[1].imshow(gt_overlay, alpha=0.4)
            axes[1].set_title(f"GT (coverage={per['gt_coverage_pct']:.1f}%)")
            axes[1].axis("off")

            axes[2].imshow(rgb)
            pred_overlay = np.zeros_like(rgb)
            pred_overlay[pred_mask > 0] = [255, 0, 0]
            axes[2].imshow(pred_overlay, alpha=0.4)
            axes[2].set_title(f"Pred IoU={per['iou']:.3f} (coverage={per['pred_coverage_pct']:.1f}%)")
            axes[2].axis("off")

            short_label = r["short"]
            plt.suptitle(f"{short_label}\n{gt_name}", fontsize=10)
            plt.tight_layout()
            safe_name = gt_name.replace("/", "_")
            plt.savefig(overlay_dir / f"{safe_name}_{short_label}.png", dpi=120, bbox_inches="tight")
            plt.close()

    print(f"\nVisualizations saved to {vis_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate LaC navigable pipeline runs against GT")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Root results directory")
    parser.add_argument("--gt_mask_dir", type=Path, default=DEFAULT_GT_MASK_DIR,
                        help="Directory with GT masks (*_mask.png)")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for evaluation results")
    parser.add_argument("--seg_method", type=str, nargs="+",
                        default=["grounding_dino", "sam3"],
                        choices=["grounding_dino", "sam3"],
                        help="Segmentation method(s) to evaluate (default: both). "
                             "Use --seg_method sam3 for SAM3-only evaluation.")
    args = parser.parse_args()

    # Build runs for the requested segmentation methods
    lac_runs = _build_lac_runs(seg_methods=args.seg_method)

    print("=" * 60)
    print("LaC NAVIGABLE PIPELINE EVALUATION")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")
    print(f"GT mask dir: {args.gt_mask_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Seg methods: {args.seg_method}")

    # Discover GT annotations
    gt_annotations = discover_gt_annotations(args.gt_mask_dir)
    print(f"GT images:   {len(gt_annotations)} (auto-discovered from {args.gt_mask_dir})")
    print(f"Runs:        {len(lac_runs)}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each run
    all_results = []
    for run in lac_runs:
        run_dir = args.results_dir / run["results_subdir"]
        if not run_dir.exists():
            print(f"\n⚠ Skipping {run['label']}: directory not found ({run_dir})")
            continue
        result = evaluate_strategy(run, args.results_dir, args.gt_mask_dir, args.output_dir, gt_annotations)
        all_results.append(result)

    if not all_results:
        print("No runs evaluated!")
        sys.exit(1)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(all_results, args.output_dir, args.results_dir, gt_annotations)

    # Save combined results
    combined_path = args.output_dir / "all_lac_evaluation.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Run':<45} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Dice':>6} {'Time':>8}")
    print("-" * 80)
    for r in all_results:
        t = r.get("timing", {}).get("mean_total_time", 0)
        print(f"{r['label']:<45} {r['mean_iou']:>6.3f} {r['mean_f1']:>6.3f} "
              f"{r['mean_precision']:>6.3f} {r['mean_recall']:>6.3f} {r['mean_dice']:>6.3f} {t:>7.1f}s")
    print("=" * 80)

    # Combined CSV
    csv_path = args.output_dir / "all_lac_per_image_metrics.csv"
    csv_fields = ["short", "gt_name", "folder", "image_id", "iou", "precision", "recall",
                  "f1", "dice", "accuracy", "pred_coverage_pct", "gt_coverage_pct",
                  "num_masks", "num_vlm_areas", "total_time", "reasoner_time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            for per in r["per_image"]:
                row = {"short": r["short"], **per}
                writer.writerow(row)

    print(f"\nResults saved to {args.output_dir}")
    print(f"  Combined JSON: {combined_path}")
    print(f"  Combined CSV:  {csv_path}")


if __name__ == "__main__":
    main()
