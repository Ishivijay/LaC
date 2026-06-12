#!/usr/bin/env python3
"""
Evaluate Free Ground Detection Pipeline runs against ground truth masks.

Discovers runs from the output directory structure:
    {strategy}/{model_tag}/{input_mode}_sam3/

Strategies: zero_shot, few_shot, two_vlm
Models: Qwen, Gemma, Qwen_Gemma (for two_vlm with different models)
Modes: rgb_only, rgb_depth_separate

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
DEFAULT_GT_MASK_DIR = WORK_DIR / "mlp_ground_truth" / "proper_annotations"
DEFAULT_DATA_DIR = (
    WORK_DIR / "mlp_dataset" / "prospthesisproject-Data" / "Code" / "Data"
)
DEFAULT_OUTPUT_DIR = WORK_DIR / "free_ground_results" / "evaluation_v2"

STRATEGIES = ["zero_shot", "few_shot", "two_vlm"]
MODELS = ["Qwen", "Gemma"]
MODES = ["rgb_only", "rgb_depth_separate"]


def discover_gt_annotations(gt_mask_dir: Path) -> Dict[str, tuple]:
    """Auto-discover GT masks from directory structure.

    Returns dict: gt_key → (folder_name, image_id)
    """
    annotations = {}
    subfolders = [d for d in gt_mask_dir.iterdir() if d.is_dir()]
    if subfolders:
        for subfolder in sorted(subfolders):
            folder_name = subfolder.name
            for mask_file in sorted(subfolder.glob("*_mask.png")):
                image_id = mask_file.stem.replace("_mask", "")
                gt_key = f"{folder_name}/{image_id}"
                annotations[gt_key] = (folder_name, image_id)
        if annotations:
            return annotations
    return annotations


def discover_runs(results_dir: Path, strategy_filter: str = None,
                   model_filter: str = None, input_mode_filter: str = None) -> List[Dict]:
    """Discover all pipeline runs from the results directory.

    Scans: {results_dir}/{strategy}/{model_tag}/{input_mode}/
    
    Args:
        results_dir: Directory containing pipeline results
        strategy_filter: Filter by strategy (e.g., sa2va, zero_shot)
        model_filter: Filter by model tag (e.g., Sa2VA-Qwen3-VL-4B)
        input_mode_filter: Filter by input mode (e.g., rgb_only, rgb_only_sa2va)
    """
    runs = []
    if not results_dir.exists():
        return runs

    VALID_MODES = {"rgb_only", "rgb_depth_separate", "rgb_only_sa2va", "rgb_depth_separate_sa2va"}
    skip_dirs = {"slurm_logs", "logs", "evaluation", "comparison", "evaluation_v2",
                 "comparison_v2", "Annotated_Ground_Truth",
                 "lac_navigable_evaluation", "lac_navigable_comparison",
                 "lac_navigable_evaluation_v2", "lac_navigable_comparison_v2"}

    for strategy_dir in sorted(results_dir.iterdir()):
        if not strategy_dir.is_dir() or strategy_dir.name in skip_dirs:
            continue
        strategy = strategy_dir.name
        
        # Filter by strategy
        if strategy_filter and strategy != strategy_filter:
            continue

        for model_dir in sorted(strategy_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_tag = model_dir.name
            
            # Filter by model
            if model_filter and model_tag != model_filter:
                continue

            for mode_dir in sorted(model_dir.iterdir()):
                if not mode_dir.is_dir() or mode_dir.name not in VALID_MODES:
                    continue
                input_mode = mode_dir.name
                
                # Filter by input mode
                if input_mode_filter and input_mode != input_mode_filter:
                    continue

                # Determine model names from tag
                parts = model_tag.split("_")
                if len(parts) == 2:
                    reasoner_model = parts[0]
                    evaluator_model = parts[1]
                else:
                    reasoner_model = model_tag
                    evaluator_model = model_tag

                label = f"{model_tag} — {strategy} ({input_mode})"
                short = f"{model_tag}_{strategy}_{input_mode}"

                runs.append({
                    "label": label,
                    "short": short,
                    "strategy": strategy,
                    "model_tag": model_tag,
                    "reasoner_model": reasoner_model,
                    "evaluator_model": evaluator_model,
                    "mode": input_mode,
                    "results_subdir": str(strategy_dir.relative_to(results_dir) / model_tag / mode_dir.name),
                })

    return runs


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def load_gt_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 127).astype(np.uint8)


def load_predicted_mask(mask_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape != target_size:
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize((target_size[1], target_size[0]), Image.NEAREST)
        mask = np.array(mask_img)
    return (mask > 127).astype(np.uint8)


def combine_predicted_masks(mask_dir: Path, target_h: int, target_w: int,
                            image_id: str = None) -> np.ndarray:
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
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = int(intersection)
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(pred.size - tp - fp - fn)

    iou = intersection / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "iou": round(float(iou), 4), "precision": round(float(precision), 4),
        "recall": round(float(recall), 4), "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "pred_coverage_pct": round(float(pred.sum() / pred.size * 100), 1),
        "gt_coverage_pct": round(float(gt.sum() / gt.size * 100), 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def load_rgb_image(folder: str, image_id: str) -> Optional[np.ndarray]:
    rgb_path = DEFAULT_DATA_DIR / folder / "sharpen_rgb" / "PNG" / f"{image_id}.png"
    if rgb_path.exists():
        return np.array(Image.open(rgb_path).convert("RGB"))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single_image(
    run: Dict, gt_name: str, folder: str, image_id: str,
    gt_mask: np.ndarray, results_dir: Path,
) -> Dict:
    h, w = gt_mask.shape
    run_dir = results_dir / run["results_subdir"]

    pred_mask = np.zeros((h, w), dtype=np.uint8)
    metadata = {}
    prediction_found = False
    num_vlm_areas = 0
    vlm_areas = []
    evaluator_areas = []

    # Load analysis JSON
    json_path = run_dir / folder / f"{image_id}_lac_analysis.json"
    if json_path.exists():
        with open(json_path) as f:
            analysis = json.load(f)
        reasoner = analysis.get("reasoner", {})
        num_vlm_areas = reasoner.get("num_areas", 0)
        r_time = reasoner.get("inference_time", None)
        s_time = analysis.get("segmentation", {}).get("inference_time", None)
        total_time = None
        if r_time is not None:
            total_time = r_time
            if s_time is not None:
                total_time += s_time

        # Extract VLM areas with bboxes
        vlm_areas = analysis.get("vlm_areas", [])

        # Extract evaluator areas with scores (two_vlm strategy)
        evaluator_data = analysis.get("evaluator", {})
        evaluator_areas = evaluator_data.get("areas_with_scores", [])

        metadata = {
            "num_vlm_areas": num_vlm_areas,
            "reasoner_time": r_time,
            "seg_time": s_time,
            "total_time": round(total_time, 2) if total_time else None,
        }

    # Load masks
    mask_dir = run_dir / folder / "masks"
    mask_pngs = []
    if mask_dir.exists():
        mask_pngs = sorted([f for f in mask_dir.glob(f"{image_id}_mask_*.png")
                           if "overlay" not in f.name])
        if mask_pngs:
            pred_mask = combine_predicted_masks(mask_dir, h, w, image_id=image_id)
            prediction_found = True

    metrics = compute_metrics(pred_mask, gt_mask) if prediction_found else {
        "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "accuracy": 0.0,
        "pred_coverage_pct": 0.0,
        "gt_coverage_pct": round(float(gt_mask.sum() / gt_mask.size * 100), 1),
        "tp": 0, "fp": 0, "fn": int(gt_mask.sum()), "tn": int(gt_mask.size - gt_mask.sum()),
    }

    return {
        "gt_name": gt_name, "folder": folder, "image_id": image_id,
        "prediction_found": prediction_found,
        "num_masks": len(mask_pngs) if prediction_found else 0,
        "num_vlm_areas": num_vlm_areas,
        "vlm_areas": vlm_areas,
        "evaluator_areas": evaluator_areas,
        **metrics, **metadata,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-strategy evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(run: Dict, results_dir: Path, gt_mask_dir: Path,
                      output_base: Path, gt_annotations: Dict) -> Dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {run['label']}")
    print(f"  subdir: {run['results_subdir']}")
    print(f"{'='*60}")

    image_results = []
    for gt_name, (folder, image_id) in gt_annotations.items():
        gt_mask_path = gt_mask_dir / folder / f"{image_id}_mask.png"
        if not gt_mask_path.exists():
            continue
        gt_mask = load_gt_mask(gt_mask_path)
        result = evaluate_single_image(run, gt_name, folder, image_id, gt_mask, results_dir)
        image_results.append(result)

        iou = result["iou"]
        f1 = result["f1"]
        status = "✓" if result["prediction_found"] else "✗"
        t = result.get("total_time", "?")
        print(f"  {status} {gt_name}: IoU={iou:.3f}, F1={f1:.3f}, time={t}s")

    if not image_results:
        print("  No images evaluated!")
        return {"label": run["label"], "short": run["short"], "error": "no_images"}

    # Aggregate
    agg = {
        "label": run["label"], "short": run["short"],
        "strategy": run["strategy"], "model_tag": run["model_tag"],
        "mode": run["mode"],
        "num_images": len(image_results),
        "predictions_found": sum(1 for r in image_results if r["prediction_found"]),
        "mean_iou": round(float(np.mean([r["iou"] for r in image_results])), 4),
        "mean_precision": round(float(np.mean([r["precision"] for r in image_results])), 4),
        "mean_recall": round(float(np.mean([r["recall"] for r in image_results])), 4),
        "mean_f1": round(float(np.mean([r["f1"] for r in image_results])), 4),
        "mean_accuracy": round(float(np.mean([r["accuracy"] for r in image_results])), 4),
        "per_image": image_results,
    }

    total_times = [r["total_time"] for r in image_results if r.get("total_time") is not None]
    if total_times:
        agg["timing"] = {
            "mean_total_time": round(float(np.mean(total_times)), 2),
            "median_total_time": round(float(np.median(total_times)), 2),
        }

    # Save per-strategy results
    strategy_dir = output_base / run["short"]
    strategy_dir.mkdir(parents=True, exist_ok=True)
    with open(strategy_dir / "evaluation.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)

    csv_path = strategy_dir / "per_image_metrics.csv"
    csv_fields = ["gt_name", "folder", "image_id", "iou", "precision", "recall",
                  "f1", "accuracy", "pred_coverage_pct", "gt_coverage_pct",
                  "num_masks", "num_vlm_areas", "total_time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in image_results:
            writer.writerow(r)

    print(f"\n  Summary: mean_IoU={agg['mean_iou']:.4f}, mean_F1={agg['mean_f1']:.4f}")
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline runs against GT")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--gt_mask_dir", type=Path, default=DEFAULT_GT_MASK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", type=str, default=None,
                        help="Filter by strategy (e.g., sa2va, zero_shot, few_shot, two_vlm)")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model tag (e.g., Sa2VA-Qwen3-VL-4B)")
    parser.add_argument("--input_mode", type=str, default=None,
                        help="Filter by input mode (e.g., rgb_only, rgb_depth_separate, rgb_only_sa2va)")
    args = parser.parse_args()

    # Discover runs with filters
    runs = discover_runs(args.results_dir,
                         strategy_filter=args.strategy,
                         model_filter=args.model,
                         input_mode_filter=args.input_mode)

    print("=" * 60)
    print("FREE GROUND DETECTION — EVALUATION")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")
    print(f"GT mask dir: {args.gt_mask_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Runs found:  {len(runs)}")

    if not runs:
        print("No runs found! Check results directory structure.")
        sys.exit(1)

    for run in runs:
        print(f"  {run['label']}  →  {run['results_subdir']}")

    # Discover GT annotations
    gt_annotations = discover_gt_annotations(args.gt_mask_dir)
    print(f"GT images:   {len(gt_annotations)}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each run
    all_results = []
    for run in runs:
        run_dir = args.results_dir / run["results_subdir"]
        if not run_dir.exists():
            print(f"\n⚠ Skipping {run['label']}: directory not found")
            continue
        result = evaluate_strategy(run, args.results_dir, args.gt_mask_dir,
                                   args.output_dir, gt_annotations)
        all_results.append(result)

    if not all_results:
        print("No runs evaluated!")
        sys.exit(1)

    # Save combined results
    combined_path = args.output_dir / "all_evaluation.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Run':<50} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['label']:<50} {r['mean_iou']:>6.3f} {r['mean_f1']:>6.3f} "
              f"{r['mean_precision']:>6.3f} {r['mean_recall']:>6.3f}")
    print("=" * 90)

    # Combined CSV
    csv_path = args.output_dir / "all_per_image_metrics.csv"
    csv_fields = ["short", "gt_name", "folder", "image_id", "iou", "precision", "recall",
                  "f1", "accuracy", "pred_coverage_pct", "gt_coverage_pct",
                  "num_masks", "num_vlm_areas", "total_time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            for per in r.get("per_image", []):
                writer.writerow({"short": r["short"], **per})

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
