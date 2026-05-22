#!/usr/bin/env python3
"""
Compare LaC navigable-area pipeline strategies across models and modes.

Generates comparison tables, charts, and reports for the 4 LaC navigable runs:
  - Qwen2.5-VL-7B-Instruct: rgb_only, rgb_depth_overlay
  - gemma-4-E4B-it:         rgb_only, rgb_depth_overlay

Comparison dimensions:
  - Model comparison:  Qwen vs Gemma (per mode)
  - Mode comparison:   rgb_only vs rgb_depth_overlay (per model)
  - Overall ranking

Ground truth masks: sam3_output_v7/*_mask.png

Usage:
    python3 compare_lac.py
    python3 compare_lac.py --results_dir /path/to/free_ground_results
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
DEFAULT_OUTPUT_DIR = WORK_DIR / "free_ground_results" / "lac_navigable_comparison"


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
                image_id = mask_file.stem.replace("_mask", "")
                gt_key = f"{folder_name}/{image_id}"
                annotations[gt_key] = (folder_name, image_id)
        if annotations:
            return annotations

    # Fallback: flat format with {prefix}_{image_id}_mask.png
    SAM3_MAPPING = {
        "LA_Downstairs_image28": ("lms_kamal_LA_downstairs_Nopeople_1", "image28"),
        "LA_Upstairs_image188":  ("lms_kamal_LA_upstairs_Nopeople_1", "image188"),
        "LA_Upstairs_image86":   ("lms_kamal_LA_upstairs_Nopeople_1", "image86"),
        "LB_Upstairs_image147":  ("lms_kamal_LB_upstairs_Nopeople_2", "image147"),
        "RA_Downstairs_image36": ("lms_kamal_RA_downstairs_Nopeople_1", "image36"),
        "RA_Upstairs_image28":   ("lms_kamal_RA_upstairs_Nopeople_1", "image28"),
        "RB_Downstairs_image95": ("lms_kamal_RB_downstairs_Nopeople_1", "image95"),
    }
    for mask_file in sorted(gt_mask_dir.glob("*_mask.png")):
        stem = mask_file.stem.replace("_mask", "")
        if stem in SAM3_MAPPING:
            annotations[stem] = SAM3_MAPPING[stem]

    return annotations

MODELS = ["Qwen2.5-VL-7B-Instruct", "gemma-4-E4B-it"]
MODES = ["rgb_only", "rgb_depth_separate"]

_MODE_LABELS = {
    "rgb_only": "RGB only",
    "rgb_depth_separate": "RGB+D separate",
}


def _build_strategies(seg_methods=None):
    """Build strategy definitions for comparison.

    Args:
        seg_methods: List of segmentation methods to include.
                     Defaults to ["grounding_dino", "sam3"].
                     Pass ["sam3"] for SAM3-only comparison.
    """
    if seg_methods is None:
        seg_methods = ["grounding_dino", "sam3"]
    strategies = []
    for model in MODELS:
        for mode in MODES:
            mode_label = _MODE_LABELS.get(mode, mode)
            for seg_method in seg_methods:
                seg_label = "G-DINO+SAM" if seg_method == "grounding_dino" else "SAM3"
                strategies.append({
                    "label": f"{model} — {mode_label}, {seg_label}",
                    "short": f"{model}_{mode}_{seg_method}",
                    "model": model,
                    "mode": mode,
                    "results_subdir": f"{model}_LaC_navigable/{mode}_{seg_method}",
                })
    return strategies


STRATEGIES = _build_strategies()

COLORS = {
    "Qwen2.5-VL-7B-Instruct": {
        "rgb_only": "#1976D2",
        "rgb_depth_separate": "#0D47A1",
    },
    "gemma-4-E4B-it": {
        "rgb_only": "#D32F2F",
        "rgb_depth_separate": "#B71C1C",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions (shared with evaluate_lac.py)
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
    """Combine individual mask PNGs into a single binary mask.
    Args:
        image_id: If provided, only combine masks starting with ``{image_id}_mask_``.
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
    dice = 2 * intersection / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "iou": round(float(iou), 4), "precision": round(float(precision), 4),
        "recall": round(float(recall), 4), "f1": round(float(f1), 4),
        "dice": round(float(dice), 4), "accuracy": round(float(accuracy), 4),
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
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(strategy: Dict, results_dir: Path, gt_mask_dir: Path, gt_annotations: Dict) -> List[Dict]:
    """Evaluate a single strategy, return per-image results."""
    run_dir = results_dir / strategy["results_subdir"]
    if not run_dir.exists():
        print(f"  ⚠ Not found: {run_dir}")
        return []

    results = []
    for gt_name, (folder, image_id) in gt_annotations.items():
        # Support both subfolder and flat GT mask formats
        gt_mask_path = gt_mask_dir / folder / f"{image_id}_mask.png"
        if not gt_mask_path.exists():
            gt_mask_path = gt_mask_dir / f"{gt_name}_mask.png"
        if not gt_mask_path.exists():
            continue
        gt_mask = load_gt_mask(gt_mask_path)
        h, w = gt_mask.shape

        # Load prediction
        mask_dir = run_dir / folder / "masks"
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        num_masks = 0
        if mask_dir.exists():
            mask_pngs = [f for f in mask_dir.glob(f"{image_id}_mask_*.png")
                         if "overlay" not in f.name]
            if mask_pngs:
                pred_mask = combine_predicted_masks(mask_dir, h, w, image_id=image_id)
                num_masks = len(mask_pngs)

        metrics = compute_metrics(pred_mask, gt_mask)

        # Load timing from JSON
        json_path = run_dir / folder / f"{image_id}_lac_analysis.json"
        timing = {}
        if json_path.exists():
            with open(json_path) as f:
                analysis = json.load(f)
            r_time = analysis.get("reasoner", {}).get("inference_time")
            s_time = analysis.get("segmentation", {}).get("inference_time")
            c_time = analysis.get("cost_map", {}).get("inference_time")
            total = sum(x for x in [r_time, s_time, c_time] if x is not None)
            timing = {
                "reasoner_time": r_time, "seg_time": s_time,
                "costmap_time": c_time, "total_time": round(total, 2) if total else None,
            }

        results.append({
            "gt_name": gt_name, "folder": folder, "image_id": image_id,
            "strategy": strategy["short"], "model": strategy["model"],
            "mode": strategy["mode"], "num_masks": num_masks,
            **metrics, **timing,
        })
    return results


def load_or_compute_evaluations(results_dir: Path, gt_mask_dir: Path, gt_annotations: Dict,
                                strategies: list = None) -> Dict[str, List[Dict]]:
    """Evaluate all strategies, return {short: [per_image_results]}."""
    if strategies is None:
        strategies = STRATEGIES
    all_evals = {}
    for s in strategies:
        print(f"Evaluating {s['label']}...")
        evals = evaluate_strategy(s, results_dir, gt_mask_dir, gt_annotations)
        if evals:
            all_evals[s["short"]] = evals
    return all_evals


# ──────────────────────────────────────────────────────────────────────────────
# Comparison tables
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_tables(all_evals: Dict[str, List[Dict]]):
    """Print formatted comparison tables."""
    print("\n" + "=" * 90)
    print("TABLE 1: Overall Strategy Comparison")
    print("=" * 90)
    header = f"{'Strategy':<40} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Dice':>6} {'Acc':>6} {'Time':>7}"
    print(header)
    print("-" * 90)

    agg_data = {}
    for short, evals in all_evals.items():
        ious = [e["iou"] for e in evals]
        agg = {
            "mean_iou": np.mean(ious), "mean_f1": np.mean([e["f1"] for e in evals]),
            "mean_prec": np.mean([e["precision"] for e in evals]),
            "mean_rec": np.mean([e["recall"] for e in evals]),
            "mean_dice": np.mean([e["dice"] for e in evals]),
            "mean_acc": np.mean([e["accuracy"] for e in evals]),
        }
        times = [e["total_time"] for e in evals if e.get("total_time") is not None]
        agg["mean_time"] = np.mean(times) if times else 0
        agg_data[short] = agg
        t = agg["mean_time"]
        print(f"{short:<40} {agg['mean_iou']:>6.3f} {agg['mean_f1']:>6.3f} "
              f"{agg['mean_prec']:>6.3f} {agg['mean_rec']:>6.3f} {agg['mean_dice']:>6.3f} "
              f"{agg['mean_acc']:>6.3f} {t:>6.1f}s")
    print("=" * 90)

    # Table 2: Model comparison (averaged over modes)
    print("\n" + "=" * 60)
    print("TABLE 2: Model Comparison (avg over modes)")
    print("=" * 60)
    print(f"{'Model':<30} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print("-" * 60)
    for model in MODELS:
        model_evals = [e for s, evals in all_evals.items() for e in evals if e["model"] == model]
        if model_evals:
            print(f"{model:<30} {np.mean([e['iou'] for e in model_evals]):>6.3f} "
                  f"{np.mean([e['f1'] for e in model_evals]):>6.3f} "
                  f"{np.mean([e['precision'] for e in model_evals]):>6.3f} "
                  f"{np.mean([e['recall'] for e in model_evals]):>6.3f}")
    print("=" * 60)

    # Table 3: Mode comparison (averaged over models)
    print("\n" + "=" * 60)
    print("TABLE 3: Mode Comparison (avg over models)")
    print("=" * 60)
    print(f"{'Mode':<25} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print("-" * 60)
    for mode in MODES:
        mode_evals = [e for s, evals in all_evals.items() for e in evals if e["mode"] == mode]
        if mode_evals:
            print(f"{mode:<25} {np.mean([e['iou'] for e in mode_evals]):>6.3f} "
                  f"{np.mean([e['f1'] for e in mode_evals]):>6.3f} "
                  f"{np.mean([e['precision'] for e in mode_evals]):>6.3f} "
                  f"{np.mean([e['recall'] for e in mode_evals]):>6.3f}")
    print("=" * 60)

    # Table 4: Per-image IoU breakdown
    gt_names = sorted(set(k for evals in all_evals.values() for r in evals for k in [r["gt_name"]]))
    print("\n" + "=" * 100)
    print("TABLE 4: Per-Image IoU")
    print("=" * 100)
    header = f"{'GT Image':<30}" + "".join(f"{s.replace('_','\n')[:20]:>12}" for s in all_evals.keys())
    print(header)
    print("-" * 100)
    for gt_name in gt_names:
        row = f"{gt_name:<30}"
        for short, evals in all_evals.items():
            per = next((e for e in evals if e["gt_name"] == gt_name), None)
            iou = per["iou"] if per else 0.0
            row += f"{iou:>12.3f}"
        print(row)
    print("=" * 100)

    return agg_data


# ──────────────────────────────────────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────────────────────────────────────

def generate_comparison_visualizations(all_evals: Dict[str, List[Dict]], output_dir: Path):
    """Generate comparison charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping charts")
        return

    vis_dir = output_dir / "charts"
    vis_dir.mkdir(parents=True, exist_ok=True)
    gt_names = sorted(set(k for evals in all_evals.values() for r in evals for k in [r["gt_name"]]))

    # ── 1. Grouped bar chart: IoU per image ──
    fig, ax = plt.subplots(figsize=(14, 6))
    n_strategies = len(all_evals)
    n_images = len(gt_names)
    bar_width = 0.8 / max(n_strategies, 1)
    x = np.arange(n_images)

    for i, (short, evals) in enumerate(all_evals.items()):
        ious = []
        for gt_name in gt_names:
            per = next((e for e in evals if e["gt_name"] == gt_name), None)
            ious.append(per["iou"] if per else 0.0)
        model = evals[0]["model"] if evals else ""
        mode = evals[0]["mode"] if evals else ""
        color = COLORS.get(model, {}).get(mode, f"C{i}")
        label = short.replace("_", " ")
        ax.bar(x + i * bar_width, ious, bar_width, label=label, color=color, alpha=0.85, edgecolor="black")

    ax.set_xticks(x + bar_width * (n_strategies - 1) / 2)
    ax.set_xticklabels([n.replace("_", "\n") for n in gt_names], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title("LaC Navigable — Per-Image IoU by Strategy")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(vis_dir / "per_image_iou_comparison.png", dpi=150)
    plt.close()

    # ── 2. Model comparison (avg over modes) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, metric in enumerate(["iou", "f1"]):
        ax = axes[ax_idx]
        model_vals = {}
        for model in MODELS:
            vals = [e[metric] for s, evals in all_evals.items() for e in evals if e["model"] == model]
            model_vals[model] = np.mean(vals) if vals else 0

        colors_m = ["#1976D2" if "Qwen" in m else "#D32F2F" for m in model_vals.keys()]
        bars = ax.bar(model_vals.keys(), model_vals.values(), color=colors_m, alpha=0.85, edgecolor="black")
        for bar, val in zip(bars, model_vals.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Model Comparison — Mean {metric.upper()}")
        ax.set_ylim(0, max(max(model_vals.values(), default=0.5) * 1.3, 0.5))
    plt.tight_layout()
    plt.savefig(vis_dir / "model_comparison.png", dpi=150)
    plt.close()

    # ── 3. Mode comparison (avg over models) ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax_idx, metric in enumerate(["iou", "f1"]):
        ax = axes[ax_idx]
        mode_vals = {}
        for mode in MODES:
            vals = [e[metric] for s, evals in all_evals.items() for e in evals if e["mode"] == mode]
            mode_vals[mode] = np.mean(vals) if vals else 0

        colors_mo = ["#4CAF50", "#FF9800"]
        bars = ax.bar(mode_vals.keys(), mode_vals.values(), color=colors_mo, alpha=0.85, edgecolor="black")
        for bar, val in zip(bars, mode_vals.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Mode Comparison — Mean {metric.upper()}")
        ax.set_ylim(0, max(max(mode_vals.values(), default=0.5) * 1.3, 0.5))
    plt.tight_layout()
    plt.savefig(vis_dir / "mode_comparison.png", dpi=150)
    plt.close()

    # ── 4. Speed vs Accuracy scatter ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for short, evals in all_evals.items():
        ious = [e["iou"] for e in evals]
        times = [e["total_time"] for e in evals if e.get("total_time") is not None]
        if times:
            mean_iou = np.mean(ious)
            mean_time = np.mean(times)
            model = evals[0]["model"]
            color = "#1976D2" if "Qwen" in model else "#D32F2F"
            marker = "o" if "rgb_only" in short else "s"
            ax.scatter(mean_time, mean_iou, s=150, c=color, marker=marker, edgecolors="black",
                       label=short.replace("_", " "), zorder=5)
            ax.annotate(short.replace("_", "\n"), (mean_time, mean_iou),
                        textcoords="offset points", xytext=(10, 5), fontsize=7)
    ax.set_xlabel("Mean Inference Time (seconds)")
    ax.set_ylabel("Mean IoU")
    ax.set_title("LaC Navigable — Speed vs Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(vis_dir / "speed_vs_accuracy.png", dpi=150)
    plt.close()

    # ── 5. Metric heatmap ──
    metrics_to_show = ["iou", "precision", "recall", "f1", "dice"]
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    row_labels = []
    for short, evals in all_evals.items():
        row_labels.append(short.replace("_", " "))
        row = [np.mean([e[m] for e in evals]) for m in metrics_to_show]
        data.append(row)
    data = np.array(data)
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=max(0.5, data.max()))
    ax.set_xticks(range(len(metrics_to_show)))
    ax.set_xticklabels([m.upper() for m in metrics_to_show])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(len(row_labels)):
        for j in range(len(metrics_to_show)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=9,
                    color="black" if data[i,j] > 0.5 else "gray")
    plt.colorbar(im, ax=ax)
    ax.set_title("LaC Navigable — Metric Heatmap")
    plt.tight_layout()
    plt.savefig(vis_dir / "metric_heatmap.png", dpi=150)
    plt.close()

    print(f"Charts saved to {vis_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────────────

def save_comparison_results(all_evals: Dict[str, List[Dict]], agg_data: Dict, output_dir: Path):
    """Save comparison results as JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "num_strategies": len(all_evals),
        "num_gt_images": len(set(k for evals in all_evals.values() for r in evals for k in [r["gt_name"]])),
        "aggregate": {short: {k: round(float(v), 4) for k, v in agg.items()}
                      for short, agg in agg_data.items()},
        "per_image": {short: evals for short, evals in all_evals.items()},
    }
    with open(output_dir / "lac_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Per-image CSV
    csv_path = output_dir / "lac_per_image_comparison.csv"
    fields = ["strategy", "model", "mode", "gt_name", "folder", "image_id",
              "iou", "precision", "recall", "f1", "dice", "accuracy",
              "pred_coverage_pct", "gt_coverage_pct", "num_masks", "total_time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for short, evals in all_evals.items():
            for e in evals:
                writer.writerow(e)

    # Aggregate CSV
    agg_csv = output_dir / "lac_aggregate_comparison.csv"
    agg_fields = ["short", "mean_iou", "mean_f1", "mean_prec", "mean_rec",
                  "mean_dice", "mean_acc", "mean_time"]
    with open(agg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for short, agg in agg_data.items():
            writer.writerow({"short": short, **agg})

    print(f"\nResults saved to {output_dir}")
    print(f"  JSON: {output_dir / 'lac_comparison.json'}")
    print(f"  Per-image CSV: {csv_path}")
    print(f"  Aggregate CSV: {agg_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, log_path: Path):
        self.log = open(log_path, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.log.write(data)

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description="Compare LaC navigable pipeline strategies")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--gt_mask_dir", type=Path, default=DEFAULT_GT_MASK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seg_method", type=str, nargs="+",
                        default=["grounding_dino", "sam3"],
                        choices=["grounding_dino", "sam3"],
                        help="Segmentation method(s) to compare (default: both). "
                             "Use --seg_method sam3 for SAM3-only comparison.")
    args = parser.parse_args()

    # Build strategies for the requested segmentation methods
    strategies = _build_strategies(seg_methods=args.seg_method)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Tee output to log
    log_path = args.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = Tee(log_path)
    sys.stdout = tee

    print("=" * 60)
    print("LaC NAVIGABLE PIPELINE — STRATEGY COMPARISON")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")
    print(f"GT mask dir: {args.gt_mask_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Seg methods: {args.seg_method}")

    # Discover GT annotations
    gt_annotations = discover_gt_annotations(args.gt_mask_dir)
    print(f"Strategies:  {len(strategies)}")
    print(f"GT images:   {len(gt_annotations)} (auto-discovered)")
    print()

    # Evaluate all strategies
    all_evals = load_or_compute_evaluations(
        args.results_dir, args.gt_mask_dir, gt_annotations, strategies=strategies,
    )
    if not all_evals:
        print("No strategies could be evaluated!")
        sys.exit(1)

    # Print comparison tables
    agg_data = print_comparison_tables(all_evals)

    # Generate visualizations
    print("\nGenerating comparison visualizations...")
    generate_comparison_visualizations(all_evals, args.output_dir)

    # Save results
    save_comparison_results(all_evals, agg_data, args.output_dir)

    # Restore stdout
    sys.stdout = tee.stdout
    tee.close()
    print(f"Comparison complete. Log saved to {log_path}")


if __name__ == "__main__":
    main()
