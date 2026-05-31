#!/usr/bin/env python3
"""
Compare Free Ground Detection Pipeline strategies across models and modes.

Generates comparison tables, charts, and reports for all discovered runs.

Comparison dimensions:
  - Strategy: zero_shot vs few_shot vs two_vlm
  - Model: Qwen vs Gemma vs Qwen+Gemma
  - Mode: rgb_only vs rgb_depth_separate

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
DEFAULT_GT_MASK_DIR = WORK_DIR / "mlp_ground_truth" / "proper_annotations"
DEFAULT_DATA_DIR = (
    WORK_DIR / "mlp_dataset" / "prospthesisproject-Data" / "Code" / "Data"
)
DEFAULT_OUTPUT_DIR = WORK_DIR / "free_ground_results" / "comparison_v2"

STRATEGIES = ["zero_shot", "few_shot", "two_vlm"]
MODELS = ["Qwen", "Gemma"]
MODES = ["rgb_only", "rgb_depth_separate"]

COLORS = {
    "zero_shot": "#2196F3",
    "few_shot": "#4CAF50",
    "two_vlm": "#FF5722",
}


def discover_gt_annotations(gt_mask_dir: Path) -> Dict[str, tuple]:
    annotations = {}
    subfolders = [d for d in gt_mask_dir.iterdir() if d.is_dir()]
    if subfolders:
        for subfolder in sorted(subfolders):
            for mask_file in sorted(subfolder.glob("*_mask.png")):
                image_id = mask_file.stem.replace("_mask", "")
                gt_key = f"{subfolder.name}/{image_id}"
                annotations[gt_key] = (subfolder.name, image_id)
    return annotations


def discover_runs(results_dir: Path) -> List[Dict]:
    """Discover all pipeline runs from the results directory.

    Scans: {results_dir}/{strategy}/{model_tag}/{input_mode}/
    """
    runs = []
    if not results_dir.exists():
        return runs

    VALID_MODES = {"rgb_only", "rgb_depth_separate"}
    skip_dirs = {"slurm_logs", "logs", "evaluation", "comparison", "evaluation_v2",
                 "comparison_v2", "Annotated_Ground_Truth",
                 "lac_navigable_evaluation", "lac_navigable_comparison",
                 "lac_navigable_evaluation_v2", "lac_navigable_comparison_v2"}

    for strategy_dir in sorted(results_dir.iterdir()):
        if not strategy_dir.is_dir() or strategy_dir.name in skip_dirs:
            continue
        strategy = strategy_dir.name

        for model_dir in sorted(strategy_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_tag = model_dir.name

            for mode_dir in sorted(model_dir.iterdir()):
                if not mode_dir.is_dir() or mode_dir.name not in VALID_MODES:
                    continue
                input_mode = mode_dir.name

                parts = model_tag.split("_")
                reasoner_model = parts[0]
                evaluator_model = parts[1] if len(parts) == 2 else parts[0]

                runs.append({
                    "label": f"{model_tag} — {strategy} ({input_mode})",
                    "short": f"{model_tag}_{strategy}_{input_mode}",
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
    return (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)


def load_predicted_mask(mask_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape != target_size:
        mask = np.array(Image.fromarray(mask).resize(
            (target_size[1], target_size[0]), Image.NEAREST))
    return (mask > 127).astype(np.uint8)


def combine_predicted_masks(mask_dir: Path, target_h: int, target_w: int,
                            image_id: str = None) -> np.ndarray:
    combined = np.zeros((target_h, target_w), dtype=np.uint8)
    mask_files = [f for f in mask_dir.glob("*.png") if "overlay" not in f.name]
    if image_id:
        mask_files = [f for f in mask_files if f.name.startswith(f"{image_id}_mask_")]
    for mf in mask_files:
        combined = np.maximum(combined, load_predicted_mask(mf, (target_h, target_w)))
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


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(strategy: Dict, results_dir: Path,
                      gt_mask_dir: Path, gt_annotations: Dict) -> List[Dict]:
    run_dir = results_dir / strategy["results_subdir"]
    if not run_dir.exists():
        print(f"  ⚠ Not found: {run_dir}")
        return []

    results = []
    for gt_name, (folder, image_id) in gt_annotations.items():
        gt_mask_path = gt_mask_dir / folder / f"{image_id}_mask.png"
        if not gt_mask_path.exists():
            continue
        gt_mask = load_gt_mask(gt_mask_path)
        h, w = gt_mask.shape

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

        # Load timing
        json_path = run_dir / folder / f"{image_id}_lac_analysis.json"
        timing = {}
        if json_path.exists():
            with open(json_path) as f:
                analysis = json.load(f)
            r_time = analysis.get("reasoner", {}).get("inference_time")
            s_time = analysis.get("segmentation", {}).get("inference_time")
            total = sum(x for x in [r_time, s_time] if x is not None)
            timing = {
                "reasoner_time": r_time, "seg_time": s_time,
                "total_time": round(total, 2) if total else None,
            }

        results.append({
            "gt_name": gt_name, "folder": folder, "image_id": image_id,
            "strategy_name": strategy["short"], "strategy": strategy["strategy"],
            "model_tag": strategy["model_tag"], "mode": strategy["mode"],
            "num_masks": num_masks, **metrics, **timing,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Comparison tables
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_tables(all_evals: Dict[str, List[Dict]]):
    print("\n" + "=" * 90)
    print("TABLE 1: Overall Strategy Comparison")
    print("=" * 90)
    print(f"{'Strategy':<45} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Time':>7}")
    print("-" * 90)

    agg_data = {}
    for short, evals in all_evals.items():
        ious = [e["iou"] for e in evals]
        times = [e["total_time"] for e in evals if e.get("total_time") is not None]
        agg = {
            "mean_iou": float(np.mean(ious)),
            "mean_f1": float(np.mean([e["f1"] for e in evals])),
            "mean_prec": float(np.mean([e["precision"] for e in evals])),
            "mean_rec": float(np.mean([e["recall"] for e in evals])),
            "mean_time": float(np.mean(times)) if times else 0,
        }
        agg_data[short] = agg
        t = agg["mean_time"]
        print(f"{short:<45} {agg['mean_iou']:>6.3f} {agg['mean_f1']:>6.3f} "
              f"{agg['mean_prec']:>6.3f} {agg['mean_rec']:>6.3f} {t:>6.1f}s")
    print("=" * 90)

    # Table 2: Strategy comparison (avg over models/modes)
    strategies_found = sorted(set(e["strategy"] for evals in all_evals.values() for e in evals))
    if len(strategies_found) > 1:
        print("\n" + "=" * 60)
        print("TABLE 2: Strategy Comparison (avg over all)")
        print("=" * 60)
        print(f"{'Strategy':<20} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
        print("-" * 60)
        for strat in strategies_found:
            s_evals = [e for evals in all_evals.values() for e in evals if e["strategy"] == strat]
            if s_evals:
                print(f"{strat:<20} {np.mean([e['iou'] for e in s_evals]):>6.3f} "
                      f"{np.mean([e['f1'] for e in s_evals]):>6.3f} "
                      f"{np.mean([e['precision'] for e in s_evals]):>6.3f} "
                      f"{np.mean([e['recall'] for e in s_evals]):>6.3f}")
        print("=" * 60)

    # Table 3: Model comparison
    models_found = sorted(set(e["model_tag"] for evals in all_evals.values() for e in evals))
    if len(models_found) > 1:
        print("\n" + "=" * 60)
        print("TABLE 3: Model Comparison (avg over strategies/modes)")
        print("=" * 60)
        print(f"{'Model':<25} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
        print("-" * 60)
        for model in models_found:
            m_evals = [e for evals in all_evals.values() for e in evals if e["model_tag"] == model]
            if m_evals:
                print(f"{model:<25} {np.mean([e['iou'] for e in m_evals]):>6.3f} "
                      f"{np.mean([e['f1'] for e in m_evals]):>6.3f} "
                      f"{np.mean([e['precision'] for e in m_evals]):>6.3f} "
                      f"{np.mean([e['recall'] for e in m_evals]):>6.3f}")
        print("=" * 60)

    return agg_data


# ──────────────────────────────────────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────────────────────────────────────

def generate_comparison_visualizations(all_evals: Dict[str, List[Dict]], output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping charts")
        return

    vis_dir = output_dir / "charts"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Grouped bar chart: IoU per strategy
    fig, ax = plt.subplots(figsize=(12, 6))
    shorts = list(all_evals.keys())
    mean_ious = [np.mean([e["iou"] for e in all_evals[s]]) for s in shorts]
    colors = [COLORS.get(all_evals[s][0]["strategy"], "#999") for s in shorts]
    bars = ax.bar(range(len(shorts)), mean_ious, color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(shorts)))
    ax.set_xticklabels([s.replace("_", "\n") for s in shorts], fontsize=8, ha="center")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Mean IoU by Strategy/Model/Mode")
    for bar, val in zip(bars, mean_ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(vis_dir / "iou_comparison.png", dpi=150)
    plt.close()

    # 2. Metric heatmap
    metrics_to_show = ["iou", "precision", "recall", "f1", "accuracy"]
    fig, ax = plt.subplots(figsize=(10, max(4, len(shorts) * 0.8)))
    data = np.array([[np.mean([e[m] for e in all_evals[s]]) for m in metrics_to_show] for s in shorts])
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=max(0.5, data.max()))
    ax.set_xticks(range(len(metrics_to_show)))
    ax.set_xticklabels([m.upper() for m in metrics_to_show])
    ax.set_yticks(range(len(shorts)))
    ax.set_yticklabels([s.replace("_", " ") for s in shorts], fontsize=8)
    for i in range(len(shorts)):
        for j in range(len(metrics_to_show)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=9,
                    color="black" if data[i,j] > 0.5 else "gray")
    plt.colorbar(im, ax=ax)
    ax.set_title("Metric Heatmap")
    plt.tight_layout()
    plt.savefig(vis_dir / "metric_heatmap.png", dpi=150)
    plt.close()

    # 3. Strategy comparison: 4 approaches across metrics
    # Group: zero_shot, few_shot, two_vlm (same), two_vlm (diff)
    strategy_groups = {
        "Zero-Shot\n(1 VLM)": [],
        "Few-Shot\n(1 VLM + examples)": [],
        "Two-VLM\n(same model)": [],
        "Two-VLM\n(different models)": [],
    }
    for short, evals in all_evals.items():
        if not evals:
            continue
        strategy = evals[0].get("strategy", "")
        model_tag = evals[0].get("model_tag", "")
        if strategy == "zero_shot":
            strategy_groups["Zero-Shot\n(1 VLM)"].extend(evals)
        elif strategy == "few_shot":
            strategy_groups["Few-Shot\n(1 VLM + examples)"].extend(evals)
        elif strategy == "two_vlm":
            if "_" in model_tag:  # e.g., "Qwen_Gemma"
                strategy_groups["Two-VLM\n(different models)"].extend(evals)
            else:
                strategy_groups["Two-VLM\n(same model)"].extend(evals)

    # Filter empty groups
    strategy_groups = {k: v for k, v in strategy_groups.items() if v}

    if strategy_groups:
        strat_metrics = ["iou", "precision", "recall", "f1", "accuracy"]
        strat_names = list(strategy_groups.keys())
        strat_data = {}
        for metric in strat_metrics:
            strat_data[metric] = []
            for name in strat_names:
                evals = strategy_groups[name]
                strat_data[metric].append(np.mean([e[metric] for e in evals]) if evals else 0)

        x = np.arange(len(strat_names))
        width = 0.13
        strat_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

        fig, ax = plt.subplots(figsize=(14, 7))
        for i, metric in enumerate(strat_metrics):
            offset = (i - len(strat_metrics) / 2 + 0.5) * width
            bars = ax.bar(x + offset, strat_data[metric], width,
                          label=metric.upper(), color=strat_colors[i], alpha=0.85,
                          edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, strat_data[metric]):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(strat_names, fontsize=10)
        ax.set_ylabel("Score")
        ax.set_title("Strategy Comparison: Zero-Shot vs Few-Shot vs Two-VLM (same) vs Two-VLM (diff)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, max(max(v) for v in strat_data.values()) * 1.2 + 0.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(vis_dir / "strategy_comparison.png", dpi=150)
        plt.close()

        # 4. Radar/spider chart for strategy comparison
        angles = np.linspace(0, 2 * np.pi, len(strat_metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        radar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, name in enumerate(strat_names):
            values = [strat_data[m][i] for m in strat_metrics]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=name.replace("\n", " "),
                    color=radar_colors[i % len(radar_colors)])
            ax.fill(angles, values, alpha=0.1, color=radar_colors[i % len(radar_colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in strat_metrics], fontsize=10)
        ax.set_title("Strategy Comparison (Radar)", fontsize=12, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        plt.savefig(vis_dir / "strategy_radar.png", dpi=150)
        plt.close()

    print(f"Charts saved to {vis_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────────────

def save_comparison_results(all_evals: Dict[str, List[Dict]], agg_data: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_strategies": len(all_evals),
            "aggregate": {short: {k: round(float(v), 4) for k, v in agg.items()}
                          for short, agg in agg_data.items()},
            "per_image": {short: evals for short, evals in all_evals.items()},
        }, f, indent=2, default=str)

    csv_path = output_dir / "per_image_comparison.csv"
    fields = ["strategy_name", "strategy", "model_tag", "mode", "gt_name", "folder",
              "image_id", "iou", "precision", "recall", "f1", "dice", "accuracy",
              "pred_coverage_pct", "gt_coverage_pct", "num_masks", "total_time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for short, evals in all_evals.items():
            for e in evals:
                writer.writerow(e)

    print(f"\nResults saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

class Tee:
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
    parser = argparse.ArgumentParser(description="Compare pipeline strategies")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--gt_mask_dir", type=Path, default=DEFAULT_GT_MASK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Tee output to log
    log_path = args.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = Tee(log_path)
    sys.stdout = tee

    # Discover runs
    runs = discover_runs(args.results_dir)

    print("=" * 60)
    print("FREE GROUND DETECTION — STRATEGY COMPARISON")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")
    print(f"GT mask dir: {args.gt_mask_dir}")
    print(f"Runs found:  {len(runs)}")

    if not runs:
        print("No runs found!")
        sys.exit(1)

    # Discover GT annotations
    gt_annotations = discover_gt_annotations(args.gt_mask_dir)
    print(f"GT images:   {len(gt_annotations)}")
    print()

    # Evaluate all
    all_evals = {}
    for run in runs:
        print(f"Evaluating {run['label']}...")
        evals = evaluate_strategy(run, args.results_dir, args.gt_mask_dir, gt_annotations)
        if evals:
            all_evals[run["short"]] = evals

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

    sys.stdout = tee.stdout
    tee.close()
    print(f"Comparison complete. Log saved to {log_path}")


if __name__ == "__main__":
    main()
