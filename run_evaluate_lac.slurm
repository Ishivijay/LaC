#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Run LaC Navigable Evaluation & Comparison (no GPU needed)
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash run_evaluate_lac.slurm              # submit both jobs
#   bash run_evaluate_lac.slurm --dry-run     # show commands without submitting
#
# Output:
#   $WORK/free_ground_results/lac_navigable_evaluation/
#   $WORK/free_ground_results/lac_navigable_comparison/
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK:-/home/woody/iwnt/iwnt164h}"
PY="/usr/bin/python3"
RESULTS_DIR="$WORK_DIR/free_ground_results"
GT_DIR="/home/woody/iwnt/iwnt164h/mlp_dataset/prospthesisproject-Data/Code/annotated_samples/sam3_output_v7"

DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
    esac
done

mkdir -p "$RESULTS_DIR/slurm_logs"

if $DRY_RUN; then
    echo "DRY-RUN: Would submit 2 evaluation jobs"
    echo ""
    echo "  1. evaluate_lac.py → $RESULTS_DIR/lac_navigable_evaluation/"
    echo "  2. compare_lac.py  → $RESULTS_DIR/lac_navigable_comparison/"
    exit 0
fi

# ── Job 1: Per-strategy evaluation ──
EVAL_LOG="$RESULTS_DIR/slurm_logs/lac_evaluate_%j.out"
JOB1=$(sbatch \
    --partition=work \
    --gres=gpu:1 \
    --time=00:30:00 \
    --job-name=lac_evaluate \
    --output="$EVAL_LOG" \
    --export=ALL \
    --wrap="cd $SCRIPT_DIR && $PY evaluate_lac.py \
        --results_dir $RESULTS_DIR \
        --gt_mask_dir $GT_DIR \
        --output_dir $RESULTS_DIR/lac_navigable_evaluation" \
    | tail -1 | awk '{print $NF}')
echo "Submitted evaluate_lac → Job ID: $JOB1"

# ── Job 2: Cross-strategy comparison (depends on job 1) ──
COMP_LOG="$RESULTS_DIR/slurm_logs/lac_compare_%j.out"
JOB2=$(sbatch \
    --partition=work \
    --gres=gpu:1 \
    --time=00:30:00 \
    --job-name=lac_compare \
    --output="$COMP_LOG" \
    --export=ALL \
    --dependency=afterok:$JOB1 \
    --wrap="cd $SCRIPT_DIR && $PY compare_lac.py \
        --results_dir $RESULTS_DIR \
        --gt_mask_dir $GT_DIR \
        --output_dir $RESULTS_DIR/lac_navigable_comparison" \
    | tail -1 | awk '{print $NF}')
echo "Submitted compare_lac → Job ID: $JOB2 (depends on $JOB1)"

echo ""
echo "Output directories:"
echo "  Evaluation: $RESULTS_DIR/lac_navigable_evaluation/"
echo "  Comparison: $RESULTS_DIR/lac_navigable_comparison/"
echo ""
echo "Monitor: squeue -u \$(whoami)"
