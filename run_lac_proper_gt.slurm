#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LaC Pipeline with SAM3 — Run for new GT images (proper_annotations)
# ──────────────────────────────────────────────────────────────────────────────
# Runs the pipeline for images in the new GT directory:
#   $WORK/mlp_ground_truth/proper_annotations/
#
# Modes: rgb_only + rgb_depth_separate (NO overlay)
# Models: Qwen + Gemma
# Segmentation: SAM3
#
# Usage:
#   bash run_lac_proper_gt.slurm              # submit all 4 jobs
#   bash run_lac_proper_gt.slurm --dry-run     # show commands without submitting
#   bash run_lac_proper_gt.slurm --clean       # clean output dirs before running
#   bash run_lac_proper_gt.slurm --model Qwen2.5-VL-7B-Instruct  # single model
#   bash run_lac_proper_gt.slurm --mode rgb_only  # single mode
#
# Output stored in:
#   $WORK/free_ground_results/{model}_LaC_navigable/rgb_only_sam3/
#   $WORK/free_ground_results/{model}_LaC_navigable/rgb_depth_separate_sam3/
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAC_DIR="$SCRIPT_DIR"
WORK_DIR="${WORK:-/home/woody/iwnt/iwnt164h}"
PY="$WORK_DIR/venvs/marigold/bin/python3"
RESULTS_DIR="$WORK_DIR/free_ground_results"
GT_DIR="$WORK_DIR/mlp_ground_truth/proper_annotations"

MODELS=("Qwen2.5-VL-7B-Instruct" "gemma-4-E4B-it")
MODES=("rgb_only" "rgb_depth_separate")

CLEAN_FLAG=""
DRY_RUN=false
SINGLE_MODEL=""
SINGLE_MODE=""

for arg in "$@"; do
    case $arg in
        --dry-run)  DRY_RUN=true ;;
        --clean)    CLEAN_FLAG="--clean" ;;
        --model)    shift; SINGLE_MODEL="$1" ;;
        --mode)     shift; SINGLE_MODE="$1" ;;
    esac
done

if [[ -n "$SINGLE_MODEL" ]]; then
    MODELS=("$SINGLE_MODEL")
fi
if [[ -n "$SINGLE_MODE" ]]; then
    MODES=("$SINGLE_MODE")
fi

mkdir -p "$RESULTS_DIR/slurm_logs"

declare -a JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    if [[ "$MODEL" == "gemma-4-E4B-it" ]]; then
        PARTITION="v100"
        GPU_SPEC="gpu:v100:1"
        # 1391 images × ~20s = ~7.7h → 10h to be safe
        TIME_LIMIT="10:00:00"
    else
        PARTITION="rtx3080"
        GPU_SPEC="gpu:1"
        # 1391 images × ~14s = ~5.4h → 8h to be safe
        TIME_LIMIT="08:00:00"
    fi

    for MODE in "${MODES[@]}"; do
        JOB_NAME="lac_gt_${MODEL}_${MODE}"
        LOG_OUT="$RESULTS_DIR/slurm_logs/${JOB_NAME}_%j.out"

        WRAP_CMD="export HF_HOME=$WORK_DIR/.cache/huggingface && export HF_HUB_OFFLINE=1 && cd $LAC_DIR && $PY lac_pipeline.py --model $MODEL --input_mode $MODE --prompt_dir prompts_navigable --output_suffix _navigable --gt_dir $GT_DIR $CLEAN_FLAG"

        if $DRY_RUN; then
            echo "DRY-RUN:"
            echo "  sbatch \\"
            echo "    --partition=${PARTITION} \\"
            echo "    --gres=${GPU_SPEC} \\"
            echo "    --time=${TIME_LIMIT} \\"
            echo "    --job-name=${JOB_NAME} \\"
            echo "    --output=${LOG_OUT} \\"
            echo "    --export=ALL \\"
            echo "    --wrap=\"${WRAP_CMD}\""
            echo ""
        else
            JOB_ID=$(sbatch \
                --partition="${PARTITION}" \
                --gres="${GPU_SPEC}" \
                --time="${TIME_LIMIT}" \
                --job-name="${JOB_NAME}" \
                --output="${LOG_OUT}" \
                --export=ALL \
                --wrap="${WRAP_CMD}" \
                | tail -1 | awk '{print $NF}')
            echo "Submitted ${JOB_NAME} → Job ID: ${JOB_ID}"
            JOB_IDS+=("${JOB_ID}")
        fi
    done
done

if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"
    echo ""
    echo "GT directory: $GT_DIR"
    echo "Output directories:"
    for MODEL in "${MODELS[@]}"; do
        for MODE in "${MODES[@]}"; do
            echo "  $RESULTS_DIR/${MODEL}_LaC_navigable/${MODE}_sam3/"
        done
    done
fi
