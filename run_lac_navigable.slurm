#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LaC Navigable-Area Pipeline — Both Models (Qwen + Gemma)
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash run_lac_navigable.slurm              # submit all 4 jobs
#   bash run_lac_navigable.slurm --dry-run     # show commands without submitting
#   bash run_lac_navigable.slurm --clean       # clean output dirs before running
#
# Output stored in: $WORK/free_ground_results/{model}_LaC_navigable/{mode}_grounding_dino/
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAC_DIR="$SCRIPT_DIR"
WORK_DIR="${WORK:-/home/woody/iwnt/iwnt164h}"
HOME_DIR="/home/hpc/iwnt/iwnt164h"
PY="$WORK_DIR/venvs/marigold/bin/python3"
RESULTS_DIR="$WORK_DIR/free_ground_results"
GT_DIR="$RESULTS_DIR/Annotated_Ground_Truth"

# Navigable-area Grounding-DINO prompt — specific to walkable surfaces only
GDINO_PROMPT="floor . ground . stairs . step ."

MODELS=("Qwen2.5-VL-7B-Instruct" "gemma-4-E4B-it")
MODES=("rgb_only" "rgb_depth_overlay")

CLEAN_FLAG=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --clean)   CLEAN_FLAG="--clean" ;;
    esac
done

# Ensure log directory exists
mkdir -p "$RESULTS_DIR/slurm_logs"

declare -a JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    # Choose partition based on model
    if [[ "$MODEL" == "gemma-4-E4B-it" ]]; then
        PARTITION="v100"
        GPU_SPEC="gpu:v100:1"
    else
        PARTITION="rtx3080"
        GPU_SPEC="gpu:1"
    fi

    for MODE in "${MODES[@]}"; do
        JOB_NAME="lac_nav_${MODEL}_${MODE}"
        LOG_OUT="$RESULTS_DIR/slurm_logs/${JOB_NAME}_%j.out"

        # Build the wrap command with proper environment
        WRAP_CMD="export HF_HOME=$WORK_DIR/.cache/huggingface && export HF_HUB_OFFLINE=1 && cd $LAC_DIR && $PY lac_pipeline.py --model $MODEL --input_mode $MODE --segmentation_method grounding_dino --grounding_dino_text_prompt '$GDINO_PROMPT' --prompt_dir prompts_navigable --output_suffix _navigable --gt_dir $GT_DIR $CLEAN_FLAG"

        if $DRY_RUN; then
            echo "DRY-RUN:"
            echo "  sbatch \\"
            echo "    --partition=${PARTITION} \\"
            echo "    --gres=${GPU_SPEC} \\"
            echo "    --time=01:00:00 \\"
            echo "    --job-name=${JOB_NAME} \\"
            echo "    --output=${LOG_OUT} \\"
            echo "    --export=ALL \\"
            echo "    --wrap=\"${WRAP_CMD}\""
            echo ""
        else
            JOB_ID=$(sbatch \
                --partition="${PARTITION}" \
                --gres="${GPU_SPEC}" \
                --time=01:00:00 \
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
    echo "Output directories:"
    for MODEL in "${MODELS[@]}"; do
        for MODE in "${MODES[@]}"; do
            echo "  \$WORK/free_ground_results/${MODEL}_LaC_navigable/${MODE}_grounding_dino/"
        done
    done
    echo ""
    echo "Monitor: squeue -u \$(whoami)"
fi
