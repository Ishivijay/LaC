#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Unified Free Ground Detection Pipeline — SLURM Launcher
# ──────────────────────────────────────────────────────────────────────────────
# Supports all strategies: zero_shot, few_shot, two_vlm
#
# Usage:
#   # Submit a single run:
#   bash run_pipeline.slurm --strategy zero_shot --model Qwen2.5-VL-7B-Instruct --mode rgb_only
#
#   # Two-VLM with different models:
#   bash run_pipeline.slurm --strategy two_vlm --reasoner Qwen2.5-VL-7B-Instruct --evaluator gemma-4-E4B-it --mode rgb_depth_separate
#
#   # Submit ALL combinations:
#   bash run_pipeline.slurm --all
#
#   # Dry run (show commands without submitting):
#   bash run_pipeline.slurm --all --dry-run
#
# Output: $WORK/free_ground_results/{strategy}/{model_tag}/{input_mode}_sam3/
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK:-/home/woody/iwnt/iwnt164h}"
PY="$WORK_DIR/venvs/marigold/bin/python3"
RESULTS_DIR="$WORK_DIR/free_ground_results"
GT_DIR="$WORK_DIR/mlp_ground_truth/proper_annotations"

# Defaults
STRATEGY=""
MODEL=""
REASONER=""
EVALUATOR=""
MODE=""
FEW_SHOT_DIR=""
CLEAN_FLAG=""
DRY_RUN=false
RUN_ALL=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)   shift; STRATEGY="$1" ;;
        --model)      shift; MODEL="$1" ;;
        --reasoner)   shift; REASONER="$1" ;;
        --evaluator)  shift; EVALUATOR="$1" ;;
        --mode)       shift; MODE="$1" ;;
        --few_shot_dir) shift; FEW_SHOT_DIR="$1" ;;
        --clean)      CLEAN_FLAG="--clean" ;;
        --dry-run)    DRY_RUN=true ;;
        --all)        RUN_ALL=true ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ── Helper: submit a single job ──────────────────────────────────────────────
submit_job() {
    local strategy="$1"
    local model="$2"
    local mode="$3"
    local extra_args="${4:-}"

    # Choose partition based on model + strategy
    local partition gpu_spec time_limit
    if [[ "$model" == *"gemma"* ]]; then
        partition="v100"
        gpu_spec="gpu:v100:1"
        time_limit="20:00:00"
    elif [[ "$strategy" == "few_shot" ]]; then
        # Few-shot needs more VRAM (3 example images in prompt) → use v100
        partition="v100"
        gpu_spec="gpu:v100:1"
        time_limit="20:00:00"
    else
        partition="rtx3080"
        gpu_spec="gpu:1"
        time_limit="20:00:00"
    fi

    # Short name for job
    local short_model=$(echo "$model" | sed 's/Qwen2.5-VL-7B-Instruct/Qwen/' | sed 's/gemma-4-E4B-it/Gemma/')
    local job_name="fg_${strategy}_${short_model}_${mode}"
    local log_out="$RESULTS_DIR/slurm_logs/${job_name}_%j.out"

    local wrap_cmd="export HF_HOME=$WORK_DIR/.cache/huggingface && export HF_HUB_OFFLINE=1 && cd $SCRIPT_DIR && $PY lac_pipeline.py --strategy $strategy --model $model --input_mode $mode --prompt_dir prompts_navigable --gt_dir $GT_DIR $extra_args $CLEAN_FLAG"

    if $DRY_RUN; then
        echo "DRY-RUN:"
        echo "  sbatch \\"
        echo "    --partition=${partition} \\"
        echo "    --gres=${gpu_spec} \\"
        echo "    --time=${time_limit} \\"
        echo "    --job-name=${job_name} \\"
        echo "    --output=${log_out} \\"
        echo "    --export=ALL \\"
        echo "    --wrap=\"${wrap_cmd}\""
        echo ""
    else
        mkdir -p "$RESULTS_DIR/slurm_logs"
        local job_id=$(sbatch \
            --partition="${partition}" \
            --gres="${gpu_spec}" \
            --time="${time_limit}" \
            --job-name="${job_name}" \
            --output="${log_out}" \
            --export=ALL \
            --wrap="${wrap_cmd}" \
            | tail -1 | awk '{print $NF}')
        echo "Submitted ${job_name} → Job ID: ${job_id}"
        echo "$job_id"
    fi
}

# ── Helper: submit a two_vlm job with different models ───────────────────────
submit_two_vlm_job() {
    local reasoner="$1"
    local evaluator="$2"
    local mode="$3"

    # Partition based on the LARGER model (may need to load both)
    local partition gpu_spec time_limit
    if [[ "$reasoner" == *"gemma"* ]] || [[ "$evaluator" == *"gemma"* ]]; then
        partition="v100"
        gpu_spec="gpu:v100:1"
        time_limit="20:00:00"
    else
        partition="rtx3080"
        gpu_spec="gpu:1"
        time_limit="20:00:00"
    fi

    local r_short=$(echo "$reasoner" | sed 's/Qwen2.5-VL-7B-Instruct/Qwen/' | sed 's/gemma-4-E4B-it/Gemma/')
    local e_short=$(echo "$evaluator" | sed 's/Qwen2.5-VL-7B-Instruct/Qwen/' | sed 's/gemma-4-E4B-it/Gemma/')
    local job_name="fg_two_vlm_${r_short}_${e_short}_${mode}"
    local log_out="$RESULTS_DIR/slurm_logs/${job_name}_%j.out"

    local wrap_cmd="export HF_HOME=$WORK_DIR/.cache/huggingface && export HF_HUB_OFFLINE=1 && cd $SCRIPT_DIR && $PY lac_pipeline.py --strategy two_vlm --reasoner_model $reasoner --evaluator_model $evaluator --input_mode $mode --prompt_dir prompts_navigable --gt_dir $GT_DIR $CLEAN_FLAG"

    if $DRY_RUN; then
        echo "DRY-RUN:"
        echo "  sbatch \\"
        echo "    --partition=${partition} \\"
        echo "    --gres=${gpu_spec} \\"
        echo "    --time=${time_limit} \\"
        echo "    --job-name=${job_name} \\"
        echo "    --output=${log_out} \\"
        echo "    --export=ALL \\"
        echo "    --wrap=\"${wrap_cmd}\""
        echo ""
    else
        mkdir -p "$RESULTS_DIR/slurm_logs"
        local job_id=$(sbatch \
            --partition="${partition}" \
            --gres="${gpu_spec}" \
            --time="${time_limit}" \
            --job-name="${job_name}" \
            --output="${log_out}" \
            --export=ALL \
            --wrap="${wrap_cmd}" \
            | tail -1 | awk '{print $NF}')
        echo "Submitted ${job_name} → Job ID: ${job_id}"
        echo "$job_id"
    fi
}

# ── Main dispatch ────────────────────────────────────────────────────────────
declare -a ALL_JOBS=()

if $RUN_ALL; then
    echo "Submitting ALL pipeline combinations..."
    echo ""

    MODELS=("Qwen2.5-VL-7B-Instruct" "gemma-4-E4B-it")
    MODES=("rgb_only" "rgb_depth_separate")

    # 1. Zero-shot: each model × each mode
    echo "=== Zero-shot ==="
    for model in "${MODELS[@]}"; do
        for mode in "${MODES[@]}"; do
            submit_job "zero_shot" "$model" "$mode"
        done
    done

    # 2. Few-shot: each model × each mode
    echo "=== Few-shot ==="
    local_few_shot="${FEW_SHOT_DIR:-$GT_DIR}"
    for model in "${MODELS[@]}"; do
        for mode in "${MODES[@]}"; do
            submit_job "few_shot" "$model" "$mode" "--few_shot_dir $local_few_shot"
        done
    done

    # 3. Two-VLM (same model): each model × each mode
    echo "=== Two-VLM (same) ==="
    for model in "${MODELS[@]}"; do
        for mode in "${MODES[@]}"; do
            submit_job "two_vlm" "$model" "$mode"
        done
    done

    # 4. Two-VLM (different models): each combination × each mode
    echo "=== Two-VLM (different) ==="
    for mode in "${MODES[@]}"; do
        submit_two_vlm_job "Qwen2.5-VL-7B-Instruct" "gemma-4-E4B-it" "$mode"
        submit_two_vlm_job "gemma-4-E4B-it" "Qwen2.5-VL-7B-Instruct" "$mode"
    done

    echo ""
    echo "All jobs submitted. Monitor: squeue -u \$(whoami)"

elif [[ -n "$STRATEGY" ]]; then
    # Single run
    if [[ -z "$MODE" ]]; then
        echo "Error: --mode is required for single run"
        exit 1
    fi

    if [[ "$STRATEGY" == "two_vlm" ]] && [[ -n "$REASONER" ]] && [[ -n "$EVALUATOR" ]]; then
        # Two-VLM with different models
        submit_two_vlm_job "$REASONER" "$EVALUATOR" "$MODE"
    else
        # Use --model or default
        MODEL="${MODEL:-Qwen2.5-VL-7B-Instruct}"
        local_args=""
        if [[ "$STRATEGY" == "few_shot" ]] && [[ -n "$FEW_SHOT_DIR" ]]; then
            local_args="--few_shot_dir $FEW_SHOT_DIR"
        fi
        submit_job "$STRATEGY" "$MODEL" "$MODE" "$local_args"
    fi
else
    echo "Usage:"
    echo "  bash run_pipeline.slurm --all [--dry-run] [--clean]"
    echo "  bash run_pipeline.slurm --strategy <zero_shot|few_shot|two_vlm> --model <model> --mode <mode> [--clean]"
    echo "  bash run_pipeline.slurm --strategy two_vlm --reasoner <model> --evaluator <model> --mode <mode> [--clean]"
    exit 1
fi
