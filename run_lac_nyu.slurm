#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LAC Pipeline on NYU Depth V2 Test Set — SLURM Launcher
# ──────────────────────────────────────────────────────────────────────────────
#
# Runs YOUR ORIGINAL lac_pipeline.py on the NYU Depth V2 test set.
# Uses sbatch --wrap to dynamically set partition and GPU type.
#
# Usage:
#   bash run_lac_nyu.slurm
#   bash run_lac_nyu.slurm --quick-test
#   bash run_lac_nyu.slurm --strategy zero_shot --mode rgb_only
#   bash run_lac_nyu.slurm --model gemma-4-E4B-it
#   bash run_lac_nyu.slurm --prepare-only
#   bash run_lac_nyu.slurm --all   # Run all combinations
#
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK:-/home/woody/iwnt/iwnt164h}"
PY="$WORK_DIR/venvs/marigold/bin/python3"
RESULTS_DIR="$WORK_DIR/nyu_lac_results"
GT_DIR="$WORK_DIR/nyu_testset_gt/floor_masks"
NYU_DIR="$WORK_DIR/nyu_testset"
LAC_DIR="$SCRIPT_DIR"

mkdir -p "$RESULTS_DIR/slurm_logs"

# ── Defaults ──────────────────────────────────────────────────────────────────
STRATEGY="two_vlm"
MODEL="Qwen2.5-VL-7B-Instruct"
MODE="rgb_only"
NUM_IMAGES=""
QUICK_TEST=false
PREPARE_ONLY=false
EVALUATE=true
CLEAN_FLAG=""
DRY_RUN=false
RUN_ALL=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)       shift; STRATEGY="$1" ;;
        --model)          shift; MODEL="$1" ;;
        --mode)           shift; MODE="$1" ;;
        --num_images)     shift; NUM_IMAGES="$1" ;;
        --quick-test)     QUICK_TEST=true ;;
        --prepare-only)   PREPARE_ONLY=true ;;
        --evaluate)       EVALUATE=true ;;
        --no-evaluate)    EVALUATE=false ;;
        --clean)          CLEAN_FLAG="--clean" ;;
        --dry-run)        DRY_RUN=true ;;
        --all)            RUN_ALL=true ;;
        --partition)      shift; PARTITION_OVERRIDE="$1" ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Select partition and GPU ─────────────────────────────────────────────────
if [[ -n "${PARTITION_OVERRIDE:-}" ]]; then
    PARTITION="$PARTITION_OVERRIDE"
    if [[ "$PARTITION" == "v100" ]]; then
        GPU_SPEC="gpu:v100:1"
    else
        GPU_SPEC="gpu:1"
    fi
    TIME_LIMIT="12:00:00"
elif [[ "$MODEL" == "gemma-4-E4B-it" ]]; then
    PARTITION="v100"
    GPU_SPEC="gpu:v100:1"
    TIME_LIMIT="05:00:00"
else
    PARTITION="rtx3080"
    GPU_SPEC="gpu:1"
    TIME_LIMIT="12:00:00"
fi

# ── Prepare data (runs locally, not on cluster) ──────────────────────────────
echo "============================================================"
echo "Step 1: Preparing NYU data for LAC pipeline..."
echo "============================================================"

PREPARE_ARGS="--nyu_dir $NYU_DIR --output_base $WORK_DIR"
if [ -n "$NUM_IMAGES" ]; then
    PREPARE_ARGS="$PREPARE_ARGS --num_images $NUM_IMAGES"
fi
if [ "$QUICK_TEST" = true ]; then
    PREPARE_ARGS="$PREPARE_ARGS --num_images 5"
fi

$PY "$SCRIPT_DIR/prepare_nyu_for_lac.py" $PREPARE_ARGS

if [ "$PREPARE_ONLY" = true ]; then
    echo "Data preparation only — done."
    exit 0
fi

# ── Extract floor GT masks (if not already done) ─────────────────────────────
if [ ! -d "$GT_DIR" ] || [ -z "$(ls -A "$GT_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "============================================================"
    echo "Step 2: Extracting floor ground truth masks..."
    echo "============================================================"
    $PY "$SCRIPT_DIR/extract_nyu_floor_gt.py" \
        --mat_file "$WORK_DIR/nyuv2/nyu_depth_v2_labeled.mat" \
        --split_file "/home/hpc/iwnt/iwnt164h/Marigold/data_split/nyu_depth/labeled/filename_list_test.txt" \
        --output_dir "$WORK_DIR/nyu_testset_gt"
else
    echo "Floor GT masks already exist — skipping extraction."
fi

# ── Build pipeline command ────────────────────────────────────────────────────
build_pipeline_cmd() {
    local strategy="$1"
    local model="$2"
    local mode="$3"

    local cmd="export HF_HOME=$WORK_DIR/.cache/huggingface && export HF_HUB_OFFLINE=1 && cd $LAC_DIR && $PY lac_pipeline.py --config nyu_lac_config.yaml --strategy $strategy --model $model --input_mode $mode"

    if [ -n "$NUM_IMAGES" ]; then
        cmd="$cmd --num_images $NUM_IMAGES"
    fi
    if [ "$QUICK_TEST" = true ]; then
        cmd="$cmd --quick_test"
    fi
    if [ -n "$CLEAN_FLAG" ]; then
        cmd="$cmd $CLEAN_FLAG"
    fi

    # Add evaluation step after pipeline
    if [ "$EVALUATE" = true ]; then
        local short_model=""
        case "$model" in
            Qwen2.5-VL-7B-Instruct) short_model="Qwen" ;;
            gemma-4-E4B-it) short_model="Gemma" ;;
            *) short_model="$model" ;;
        esac
        local result_dir="$RESULTS_DIR/$strategy/$short_model/${mode}_sam3"
        cmd="$cmd && $PY $LAC_DIR/evaluate_lac_nyu.py --results_dir $result_dir --gt_mask_dir $GT_DIR --output_dir $result_dir/evaluation"
    fi

    echo "$cmd"
}

# ── Submit jobs ───────────────────────────────────────────────────────────────
submit_job() {
    local strategy="$1"
    local model="$2"
    local mode="$3"

    # Select partition based on model
    local partition="$PARTITION"
    local gpu_spec="$GPU_SPEC"
    local time_limit="$TIME_LIMIT"
    if [[ "$model" == "gemma-4-E4B-it" ]]; then
        partition="v100"
        gpu_spec="gpu:v100:1"
        time_limit="05:00:00"
    else
        partition="rtx3080"
        gpu_spec="gpu:1"
        time_limit="12:00:00"
    fi

    local short_model=""
    case "$model" in
        Qwen2.5-VL-7B-Instruct) short_model="Qwen" ;;
        gemma-4-E4B-it) short_model="Gemma" ;;
        *) short_model="$model" ;;
    esac

    local job_name="lac_nyu_${short_model}_${mode}"
    local log_out="$RESULTS_DIR/slurm_logs/${job_name}_%j.out"
    local log_err="$RESULTS_DIR/slurm_logs/${job_name}_%j.err"
    local wrap_cmd=$(build_pipeline_cmd "$strategy" "$model" "$mode")

    if $DRY_RUN; then
        echo "  [DRY RUN] sbatch \\"
        echo "    --partition=${partition} \\"
        echo "    --gres=${gpu_spec} \\"
        echo "    --time=${time_limit} \\"
        echo "    --job-name=${job_name} \\"
        echo "    --output=${log_out} \\"
        echo "    --error=${log_err} \\"
        echo "    --cpus-per-task=4 \\"
        echo "    --wrap=\"${wrap_cmd}\""
        echo ""
    else
        JOB_ID=$(sbatch \
            --partition="${partition}" \
            --gres="${gpu_spec}" \
            --time="${time_limit}" \
            --job-name="${job_name}" \
            --output="${log_out}" \
            --error="${log_err}" \
            --cpus-per-task=4 \
            --wrap="${wrap_cmd}" \
            | awk '{print $NF}')
        echo "  Submitted ${short_model} ${mode}: Job $JOB_ID (partition=${partition}, gres=${gpu_spec})"
    fi
}

echo ""
echo "============================================================"
echo "Step 3: Submitting LAC pipeline jobs..."
echo "============================================================"

if $RUN_ALL; then
    echo "Running ALL combinations..."
    for strategy in two_vlm zero_shot; do
        for model in Qwen2.5-VL-7B-Instruct gemma-4-E4B-it; do
            for mode in rgb_only rgb_depth_separate; do
                submit_job "$strategy" "$model" "$mode"
            done
        done
    done
else
    submit_job "$STRATEGY" "$MODEL" "$MODE"
fi

echo ""
echo "============================================================"
echo "Jobs submitted! Monitor with: squeue -u \$USER"
echo "Results will be in: $RESULTS_DIR"
echo "============================================================"
