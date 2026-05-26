#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LAC Pipeline on NYU Depth V2 Test Set — SLURM Launcher
# ──────────────────────────────────────────────────────────────────────────────
#
# Runs YOUR ORIGINAL lac_pipeline.py on the NYU Depth V2 test set.
# Steps:
#   1. Prepares NYU data into lms_kamal_* format (symlinks + depth conversion)
#   2. Extracts floor ground truth masks from NYU labeled data
#   3. Runs the original lac_pipeline.py
#   4. Evaluates results against floor GT
#
# Usage:
#   sbatch run_lac_nyu.slurm
#   sbatch run_lac_nyu.slurm --quick-test
#   sbatch run_lac_nyu.slurm --strategy zero_shot --mode rgb_only
#   sbatch run_lac_nyu.slurm --model Qwen2.5-VL-7B-Instruct
#   sbatch run_lac_nyu.slurm --prepare-only   # Just prepare data, don't run pipeline
#
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=lac_nyu
#SBATCH --output=/home/woody/iwnt/iwnt164h/nyu_lac_results/slurm_logs/%j.out
#SBATCH --error=/home/woody/iwnt/iwnt164h/nyu_lac_results/slurm_logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=rtx3080
## #SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK:-/home/woody/iwnt/iwnt164h}"
PY="$WORK_DIR/venvs/marigold/bin/python3"
RESULTS_DIR="$WORK_DIR/nyu_lac_results"
GT_DIR="$WORK_DIR/nyu_testset_gt/floor_masks"
NYU_DIR="$WORK_DIR/nyu_testset"

mkdir -p "$RESULTS_DIR/slurm_logs"

# ── Defaults ──────────────────────────────────────────────────────────────────
STRATEGY="two_vlm"
MODEL="Qwen2.5-VL-7B-Instruct"
MODE="rgb_depth_separate"
NUM_IMAGES=""
QUICK_TEST=false
PREPARE_ONLY=false
EVALUATE=true
CLEAN_FLAG=""

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
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================================"
echo "LAC Pipeline — NYU Depth V2 Test Set"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "Strategy:      $STRATEGY"
echo "Model:         $MODEL"
echo "Input mode:    $MODE"
echo "Python:        $PY"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ── Step 1: Prepare NYU data ─────────────────────────────────────────────────
echo ""
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

# ── Step 2: Extract floor GT masks ───────────────────────────────────────────
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

# ── Step 3: Run original LAC pipeline ────────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 3: Running original LAC pipeline..."
echo "============================================================"

PIPELINE_CMD="$PY $SCRIPT_DIR/lac_pipeline.py \
    --config $SCRIPT_DIR/nyu_lac_config.yaml \
    --strategy $STRATEGY \
    --model $MODEL \
    --input_mode $MODE"

if [ "$QUICK_TEST" = true ]; then
    PIPELINE_CMD="$PIPELINE_CMD --quick_test"
fi

if [ -n "$CLEAN_FLAG" ]; then
    PIPELINE_CMD="$PIPELINE_CMD $CLEAN_FLAG"
fi

echo "Command: $PIPELINE_CMD"
echo ""

eval $PIPELINE_CMD

echo ""
echo "============================================================"
echo "Pipeline complete! Results in: $RESULTS_DIR"
echo "============================================================"

# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
if [ "$EVALUATE" = true ]; then
    echo ""
    echo "============================================================"
    echo "Step 4: Evaluating against floor ground truth..."
    echo "============================================================"

    # Find the results directory for this run
    # Structure: {strategy}/{model_tag}/{input_mode}_sam3/
    SHORT_MODEL=""
    case "$MODEL" in
        Qwen2.5-VL-7B-Instruct) SHORT_MODEL="Qwen" ;;
        gemma-4-E4B-it) SHORT_MODEL="Gemma" ;;
        *) SHORT_MODEL="$MODEL" ;;
    esac

    RESULT_SUBDIR="$RESULTS_DIR/$STRATEGY/$SHORT_MODEL/${MODE}_sam3"
    if [ -d "$RESULT_SUBDIR" ]; then
        $PY "$SCRIPT_DIR/evaluate_lac_nyu.py" \
            --results_dir "$RESULT_SUBDIR" \
            --gt_mask_dir "$GT_DIR" \
            --output_dir "$RESULT_SUBDIR/evaluation"
    else
        echo "Results directory not found: $RESULT_SUBDIR"
        echo "Trying to auto-detect..."
        # Try to find the latest results
        LATEST=$(find "$RESULTS_DIR" -name "all_results.csv" -type f -newer "$RESULTS_DIR" 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            FOUND_DIR=$(dirname "$LATEST")
            echo "Found: $FOUND_DIR"
            $PY "$SCRIPT_DIR/evaluate_lac_nyu.py" \
                --results_dir "$FOUND_DIR" \
                --gt_mask_dir "$GT_DIR" \
                --output_dir "$FOUND_DIR/evaluation"
        else
            echo "ERROR: Could not find results to evaluate."
        fi
    fi
fi

echo ""
echo "All done! Job $SLURM_JOB_ID complete."
