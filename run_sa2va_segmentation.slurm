#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Sa2VA Image Segmentation Test — SLURM Launcher
# ──────────────────────────────────────────────────────────────────────────────
# Runs Sa2VA over LaC-style RGB/depth folders and saves LaC-like outputs.
#
# Usage:
#   bash run_sa2va_segmentation.slurm \
#     --base-dir /path/to/Data \
#     --output-root /path/to/free_ground_results \
#     --input-mode rgb_depth_separate
#
# Optional:
#   --folders lms_kamal_RB_downstairs_Nopeople_1 lms_kamal_LA_upstairs_Nopeople_1
#   --num-images 5
#   --image-max-side 896
#   --model-id ByteDance/Sa2VA-Qwen3-VL-4B
#   --model-tag Sa2VA-Qwen3-VL-4B
#   --strategy sa2va
#   --partition v100
#   --gres gpu:v100:1
#   --time 04:00:00
#   --dry-run
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK:-/home/woody/iwnt/iwnt164h}"
PY="$WORK_DIR/venvs/marigold/bin/python3"
RESULTS_DIR="$WORK_DIR/free_ground_results"
HF_CACHE="$WORK_DIR/.cache/huggingface"

BASE_DIR="${WORK_DIR}/mlp_dataset/prospthesisproject-Data/Code/Data"
OUTPUT_ROOT="$RESULTS_DIR"
INPUT_MODE="rgb_depth_separate"
MODEL_ID='ByteDance/Sa2VA-Qwen3-VL-4B'
MODEL_TAG='Sa2VA-Qwen3-VL-4B'
STRATEGY='sa2va'
NUM_IMAGES="0"
IMAGE_MAX_SIDE="896"
PARTITION="v100"
GRES="gpu:v100:1"
TIME_LIMIT="04:00:00"
DRY_RUN=false
FOLDERS=()
GT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir) shift; BASE_DIR="$1" ;;
        --output-root) shift; OUTPUT_ROOT="$1" ;;
        --input-mode) shift; INPUT_MODE="$1" ;;
        --model-id) shift; MODEL_ID="$1" ;;
        --model-tag) shift; MODEL_TAG="$1" ;;
        --strategy) shift; STRATEGY="$1" ;;
        --num-images) shift; NUM_IMAGES="$1" ;;
        --image-max-side) shift; IMAGE_MAX_SIDE="$1" ;;
        --gt-dir) shift; GT_DIR="$1" ;;
        --folders)
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                FOLDERS+=("$1")
                shift
            done
            continue
            ;;
        --partition) shift; PARTITION="$1" ;;
        --gres) shift; GRES="$1" ;;
        --time) shift; TIME_LIMIT="$1" ;;
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            cat <<'EOF'
Usage:
  bash run_sa2va_segmentation.slurm --base-dir DIR --output-root DIR [options]

Required:
  --base-dir DIR      Base dataset directory containing lms_kamal_* folders
  --output-root DIR   Root directory for results

Options:
    --input-mode MODE   rgb_only | rgb_depth_separate
  --folders ...       Optional subset of lms_kamal_* folders to process
    --gt-dir DIR        Optional annotation dir to filter image IDs
  --num-images N      Limit images per folder (default: 0 = all)
    --image-max-side N  Resize the model input longest side to N pixels (default: 896)
  --model-id ID       Hugging Face model ID (default: ByteDance/Sa2VA-Qwen3-VL-4B)
  --model-tag NAME    Output model tag (default: Sa2VA-Qwen3-VL-4B)
  --strategy NAME     Top-level output folder (default: sa2va)
  --partition NAME    SLURM partition (default: v100)
  --gres SPEC         SLURM GPU spec (default: gpu:v100:1)
  --time HH:MM:SS     Wall time (default: 04:00:00)
  --dry-run           Print sbatch command without submitting
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

mkdir -p "$RESULTS_DIR/slurm_logs"

JOB_NAME="sa2va_segmentation"
LOG_OUT="$RESULTS_DIR/slurm_logs/${JOB_NAME}_%j.out"

printf -v HF_HOME_Q '%q' "$HF_CACHE"
printf -v SCRIPT_DIR_Q '%q' "$SCRIPT_DIR"
printf -v PY_Q '%q' "$PY"
printf -v TEST_SCRIPT_Q '%q' "$SCRIPT_DIR/test_sa2va_lacstyle.py"
printf -v BASE_DIR_Q '%q' "$BASE_DIR"
printf -v OUTPUT_ROOT_Q '%q' "$OUTPUT_ROOT"
printf -v INPUT_MODE_Q '%q' "$INPUT_MODE"
printf -v MODEL_ID_Q '%q' "$MODEL_ID"
printf -v MODEL_TAG_Q '%q' "$MODEL_TAG"
printf -v STRATEGY_Q '%q' "$STRATEGY"
printf -v NUM_IMAGES_Q '%q' "$NUM_IMAGES"
printf -v IMAGE_MAX_SIDE_Q '%q' "$IMAGE_MAX_SIDE"
printf -v GT_DIR_Q '%q' "$GT_DIR"

FOLDER_ARGS=()
for folder in "${FOLDERS[@]}"; do
    printf -v folder_q '%q' "$folder"
    FOLDER_ARGS+=("$folder_q")
done

WRAP_CMD="export HF_HOME=$HF_HOME_Q && cd $SCRIPT_DIR_Q && $PY_Q $TEST_SCRIPT_Q --base-dir $BASE_DIR_Q --output-root $OUTPUT_ROOT_Q --input-mode $INPUT_MODE_Q --model-id $MODEL_ID_Q --model-tag $MODEL_TAG_Q --strategy $STRATEGY_Q --num-images $NUM_IMAGES_Q --image-max-side $IMAGE_MAX_SIDE_Q"
if [[ ${#FOLDER_ARGS[@]} -gt 0 ]]; then
    WRAP_CMD+=" --folders ${FOLDER_ARGS[*]}"
fi
if [[ -n "$GT_DIR" ]]; then
    WRAP_CMD+=" --gt-dir $GT_DIR_Q"
fi

if $DRY_RUN; then
    echo "DRY-RUN:"
    echo "  sbatch"
    echo "    --partition=${PARTITION}"
    echo "    --gres=${GRES}"
    echo "    --time=${TIME_LIMIT}"
    echo "    --job-name=${JOB_NAME}"
    echo "    --output=${LOG_OUT}"
    echo "    --export=ALL"
    echo "    --wrap=${WRAP_CMD}"
else
    JOB_ID=$(sbatch \
        --partition="${PARTITION}" \
        --gres="${GRES}" \
        --time="${TIME_LIMIT}" \
        --job-name="${JOB_NAME}" \
        --output="${LOG_OUT}" \
        --export=ALL \
        --wrap="${WRAP_CMD}" \
        | tail -1 | awk '{print $NF}')
    echo "Submitted ${JOB_NAME} → Job ID: ${JOB_ID}"
fi