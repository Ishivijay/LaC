#!/bin/bash
#SBATCH --job-name=lac_free_ground
#SBATCH --output=logs/lac_free_ground_%j.out
#SBATCH --error=logs/lac_free_ground_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE

# GPU partition — change to match your model's VRAM needs:
#   V100 32GB:  Qwen2.5-VL-7B (4-bit), Qwen3.5-4B (4-bit), Llama-3.2-11B (4-bit)
#   A100 80GB:  Qwen3.5-27B (8-bit), gemma-4-31B-it (8-bit)
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1

# ==============================================================================
# LaC Free Ground Detection Pipeline - SLURM Job Script
# ==============================================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "=============================================="

# Create log directory
mkdir -p logs

# ==============================================================================
# Environment — ensure HuggingFace uses $WORK for cache
# ==============================================================================
export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_OFFLINE=1

echo "HF_HOME: $HF_HOME"
echo ""

# Use venv Python directly (avoids source activate issues)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$WORK/venvs/marigold/bin/python3"

# Print GPU info
nvidia-smi

# Print Python environment info
echo ""
echo "Python: $PY"
echo "PyTorch: $($PY -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $($PY -c 'import torch; print(torch.cuda.is_available())')"
echo "Transformers: $($PY -c 'import transformers; print(transformers.__version__)')"
echo ""

# ==============================================================================
# Run the LaC pipeline
# ==============================================================================
cd "$SCRIPT_DIR"

# Default: Quick test with 3 images per folder using Grounding-DINO
$PY lac_pipeline.py --config lac_config.yaml --quick_test --verbose

# --- Uncomment below for specific configurations ---

# Full processing with default config (Grounding-DINO):
# $PY lac_pipeline.py --config lac_config.yaml --verbose

# Specific images (the 6 annotated ones for comparison):
# $PY lac_pipeline.py --config lac_config.yaml --specific_images image28 image30 image36 image99 image188 image207 --verbose

# Different segmentation methods:
# $PY lac_pipeline.py --config lac_config.yaml --segmentation_method sam --quick_test
# $PY lac_pipeline.py --config lac_config.yaml --segmentation_method vlm_only --quick_test

# Different VLM models:
# $PY lac_pipeline.py --config lac_config.yaml --reasoner_model Qwen3.5-4B --quick_test
# $PY lac_pipeline.py --config lac_config.yaml --reasoner_model Qwen3.5-27B --quick_test
# $PY lac_pipeline.py --config lac_config.yaml --reasoner_model Llama-3.2-11B-Vision-Instruct --quick_test
# $PY lac_pipeline.py --config lac_config.yaml --reasoner_model gemma-4-31B-it --quick_test

# Clean previous output and re-run:
# $PY lac_pipeline.py --config lac_config.yaml --clean --quick_test --verbose

# Run only specific stages:
# $PY lac_pipeline.py --config lac_config.yaml --stages reasoner,evaluator --quick_test

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
