#!/bin/bash
# Unified Walkable Area Detection - All Experiments
# ===================================================
# This script runs all experiments with different strategies, models, and input modes
# for identifying walkable areas in indoor scenes.

# Set default values
WORK_DIR="${WORK:-/home/woody/iwnt/iwnt164h}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/walkable_area_pipeline.py"
DATA_DIR="${DATA_DIR:-/home/woody/iwnt/iwnt164h/mlp_dataset/prospthesisproject-Data/Code/Data}"
OUTPUT_DIR="${WORK_DIR}/free_ground_results"

# Available models
MODELS=(
    "Qwen2.5-VL-7B-Instruct"
    "Qwen3.5-2B"
    "Qwen3-VL-4B-Instruct"
    "gemma-4-E4B-it"
    "ByteDance/Sa2VA-Qwen3-VL-4B"
)

# Available strategies
STRATEGIES=("zero_shot" "few_shot" "two_vlm" "sa2va")

# Available input modes
INPUT_MODES=("rgb_only" "rgb_depth_separate")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run a single experiment
run_experiment() {
    local strategy=$1
    local model=$2
    local input_mode=$3
    local additional_args=$4
    
    print_info "Running: strategy=$strategy, model=$model, input_mode=$input_mode"
    
    local cmd="python $PIPELINE_SCRIPT \
        --strategy $strategy \
        --model $model \
        --input_mode $input_mode \
        $additional_args"
    
    echo "$cmd"
    eval "$cmd"
    
    if [ $? -eq 0 ]; then
        print_success "Completed: strategy=$strategy, model=$model, input_mode=$input_mode"
    else
        print_error "Failed: strategy=$strategy, model=$model, input_mode=$input_mode"
    fi
    echo ""
}

# Function to run all experiments for a specific strategy
run_strategy_experiments() {
    local strategy=$1
    
    print_info "========================================"
    print_info "Running all $strategy experiments"
    print_info "========================================"
    
    for model in "${MODELS[@]}"; do
        for input_mode in "${INPUT_MODES[@]}"; do
            # Skip incompatible combinations
            if [ "$strategy" == "sa2va" ] && [[ "$model" != *"Sa2VA"* ]]; then
                print_warning "Skipping: SA2VA strategy only works with SA2VA models"
                continue
            fi
            
            if [ "$strategy" != "sa2va" ] && [[ "$model" == *"Sa2VA"* ]]; then
                print_warning "Skipping: SA2VA models only work with sa2va strategy"
                continue
            fi
            
            run_experiment "$strategy" "$model" "$input_mode" ""
        done
    done
}

# Function to run specific experiments
run_specific_experiments() {
    print_info "========================================"
    print_info "Running specific experiments"
    print_info "========================================"
    
    # Zero-shot experiments
    print_info "Zero-shot experiments..."
    run_experiment "zero_shot" "Qwen2.5-VL-7B-Instruct" "rgb_only" ""
    run_experiment "zero_shot" "Qwen2.5-VL-7B-Instruct" "rgb_depth_separate" ""
    run_experiment "zero_shot" "Qwen3.5-2B" "rgb_only" ""
    run_experiment "zero_shot" "Qwen3.5-2B" "rgb_depth_separate" ""
    run_experiment "zero_shot" "Qwen3-VL-4B-Instruct" "rgb_only" ""
    run_experiment "zero_shot" "Qwen3-VL-4B-Instruct" "rgb_depth_separate" ""
    run_experiment "zero_shot" "gemma-4-E4B-it" "rgb_only" ""
    run_experiment "zero_shot" "gemma-4-E4B-it" "rgb_depth_separate" ""
    
    # Few-shot experiments
    print_info "Few-shot experiments..."
    run_experiment "few_shot" "Qwen2.5-VL-7B-Instruct" "rgb_depth_separate" "--few_shot_dir $DATA_DIR/annotations --num_examples 3"
    run_experiment "few_shot" "Qwen3.5-2B" "rgb_depth_separate" "--few_shot_dir $DATA_DIR/annotations --num_examples 3"
    run_experiment "few_shot" "Qwen3-VL-4B-Instruct" "rgb_depth_separate" "--few_shot_dir $DATA_DIR/annotations --num_examples 3"
    
    # Two-VLM experiments
    print_info "Two-VLM experiments..."
    run_experiment "two_vlm" "Qwen2.5-VL-7B-Instruct" "rgb_depth_separate" ""
    run_experiment "two_vlm" "Qwen3.5-2B" "rgb_depth_separate" ""
    run_experiment "two_vlm" "Qwen3-VL-4B-Instruct" "rgb_depth_separate" ""
    run_experiment "two_vlm" "Qwen2.5-VL-7B-Instruct" "rgb_depth_separate" "--reasoner_model Qwen2.5-VL-7B-Instruct --evaluator_model gemma-4-E4B-it"
    
    # SA2VA experiments
    print_info "SA2VA experiments..."
    run_experiment "sa2va" "ByteDance/Sa2VA-Qwen3-VL-4B" "rgb_only" ""
    run_experiment "sa2va" "ByteDance/Sa2VA-Qwen3-VL-4B" "rgb_depth_separate" ""
}

# Function to run evaluation
run_evaluation() {
    print_info "========================================"
    print_info "Running evaluation"
    print_info "========================================"
    
    # Evaluate all results
    if [ -f "$SCRIPT_DIR/evaluate_all.py" ]; then
        print_info "Running evaluate_all.py..."
        python "$SCRIPT_DIR/evaluate_all.py" --output_dir "$OUTPUT_DIR"
    else
        print_warning "evaluate_all.py not found, skipping evaluation"
    fi
}

# Function to run comparison
run_comparison() {
    print_info "========================================"
    print_info "Running comparison"
    print_info "========================================"
    
    # Compare different strategies
    if [ -f "$SCRIPT_DIR/compare_strategies.py" ]; then
        print_info "Running compare_strategies.py..."
        python "$SCRIPT_DIR/compare_strategies.py" --output_dir "$OUTPUT_DIR"
    else
        print_warning "compare_strategies.py not found, skipping comparison"
    fi
}

# Function to show help
show_help() {
    cat << EOF
Unified Walkable Area Detection - All Experiments
==================================================

Usage: $0 [OPTIONS]

Options:
    -a, --all              Run all experiments for all strategies and models
    -s, --specific         Run specific experiments (recommended subset)
    -z, --zero-shot        Run zero-shot experiments only
    -f, --few-shot         Run few-shot experiments only
    -t, --two-vlm          Run two-VLM experiments only
    -v, --sa2va            Run SA2VA experiments only
    -e, --evaluate         Run evaluation on existing results
    -c, --compare          Run comparison between strategies
    -h, --help             Show this help message

Examples:
    # Run all experiments
    $0 --all

    # Run specific experiments (recommended)
    $0 --specific

    # Run zero-shot experiments only
    $0 --zero-shot

    # Run evaluation
    $0 --evaluate

    # Run comparison
    $0 --compare

Models:
    - Qwen2.5-VL-7B-Instruct
    - Qwen3.5-2B
    - Qwen3-VL-4B-Instruct
    - gemma-4-E4B-it
    - ByteDance/Sa2VA-Qwen3-VL-4B

Strategies:
    - zero_shot: Single VLM → SAM3
    - few_shot: Single VLM with examples → SAM3
    - two_vlm: Reasoner VLM + Evaluator VLM → SAM3
    - sa2va: SA2VA model (direct segmentation)

Input Modes:
    - rgb_only: RGB image only
    - rgb_depth_separate: RGB and depth images as separate inputs

Output:
    Results are saved to: $OUTPUT_DIR/{strategy}/{model_tag}/{input_mode}_{method}/

EOF
}

# Main script logic
main() {
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                print_info "Running all experiments for all strategies..."
                for strategy in "${STRATEGIES[@]}"; do
                    run_strategy_experiments "$strategy"
                done
                shift
                ;;
            -s|--specific)
                run_specific_experiments
                shift
                ;;
            -z|--zero-shot)
                run_strategy_experiments "zero_shot"
                shift
                ;;
            -f|--few-shot)
                run_strategy_experiments "few_shot"
                shift
                ;;
            -t|--two-vlm)
                run_strategy_experiments "two_vlm"
                shift
                ;;
            -v|--sa2va)
                run_strategy_experiments "sa2va"
                shift
                ;;
            -e|--evaluate)
                run_evaluation
                shift
                ;;
            -c|--compare)
                run_comparison
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Run main function
main "$@"