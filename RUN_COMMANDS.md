# LaC Pipeline — Run Commands Reference

All commands use the venv Python directly (avoids `source activate` issues):

```bash
export PY="$WORK/venvs/marigold/bin/python3"
export DIR="/home/hpc/iwnt/iwnt164h/lac_free_ground"
```

## Available Models

| Model Name | VRAM (4-bit) | Partition |
|---|---|---|
| `Qwen2.5-VL-7B-Instruct` | ~6 GB | V100, RTX2080Ti |
| `Qwen3.5-4B` | ~4 GB | V100, RTX2080Ti |
| `Qwen3.5-27B` | ~18 GB | V100 (8-bit), A100 |
| `Llama-3.2-11B-Vision-Instruct` | ~7 GB | V100 |
| `gemma-4-31B-it` | ~18 GB | A100 (recommended) |

## Segmentation Methods

| Method | Flag | Description |
|---|---|---|
| Grounding-DINO + SAM | `--segmentation_method grounding_dino` | Text-prompt detection → SAM mask (best quality) |
| SAM only | `--segmentation_method sam` | VLM bbox → SAM mask |
| VLM bbox only | `--segmentation_method vlm_only` | VLM bbox → rectangular mask (no extra model) |

## Input Modes

| Mode | Flag | Description |
|---|---|---|
| RGB + Depth overlay | `--input_mode rgb_depth_overlay` | Alpha-blended depth on RGB |
| RGB only | `--input_mode rgb_only` | Plain RGB image only |

---

## Quick Test (3 images per folder)

### Qwen2.5-VL-7B — all combos

```bash
# Grounding-DINO + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method grounding_dino --input_mode rgb_depth_overlay

# Grounding-DINO + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method grounding_dino --input_mode rgb_only

# SAM + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method sam --input_mode rgb_depth_overlay

# SAM + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method sam --input_mode rgb_only

# VLM only + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method vlm_only --input_mode rgb_depth_overlay

# VLM only + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method vlm_only --input_mode rgb_only
```

---

### Qwen3.5-4B — all combos

```bash
# Grounding-DINO + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-4B --segmentation_method grounding_dino --input_mode rgb_depth_overlay

# Grounding-DINO + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-4B --segmentation_method grounding_dino --input_mode rgb_only

# SAM + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-4B --segmentation_method sam --input_mode rgb_depth_overlay

# SAM + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-4B --segmentation_method sam --input_mode rgb_only

# VLM only + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-4B --segmentation_method vlm_only --input_mode rgb_depth_overlay

# VLM only + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-4B --segmentation_method vlm_only --input_mode rgb_only
```

---

### Qwen3.5-27B — all combos (needs V100 32GB or A100)

```bash
# Grounding-DINO + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-27B --segmentation_method grounding_dino --input_mode rgb_depth_overlay

# Grounding-DINO + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-27B --segmentation_method grounding_dino --input_mode rgb_only

# SAM + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-27B --segmentation_method sam --input_mode rgb_depth_overlay

# SAM + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-27B --segmentation_method sam --input_mode rgb_only

# VLM only + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-27B --segmentation_method vlm_only --input_mode rgb_depth_overlay

# VLM only + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Qwen3.5-27B --segmentation_method vlm_only --input_mode rgb_only
```

---

### Llama-3.2-11B-Vision-Instruct — all combos

```bash
# Grounding-DINO + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Llama-3.2-11B-Vision-Instruct --segmentation_method grounding_dino --input_mode rgb_depth_overlay

# Grounding-DINO + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Llama-3.2-11B-Vision-Instruct --segmentation_method grounding_dino --input_mode rgb_only

# SAM + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Llama-3.2-11B-Vision-Instruct --segmentation_method sam --input_mode rgb_depth_overlay

# SAM + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Llama-3.2-11B-Vision-Instruct --segmentation_method sam --input_mode rgb_only

# VLM only + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Llama-3.2-11B-Vision-Instruct --segmentation_method vlm_only --input_mode rgb_depth_overlay

# VLM only + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model Llama-3.2-11B-Vision-Instruct --segmentation_method vlm_only --input_mode rgb_only
```

---

### gemma-4-31B-it — all combos (needs A100 80GB)

```bash
# Grounding-DINO + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model gemma-4-31B-it --segmentation_method grounding_dino --input_mode rgb_depth_overlay

# Grounding-DINO + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model gemma-4-31B-it --segmentation_method grounding_dino --input_mode rgb_only

# SAM + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model gemma-4-31B-it --segmentation_method sam --input_mode rgb_depth_overlay

# SAM + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model gemma-4-31B-it --segmentation_method sam --input_mode rgb_only

# VLM only + RGB+Depth
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model gemma-4-31B-it --segmentation_method vlm_only --input_mode rgb_depth_overlay

# VLM only + RGB only
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --reasoner_model gemma-4-31B-it --segmentation_method vlm_only --input_mode rgb_only
```

---

## Run on Annotated Images (for ground truth comparison)

Replace `--quick_test` with `--specific_images`:

```bash
# Qwen2.5 + Grounding-DINO + RGB+Depth on annotated images
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --verbose \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method grounding_dino \
  --input_mode rgb_depth_overlay \
  --specific_images image28 image30 image36 image99 image188 image207

# Qwen2.5 + Grounding-DINO + RGB only on annotated images
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --verbose \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method grounding_dino \
  --input_mode rgb_only \
  --specific_images image28 image30 image36 image99 image188 image207
```

---

## Full Run (all images, all folders)

```bash
# Remove --quick_test to process ALL images
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --verbose \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method grounding_dino \
  --input_mode rgb_depth_overlay
```

---

## Via SLURM (compute node — recommended for large runs)

```bash
sbatch /home/hpc/iwnt/iwnt164h/lac_free_ground/run_lac.slurm
```

---

## Clean Previous Output Before Re-running

Add `--clean` to any command:

```bash
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose --clean \
  --reasoner_model Qwen2.5-VL-7B-Instruct --segmentation_method grounding_dino \
  --input_mode rgb_depth_overlay
```

---

## Run Only Specific Stages

```bash
# Only reasoner + evaluator (skip segmentation and cost map)
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --stages reasoner,evaluator

# Only segmentation + cost map (reuse existing reasoner/evaluator JSON outputs)
$PY $DIR/lac_pipeline.py --config $DIR/lac_config.yaml --quick_test --verbose \
  --stages segmentation,costmap
```
