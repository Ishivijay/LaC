#!/usr/bin/env python3
"""Test script: Compare current evaluator vs V2 combined evaluator on a single image.

Runs the full pipeline (reasoner → evaluator) on one image using:
1. Current evaluator prompt (short, scoring-only)
2. V2 evaluator prompt (same as reasoner + scoring + reasoner output)

Then prints both outputs side-by-side for comparison.

Usage:
    python3 test_evaluator_v2.py --image image24 --folder lms_kamal_LA_downstairs_Nopeople_1
    python3 test_evaluator_v2.py --image image137 --model gemma-4-E4B-it
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "free_ground_pipeline"))

from pipeline import (
    load_config,
    discover_folders,
    discover_image_pairs,
    load_image_for_vlm,
    load_vlm_model,
    run_inference,
)


def set_prompt_dir(prompt_dir: Path):
    """Set the prompt directory for load_prompt()."""
    import lac_pipeline as lp
    lp._PROMPT_DIR = prompt_dir


def load_prompt(filename: str) -> str:
    """Load a prompt file from the configured prompt directory."""
    import lac_pipeline as lp
    prompt_dir = getattr(lp, "_PROMPT_DIR", Path(__file__).parent / "prompts_navigable")
    path = prompt_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().strip()


def run_reasoner(model, processor, config, rgb_image, depth_image):
    """Run Stage 1: Free Ground Reasoner."""
    import lac_pipeline as lp
    return lp.run_free_ground_reasoner(model, processor, config, rgb_image, depth_image)


def run_evaluator_v1(model, processor, config, reasoner_output, rgb_image, depth_image):
    """Run Stage 2 with CURRENT evaluator prompt."""
    import lac_pipeline as lp
    return lp.run_traversability_evaluator(
        model, processor, config, reasoner_output, rgb_image, depth_image
    )


def run_evaluator_v2(model, processor, config, reasoner_output, rgb_image, depth_image):
    """Run Stage 2 with V2 combined evaluator prompt."""
    input_mode = config["pipeline"].get("input_mode", "rgb_depth_separate")

    # Load V2 prompt template
    system_template = load_prompt("traversability_evaluator_system_v2.txt")

    # Fill in placeholders with reasoner output
    free_ground_areas = reasoner_output.get("free_ground_areas", [])
    navigability_reasoning = reasoner_output.get("navigability_reasoning", "None")

    system_prompt = system_template.replace(
        "{free_ground_areas}", json.dumps(free_ground_areas, indent=2)
    )
    system_prompt = system_prompt.replace(
        "{navigability_reasoning}", str(navigability_reasoning)
    )

    # Build user content
    if input_mode == "rgb_depth_separate" and depth_image is not None:
        user_prompt = load_prompt("traversability_evaluator_user_depth.txt")
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": rgb_image},
            {"type": "image", "image": depth_image},
        ]
    else:
        user_prompt = load_prompt("traversability_evaluator_user.txt")
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": rgb_image},
        ]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    response = run_inference(model, processor, messages, {"model": config["model"]["evaluator"]})
    return response


def parse_json_from_response(response: str) -> dict:
    """Extract JSON from VLM response."""
    import re
    cleaned = re.sub(r"<think\b[^>]*>.*?</think?>", "", response, flags=re.DOTALL)
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
    return {"raw_response": cleaned, "parse_error": True}


def main():
    parser = argparse.ArgumentParser(description="Test evaluator V2 on a single image")
    parser.add_argument("--image", type=str, default="image24",
                        help="Image ID to test (default: image24)")
    parser.add_argument("--folder", type=str, default="lms_kamal_LA_downstairs_Nopeople_1",
                        help="Folder name (default: lms_kamal_LA_downstairs_Nopeople_1)")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct",
                        help="Model name (default: Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--input_mode", type=str, default="rgb_depth_separate",
                        choices=["rgb_only", "rgb_depth_separate"],
                        help="Input mode (default: rgb_depth_separate)")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "lac_config.yaml"),
                        help="Config file path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config["pipeline"]["input_mode"] = args.input_mode

    # Override model
    from pipeline import MODEL_REGISTRY
    if args.model in MODEL_REGISTRY:
        config["model"]["reasoner"]["name"] = args.model
        config["model"]["reasoner"]["hf_model_id"] = MODEL_REGISTRY[args.model]
        config["model"]["evaluator"]["name"] = args.model
        config["model"]["evaluator"]["hf_model_id"] = MODEL_REGISTRY[args.model]
    else:
        print(f"Model '{args.model}' not in registry. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # Set prompt directory
    prompt_dir = Path(__file__).parent / "prompts_navigable"
    set_prompt_dir(prompt_dir)

    # Find image
    base_dir = config["data"]["base_dir"]
    folder_path = Path(base_dir) / args.folder
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        sys.exit(1)

    rgb_sub = config["data"]["rgb_subfolder"]
    depth_sub = config["data"]["depth_subfolder"]
    rgb_path = folder_path / rgb_sub / f"{args.image}.png"
    depth_path = folder_path / depth_sub / f"{args.image}_depth_colored.png"

    if not rgb_path.exists():
        # Try jpg
        rgb_path = folder_path / rgb_sub / f"{args.image}.jpg"
    if not depth_path.exists():
        depth_path = folder_path / depth_sub / f"{args.image}_depth.png"
    if not depth_path.exists():
        depth_path = folder_path / depth_sub / f"{args.image}.png"

    if not rgb_path.exists():
        print(f"RGB image not found: {folder_path / rgb_sub / args.image}.[png|jpg]")
        sys.exit(1)

    print("=" * 70)
    print("EVALUATOR V2 COMPARISON TEST")
    print("=" * 70)
    print(f"Image:   {args.folder}/{args.image}")
    print(f"Model:   {args.model}")
    print(f"Mode:    {args.input_mode}")
    print(f"RGB:     {rgb_path}")
    print(f"Depth:   {depth_path if depth_path.exists() else 'NOT FOUND'}")
    print()

    # Load images
    rgb_image = load_image_for_vlm(rgb_path)
    depth_image = None
    if args.input_mode != "rgb_only" and depth_path.exists():
        depth_image = load_image_for_vlm(depth_path)

    # Load model
    print("Loading model...")
    model_config = {"model": config["model"]["reasoner"]}
    model, processor = load_vlm_model(model_config)
    print("Model loaded.\n")

    # ── Stage 1: Reasoner ──
    print("─" * 70)
    print("STAGE 1: FREE GROUND REASONER")
    print("─" * 70)
    t0 = time.time()
    reasoner_output = run_reasoner(model, processor, config, rgb_image, depth_image)
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s")
    print(f"Areas found: {len(reasoner_output.get('free_ground_areas', []))}")
    print(json.dumps(reasoner_output, indent=2, default=str))
    print()

    # ── Stage 2a: Current Evaluator (V1) ──
    print("─" * 70)
    print("STAGE 2a: CURRENT EVALUATOR (V1 — short scoring prompt)")
    print("─" * 70)
    t0 = time.time()
    v1_response = run_evaluator_v1(model, processor, config, reasoner_output, rgb_image, depth_image)
    t1 = time.time()
    v1_parsed = parse_json_from_response(v1_response) if isinstance(v1_response, str) else v1_response
    print(f"Time: {t1-t0:.1f}s")
    print(json.dumps(v1_parsed, indent=2, default=str))
    print()

    # ── Stage 2b: V2 Combined Evaluator ──
    print("─" * 70)
    print("STAGE 2b: V2 COMBINED EVALUATOR (same as reasoner + scoring)")
    print("─" * 70)
    t0 = time.time()
    v2_raw = run_evaluator_v2(model, processor, config, reasoner_output, rgb_image, depth_image)
    t1 = time.time()
    v2_parsed = parse_json_from_response(v2_raw) if isinstance(v2_raw, str) else v2_raw
    print(f"Time: {t1-t0:.1f}s")
    print(json.dumps(v2_parsed, indent=2, default=str))
    print()

    # ── Comparison ──
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    v1_areas = v1_parsed.get("free_ground_areas", []) if isinstance(v1_parsed, dict) else []
    v2_areas = v2_parsed.get("free_ground_areas", []) if isinstance(v2_parsed, dict) else []
    r_areas = reasoner_output.get("free_ground_areas", [])

    print(f"Reasoner found:      {len(r_areas)} areas")
    print(f"V1 evaluator output: {len(v1_areas)} areas (scoring only, no bboxes)")
    print(f"V2 evaluator output: {len(v2_areas)} areas (full output with bboxes + scores)")
    print()

    # V2 areas with bboxes and scores
    if v2_areas:
        print("V2 Areas (name | type | bbox | score):")
        for a in v2_areas:
            name = a.get("name", "?")
            atype = a.get("type", "?")
            bbox = a.get("bbox", {})
            score = a.get("traversability_score", "?")
            print(f"  {name:30s} | {atype:12s} | {bbox} | score={score}")
    print()

    # V1 scores
    v1_scores = v1_parsed.get("traversability_score", {}) if isinstance(v1_parsed, dict) else {}
    if v1_scores:
        print("V1 Scores:")
        for name, score in v1_scores.items():
            print(f"  {name}: {score}")

    # Save outputs
    output_dir = Path(__file__).parent / "test_evaluator_v2_output"
    output_dir.mkdir(exist_ok=True)
    safe_name = f"{args.folder}_{args.image}"

    with open(output_dir / f"{safe_name}_reasoner.json", "w") as f:
        json.dump(reasoner_output, f, indent=2, default=str)
    with open(output_dir / f"{safe_name}_evaluator_v1.json", "w") as f:
        json.dump(v1_parsed, f, indent=2, default=str)
    with open(output_dir / f"{safe_name}_evaluator_v2.json", "w") as f:
        json.dump(v2_parsed, f, indent=2, default=str)

    print(f"\nOutputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
