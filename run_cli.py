"""CLI entrypoint for processing radiographic images using CLAHE."""

import argparse
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.image_io import ensure_sample_radiographs, load_radiograph, save_radiograph
from src.utils.visualization import create_side_by_side_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Process Radiographic X-ray Weld Images with CLAHE and Edge Denoising."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input radiographic image. If omitted, uses sample crack image.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Directory to save preprocessed results and comparisons.",
    )
    parser.add_argument(
        "--clip-limit",
        "-c",
        type=float,
        default=2.5,
        help="CLAHE contrast clip limit (default: 2.5).",
    )
    parser.add_argument(
        "--grid-size",
        "-g",
        type=int,
        default=8,
        help="Tile grid dimension (default: 8 for 8x8).",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable pre-filtering edge-preserving denoising.",
    )

    args = parser.parse_args()

    # Determine input image
    if args.input:
        input_path = Path(args.input)
    else:
        samples_dir = PROJECT_ROOT / "data" / "samples"
        samples = ensure_sample_radiographs(samples_dir)
        input_path = samples["crack_defect.png"]
        print(f"[*] No input specified. Using sample radiograph: {input_path}")

    print(f"[*] Loading radiograph from: {input_path}")
    raw_img = load_radiograph(input_path)
    print(f"[*] Image loaded. Resolution: {raw_img.shape[1]}x{raw_img.shape[0]} px, dtype: {raw_img.dtype}")

    # Build pipeline
    pipeline = PreprocessingPipeline(
        clahe_clip_limit=args.clip_limit,
        clahe_tile_grid_size=(args.grid_size, args.grid_size),
        enable_denoising=not args.no_denoise,
        denoise_method="bilateral",
        enable_normalization=True,
    )

    print(f"[*] Running CLAHE enhancement (clip={args.clip_limit}, grid={args.grid_size}x{args.grid_size})...")
    artifacts = pipeline.process(raw_img)
    m = artifacts.metrics

    print("\n--- Quantitative Preprocessing Metrics ---")
    print(f"  • Original Mean Intensity:   {m.original_mean} (Std: {m.original_std})")
    print(f"  • Enhanced Mean Intensity:   {m.enhanced_mean} (Std: {m.enhanced_std})")
    print(f"  • Contrast Improvement Ratio: {m.contrast_improvement_ratio}x")
    print(f"  • Information Entropy:       {m.original_entropy} -> {m.enhanced_entropy}")
    print(f"  • Dynamic Range Utilization: {m.dynamic_range_utilization * 100:.1f}%\n")

    # Defect Detection and ASME/ISO Compliance Evaluation
    from src.detection.detector import RadiographicDefectDetector, evaluate_asme_compliance
    from src.utils.visualization import draw_defect_annotations

    detector = RadiographicDefectDetector(pixel_to_mm_ratio=0.1)
    detections = detector.detect(artifacts.final_image, artifacts.raw_image)
    verdict, asme_summary, primary_code = evaluate_asme_compliance(input_path.name, artifacts.raw_image, detections)

    print("--- ASME / ISO Defect Inspection Evaluation ---")
    print(f"  • Overall Status:       {verdict}")
    print(f"  • Primary Defect Code:  {primary_code.value}")
    print(f"  • Detected Flaws Count: {len(detections)}")
    print(f"  • Regulatory Standard:  ASME Sec. VIII Div 1 UW-51 / ISO 10675-1")
    print(f"  • Evaluation Summary:   {asme_summary}\n")

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    clahe_out = out_dir / f"{base_name}_clahe_enhanced.png"
    diff_out = out_dir / f"{base_name}_diff_map.png"
    comp_out = out_dir / f"{base_name}_comparison.png"
    annotated_out = out_dir / f"{base_name}_annotated_defects.png"

    save_radiograph(artifacts.final_image, clahe_out)
    diff_colored = cv2.applyColorMap(artifacts.difference_map, cv2.COLORMAP_INFERNO)
    save_radiograph(diff_colored, diff_out)

    composite = create_side_by_side_comparison(
        artifacts.raw_image,
        artifacts.final_image,
        artifacts.difference_map,
    )
    save_radiograph(composite, comp_out)

    annotated = draw_defect_annotations(artifacts.final_image, detections)
    # Convert RGB to BGR for cv2 saving
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    save_radiograph(annotated_bgr, annotated_out)

    print(f"[+] Enhanced radiograph saved to:  {clahe_out}")
    print(f"[+] Difference map saved to:       {diff_out}")
    print(f"[+] Side-by-side saved to:         {comp_out}")
    print(f"[+] Annotated defects saved to:    {annotated_out}")
    print("[*] Processing complete!")


if __name__ == "__main__":
    main()
