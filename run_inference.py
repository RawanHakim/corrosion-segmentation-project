"""Batch inference on static images using the enhanced SegFormer pipeline.

Place images in the ``images/`` folder (relative to this script) and run::

    python run_inference.py

Results are written to ``outputs/``: mask, overlay with region bounding boxes,
and a per-image JSON report with severity grading and region analysis.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realtime_corrosion import (
    load_model,
    preprocess_frame,
    run_inference,
    postprocess_predictions,
    calculate_metrics,
    create_overlay,
    analyze_regions,
    grade_severity,
    check_frame_quality,
)

CHECKPOINT_PATH = str(SCRIPT_DIR / "segformer_corrosion.pth")
IMAGES_DIR = SCRIPT_DIR / "images"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device, checkpoint_path=CHECKPOINT_PATH)

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts)

    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        return

    print(f"Found {len(image_paths)} image(s).\n")

    summary = []

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Skipping (cannot read): {img_path}")
            continue

        quality_ok, quality_issues = check_frame_quality(img_bgr)
        if not quality_ok:
            print(f"  Quality issues: {', '.join(quality_issues)} — processing anyway")

        tensor, original_size = preprocess_frame(img_bgr, device)
        preds, timing = run_inference(model, tensor, original_size, device)
        preds = postprocess_predictions(preds)
        metrics = calculate_metrics(preds, img_bgr.shape)
        regions = analyze_regions(preds)
        grade, grade_desc = grade_severity(metrics)

        overlay = create_overlay(img_bgr, preds, regions=regions)

        stem = img_path.stem

        mask_gray = preds.astype(np.uint8) * 127
        cv2.imwrite(str(OUTPUTS_DIR / f"{stem}_mask.png"), mask_gray)
        cv2.imwrite(str(OUTPUTS_DIR / f"{stem}_overlay.png"), overlay)

        report = {
            "image": img_path.name,
            "quality": {"ok": quality_ok, "issues": quality_issues},
            "metrics": {
                "pixel_counts": metrics["pixel_counts"],
                "area_percent": metrics["area_percent"],
            },
            "severity": {"grade": grade, "description": grade_desc},
            "regions": {"count": len(regions), "details": regions},
            "inference_time_ms": round(timing["total_inference"], 2),
        }
        with open(OUTPUTS_DIR / f"{stem}_report.json", "w") as f:
            json.dump(report, f, indent=2)

        summary.append({
            "image": img_path.name,
            "grade": grade,
            "corrosion": metrics["area_percent"]["corroded_total"],
            "regions": len(regions),
        })

        print(
            f"  Grade: {grade} | "
            f"Corrosion: {metrics['area_percent']['corroded_total']:.2f}% | "
            f"Regions: {len(regions)}"
        )
        print(f"  Saved: {stem}_mask.png, {stem}_overlay.png, {stem}_report.json\n")

    print("=" * 50)
    print("BATCH SUMMARY")
    print("=" * 50)
    for s in summary:
        print(
            f"  [{s['grade']:>8}] {s['image']} — "
            f"{s['corrosion']:.2f}% corrosion, {s['regions']} regions"
        )
    print(f"\nTotal: {len(summary)} images processed. Done!")


if __name__ == "__main__":
    main()
