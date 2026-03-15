import json
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

try:
    import board
    import busio
    import adafruit_vl53l1x
    VL53L1X_AVAILABLE = True
except ImportError:
    VL53L1X_AVAILABLE = False

# ==================== CONFIGURATION ====================
CHECKPOINT_PATH = "segformer_corrosion.pth"
OUTPUTS_DIR = Path("realtime_outputs")
IMAGE_SIZE = 512
TARGET_FPS = 30
CORROSION_THRESHOLD_PERCENT = 2.0
COOLDOWN_SECONDS = 4
CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.7
USE_FP16 = True
MIN_REGION_AREA = 500
SCENE_CHANGE_THRESHOLD = 15.0
TEMPORAL_BUFFER_SIZE = 5
BLUR_THRESHOLD = 50.0
BRIGHTNESS_MIN = 30
BRIGHTNESS_MAX = 230

IMX219_HFOV_DEG = 62.2
IMX219_VFOV_DEG = 51.1

DISPLAY_WIDTH = 1280
FONT = cv2.FONT_HERSHEY_SIMPLEX

GRADE_COLORS = {
    "CRITICAL": (0, 0, 255),
    "HIGH": (0, 128, 255),
    "MODERATE": (0, 255, 255),
    "LOW": (0, 255, 0),
    "CLEAN": (255, 255, 255),
}
# =======================================================


# ==================== VL53L1X SENSOR ====================

def init_vl53l1x():
    """Initialise VL53L1X ToF sensor over I2C.
    Returns sensor object, or None if unavailable."""
    if not VL53L1X_AVAILABLE:
        print("[VL53L1X] Library not installed — running without distance sensor.")
        return None
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_vl53l1x.VL53L1X(i2c)
        sensor.distance_mode = 1
        sensor.timing_budget = 50
        sensor.start_ranging()
        print("[VL53L1X] Sensor initialised successfully.")
        return sensor
    except Exception as e:
        print(f"[VL53L1X] Sensor init failed — running without distance sensor. ({e})")
        return None


def read_distance_mm(sensor):
    """Read distance from VL53L1X. Returns mm as float, or None if unavailable."""
    if sensor is None:
        return None
    try:
        if sensor.data_ready:
            d = sensor.distance
            sensor.clear_interrupt()
            if d is None or d <= 0:
                return None
            return d * 1000.0
    except Exception:
        pass
    return None


# ==================== AREA COMPUTATION ====================

def corrosion_area_from_mask(mask01, distance_mm, hfov_deg, vfov_deg):
    """Compute real-world corrosion area in m² from a binary mask."""
    h, w = mask01.shape[:2]
    Z = distance_mm / 1000.0

    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)

    scene_w_m = 2.0 * Z * math.tan(hfov / 2.0)
    scene_h_m = 2.0 * Z * math.tan(vfov / 2.0)

    m_per_px_x = scene_w_m / w
    m_per_px_y = scene_h_m / h

    pixel_count = int(mask01.sum())
    area_m2 = pixel_count * (m_per_px_x * m_per_px_y)
    return area_m2


# ==================== FRAME QUALITY CHECK ====================

def check_frame_quality(frame, blur_threshold=None, brightness_min=None, brightness_max=None):
    """Check if a frame is suitable for inference.
    Returns (is_ok: bool, issues: list[str])."""
    if blur_threshold is None:
        blur_threshold = BLUR_THRESHOLD
    if brightness_min is None:
        brightness_min = BRIGHTNESS_MIN
    if brightness_max is None:
        brightness_max = BRIGHTNESS_MAX

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = float(gray.mean())

    issues = []
    if laplacian_var < blur_threshold:
        issues.append("BLURRY")
    if mean_brightness < brightness_min:
        issues.append("TOO_DARK")
    if mean_brightness > brightness_max:
        issues.append("OVEREXPOSED")

    return len(issues) == 0, issues


# ==================== SCENE-CHANGE DETECTION ====================

def compute_frame_similarity(frame_a, frame_b, thumbnail_size=(64, 64)):
    """Mean absolute difference between two frames as 64x64 grayscale thumbnails.
    Returns a value in [0, 255]: 0 = identical, 255 = completely different."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    small_a = cv2.resize(gray_a, thumbnail_size).astype(np.float32)
    small_b = cv2.resize(gray_b, thumbnail_size).astype(np.float32)
    return float(np.mean(np.abs(small_a - small_b)))


# ==================== MODEL ====================

def load_model(device, checkpoint_path=None, use_fp16=None):
    """Load SegFormer model with trained weights.

    Args:
        device: torch device.
        checkpoint_path: path to .pth file (default: CHECKPOINT_PATH constant).
        use_fp16: enable half-precision (default: USE_FP16 config when CUDA).
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    if use_fp16 is None:
        use_fp16 = USE_FP16 and device.type == "cuda"

    print("Loading SegFormer model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.eval()

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    if use_fp16:
        model.half()
        print("FP16 inference enabled.")

    print("Model loaded successfully.\n")
    return model


def preprocess_frame(frame, device, use_fp16=None):
    """Preprocess BGR frame for model inference.
    Returns: tensor [1, 3, IMAGE_SIZE, IMAGE_SIZE], original (h, w)."""
    if use_fp16 is None:
        use_fp16 = USE_FP16 and device.type == "cuda"

    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    norm = resized.astype(np.float32) / 255.0
    chw = np.transpose(norm, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    if use_fp16:
        tensor = tensor.half()
    return tensor, (h, w)


def run_inference(model, tensor, original_size, device, confidence_threshold=None):
    """Run model inference with confidence filtering.

    Pixels whose max-class probability is below *confidence_threshold* are
    forced to class 0 (background), eliminating low-confidence noise.

    Returns: (preds [H,W] uint8, timing_info dict)
    """
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD

    h, w = original_size
    inference_start = time.time()

    with torch.no_grad():
        forward_start = time.time()
        logits = model(pixel_values=tensor).logits
        forward_time = time.time() - forward_start

        interp_start = time.time()
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        interp_time = time.time() - interp_start

        classify_start = time.time()
        if confidence_threshold > 0:
            probs = F.softmax(logits, dim=1)
            confidence_vals, raw_preds = probs.max(dim=1)
            preds = raw_preds[0].cpu().numpy().astype(np.uint8)
            conf_map = confidence_vals[0].cpu().numpy()
            preds[conf_map < confidence_threshold] = 0
        else:
            preds = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        classify_time = time.time() - classify_start

    total_inference_time = time.time() - inference_start

    timing_info = {
        "forward_pass": forward_time * 1000,
        "interpolation": interp_time * 1000,
        "classify": classify_time * 1000,
        "total_inference": total_inference_time * 1000,
    }

    return preds, timing_info


# ==================== POST-PROCESSING ====================

def postprocess_predictions(preds, min_region_area=None):
    """Morphological cleanup: open/close to remove speckles and fill holes,
    then discard connected components smaller than *min_region_area* pixels."""
    if min_region_area is None:
        min_region_area = MIN_REGION_AREA

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = np.zeros_like(preds)

    for cls in [1, 2]:
        cls_mask = (preds == cls).astype(np.uint8)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_OPEN, kernel)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel)

        if min_region_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_region_area:
                    cls_mask[labels == i] = 0

        cleaned[cls_mask == 1] = cls

    return cleaned


# ==================== METRICS ====================

def calculate_metrics(preds, frame_shape):
    """Pixel counts and area percentages from a prediction mask."""
    h, w = frame_shape[:2]
    total_pixels = h * w

    class0_pixels = np.sum(preds == 0)
    class1_pixels = np.sum(preds == 1)
    class2_pixels = np.sum(preds == 2)

    class1_percent = (class1_pixels / total_pixels) * 100
    class2_percent = (class2_pixels / total_pixels) * 100
    corroded_total_percent = class1_percent + class2_percent

    return {
        "total_pixels": int(total_pixels),
        "pixel_counts": {
            "class0": int(class0_pixels),
            "class1": int(class1_pixels),
            "class2": int(class2_pixels),
        },
        "area_percent": {
            "class1": round(class1_percent, 2),
            "class2": round(class2_percent, 2),
            "corroded_total": round(corroded_total_percent, 2),
        },
    }


# ==================== REGION ANALYSIS ====================

def analyze_regions(preds, distance_mm=None, hfov_deg=None, vfov_deg=None):
    """Detect individual corrosion regions via connected-component analysis.

    Returns a list of region dicts sorted largest-first, each containing
    bounding box, centroid, pixel area, severity breakdown, and optionally
    real-world area in cm² when *distance_mm* is provided.
    """
    if hfov_deg is None:
        hfov_deg = IMX219_HFOV_DEG
    if vfov_deg is None:
        vfov_deg = IMX219_VFOV_DEG

    binary = ((preds == 1) | (preds == 2)).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    h, w = preds.shape[:2]
    m_per_px_x = None
    m_per_px_y = None
    if distance_mm is not None and distance_mm > 0:
        Z = distance_mm / 1000.0
        scene_w = 2.0 * Z * math.tan(math.radians(hfov_deg) / 2.0)
        scene_h = 2.0 * Z * math.tan(math.radians(vfov_deg) / 2.0)
        m_per_px_x = scene_w / w
        m_per_px_y = scene_h / h

    regions = []
    for i in range(1, num_labels):
        x, y, rw, rh, area_px = stats[i]
        region_pixels = preds[labels == i]
        severe_count = int(np.sum(region_pixels == 2))
        fair_count = int(np.sum(region_pixels == 1))

        region = {
            "id": i,
            "bbox": {"x": int(x), "y": int(y), "w": int(rw), "h": int(rh)},
            "area_pixels": int(area_px),
            "centroid": {
                "x": round(float(centroids[i][0]), 1),
                "y": round(float(centroids[i][1]), 1),
            },
            "severity_breakdown": {"fair_pixels": fair_count, "severe_pixels": severe_count},
            "dominant_severity": "severe" if severe_count > fair_count else "fair",
        }

        if m_per_px_x is not None:
            area_m2 = area_px * m_per_px_x * m_per_px_y
            region["area_cm2"] = round(area_m2 * 10000, 4)

        regions.append(region)

    return sorted(regions, key=lambda r: r["area_pixels"], reverse=True)


# ==================== SEVERITY GRADING ====================

def grade_severity(metrics):
    """Map corrosion percentages to an actionable severity grade.
    Returns (grade_str, description)."""
    total = metrics["area_percent"]["corroded_total"]
    severe = metrics["area_percent"]["class2"]

    if severe > 5.0 or total > 15.0:
        return "CRITICAL", "Immediate maintenance required"
    if severe > 2.0 or total > 8.0:
        return "HIGH", "Schedule maintenance soon"
    if total > 3.0:
        return "MODERATE", "Monitor closely"
    if total > 0.3:
        return "LOW", "Minor corrosion detected"
    return "CLEAN", "No significant corrosion"


# ==================== TEMPORAL SMOOTHING ====================

class TemporalSmoother:
    """Rolling pixel-wise mode filter across recent prediction frames,
    eliminating single-frame flicker without adding latency."""

    def __init__(self, buffer_size=None, num_classes=3):
        if buffer_size is None:
            buffer_size = TEMPORAL_BUFFER_SIZE
        self._buffer = deque(maxlen=buffer_size)
        self._num_classes = num_classes

    def smooth(self, preds):
        if self._buffer and self._buffer[-1].shape != preds.shape:
            self._buffer.clear()
        self._buffer.append(preds.copy())
        if len(self._buffer) < 3:
            return preds
        stacked = np.stack(list(self._buffer), axis=0)
        h, w = preds.shape[:2]
        counts = np.zeros((self._num_classes, h, w), dtype=np.int32)
        for c in range(self._num_classes):
            counts[c] = np.sum(stacked == c, axis=0)
        return np.argmax(counts, axis=0).astype(np.uint8)

    def reset(self):
        self._buffer.clear()


# ==================== OVERLAY / DISPLAY ====================

def create_overlay(frame, preds, alpha=0.3, regions=None):
    """Blend green (fair) / red (severe) colour mask onto frame.
    Optionally draw labelled bounding boxes for each detected region."""
    h, w = frame.shape[:2]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[preds == 1] = (0, 255, 0)
    color_mask[preds == 2] = (0, 0, 255)

    overlay = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

    if regions:
        for region in regions:
            bbox = region["bbox"]
            color = (0, 0, 255) if region["dominant_severity"] == "severe" else (0, 255, 0)
            x, y, rw, rh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            cv2.rectangle(overlay, (x, y), (x + rw, y + rh), color, 2)
            label = f"R{region['id']}: {region['dominant_severity']}"
            if "area_cm2" in region:
                label += f" ({region['area_cm2']:.1f}cm2)"
            cv2.putText(overlay, label, (x, max(y - 5, 15)), FONT, 0.4, color, 1)

    return overlay


def get_unique_filename(base_path, extension):
    """Generate unique filename by appending _1, _2, etc. if file exists."""
    path = Path(f"{base_path}{extension}")
    if not path.exists():
        return str(path)
    counter = 1
    while True:
        path = Path(f"{base_path}_{counter}{extension}")
        if not path.exists():
            return str(path)
        counter += 1


def save_detection(frame, overlay, metrics, timestamp, regions=None, grade_info=None):
    """Save overlay image and JSON metrics for a triggered detection."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    base_name = OUTPUTS_DIR / timestamp

    overlay_path = get_unique_filename(f"{base_name}", ".jpg")
    json_path = get_unique_filename(base_name, ".json")

    cv2.imwrite(overlay_path, overlay)

    h, w = frame.shape[:2]
    json_data = {
        "timestamp": timestamp,
        "frame_size": {"width": w, "height": h},
        "pixel_counts": metrics["pixel_counts"],
        "area_percent": metrics["area_percent"],
        "trigger": {
            "threshold_percent": CORROSION_THRESHOLD_PERCENT,
            "triggered": True,
        },
        "saved_files": {"image": Path(overlay_path).name},
    }

    if grade_info:
        json_data["severity"] = {"grade": grade_info[0], "description": grade_info[1]}

    if regions:
        json_data["regions"] = {"count": len(regions), "details": regions}

    if metrics.get("distance_mm") is not None:
        json_data["distance_mm"] = metrics["distance_mm"]
    if metrics.get("area_cm2") is not None:
        json_data["total_corrosion_area_cm2"] = metrics["area_cm2"]

    if "timing" in metrics:
        json_data["performance"] = {
            "timing": metrics["timing"],
            "gpu": metrics.get("gpu", {}),
        }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Saved detection: {Path(json_path).name}")

    return {"image": Path(overlay_path).name, "json": Path(json_path).name}


# ==================== SESSION REPORT ====================

def save_session_report(output_dir, session_start, detections, total_frames, skipped_frames):
    """Write a JSON summary of the entire scanning session on exit."""
    output_dir.mkdir(exist_ok=True)
    session_end = datetime.now()
    duration = (session_end - session_start).total_seconds()

    worst_grade = "CLEAN"
    grade_order = ["CLEAN", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    for d in detections:
        g = d.get("grade", "CLEAN")
        if grade_order.index(g) > grade_order.index(worst_grade):
            worst_grade = g

    report = {
        "session_start": session_start.isoformat(),
        "session_end": session_end.isoformat(),
        "duration_seconds": round(duration, 1),
        "total_frames_processed": total_frames,
        "frames_skipped_quality": skipped_frames,
        "total_detections_saved": len(detections),
        "worst_grade": worst_grade,
        "detections": detections,
    }

    report_name = f"session_report_{session_start.strftime('%Y%m%d_%H%M%S')}.json"
    report_path = output_dir / report_name
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSession report saved: {report_path.name}")
    return str(report_path)


# ==================== HUD INFO OVERLAY ====================

def add_info_overlay(display_frame, fps, metrics, last_save_time, cooldown_active, processing_fps):
    """Draw a semi-transparent HUD with all live telemetry."""
    h, w = display_frame.shape[:2]

    panel_height = 290
    panel = display_frame.copy()
    cv2.rectangle(panel, (0, 0), (w, panel_height), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(panel, 0.7, display_frame, 0.3, 0)

    y = 20
    lh = 22

    # Row 0 — FPS
    cv2.putText(display_frame,
                f"Display FPS: {fps:.1f} | Processing: {processing_fps:.1f} FPS",
                (10, y), FONT, 0.55, (0, 255, 0), 2)

    # Row 1 — Timing
    if "timing" in metrics:
        txt = (f"Frame: {metrics['timing']['total_frame_ms']:.1f}ms | "
               f"Inference: {metrics['timing']['inference_ms']:.1f}ms | "
               f"Max FPS: {metrics['timing']['fps_capacity']:.1f}")
        cv2.putText(display_frame, txt, (10, y + lh), FONT, 0.5, (100, 255, 255), 1)

    # Row 2 — GPU (conditional extra row)
    if "gpu" in metrics and metrics["gpu"]["memory_allocated_mb"] > 0:
        gpu_txt = f"GPU: {metrics['gpu']['device']} | Mem: {metrics['gpu']['memory_allocated_mb']:.1f}MB"
        cv2.putText(display_frame, gpu_txt, (10, y + 2 * lh), FONT, 0.5, (255, 200, 100), 1)
        y += lh

    # Row 3 — Severity grade
    grade = metrics.get("severity_grade", "CLEAN")
    grade_color = GRADE_COLORS.get(grade, (255, 255, 255))
    grade_desc = metrics.get("severity_description", "")
    cv2.putText(display_frame, f"Grade: {grade} — {grade_desc}",
                (10, y + 3 * lh), FONT, 0.6, grade_color, 2)

    # Row 4 — Corrosion %
    corr_pct = metrics["area_percent"]["corroded_total"]
    pct_color = (0, 0, 255) if corr_pct >= CORROSION_THRESHOLD_PERCENT else (255, 255, 255)
    cv2.putText(display_frame, f"Corrosion: {corr_pct:.2f}%",
                (10, y + 4 * lh), FONT, 0.6, pct_color, 2)

    # Row 5 — Class breakdown
    cv2.putText(display_frame,
                f"Fair: {metrics['area_percent']['class1']:.2f}% | "
                f"Severe: {metrics['area_percent']['class2']:.2f}%",
                (10, y + 5 * lh), FONT, 0.55, (255, 255, 255), 1)

    # Row 6 — Regions
    regions = metrics.get("regions", [])
    region_txt = f"Regions: {len(regions)} detected"
    if regions and "area_cm2" in regions[0]:
        region_txt += f" | Largest: {regions[0]['area_cm2']:.1f}cm2"
    cv2.putText(display_frame, region_txt, (10, y + 6 * lh), FONT, 0.55, (255, 200, 255), 1)

    # Row 7 — Distance
    dist_mm = metrics.get("distance_mm")
    if dist_mm is not None:
        dist_txt = f"Distance: {dist_mm:.0f} mm"
        dist_col = (0, 220, 255)
    else:
        dist_txt = "Distance: N/A (no sensor)"
        dist_col = (130, 130, 130)
    cv2.putText(display_frame, dist_txt, (10, y + 7 * lh), FONT, 0.55, dist_col, 1)

    # Row 8 — Real-world area
    area_cm2 = metrics.get("area_cm2")
    if area_cm2 is not None:
        area_txt = f"Corrosion Area: {area_cm2:.2f} cm2"
        area_col = (0, 0, 255) if corr_pct >= CORROSION_THRESHOLD_PERCENT else (200, 200, 200)
    else:
        area_txt = "Corrosion Area: N/A (no sensor)"
        area_col = (130, 130, 130)
    cv2.putText(display_frame, area_txt, (10, y + 8 * lh), FONT, 0.55, area_col, 2)

    # Row 9 — Quality
    quality_ok = metrics.get("quality_ok", True)
    quality_issues = metrics.get("quality_issues", [])
    if quality_ok:
        q_txt, q_col = "Quality: OK", (0, 255, 0)
    else:
        q_txt, q_col = "Quality: " + ", ".join(quality_issues), (0, 0, 255)
    cv2.putText(display_frame, q_txt, (10, y + 9 * lh), FONT, 0.55, q_col, 2)

    # Row 10 — Cooldown (conditional)
    if cooldown_active:
        remaining = COOLDOWN_SECONDS - (time.time() - last_save_time)
        cv2.putText(display_frame, f"Cooldown: {remaining:.1f}s",
                    (10, y + 10 * lh), FONT, 0.55, (0, 165, 255), 2)

    # Quit hint (bottom-right)
    cv2.putText(display_frame, "Press 'q' to quit",
                (w - 200, h - 20), FONT, 0.55, (255, 255, 255), 1)

    return display_frame


# ==================== MAIN ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_fp16 = USE_FP16 and device.type == "cuda"
    print(f"FP16 mode: {'enabled' if use_fp16 else 'disabled'}\n")

    model = load_model(device)

    # Warm-up pass so the first real frame isn't penalised by CUDA kernel compilation
    if device.type == "cuda":
        print("Running warm-up inference...")
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        if use_fp16:
            dummy = dummy.half()
        with torch.no_grad():
            model(pixel_values=dummy)
        print("Warm-up complete.\n")

    tof_sensor = init_vl53l1x()
    smoother = TemporalSmoother()

    # Session tracking
    session_start = datetime.now()
    session_detections = []
    total_frames_processed = 0
    skipped_quality_count = 0

    print(f"Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    print("Camera opened successfully!")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Corrosion threshold: {CORROSION_THRESHOLD_PERCENT}%")
    print(f"Cooldown: {COOLDOWN_SECONDS}s")
    print(f"Scene-change threshold: {SCENE_CHANGE_THRESHOLD}")
    print(f"Temporal buffer: {TEMPORAL_BUFFER_SIZE} frames")
    print(f"Min region area: {MIN_REGION_AREA}px")
    print(f"Output directory: {OUTPUTS_DIR.absolute()}")
    print("\nStarting real-time corrosion detection...\n")

    window_name = "Real-Time Corrosion Scanner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    frame_interval = 1.0 / TARGET_FPS
    last_process_time = 0.0
    last_save_time = 0.0

    fps_display = 0
    fps_counter = 0
    fps_timer = time.time()

    processing_fps = 0
    processing_counter = 0
    processing_timer = time.time()

    current_preds = None
    current_metrics = None
    last_saved_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            current_time = time.time()

            # ---- Process at target FPS ----
            if current_time - last_process_time >= frame_interval:
                last_process_time = current_time
                fps_counter += 1
                processing_counter += 1
                total_frames_processed += 1

                # Quality gate
                quality_ok, quality_issues = check_frame_quality(frame)

                if not quality_ok:
                    skipped_quality_count += 1
                    if current_metrics is not None:
                        current_metrics["quality_ok"] = False
                        current_metrics["quality_issues"] = quality_issues
                else:
                    # --- Full inference pipeline ---
                    preprocess_start = time.time()
                    tensor, original_size = preprocess_frame(frame, device)
                    preprocess_time = (time.time() - preprocess_start) * 1000

                    raw_preds, inference_timing = run_inference(
                        model, tensor, original_size, device
                    )

                    postprocess_start = time.time()
                    cleaned_preds = postprocess_predictions(raw_preds)
                    current_preds = smoother.smooth(cleaned_preds)

                    current_metrics = calculate_metrics(current_preds, frame.shape)
                    postprocess_time = (time.time() - postprocess_start) * 1000

                    # Distance + real-world area
                    distance_mm = read_distance_mm(tof_sensor)
                    area_cm2 = None
                    if distance_mm is not None:
                        binary_mask = ((current_preds == 1) | (current_preds == 2)).astype(np.uint8)
                        area_m2 = corrosion_area_from_mask(
                            binary_mask, distance_mm, IMX219_HFOV_DEG, IMX219_VFOV_DEG
                        )
                        area_cm2 = area_m2 * 10_000

                    current_metrics["distance_mm"] = (
                        round(distance_mm, 1) if distance_mm is not None else None
                    )
                    current_metrics["area_cm2"] = (
                        round(area_cm2, 4) if area_cm2 is not None else None
                    )

                    # Region analysis
                    regions = analyze_regions(current_preds, distance_mm=distance_mm)
                    current_metrics["regions"] = regions

                    # Severity grading
                    grade, grade_desc = grade_severity(current_metrics)
                    current_metrics["severity_grade"] = grade
                    current_metrics["severity_description"] = grade_desc

                    # Quality status
                    current_metrics["quality_ok"] = True
                    current_metrics["quality_issues"] = []

                    # Timing & GPU stats
                    total_frame_time = (
                        preprocess_time + inference_timing["total_inference"] + postprocess_time
                    )

                    if device.type == "cuda":
                        gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                        gpu_mem_res = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    else:
                        gpu_mem_alloc = 0
                        gpu_mem_res = 0

                    current_metrics["timing"] = {
                        "preprocess_ms": round(preprocess_time, 2),
                        "inference_ms": round(inference_timing["total_inference"], 2),
                        "forward_pass_ms": round(inference_timing["forward_pass"], 2),
                        "interpolation_ms": round(inference_timing["interpolation"], 2),
                        "classify_ms": round(inference_timing["classify"], 2),
                        "postprocess_ms": round(postprocess_time, 2),
                        "total_frame_ms": round(total_frame_time, 2),
                        "fps_capacity": (
                            round(1000 / total_frame_time, 2) if total_frame_time > 0 else 0
                        ),
                    }
                    current_metrics["gpu"] = {
                        "memory_allocated_mb": round(gpu_mem_alloc, 2),
                        "memory_reserved_mb": round(gpu_mem_res, 2),
                        "device": str(device),
                    }

                    print(
                        f"[{grade}] Frame: {total_frame_time:.1f}ms | "
                        f"Inference: {inference_timing['total_inference']:.1f}ms | "
                        f"Corrosion: {current_metrics['area_percent']['corroded_total']:.2f}% | "
                        f"Regions: {len(regions)} | GPU: {gpu_mem_alloc:.1f}MB"
                    )

                    # Save trigger
                    corroded_pct = current_metrics["area_percent"]["corroded_total"]
                    cooldown_elapsed = (current_time - last_save_time) >= COOLDOWN_SECONDS

                    if corroded_pct >= CORROSION_THRESHOLD_PERCENT and cooldown_elapsed:
                        # Scene-change gate: skip if the view hasn't changed
                        scene_changed = True
                        if last_saved_frame is not None:
                            diff = compute_frame_similarity(frame, last_saved_frame)
                            scene_changed = diff >= SCENE_CHANGE_THRESHOLD

                        if scene_changed:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            overlay_save = create_overlay(frame, current_preds, regions=regions)
                            saved = save_detection(
                                frame, overlay_save, current_metrics, timestamp,
                                regions=regions, grade_info=(grade, grade_desc),
                            )
                            last_save_time = current_time
                            last_saved_frame = frame.copy()

                            session_detections.append({
                                "timestamp": timestamp,
                                "grade": grade,
                                "corrosion_percent": corroded_pct,
                                "num_regions": len(regions),
                                "distance_mm": current_metrics.get("distance_mm"),
                                "area_cm2": current_metrics.get("area_cm2"),
                                "files": saved,
                            })

            # ---- FPS counters ----
            if current_time - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer = current_time

            if current_time - processing_timer >= 1.0:
                processing_fps = processing_counter
                processing_counter = 0
                processing_timer = current_time

            # ---- Display ----
            if current_preds is not None and current_metrics is not None:
                display_frame = create_overlay(
                    frame, current_preds, alpha=0.3,
                    regions=current_metrics.get("regions"),
                )
                cooldown_active = (current_time - last_save_time) < COOLDOWN_SECONDS
                display_frame = add_info_overlay(
                    display_frame, fps_display, current_metrics,
                    last_save_time, cooldown_active, processing_fps,
                )
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, "Initializing...",
                            (10, 30), FONT, 1, (0, 255, 255), 2)

            disp_h, disp_w = display_frame.shape[:2]
            if disp_w > DISPLAY_WIDTH:
                scale = DISPLAY_WIDTH / disp_w
                display_frame = cv2.resize(
                    display_frame, (DISPLAY_WIDTH, int(disp_h * scale))
                )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                print("\nQuitting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if total_frames_processed > 0:
            save_session_report(
                OUTPUTS_DIR, session_start, session_detections,
                total_frames_processed, skipped_quality_count,
            )

        print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()
