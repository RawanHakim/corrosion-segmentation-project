# Real-Time Webcam Corrosion Scanner

## Overview

This project provides a real-time corrosion detection system using a SegFormer semantic segmentation model. It processes webcam feed at approximately 3 FPS and automatically saves detection results when corrosion is detected.

## Features

- **Real-time webcam processing** at ~3 FPS
- **Automatic detection-triggered saving** (no manual keypresses needed)
- **JSON metrics** with pixel counts and area percentages
- **Dual image outputs**: raw frame + overlay visualization
- **Cooldown system** to prevent save spam
- **Live overlay display** with green (fair corrosion) and red (severe corrosion) regions
- **GPU acceleration** when CUDA is available

## Files

### Static Image Processing
- `run_inference.py` - Process images from `images/` folder (original pipeline)
- `segformer_corrosion.pth` - Trained model weights

### Real-Time Processing
- `realtime_corrosion.py` - **Real-time webcam scanner** (NEW)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the trained model weights (`segformer_corrosion.pth`) in the project directory.

## Usage

### Real-Time Webcam Scanner

```bash
python realtime_corrosion.py
```

**Controls:**
- Press `q` to quit

**What it does:**
1. Opens your default webcam (camera index 0)
2. Processes frames at ~3 FPS
3. Displays live overlay with corrosion visualization
4. Automatically saves outputs when corrosion exceeds threshold

**Outputs saved to `realtime_outputs/`:**
- `YYYYMMDD_HHMMSS_raw.jpg` - Original captured frame
- `YYYYMMDD_HHMMSS_overlay.jpg` - Frame with green/red overlay
- `YYYYMMDD_HHMMSS.json` - Detection metrics

**Example JSON output:**
```json
{
  "timestamp": "20260206_143512",
  "frame_size": {"width": 1280, "height": 720},
  "pixel_counts": {"class0": 800000, "class1": 9000, "class2": 3000},
  "area_percent": {"class1": 0.90, "class2": 0.30, "corroded_total": 1.20},
  "trigger": {"threshold_percent": 0.30, "triggered": true},
  "saved_files": {
    "raw": "20260206_143512_raw.jpg",
    "overlay": "20260206_143512_overlay.jpg"
  }
}
```

### Static Image Processing (Original)

```bash
python run_inference.py
```

Place images in `images/` folder. Results saved to `outputs/`.

## Configuration

Edit the configuration section at the top of `realtime_corrosion.py`:

```python
# ==================== CONFIGURATION ====================
CHECKPOINT_PATH = "segformer_corrosion.pth"
OUTPUTS_DIR = Path("realtime_outputs")
IMAGE_SIZE = 512  # Model input resolution
TARGET_FPS = 3    # Approximate processing rate
CORROSION_THRESHOLD_PERCENT = 0.3  # Minimum corrosion % to trigger save
COOLDOWN_SECONDS = 2  # Minimum time between saves (prevent spam)
CAMERA_INDEX = 0  # Default webcam
```

### Parameter Tuning Guide

#### `TARGET_FPS` (Default: 3)
- **Higher values (4-5)**: Smoother real-time feel, but may stress CPU/GPU
- **Lower values (2)**: More processing time per frame, better for slower systems
- Actual FPS depends on your hardware capabilities

#### `CORROSION_THRESHOLD_PERCENT` (Default: 0.3)
- **Higher values (1.0-2.0)**: Only trigger on significant corrosion, fewer false positives
- **Lower values (0.1-0.2)**: More sensitive, catches minor corrosion but may trigger more often
- Measured as percentage of total frame pixels

#### `COOLDOWN_SECONDS` (Default: 2)
- **Higher values (5-10)**: Fewer saves when viewing same corroded object
- **Lower values (1)**: More frequent saves, useful for scanning multiple objects quickly
- Prevents duplicate saves of the same corrosion region

#### `CAMERA_INDEX` (Default: 0)
- **0**: Default webcam (built-in laptop camera)
- **1, 2, ...**: External USB cameras if multiple cameras connected

#### `IMAGE_SIZE` (Default: 512)
- Model input size (do not change unless you retrain the model)
- Actual corrosion metrics calculated on original webcam resolution

## Model Classes

The model predicts 3 classes:
- **Class 0**: Background (no corrosion)
- **Class 1**: Fair corrosion (displayed in **green**)
- **Class 2**: Severe/poor corrosion (displayed in **red**)

## System Requirements

- Python 3.8+
- Webcam (built-in or USB)
- **GPU (optional)**: NVIDIA GPU with CUDA for faster processing
- **CPU**: Will work without GPU but slower

## Troubleshooting

**Camera not opening:**
- Check if another application is using the webcam
- Try different `CAMERA_INDEX` values (0, 1, 2, ...)
- Ensure webcam permissions are granted

**Slow performance:**
- Lower `TARGET_FPS` to 2
- Ensure CUDA is available if you have an NVIDIA GPU
- Close other GPU-intensive applications

**No detections being saved:**
- Lower `CORROSION_THRESHOLD_PERCENT` to test
- Check console output for corrosion percentage values
- Verify you're showing corroded objects to the camera

**Too many saves:**
- Increase `COOLDOWN_SECONDS`
- Increase `CORROSION_THRESHOLD_PERCENT`

## Technical Details

- **Architecture**: SegFormer-B1 fine-tuned for corrosion detection
- **Input**: 512×512 RGB images (resized from webcam feed)
- **Output**: Pixel-wise classification at original webcam resolution
- **Framework**: PyTorch + Hugging Face Transformers
- **Processing**: Bilinear interpolation for upscaling predictions to original size

## License

This is a Final Year Project (FYP) for EECE 502.
