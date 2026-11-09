# Real-Time Fall Detection Pipeline for Elderly Safety

> **End-to-end ML system** for fall detection in surveillance video - combining YOLO pose estimation, Transformer-based temporal sequence modeling, physics-informed FSM refinement, and a real-time inference dashboard. Processes live streams at **30 FPS / <33ms per frame** on CPU-only hardware.

Custom dataset co-created and published at [`samruddhi-2308/falldetectiondataset`](https://github.com/samruddhi-2308/falldetectiondataset).

---

## ML System Overview

This system solves a **multi-class temporal classification problem**: given a video stream, classify each frame sequence as *fall* or *no-fall*, with separate specialized pipelines for two fall modalities.

```
Raw Video → YOLO Pose Extraction → Temporal Feature Engineering
         → Transformer Sequence Model → Physics-Based FSM Refinement
         → Fall/No-Fall Decision + Alert Level
```

### Why two pipelines?

Fall kinematics differ fundamentally by starting posture:
- **Standing-to-Fall**: rapid center-of-mass drop, high keypoint velocity, loss of vertical orientation
- **Sleeping-to-Fall** (bed/chair): subtle motion, slower velocity signature, different joint angle patterns

A single model trained on pooled data underperforms on both. Two specialized models with posture-conditioned feature extractors achieve better precision and fewer false positives.

---

## Technical Architecture

### 1. Pose Feature Extraction
- **Model**: YOLOv11n-pose (`yolo11n-pose.pt`) - lightweight, real-time skeleton keypoint detection
- **Output**: 17-keypoint skeleton per frame (COCO format) - joint coordinates + confidence scores
- **Feature engineering**: keypoint velocity (frame-to-frame Δposition), acceleration, joint angle ratios, center-of-mass trajectory, bounding box aspect ratio change

### 2. Temporal Sequence Modeling
- **Architecture**: Transformer encoder (`transformer_standing_fall_detection.pth`) trained on sliding windows of pose sequences
- Captures **long-range temporal dependencies** in movement patterns that frame-by-frame classifiers miss
- Trained on custom annotated dataset - see [`falldetectiondataset`](https://github.com/samruddhi-2308/falldetectiondataset) for data collection methodology

### 3. Physics-Informed FSM Refinement
- A **finite-state machine** (states: `NORMAL → TRANSITION → FALL → ALERT`) with physics-based transition guards
- Guards validate biomechanical plausibility: e.g., a "fall" transition is rejected if vertical velocity is below threshold or if the body aspect ratio hasn't changed enough
- Reduces false positives from fast normal movements (e.g., bending, crouching) without sacrificing recall on true falls

### 4. Real-Time Inference Constraints
| Constraint | Value |
|---|---|
| Target throughput | 30 FPS |
| Max frame latency | <33ms |
| Hardware | CPU-only (no GPU required) |
| Execution model | Async thread pool + SSE streaming |

---

## Dataset

Custom fall detection dataset built by our team- annotated video sequences across both fall modalities.

→ [`samruddhi-2308/falldetectiondataset`](https://github.com/samruddhi-2308/falldetectiondataset) - see README for annotation protocol, class distribution, and train/val/test splits.

---

## Repository Structure

```
app/
  main.py                        # FastAPI app, job orchestration, SSE streaming
  static/index.html              # Dashboard UI

scripts/
  fall_detector.py               # Sleeping-to-Fall inference pipeline
  fall_detector_standing.py      # Standing-to-Fall inference pipeline

inference_outputs/               # Generated annotated videos + event CSVs
artifacts/                       # Training artifacts and model checkpoints
```

---

## Team

| Member | Contribution |
|---|---|
| **Samruddhi Lakare** | Standing-to-Fall pipeline - feature engineering, FSM design, async inference, integration |
| **Vedant Karne** | Sleeping-to-Fall pipeline - temporal modeling, integration. [`@vedantkarne`](https://github.com/vedantkarne) |
| + 2 teammates | Dataset collection, annotation, evaluation |

---

## Quick Start

### Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn aiofiles opencv-python torch ultralytics numpy
```

### Launch Dashboard
```powershell
.\start_dashboard.bat
```
Open: http://localhost:8000 - upload a video, select pipeline (`standing` or `sleeping`), monitor live inference.

### CLI Inference (Standing Pipeline)
```powershell
python scripts/fall_detector_standing.py <path-to-video> \
  --pose-model yolo11n-pose.pt \
  --transformer transformer_standing_fall_detection.pth \
  --output inference_outputs/standing_fall_output.mp4
```

---

## API

| Endpoint | Description |
|---|---|
| `POST /upload` | Upload video, select pipeline (`sleeping` \| `standing`) |
| `GET /progress/{job_id}` | SSE stream - real-time inference progress and state updates |

---

## Outputs

Each inference run produces:
- **Annotated video** - skeleton overlay, state label, alert level per frame
- **Event log CSV** - structured fall event records:

| Field | Description |
|---|---|
| `event_id` | Unique fall event identifier |
| `pipeline` | `sleeping` or `standing` |
| `state` | FSM state at event time |
| `video_timestamp` | HH:MM:SS in source video |
| `timestamp_sec` | Seconds from video start |
| `frame_index` | Frame number |
| `alert_level` | Severity classification |
| `logged_at` | Wall-clock timestamp |

---

## Model Files Required

Before running inference, ensure these are present in the project root:
- `yolo11n-pose.pt`
- `transformer_standing_fall_detection.pth`
- `transformer_fall_detection.pth` (sleeping pipeline)

---

## Future Work
- Quantization-aware training for further latency reduction on edge devices
- Multi-camera scene fusion
- Online learning / continual adaptation to new environments
