"""
FastAPI backend for the Fall Detection Dashboard.

Features:
- drag and drop upload UI
- selectable inference pipeline (sleeping-fall or standing-fall)
- progress streaming over SSE
- annotated video download
- Excel-friendly CSV fall log download
"""

import asyncio
import csv
import json
import os
import queue
import sys
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib import import_module
from pathlib import Path

import aiofiles
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
UPLOAD_DIR = BASE_DIR / "app" / "uploads"
OUTPUT_DIR = BASE_DIR / "inference_outputs"
STATIC_DIR = BASE_DIR / "app" / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_CONFIGS = {
    "sleeping": {
        "label": "Sleeping-Fall",
        "module": "fall_detector",
        "transformer": "transformer_fall_detection.pth",
    },
    "standing": {
        "label": "Standing-Fall",
        "module": "fall_detector_standing",
        "transformer": "transformer_standing_fall_detection.pth",
    },
}

jobs: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="Fall Detection Dashboard", version="2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _format_video_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _probe_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()
    return {"total_frames": total_frames, "fps": fps}


def _write_event_log(job_id: str):
    job = jobs[job_id]
    log_path = Path(job["log_path"])
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "pipeline",
                "state",
                "video_timestamp",
                "timestamp_sec",
                "frame_index",
                "progress_pct",
                "alert_level",
                "logged_at",
            ],
        )
        writer.writeheader()
        writer.writerows(job["events"])


def _maybe_record_event(job_id: str, state: str, meta: dict) -> dict | None:
    normalized_state = (state or "").strip().lower()
    previous_state = jobs[job_id]["last_state"].strip().lower()
    jobs[job_id]["last_state"] = state

    if normalized_state not in {"falling", "impact", "fallen"} or normalized_state == previous_state:
        return None

    event = {
        "event_id": len(jobs[job_id]["events"]) + 1,
        "pipeline": jobs[job_id]["pipeline"],
        "state": state,
        "video_timestamp": _format_video_timestamp(float(meta.get("timestamp_sec", 0.0))),
        "timestamp_sec": round(float(meta.get("timestamp_sec", 0.0)), 2),
        "frame_index": int(meta.get("frame_index", 0)),
        "progress_pct": int(meta.get("progress_pct", 0)),
        "alert_level": meta.get("alert_level", "NONE"),
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    jobs[job_id]["events"].append(event)
    _write_event_log(job_id)

    return {
        "title": f"{state} detected",
        "message": f"Fall event at {event['video_timestamp']}",
        "state": state,
        "video_timestamp": event["video_timestamp"],
    }


def _run_inference(job_id: str, video_path: str, output_path: str):
    sys.path.insert(0, str(SCRIPTS_DIR))
    pipeline_key = jobs[job_id]["pipeline"]
    pipeline_config = PIPELINE_CONFIGS[pipeline_key]
    detector_module = import_module(pipeline_config["module"])
    run_detection_with_progress = detector_module.run_detection_with_progress

    model_path = str(BASE_DIR / "yolo11n-pose.pt")
    transformer_path = str(BASE_DIR / pipeline_config["transformer"])
    q: queue.Queue = jobs[job_id]["queue"]
    video_meta = _probe_video_metadata(video_path)

    _write_event_log(job_id)

    def on_progress(pct: int, state: str, meta: dict | None = None):
        meta = dict(meta or {})
        total_frames = int(meta.get("total_frames", video_meta["total_frames"]))
        fps = int(meta.get("fps", video_meta["fps"]))
        frame_index = int(meta.get("frame_index", round((pct / 100) * total_frames))) if total_frames > 0 else 0

        meta["progress_pct"] = pct
        meta["total_frames"] = total_frames
        meta["fps"] = fps
        meta["frame_index"] = frame_index
        meta["timestamp_sec"] = float(meta.get("timestamp_sec", frame_index / max(fps, 1)))
        meta["alert_level"] = meta.get("alert_level", "NONE")
        payload = {
            "progress": pct,
            "state": state,
            "done": False,
            "pipeline": pipeline_key,
            "pipeline_label": pipeline_config["label"],
            "meta": meta,
        }
        notification = _maybe_record_event(job_id, state, meta)
        if notification is not None:
            payload["notification"] = notification
            payload["events_count"] = len(jobs[job_id]["events"])
        q.put(payload)

    try:
        jobs[job_id]["status"] = "processing"
        run_detection_with_progress(
            video_path,
            model_path,
            output_path,
            transformer_path=transformer_path,
            progress_callback=on_progress,
        )
        jobs[job_id]["status"] = "done"
        _write_event_log(job_id)
        q.put(
            {
                "progress": 100,
                "state": "Completed",
                "done": True,
                "events_count": len(jobs[job_id]["events"]),
            }
        )
    except Exception as exc:
        jobs[job_id]["status"] = "error"
        _write_event_log(job_id)
        q.put({"progress": 0, "state": f"Error: {exc}", "done": True, "error": True})


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), pipeline: str = Form("sleeping")):
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    if pipeline not in PIPELINE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unsupported pipeline: {pipeline}")

    job_id = str(uuid.uuid4())
    input_path = str(UPLOAD_DIR / f"{job_id}{ext}")
    output_path = str(OUTPUT_DIR / f"output_{pipeline}_{job_id}.mp4")
    log_path = str(OUTPUT_DIR / f"fall_events_{pipeline}_{job_id}.csv")

    async with aiofiles.open(input_path, "wb") as handle:
        while chunk := await file.read(1024 * 1024):
            await handle.write(chunk)

    original_stem = Path(file.filename).stem
    jobs[job_id] = {
        "status": "queued",
        "pipeline": pipeline,
        "pipeline_label": PIPELINE_CONFIGS[pipeline]["label"],
        "input": input_path,
        "output": output_path,
        "log_path": log_path,
        "filename": file.filename,
        "download_filename": f"{pipeline}_fall_detected_{original_stem}.mp4",
        "log_download_filename": f"{pipeline}_fall_events_{original_stem}.csv",
        "events": [],
        "last_state": "",
        "queue": queue.Queue(),
    }

    executor.submit(_run_inference, job_id, input_path, output_path)
    return {
        "job_id": job_id,
        "pipeline": pipeline,
        "pipeline_label": jobs[job_id]["pipeline_label"],
        "download_filename": jobs[job_id]["download_filename"],
        "log_download_filename": jobs[job_id]["log_download_filename"],
    }


@app.get("/progress/{job_id}")
async def progress_stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        q: queue.Queue = jobs[job_id]["queue"]
        loop = asyncio.get_event_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, lambda: q.get(timeout=0.5))
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("done"):
                    break
            except queue.Empty:
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] != "done":
        raise HTTPException(status_code=409, detail="Processing not finished yet")

    output_path = jobs[job_id]["output"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    filename = jobs[job_id]["download_filename"]
    encoded_name = urllib.parse.quote(filename, safe="")
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded_name}'
            )
        },
    )


@app.get("/download-log/{job_id}")
async def download_log(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] not in {"done", "error"}:
        raise HTTPException(status_code=409, detail="Processing not finished yet")

    log_path = jobs[job_id]["log_path"]
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    filename = jobs[job_id]["log_download_filename"]
    encoded_name = urllib.parse.quote(filename, safe="")
    return FileResponse(
        path=log_path,
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded_name}'
            )
        },
    )


@app.get("/events/{job_id}")
async def get_events(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"events": jobs[job_id]["events"], "pipeline": jobs[job_id]["pipeline"]}


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": jobs[job_id]["status"], "events_count": len(jobs[job_id]["events"])}
