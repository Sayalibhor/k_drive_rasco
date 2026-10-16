import os
import sys

# 1. Silence low-level OpenCV driver spam before importing cv2
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Enforce DirectShow priority without MSMF collisions

import cv2
import time
import datetime
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Directories Setup
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Camera_Inspection_App"
DIR_OK = os.path.join(BASE_DIR, "ok")
DIR_NOT_OK = os.path.join(BASE_DIR, "not_ok")

os.makedirs(DIR_OK, exist_ok=True)
os.makedirs(DIR_NOT_OK, exist_ok=True)

# ---------------------------------------------------------------------------
# Robust Camera Controller
# ---------------------------------------------------------------------------
class SafeCameraDevice:
    def __init__(self, index: int, name: str, width: int = 640, height: int = 480):
        self.index = index
        self.name = name
        self.width = width
        self.height = height
        
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.is_connected = False
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, name=f"Cam-{self.name}", daemon=True)
        self.thread.start()
        return self

    def _open(self):
        try:
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
        except Exception:
            pass
        return None

    def _capture_loop(self):
        self.cap = self._open()
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.is_connected = False
                time.sleep(2.0)
                self.cap = self._open()
                continue

            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.is_connected = True
                else:
                    self.is_connected = False
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    time.sleep(1.0)
            except Exception:
                self.is_connected = False
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1.0)

            time.sleep(0.025)

    def get_frame(self):
        with self.lock:
            if self.is_connected and self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


# ---------------------------------------------------------------------------
# Global Application Context
# ---------------------------------------------------------------------------
app = FastAPI(title="AEPL Vision Metrology Enterprise Console")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DiskIO")

cam_top: Optional[SafeCameraDevice] = None
cam_side: Optional[SafeCameraDevice] = None
is_shared_device = False

buffer_lock = threading.Lock()
shot_buffer: Dict[str, Optional[np.ndarray]] = {
    "cam1_shot1": None,
    "cam1_shot2": None,
    "cam2_shot1": None,
    "cam2_shot2": None,
}

@app.on_event("startup")
def startup_hardware():
    global cam_top, cam_side, is_shared_device
    print("\n--- Initializing Hardware Camera Pipeline ---")
    
    # 1. Clean probe for indices 0, 1, 2
    active_indices = []
    for test_idx in [0, 1, 2]:
        try:
            test_cap = cv2.VideoCapture(test_idx, cv2.CAP_DSHOW)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    active_indices.append(test_idx)
                test_cap.release()
        except Exception:
            pass
        time.sleep(0.1)

    print(f"Responsive Camera Ports Found: {active_indices}")

    if len(active_indices) >= 2:
        print(f"Assigning Independent Dual Cameras: Cam1 -> Port {active_indices[0]}, Cam2 -> Port {active_indices[1]}")
        cam_top = SafeCameraDevice(active_indices[0], "TopView").start()
        cam_side = SafeCameraDevice(active_indices[1], "SideView").start()
        is_shared_device = False
    elif len(active_indices) == 1:
        print(f"Single Camera Detected at Port {active_indices[0]}. Demultiplexing stream across both viewports.")
        cam_top = SafeCameraDevice(active_indices[0], "SharedPrimary").start()
        cam_side = cam_top  # Safe pointer reuse, prevents duplicate hardware open
        is_shared_device = True
    else:
        print("Warning: No camera hardware responded. Starting fallback driver at Port 0.")
        cam_top = SafeCameraDevice(0, "Fallback").start()
        cam_side = cam_top
        is_shared_device = True

    print("--- System Ready: Navigate to http://localhost:8000 ---\n")

@app.on_event("shutdown")
def shutdown_hardware():
    global cam_top, cam_side
    if cam_top:
        cam_top.stop()
    if cam_side and cam_side != cam_top:
        cam_side.stop()
    io_pool.shutdown(wait=False)

# ---------------------------------------------------------------------------
# MJPEG Streaming
# ---------------------------------------------------------------------------
def generate_mjpeg(device: Optional[SafeCameraDevice], label: str):
    while True:
        success, frame = device.get_frame() if device else (False, None)
        if not success or frame is None:
            blank = np.zeros((360, 480, 3), dtype="uint8")
            cv2.putText(blank, f"{label} OFFLINE", (110, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _, encoded = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n')
            time.sleep(0.2)
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream/cam1")
def stream_cam1():
    return StreamingResponse(generate_mjpeg(cam_top, "TOP VIEW"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/cam2")
def stream_cam2():
    return StreamingResponse(generate_mjpeg(cam_side, "SIDE VIEW"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/telemetry")
def get_telemetry():
    with buffer_lock:
        active_buffer_count = sum(1 for v in shot_buffer.values() if v is not None)
    
    top_online = bool(cam_top and cam_top.is_connected)
    side_online = bool(cam_side and cam_side.is_connected)

    return {
        "cam1": {"connected": top_online, "device": f"Port {cam_top.index if cam_top else 'N/A'}"},
        "cam2": {"connected": side_online, "device": f"Port {cam_side.index if cam_side else 'N/A'}"},
        "buffer_loaded": active_buffer_count,
        "storage": BASE_DIR
    }

class CaptureRequest(BaseModel):
    shot_key: str

@app.post("/api/capture_single")
def capture_single(req: CaptureRequest):
    key = req.shot_key
    target = cam_top if "cam1" in key else cam_side
    success, frame = target.get_frame() if target else (False, None)
    
    if not success or frame is None:
        return JSONResponse(status_code=400, content={"error": "Selected camera feed is unavailable."})

    with buffer_lock:
        shot_buffer[key] = frame

    return {"message": f"Captured {key} to buffer memory", "status": "ok"}

class DecisionRequest(BaseModel):
    decision: str

@app.post("/api/submit_decision")
def submit_decision(req: DecisionRequest):
    decision = req.decision.lower()
    if decision not in ["ok", "not_ok"]:
        return JSONResponse(status_code=400, content={"error": "Invalid decision"})

    target_dir = DIR_OK if decision == "ok" else DIR_NOT_OK
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    payload = {}
    with buffer_lock:
        for k, v in shot_buffer.items():
            if v is not None:
                payload[k] = v.copy()
                shot_buffer[k] = None

        if not payload:
            s1, f1 = cam_top.get_frame() if cam_top else (False, None)
            if s1: payload["Cam1_Top"] = f1
            s2, f2 = cam_side.get_frame() if cam_side else (False, None)
            if s2: payload["Cam2_Side"] = f2

    if not payload:
        return JSONResponse(status_code=400, content={"error": "No frames available to save"})

    def write_disk(data, folder, time_str, status_str):
        for name, frame in data.items():
            filename = f"{name}_{time_str}_{status_str.upper()}.jpg"
            cv2.imwrite(os.path.join(folder, filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    io_pool.submit(write_disk, payload, target_dir, ts, decision)

    return {
        "message": f"Saved {len(payload)} frames into {decision.upper()} directory",
        "timestamp": ts,
        "files_count": len(payload)
    }

@app.post("/api/clear_buffer")
def clear_buffer():
    with buffer_lock:
        for k in shot_buffer:
            shot_buffer[k] = None
    return {"message": "Shot buffer flushed"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")