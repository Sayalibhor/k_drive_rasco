import os
import cv2
import time
import queue
import logging
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ---------------------------------------------------------
# Enterprise Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AEPL_VisionStation")

# ---------------------------------------------------------
# Environment & Storage Infrastructure
# ---------------------------------------------------------
BASE_DIR = r"C:\Camera_Inspection_App"
DIR_OK = os.path.join(BASE_DIR, "ok")
DIR_NOT_OK = os.path.join(BASE_DIR, "not_ok")
DIR_LOGS = os.path.join(BASE_DIR, "telemetry_logs")

for directory in (DIR_OK, DIR_NOT_OK, DIR_LOGS):
    os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------
# Threaded Camera Stream Controller (Producer)
# ---------------------------------------------------------
class ThreadedCameraWorker:
    """Independent background worker for zero-latency camera capture."""
    def __init__(self, camera_index: int, name: str, target_res=(640, 480)):
        self.camera_index = camera_index
        self.name = name
        self.target_res = target_res
        
        self.cap = None
        self.latest_frame = None
        self.is_running = False
        self.is_connected = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, name=f"Worker-{self.name}", daemon=True)
        self.thread.start()
        return self

    def _init_hardware(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_res[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Strip internal V4L/DirectShow queue latency
            return cap
        return None

    def _capture_loop(self):
        logger.info(f"Initializing hardware stream: {self.name} (Port {self.camera_index})")
        self.cap = self._init_hardware()
        
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                self.is_connected = False
                time.sleep(1.0)
                # Auto-reconnection attempt
                self.cap = self._init_hardware()
                continue

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
                time.sleep(0.5)

            # Polling limiter ~40 FPS to avoid bus congestion
            time.sleep(0.025)

    def get_frame(self):
        with self.lock:
            if self.is_connected and self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logger.info(f"Released stream: {self.name}")


# ---------------------------------------------------------
# Enterprise Vision Metrology Workstation GUI
# ---------------------------------------------------------
class EnterpriseWorkstationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AEPL Precision Metrology Workstation v3.4 [Enterprise Edition]")
        self.root.geometry("1200x720")
        self.root.minsize(1080, 640)
        self.root.configure(bg="#121214")

        # Color Theme: Clean Dark Studio Palette
        self.theme = {
            "bg": "#121214",
            "surface": "#1e1e24",
            "surface_alt": "#26262e",
            "border": "#32323d",
            "text_primary": "#f8f9fa",
            "text_secondary": "#a0a0ab",
            "brand": "#3b82f6",
            "brand_hover": "#2563eb",
            "success": "#10b981",
            "danger": "#ef4444",
            "warning": "#f59e0b"
        }

        # Threadpool for Non-Blocking Disk Operations
        self.io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DiskIO")

        # Hardware Mapping (Change indices here if needed: 1 & 2 for USB cams)
        self.cam1_worker = ThreadedCameraWorker(camera_index=1, name="Camera_TopView").start()
        self.cam2_worker = ThreadedCameraWorker(camera_index=2, name="Camera_SideView").start()

        # Shot Storage Buffer
        self.buffer_lock = threading.Lock()
        self.shot_buffer = {
            "cam1_shot1": None,
            "cam1_shot2": None,
            "cam2_shot1": None,
            "cam2_shot2": None
        }

        self._init_layout()
        self._register_shortcuts()
        self._render_stream_cycle()

    def _init_layout(self):
        # 1. Header Navigation Bar
        header = tk.Frame(self.root, bg=self.theme["surface"], height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        brand_badge = tk.Label(
            header,
            text="AEPL",
            font=("Segoe UI", 10, "bold"),
            bg=self.theme["brand"],
            fg="#ffffff",
            padx=8,
            pady=2
        )
        brand_badge.pack(side="left", padx=(18, 10), pady=12)

        sys_title = tk.Label(
            header,
            text="Vision Metrology System  |  Station #04  •  Part AEPL-2101-00-401-V3",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["surface"],
            fg=self.theme["text_primary"]
        )
        sys_title.pack(side="left", pady=12)

        self.telemetry_lbl = tk.Label(
            header,
            text="ENGINE READY  •  IO THREADS: ACTIVE",
            font=("Segoe UI", 9, "bold"),
            bg=self.theme["surface_alt"],
            fg=self.theme["success"],
            padx=10,
            pady=4
        )
        self.telemetry_lbl.pack(side="right", padx=18, pady=12)

        # 2. Bottom Fixed Control Bar (Packed next to guarantee visibility)
        self.footer_status = tk.Label(
            self.root,
            text=f"Ready  |  Storage Volume: {BASE_DIR}",
            font=("Segoe UI", 8),
            bg=self.theme["surface_alt"],
            fg=self.theme["text_secondary"],
            anchor="w",
            padx=16,
            pady=4
        )
        self.footer_status.pack(side="bottom", fill="x")

        action_container = tk.Frame(self.root, bg=self.theme["bg"], height=62)
        action_container.pack(side="bottom", fill="x", padx=16, pady=(0, 8))
        action_container.pack_propagate(False)

        self.btn_ok = tk.Button(
            action_container,
            text="ACCEPT PART  [ Ctrl + O ]",
            font=("Segoe UI", 10, "bold"),
            bg=self.theme["success"],
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
            cursor="hand2",
            command=lambda: self.submit_decision("ok")
        )
        self.btn_ok.pack(side="left", expand=True, fill="both", padx=(0, 6))

        self.btn_not_ok = tk.Button(
            action_container,
            text="REJECT PART  [ Ctrl + N ]",
            font=("Segoe UI", 10, "bold"),
            bg=self.theme["danger"],
            fg="#ffffff",
            activebackground="#dc2626",
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
            cursor="hand2",
            command=lambda: self.submit_decision("not_ok")
        )
        self.btn_not_ok.pack(side="left", expand=True, fill="both", padx=6)

        btn_clear = tk.Button(
            action_container,
            text="Clear Buffer [ Ctrl + L ]",
            font=("Segoe UI", 10),
            bg=self.theme["surface_alt"],
            fg=self.theme["text_secondary"],
            activebackground=self.theme["border"],
            activeforeground=self.theme["text_primary"],
            bd=0,
            relief="flat",
            cursor="hand2",
            command=self.clear_shot_buffer
        )
        btn_clear.pack(side="right", expand=True, fill="both", padx=(6, 0))

        # 3. Main Dual Viewports
        viewport_wrapper = tk.Frame(self.root, bg=self.theme["bg"])
        viewport_wrapper.pack(fill="both", expand=True, padx=16, pady=8)
        viewport_wrapper.columnconfigure(0, weight=1, uniform="feed")
        viewport_wrapper.columnconfigure(1, weight=1, uniform="feed")

        # --- VIEWPORT 1 (TOP VIEW) ---
        self.card1 = tk.Frame(viewport_wrapper, bg=self.theme["surface"], bd=1, relief="solid", highlightbackground=self.theme["border"])
        self.card1.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        
        c1_header = tk.Frame(self.card1, bg=self.theme["surface"])
        c1_header.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(c1_header, text="PRIMARY VIEW  •  TOP ORTHOGONAL", font=("Segoe UI", 9, "bold"), fg=self.theme["text_secondary"], bg=self.theme["surface"]).pack(side="left")
        self.tag_cam1 = tk.Label(c1_header, text="CONNECTING", font=("Segoe UI", 8, "bold"), fg=self.theme["warning"], bg=self.theme["surface_alt"], padx=6, pady=2)
        self.tag_cam1.pack(side="right")

        f_box1 = tk.Frame(self.card1, bg="#000000", width=460, height=270)
        f_box1.pack(padx=12, pady=4, expand=True)
        f_box1.pack_propagate(False)
        self.lbl_stream1 = tk.Label(f_box1, bg="#000000")
        self.lbl_stream1.pack(fill="both", expand=True)

        c1_actions = tk.Frame(self.card1, bg=self.theme["surface"])
        c1_actions.pack(fill="x", padx=12, pady=(4, 10))
        self.btn_c1_s1 = tk.Button(c1_actions, text="Capture View A", font=("Segoe UI", 9, "bold"), bg=self.theme["brand"], fg="#ffffff", bd=0, height=2, cursor="hand2", command=lambda: self.capture_single_shot("cam1_shot1", 1))
        self.btn_c1_s1.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_c1_s2 = tk.Button(c1_actions, text="Capture View B", font=("Segoe UI", 9, "bold"), bg=self.theme["brand"], fg="#ffffff", bd=0, height=2, cursor="hand2", command=lambda: self.capture_single_shot("cam1_shot2", 1))
        self.btn_c1_s2.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # --- VIEWPORT 2 (SIDE VIEW) ---
        self.card2 = tk.Frame(viewport_wrapper, bg=self.theme["surface"], bd=1, relief="solid", highlightbackground=self.theme["border"])
        self.card2.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        c2_header = tk.Frame(self.card2, bg=self.theme["surface"])
        c2_header.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(c2_header, text="SECONDARY VIEW  •  LATERAL PROFILE", font=("Segoe UI", 9, "bold"), fg=self.theme["text_secondary"], bg=self.theme["surface"]).pack(side="left")
        self.tag_cam2 = tk.Label(c2_header, text="CONNECTING", font=("Segoe UI", 8, "bold"), fg=self.theme["warning"], bg=self.theme["surface_alt"], padx=6, pady=2)
        self.tag_cam2.pack(side="right")

        f_box2 = tk.Frame(self.card2, bg="#000000", width=460, height=270)
        f_box2.pack(padx=12, pady=4, expand=True)
        f_box2.pack_propagate(False)
        self.lbl_stream2 = tk.Label(f_box2, bg="#000000", text="Awaiting Video Sync...", fg=self.theme["text_secondary"], font=("Segoe UI", 9))
        self.lbl_stream2.pack(fill="both", expand=True)

        c2_actions = tk.Frame(self.card2, bg=self.theme["surface"])
        c2_actions.pack(fill="x", padx=12, pady=(4, 10))
        self.btn_c2_s1 = tk.Button(c2_actions, text="Capture Profile A", font=("Segoe UI", 9, "bold"), bg=self.theme["brand"], fg="#ffffff", bd=0, height=2, cursor="hand2", command=lambda: self.capture_single_shot("cam2_shot1", 2))
        self.btn_c2_s1.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_c2_s2 = tk.Button(c2_actions, text="Capture Profile B", font=("Segoe UI", 9, "bold"), bg=self.theme["brand"], fg="#ffffff", bd=0, height=2, cursor="hand2", command=lambda: self.capture_single_shot("cam2_shot2", 2))
        self.btn_c2_s2.pack(side="right", expand=True, fill="x", padx=(4, 0))

    def _register_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.submit_decision("ok"))
        self.root.bind("<Control-O>", lambda e: self.submit_decision("ok"))
        self.root.bind("<Control-n>", lambda e: self.submit_decision("not_ok"))
        self.root.bind("<Control-N>", lambda e: self.submit_decision("not_ok"))
        self.root.bind("<Control-l>", lambda e: self.clear_shot_buffer())
        self.root.bind("<Control-L>", lambda e: self.clear_shot_buffer())

    def _render_stream_cycle(self):
        """Ultra-lightweight rendering loop (Tkinter UI thread)."""
        # Camera 1 Pipeline
        c1_ok, f1 = self.cam1_worker.get_frame()
        if c1_ok:
            self.tag_cam1.config(text="ONLINE", fg=self.theme["success"])
            rgb1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
            img1 = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(rgb1, (460, 270))))
            self.lbl_stream1.configure(image=img1, text="")
            self.lbl_stream1.image = img1
        else:
            self.tag_cam1.config(text="OFFLINE", fg=self.theme["danger"])
            self.lbl_stream1.configure(image="", text="Top Camera Offline", fg=self.theme["text_secondary"])

        # Camera 2 Pipeline
        c2_ok, f2 = self.cam2_worker.get_frame()
        if c2_ok:
            self.tag_cam2.config(text="ONLINE", fg=self.theme["success"])
            rgb2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
            img2 = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(rgb2, (460, 270))))
            self.lbl_stream2.configure(image=img2, text="")
            self.lbl_stream2.image = img2
        else:
            self.tag_cam2.config(text="OFFLINE", fg=self.theme["danger"])
            self.lbl_stream2.configure(image="", text="Side Camera Offline", fg=self.theme["text_secondary"])

        # Repeat at ~33 FPS without taxing the CPU
        self.root.after(30, self._render_stream_cycle)

    def capture_single_shot(self, shot_key: str, cam_source: int):
        worker = self.cam1_worker if cam_source == 1 else self.cam2_worker
        ok, frame = worker.get_frame()
        if ok:
            with self.buffer_lock:
                self.shot_buffer[shot_key] = frame
            self.footer_status.config(text=f"Buffer State: Registered {shot_key}", fg=self.theme["brand"])
        else:
            messagebox.showwarning("Hardware Fault", f"Camera {cam_source} stream disconnected.")

    def submit_decision(self, verdict: str):
        """Asynchronously dispatches image save pipeline without GUI blocking."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        target_dir = DIR_OK if verdict == "ok" else DIR_NOT_OK

        payload = {}
        with self.buffer_lock:
            # Check manual buffer
            for k, v in self.shot_buffer.items():
                if v is not None:
                    payload[k] = v.copy()
            
            # If buffer empty, fallback to immediate live snapshots
            if not payload:
                ok1, f1 = self.cam1_worker.get_frame()
                if ok1: payload["Cam1_Top"] = f1
                ok2, f2 = self.cam2_worker.get_frame()
                if ok2: payload["Cam2_Side"] = f2

            # Flush buffer
            for k in self.shot_buffer:
                self.shot_buffer[k] = None

        if not payload:
            messagebox.showwarning("Acquisition Failed", "No valid image data available from cameras.")
            return

        # Fire and forget onto disk worker thread
        self.io_pool.submit(self._async_disk_writer, payload, target_dir, ts, verdict)

    def _async_disk_writer(self, payload: dict, directory: str, timestamp: str, verdict: str):
        """Runs in separate I/O thread pool."""
        try:
            saved_names = []
            for name, frame in payload.items():
                filename = f"{name}_{timestamp}_{verdict.upper()}.jpg"
                path = os.path.join(directory, filename)
                # Production encoding: High-quality JPEG without compression artifact
                cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_names.append(filename)

            logger.info(f"Committed {len(saved_names)} records -> [{verdict.upper()}]")
            # Update GUI from main loop safely
            self.root.after(0, lambda: self.footer_status.config(
                text=f"✓ Committed {len(saved_names)} records to [{verdict.upper()}]  |  Timestamp: {timestamp}",
                fg=self.theme["success"] if verdict == "ok" else self.theme["danger"]
            ))
        except Exception as e:
            logger.error(f"Critical I/O Error during commit: {str(e)}")

    def clear_shot_buffer(self):
        with self.buffer_lock:
            for k in self.shot_buffer:
                self.shot_buffer[k] = None
        self.footer_status.config(text="Buffer Flushed  •  Workstation Ready", fg=self.theme["text_secondary"])

    def shutdown(self):
        logger.info("Initiating graceful system shutdown...")
        self.cam1_worker.stop()
        self.cam2_worker.stop()
        self.io_pool.shutdown(wait=False)
        self.root.destroy()


# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    app_window = tk.Tk()
    station = EnterpriseWorkstationApp(app_window)
    app_window.protocol("WM_DELETE_WINDOW", station.shutdown)
    app_window.mainloop()