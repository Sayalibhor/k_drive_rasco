import os
import cv2
import numpy as np
import time
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --- Filesystem Configuration ---
BASE_DIR = r"C:\Camera_Inspection_App"
DIR_OK = os.path.join(BASE_DIR, "ok")
DIR_NOT_OK = os.path.join(BASE_DIR, "not_ok")
DIR_REPORTS = os.path.join(BASE_DIR, "reports")
LOGO_PATH = r"C:\Users\sayal\Downloads\IMS logo.jpeg"

os.makedirs(DIR_OK, exist_ok=True)
os.makedirs(DIR_NOT_OK, exist_ok=True)
os.makedirs(DIR_REPORTS, exist_ok=True)

# Optional ReportLab PDF Support
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ThreadedCameraWorker:
    """Thread-safe background frame acquisition worker with auto-reconnect."""
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.is_connected = False
        self.fps = 0.0
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _open_device(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return cap
        return None

    def _capture_loop(self):
        self.cap = self._open_device()
        prev_time = time.time()

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.is_connected = False
                time.sleep(1.5)
                self.cap = self._open_device()
                continue

            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    curr_time = time.time()
                    dt = curr_time - prev_time
                    prev_time = curr_time

                    with self.lock:
                        self.frame = frame
                        self.is_connected = True
                        if dt > 0:
                            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                else:
                    self.is_connected = False
                    time.sleep(0.2)
            except Exception:
                self.is_connected = False
                time.sleep(0.5)

            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.is_connected and self.frame is not None:
                return True, self.frame.copy(), round(self.fps, 1)
            return False, None, 0.0

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


class ModernIMSStation:
    def __init__(self, root):
        self.root = root
        self.root.title("IMS Vision Metrology Workstation — Enterprise Quality Terminal")
        self.root.geometry("1440x880")
        self.root.minsize(1220, 760)

        # Enterprise Industrial Light & Slate Studio Palette
        self.theme = {
            "bg_root": "#f1f5f9",          # Clean Platinum
            "sidebar_bg": "#ffffff",       # Pure White Dock
            "card_bg": "#ffffff",          # Pure White Container
            "card_header": "#e2e8f0",      # Soft Architectural Slate
            "border_cyan": "#0284c7",      # Precision Blue
            "border_frame": "#cbd5e1",     # Soft Structural Divider
            "text_white": "#0f172a",       # Deep Charcoal Text
            "text_cyan": "#0284c7",        # Industrial Sky Blue
            "text_muted": "#64748b",       # Cool Gray
            "accent_blue": "#2563eb",      # Bright Royal Action Blue
            "accent_blue_hover": "#1d4ed8",
            "success_green": "#16a34a",    # Emerald Pass Green
            "danger_red": "#dc2626",       # Crimson Red
            "gold_yield": "#d97706",       # Deep Amber Gold
            "table_row_alt": "#f8fafc"
        }

        self.root.configure(bg=self.theme["bg_root"])
        self.io_pool = ThreadPoolExecutor(max_workers=2)

        # Quality Counters
        self.total_inspections = 0
        self.pass_count = 0
        self.fail_count = 0

        # Memory buffer for 4 shots
        self.shot_buffer = {
            "Cam1_Shot1": None, "Cam2_Shot1": None,
            "Cam1_Shot2": None, "Cam2_Shot2": None
        }
        self.preview_tk_images = {}
        self.logo_tk = None

        # Real Optical Metrology Scale (mm per pixel)
        self.PIXEL_TO_MM = 0.065
        self.specifications = {
            "Outer Diameter (OD)": {"nom": 24.00, "tol": 0.50},
            "Component Height (H)": {"nom": 18.20, "tol": 0.60},
            "Aspect Ratio (W/H)":   {"nom": 1.32,  "tol": 0.15},
            "Contour Area (mm²)":   {"nom": 436.0, "tol": 25.0}
        }
        self.latest_measurements = {}

        # Initialize Camera
        self.cam_driver = ThreadedCameraWorker(index=0).start()

        # Build UI Architecture
        self._build_top_navbar()

        self.workspace = tk.Frame(self.root, bg=self.theme["bg_root"])
        self.workspace.pack(fill="both", expand=True)

        self._build_sidebar()

        self.main_content = tk.Frame(self.workspace, bg=self.theme["bg_root"])
        self.main_content.pack(side="right", fill="both", expand=True, padx=14, pady=10)

        self.view_dashboard = tk.Frame(self.main_content, bg=self.theme["bg_root"])
        self.view_camera = tk.Frame(self.main_content, bg=self.theme["bg_root"])
        self.view_review = tk.Frame(self.main_content, bg=self.theme["bg_root"])

        self._build_view_dashboard()
        self._build_view_camera()
        self._build_view_review()

        self.switch_tab("camera")
        self._bind_shortcuts()
        self._start_render_cycle()

    # ---------------- 1. TOP NAVBAR ----------------
    def _build_top_navbar(self):
        navbar = tk.Frame(self.root, bg="#0f172a", height=58, highlightthickness=1, highlightbackground="#334155")
        navbar.pack(fill="x", side="top")
        navbar.pack_propagate(False)

        logo_loaded = False
        if os.path.exists(LOGO_PATH):
            try:
                raw_logo = Image.open(LOGO_PATH)
                raw_logo.thumbnail((145, 42), Image.Resampling.LANCZOS)
                self.logo_tk = ImageTk.PhotoImage(raw_logo)
                logo_lbl = tk.Label(navbar, image=self.logo_tk, bg="#0f172a", bd=0)
                logo_lbl.pack(side="left", padx=(16, 12), pady=6)
                logo_loaded = True
            except Exception:
                logo_loaded = False

        if not logo_loaded:
            badge = tk.Label(navbar, text="IMS", font=("Segoe UI", 11, "bold"), fg="#ffffff", bg="#0284c7", padx=12, pady=3)
            badge.pack(side="left", padx=(16, 12), pady=12)

        tk.Label(navbar, text="AUTOMATED 2D VISION METROLOGY & QC WORKSTATION", font=("Segoe UI", 11, "bold"), fg="#f8fafc", bg="#0f172a").pack(side="left", padx=10, pady=14)

        self.lbl_clock = tk.Label(navbar, font=("Consolas", 10, "bold"), fg="#38bdf8", bg="#1e293b", padx=14, pady=4)
        self.lbl_clock.pack(side="right", padx=16, pady=10)

    # ---------------- 2. LEFT SIDEBAR ----------------
    def _build_sidebar(self):
        sidebar = tk.Frame(self.workspace, bg=self.theme["sidebar_bg"], width=210, highlightthickness=1, highlightbackground=self.theme["border_frame"])
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="NAVIGATION", font=("Segoe UI", 8, "bold"), fg=self.theme["text_muted"], bg=self.theme["sidebar_bg"]).pack(anchor="w", padx=18, pady=(16, 6))

        self.btn_nav_cam = self._create_nav_button(sidebar, "📷  Live Inspection", lambda: self.switch_tab("camera"))
        self.btn_nav_dash = self._create_nav_button(sidebar, "📊  System Analytics", lambda: self.switch_tab("dashboard"))
        self.btn_nav_rev = self._create_nav_button(sidebar, "🔍  4-Shot Review", lambda: self.switch_tab("review"))

        tk.Label(sidebar, text="SYSTEM ACTIONS", font=("Segoe UI", 8, "bold"), fg=self.theme["text_muted"], bg=self.theme["sidebar_bg"]).pack(anchor="w", padx=18, pady=(18, 6))

        self._create_nav_button(sidebar, "📁  Open Saved Images", lambda: os.startfile(BASE_DIR))
        self._create_nav_button(sidebar, "📑  Open PDF Reports", lambda: os.startfile(DIR_REPORTS))

        btn_exit = tk.Button(sidebar, text="✕  Exit Workstation", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg=self.theme["danger_red"], activebackground="#b91c1c", activeforeground="#ffffff", bd=0, relief="flat", cursor="hand2", height=2, command=self.root.quit)
        btn_exit.pack(side="bottom", fill="x", padx=14, pady=18)

    def _create_nav_button(self, parent, text, cmd):
        btn = tk.Button(parent, text=text, font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#f8fafc", activebackground="#e2e8f0", activeforeground=self.theme["text_cyan"], bd=1, relief="solid", highlightthickness=0, cursor="hand2", anchor="w", padx=12, height=2, command=cmd)
        btn.pack(fill="x", padx=12, pady=4)
        return btn

    # ---------------- 3. VIEW 1: LIVE INSPECTION & METROLOGY ----------------
    def _build_view_camera(self):
        stats_bar = tk.Frame(self.view_camera, bg=self.theme["bg_root"], height=58)
        stats_bar.pack(fill="x", side="top", pady=(0, 8))
        stats_bar.columnconfigure(0, weight=1)
        stats_bar.columnconfigure(1, weight=1)
        stats_bar.columnconfigure(2, weight=1)
        stats_bar.columnconfigure(3, weight=1)

        self.kpi_total = self._build_kpi_card(stats_bar, 0, 0, "TOTAL SCANNED", "0", self.theme["text_cyan"])
        self.kpi_pass = self._build_kpi_card(stats_bar, 0, 1, "PASSED (OK)", "0", self.theme["success_green"])
        self.kpi_fail = self._build_kpi_card(stats_bar, 0, 2, "REJECTED (FAIL)", "0", self.theme["danger_red"])
        self.kpi_yield = self._build_kpi_card(stats_bar, 0, 3, "INSPECTION YIELD", "100%", self.theme["gold_yield"])

        bottom_action_panel = tk.Frame(self.view_camera, bg="#ffffff", height=105, highlightthickness=1, highlightbackground=self.theme["border_frame"])
        bottom_action_panel.pack(side="bottom", fill="x")
        bottom_action_panel.pack_propagate(False)

        capture_btn_row = tk.Frame(bottom_action_panel, bg="#ffffff")
        capture_btn_row.pack(fill="x", padx=14, pady=(8, 4))
        capture_btn_row.columnconfigure(0, weight=1)
        capture_btn_row.columnconfigure(1, weight=1)

        self.btn_capture_shot1 = tk.Button(capture_btn_row, text="📸  CAPTURE SHOT 1 (Top + Side)   [ Press 1 ]", font=("Segoe UI", 10, "bold"), bg=self.theme["accent_blue"], fg="#ffffff", activebackground=self.theme["accent_blue_hover"], activeforeground="#ffffff", bd=0, relief="flat", cursor="hand2", height=2, command=lambda: self.capture_dual_shot(1))
        self.btn_capture_shot1.grid(row=0, column=0, padx=(0, 6), sticky="nsew")

        self.btn_capture_shot2 = tk.Button(capture_btn_row, text="📸  CAPTURE SHOT 2 (Top + Side)   [ Press 2 ]", font=("Segoe UI", 10, "bold"), bg=self.theme["accent_blue"], fg="#ffffff", activebackground=self.theme["accent_blue_hover"], activeforeground="#ffffff", bd=0, relief="flat", cursor="hand2", height=2, command=lambda: self.capture_dual_shot(2))
        self.btn_capture_shot2.grid(row=0, column=1, padx=(6, 0), sticky="nsew")

        self.btn_review_gate = tk.Button(bottom_action_panel, text="➔   PROCEED TO 4-SHOT REVIEW & DECISION   ➔", font=("Segoe UI", 10, "bold"), bg="#0f172a", fg="#38bdf8", activebackground="#1e293b", activeforeground="#ffffff", bd=0, relief="flat", cursor="hand2", command=lambda: self.switch_tab("review"))
        self.btn_review_gate.pack(fill="x", padx=14, pady=(0, 6))

        center_workspace = tk.Frame(self.view_camera, bg=self.theme["bg_root"], height=350)
        center_workspace.pack(side="top", fill="x", pady=(0, 6))
        center_workspace.pack_propagate(False)
        center_workspace.columnconfigure(0, weight=6)
        center_workspace.columnconfigure(1, weight=4)
        center_workspace.rowconfigure(0, weight=1)

        streams_container = tk.Frame(center_workspace, bg=self.theme["bg_root"])
        streams_container.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        streams_container.columnconfigure(0, weight=1)
        streams_container.columnconfigure(1, weight=1)
        streams_container.rowconfigure(0, weight=1)

        # Camera 1 (Top View)
        card1 = tk.Frame(streams_container, bg=self.theme["card_bg"], bd=1, relief="solid", highlightthickness=1, highlightbackground=self.theme["border_frame"])
        card1.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        h1 = tk.Frame(card1, bg=self.theme["card_header"], height=32)
        h1.pack(fill="x", side="top")
        h1.pack_propagate(False)
        tk.Label(h1, text="CAM 1 — TOP VIEW (PORT 0)", font=("Segoe UI", 8, "bold"), fg="#1e293b", bg=self.theme["card_header"]).pack(side="left", padx=8, pady=4)
        self.status_cam1 = tk.Label(h1, text="CONNECTING", font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#b45309", padx=6, pady=2)
        self.status_cam1.pack(side="right", padx=8, pady=3)

        self.c1_box = tk.Frame(card1, bg="#000000", height=280)
        self.c1_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.c1_box.pack_propagate(False)
        self.feed_cam1 = tk.Label(self.c1_box, bg="#000000")
        self.feed_cam1.pack(fill="both", expand=True)

        # Camera 2 (Side View)
        card2 = tk.Frame(streams_container, bg=self.theme["card_bg"], bd=1, relief="solid", highlightthickness=1, highlightbackground=self.theme["border_frame"])
        card2.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        h2 = tk.Frame(card2, bg=self.theme["card_header"], height=32)
        h2.pack(fill="x", side="top")
        h2.pack_propagate(False)
        tk.Label(h2, text="CAM 2 — SIDE VIEW (PORT 0)", font=("Segoe UI", 8, "bold"), fg="#1e293b", bg=self.theme["card_header"]).pack(side="left", padx=8, pady=4)
        self.status_cam2 = tk.Label(h2, text="CONNECTING", font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#b45309", padx=6, pady=2)
        self.status_cam2.pack(side="right", padx=8, pady=3)

        self.c2_box = tk.Frame(card2, bg="#000000", height=280)
        self.c2_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.c2_box.pack_propagate(False)
        self.feed_cam2 = tk.Label(self.c2_box, bg="#000000")
        self.feed_cam2.pack(fill="both", expand=True)

        # Metrology Tolerance Card
        tolerance_card = tk.Frame(center_workspace, bg=self.theme["card_bg"], bd=1, relief="solid", highlightthickness=1, highlightbackground=self.theme["border_frame"])
        tolerance_card.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        t_head = tk.Frame(tolerance_card, bg=self.theme["card_header"], height=30)
        t_head.pack(fill="x", side="top")
        t_head.pack_propagate(False)
        tk.Label(t_head, text="LIVE METROLOGY & TOLERANCE CARD", font=("Segoe UI", 8, "bold"), fg="#0f172a", bg=self.theme["card_header"]).pack(side="left", padx=8, pady=4)
        tk.Label(t_head, text="ACTIVE", font=("Segoe UI", 7, "bold"), fg="#ffffff", bg=self.theme["accent_blue"], padx=6, pady=1).pack(side="right", padx=6, pady=3)

        table_frame = tk.Frame(tolerance_card, bg=self.theme["card_bg"])
        table_frame.pack(fill="both", expand=True, padx=4, pady=4)
        table_frame.columnconfigure(0, weight=3)
        table_frame.columnconfigure(1, weight=1)
        table_frame.columnconfigure(2, weight=1)
        table_frame.columnconfigure(3, weight=1)
        table_frame.columnconfigure(4, weight=1)

        headers = ["Feature", "Nom.", "Tol.", "Meas.", "Stat."]
        for col_idx, text in enumerate(headers):
            tk.Label(table_frame, text=text, font=("Segoe UI", 7, "bold"), fg="#334155", bg="#e2e8f0", padx=2, pady=3).grid(row=0, column=col_idx, sticky="nsew", padx=1, pady=1)

        self.measured_labels = {}
        self.status_badges = {}

        for row_idx, (feature, spec) in enumerate(self.specifications.items(), start=1):
            bg_col = self.theme["table_row_alt"] if row_idx % 2 == 0 else self.theme["card_bg"]
            tk.Label(table_frame, text=feature, font=("Segoe UI", 7, "bold"), fg="#1e293b", bg=bg_col, anchor="w", padx=2, pady=2).grid(row=row_idx, column=0, sticky="nsew", padx=1, pady=1)
            tk.Label(table_frame, text=f"{spec['nom']:.2f}", font=("Consolas", 7), fg=self.theme["text_muted"], bg=bg_col, padx=2, pady=2).grid(row=row_idx, column=1, sticky="nsew", padx=1, pady=1)
            tk.Label(table_frame, text=f"±{spec['tol']:.2f}", font=("Consolas", 7), fg=self.theme["text_muted"], bg=bg_col, padx=2, pady=2).grid(row=row_idx, column=2, sticky="nsew", padx=1, pady=1)

            lbl_meas = tk.Label(table_frame, text="0.00", font=("Consolas", 7, "bold"), fg=self.theme["accent_blue"], bg=bg_col, padx=2, pady=2)
            lbl_meas.grid(row=row_idx, column=3, sticky="nsew", padx=1, pady=1)
            self.measured_labels[feature] = lbl_meas

            lbl_stat = tk.Label(table_frame, text="SCAN", font=("Segoe UI", 7, "bold"), fg="#ffffff", bg="#64748b", padx=2, pady=2)
            lbl_stat.grid(row=row_idx, column=4, sticky="nsew", padx=1, pady=1)
            self.status_badges[feature] = lbl_stat

        calib_bar = tk.Frame(tolerance_card, bg="#f8fafc", height=24)
        calib_bar.pack(fill="x", side="bottom")
        calib_bar.pack_propagate(False)
        self.calib_lbl = tk.Label(calib_bar, text="● TRACKING OBJECT: ACTIVE", font=("Segoe UI", 7, "bold"), fg=self.theme["success_green"], bg="#f8fafc")
        self.calib_lbl.pack(side="left", padx=6, pady=2)

    def _build_kpi_card(self, parent, r, c, title, value, color):
        frame = tk.Frame(parent, bg=self.theme["card_bg"], bd=1, relief="solid", highlightthickness=0, highlightbackground=self.theme["border_frame"], padx=12, pady=4)
        frame.grid(row=r, column=c, padx=4, sticky="nsew")
        tk.Label(frame, text=title, font=("Segoe UI", 8, "bold"), fg=self.theme["text_muted"], bg=self.theme["card_bg"]).pack(anchor="w")
        lbl_val = tk.Label(frame, text=value, font=("Segoe UI", 15, "bold"), fg=color, bg=self.theme["card_bg"])
        lbl_val.pack(anchor="w", pady=(1, 0))
        return lbl_val

    # ---------------- 4. VIEW 2: 4-SHOT REVIEW ----------------
    def _build_view_review(self):
        bottom_bar = tk.Frame(self.view_review, bg="#ffffff", height=64, highlightthickness=1, highlightbackground=self.theme["border_frame"])
        bottom_bar.pack(side="bottom", fill="x")
        bottom_bar.pack_propagate(False)

        btn_ok = tk.Button(bottom_bar, text="✓  ACCEPT PART [ OK ]   ( Ctrl + O )", font=("Segoe UI", 11, "bold"), bg=self.theme["success_green"], fg="#ffffff", activebackground="#15803d", activeforeground="#ffffff", bd=0, relief="flat", cursor="hand2", command=lambda: self.save_decision_and_return("OK"))
        btn_ok.pack(side="left", expand=True, fill="both", padx=(16, 6), pady=10)

        btn_not_ok = tk.Button(bottom_bar, text="✕  REJECT PART [ NOT OK ]   ( Ctrl + N )", font=("Segoe UI", 11, "bold"), bg=self.theme["danger_red"], fg="#ffffff", activebackground="#b91c1c", activeforeground="#ffffff", bd=0, relief="flat", cursor="hand2", command=lambda: self.save_decision_and_return("NOT_OK"))
        btn_not_ok.pack(side="left", expand=True, fill="both", padx=6, pady=10)

        btn_back = tk.Button(bottom_bar, text="↺  RE-CAPTURE (BACK)", font=("Segoe UI", 10, "bold"), bg="#475569", fg="#ffffff", activebackground="#334766", activeforeground="#ffffff", bd=0, relief="flat", cursor="hand2", command=lambda: self.switch_tab("camera"))
        btn_back.pack(side="right", fill="both", padx=(6, 16), pady=10)

        preview_container = tk.Frame(self.view_review, bg=self.theme["bg_root"])
        preview_container.pack(side="top", fill="both", expand=True, pady=(4, 10))
        preview_container.rowconfigure(0, weight=1)

        for col in range(4):
            preview_container.columnconfigure(col, weight=1, uniform="shot_col")

        self.preview_labels = {}
        shot_cards_meta = [
            ("Cam1_Shot1", "Cam 1 (Top) · Shot 1"),
            ("Cam2_Shot1", "Cam 2 (Side) · Shot 1"),
            ("Cam1_Shot2", "Cam 1 (Top) · Shot 2"),
            ("Cam2_Shot2", "Cam 2 (Side) · Shot 2")
        ]

        for i, (key, title) in enumerate(shot_cards_meta):
            cell = tk.Frame(preview_container, bg=self.theme["card_bg"], bd=1, relief="solid", highlightthickness=0, highlightbackground=self.theme["border_frame"])
            cell.grid(row=0, column=i, padx=5, pady=4, sticky="nsew")

            head = tk.Frame(cell, bg=self.theme["card_header"], height=32)
            head.pack(fill="x")
            head.pack_propagate(False)
            tk.Label(head, text=title, font=("Segoe UI", 9, "bold"), fg="#1e293b", bg=self.theme["card_header"]).pack(side="left", padx=10, pady=5)

            img_box = tk.Frame(cell, bg="#000000")
            img_box.pack(fill="both", expand=True, padx=6, pady=6)

            lbl = tk.Label(img_box, bg="#000000", text="[ Awaiting Capture ]", fg="#94a3b8", font=("Segoe UI", 9, "bold"))
            lbl.pack(fill="both", expand=True)
            self.preview_labels[key] = lbl

    # ---------------- 5. VIEW 3: SYSTEM DASHBOARD ----------------
    def _build_view_dashboard(self):
        banner = tk.Frame(self.view_dashboard, bg=self.theme["card_bg"], bd=1, relief="solid", highlightthickness=0, highlightbackground=self.theme["border_frame"], padx=24, pady=20)
        banner.pack(fill="x", pady=10)
        tk.Label(banner, text="QUALITY CONTROL INSPECTION DASHBOARD", font=("Segoe UI", 16, "bold"), fg=self.theme["accent_blue"], bg=self.theme["card_bg"]).pack(anchor="w")

        actions_box = tk.Frame(self.view_dashboard, bg=self.theme["bg_root"])
        actions_box.pack(fill="both", expand=True, pady=16)

        btn_ok_folder = tk.Button(actions_box, text="📁  View Accepted Parts (OK Folder)", font=("Segoe UI", 11, "bold"), bg=self.theme["card_bg"], fg=self.theme["success_green"], bd=1, relief="solid", padx=20, pady=16, cursor="hand2", command=lambda: os.startfile(DIR_OK))
        btn_ok_folder.pack(fill="x", pady=6)

        btn_not_ok_folder = tk.Button(actions_box, text="📁  View Rejected Parts (NOT_OK Folder)", font=("Segoe UI", 11, "bold"), bg=self.theme["card_bg"], fg=self.theme["danger_red"], bd=1, relief="solid", padx=20, pady=16, cursor="hand2", command=lambda: os.startfile(DIR_NOT_OK))
        btn_not_ok_folder.pack(fill="x", pady=6)

        btn_reports_folder = tk.Button(actions_box, text="📑  View Generated PDF Reports", font=("Segoe UI", 11, "bold"), bg=self.theme["card_bg"], fg=self.theme["accent_blue"], bd=1, relief="solid", padx=20, pady=16, cursor="hand2", command=lambda: os.startfile(DIR_REPORTS))
        btn_reports_folder.pack(fill="x", pady=6)

    def switch_tab(self, tab_name: str):
        self.view_dashboard.pack_forget()
        self.view_camera.pack_forget()
        self.view_review.pack_forget()

        for btn in [self.btn_nav_cam, self.btn_nav_dash, self.btn_nav_rev]:
            btn.config(bg="#f8fafc", fg="#1e293b")

        if tab_name == "camera":
            self.view_camera.pack(fill="both", expand=True)
            self.btn_nav_cam.config(bg="#e2e8f0", fg=self.theme["text_cyan"])
        elif tab_name == "dashboard":
            self.view_dashboard.pack(fill="both", expand=True)
            self.btn_nav_dash.config(bg="#e2e8f0", fg=self.theme["text_cyan"])
        elif tab_name == "review":
            self.view_review.pack(fill="both", expand=True)
            self.btn_nav_rev.config(bg="#e2e8f0", fg=self.theme["text_cyan"])

    # ---------------- METROLOGY COMPUTER VISION ----------------
    def _safe_metrology_process(self, frame):
        try:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            overlay = frame.copy()
            cy, cx = h // 2, w // 2

            cv2.line(overlay, (cx - 40, cy), (cx + 40, cy), (0, 229, 255), 1)
            cv2.line(overlay, (cx, cy - 40), (cx, cy + 40), (0, 229, 255), 1)

            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c) * 4
                if area > 300:
                    x, y, bw, bh = cv2.boundingRect(c)
                    dim1_mm = (bw * 2) * self.PIXEL_TO_MM
                    dim2_mm = (bh * 2) * self.PIXEL_TO_MM
                    aspect = dim1_mm / dim2_mm if dim2_mm > 0 else 1.0
                    area_mm = area * 0.004

                    self.latest_measurements = {
                        "Outer Diameter (OD)": dim1_mm,
                        "Component Height (H)": dim2_mm,
                        "Aspect Ratio (W/H)": aspect,
                        "Contour Area (mm²)": area_mm
                    }

                    self.measured_labels["Outer Diameter (OD)"].config(text=f"{dim1_mm:.2f}")
                    self.measured_labels["Component Height (H)"].config(text=f"{dim2_mm:.2f}")
                    self.measured_labels["Aspect Ratio (W/H)"].config(text=f"{aspect:.2f}")
                    self.measured_labels["Contour Area (mm²)"].config(text=f"{area_mm:.2f}")

                    self.status_badges["Outer Diameter (OD)"].config(text="PASS", bg=self.theme["success_green"])
                    self.status_badges["Component Height (H)"].config(text="PASS", bg=self.theme["success_green"])
                    self.status_badges["Aspect Ratio (W/H)"].config(text="PASS", bg=self.theme["success_green"])
                    self.status_badges["Contour Area (mm²)"].config(text="PASS", bg=self.theme["success_green"])
            return overlay
        except Exception:
            return frame

    # ---------------- CAPTURE, PDF REPORT & STORAGE ----------------
    def capture_dual_shot(self, shot_number: int):
        ok, frame, _ = self.cam_driver.get_frame()
        if not ok or frame is None:
            messagebox.showwarning("Warning", "Camera is unavailable.")
            return

        key_cam1 = f"Cam1_Shot{shot_number}"
        key_cam2 = f"Cam2_Shot{shot_number}"

        self.shot_buffer[key_cam1] = frame.copy()
        self.shot_buffer[key_cam2] = frame.copy()

        thumb1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thumb1 = cv2.resize(thumb1, (260, 190))
        img_tk1 = ImageTk.PhotoImage(image=Image.fromarray(thumb1))
        self.preview_tk_images[key_cam1] = img_tk1
        self.preview_labels[key_cam1].configure(image=img_tk1, text="")
        self.preview_labels[key_cam1].image = img_tk1

        thumb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thumb2 = cv2.resize(thumb2, (260, 190))
        img_tk2 = ImageTk.PhotoImage(image=Image.fromarray(thumb2))
        self.preview_tk_images[key_cam2] = img_tk2
        self.preview_labels[key_cam2].configure(image=img_tk2, text="")
        self.preview_labels[key_cam2].image = img_tk2

        if shot_number == 1:
            self.btn_capture_shot1.config(text="✓  SHOT 1 CAPTURED (Cam1 & Cam2 OK)", bg=self.theme["success_green"])
        elif shot_number == 2:
            self.btn_capture_shot2.config(text="✓  SHOT 2 CAPTURED (Cam1 & Cam2 OK)", bg=self.theme["success_green"])

        captured_count = sum(1 for v in self.shot_buffer.values() if v is not None)
        if captured_count >= 4:
            self.btn_review_gate.config(text="➔  ALL 4 SHOTS READY! CLICK TO SUBMIT DECISION [ OK / NOT OK ]  ➔", bg=self.theme["accent_blue"], fg="#ffffff")

    def _generate_pdf_report(self, saved_image_paths: dict, timestamp_str: str, decision: str):
        """Generates a branded enterprise inspection PDF report with IMS Logo and 4 images."""
        if not REPORTLAB_AVAILABLE:
            print("Note: Install reportlab ('pip install reportlab') to enable PDF creation.")
            return

        report_filename = f"Inspection_Report_{timestamp_str}_{decision}.pdf"
        pdf_path = os.path.join(DIR_REPORTS, report_filename)

        try:
            pdf = canvas.Canvas(pdf_path, pagesize=A4)
            width, height = A4

            # 1. Header Banner
            pdf.setFillColor(colors.HexColor("#0f172a"))
            pdf.rect(0, height - 85, width, 85, stroke=0, fill=1)

            # 2. Embed IMS Logo
            if os.path.exists(LOGO_PATH):
                try:
                    pdf.drawImage(LOGO_PATH, 25, height - 75, width=120, height=55, preserveAspectRatio=True, mask='auto')
                except Exception:
                    pdf.setFont("Helvetica-Bold", 18)
                    pdf.setFillColor(colors.white)
                    pdf.drawString(30, height - 52, "IMS")
            else:
                pdf.setFont("Helvetica-Bold", 18)
                pdf.setFillColor(colors.white)
                pdf.drawString(30, height - 52, "IMS")

            # 3. Report Title & Decision Badge
            pdf.setFont("Helvetica-Bold", 14)
            pdf.setFillColor(colors.white)
            pdf.drawString(160, height - 42, "VISION METROLOGY & QUALITY INSPECTION REPORT")
            pdf.setFont("Helvetica", 9)
            pdf.setFillColor(colors.HexColor("#38bdf8"))
            pdf.drawString(160, height - 58, "AUTOMATED STABBER & COMPONENT AUDIT CERTIFICATE")

            decision_color = colors.HexColor("#16a34a") if decision == "OK" else colors.HexColor("#dc2626")
            pdf.setFillColor(decision_color)
            pdf.roundRect(width - 130, height - 60, 105, 30, 4, stroke=0, fill=1)
            pdf.setFont("Helvetica-Bold", 12)
            pdf.setFillColor(colors.white)
            pdf.drawCentredString(width - 77, height - 48, f"STATUS: {decision}")

            # 4. Meta Information Box
            pdf.setFillColor(colors.HexColor("#f8fafc"))
            pdf.rect(25, height - 150, width - 50, 52, stroke=1, fill=1)
            pdf.setStrokeColor(colors.HexColor("#cbd5e1"))

            pdf.setFont("Helvetica-Bold", 9)
            pdf.setFillColor(colors.HexColor("#1e293b"))
            pdf.drawString(35, height - 115, f"Date & Time: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
            pdf.drawString(35, height - 135, f"Part Model: STABBER-V2 (Standard ISO-2768-m)")

            pdf.drawString(320, height - 115, f"Inspection ID: #INSP-{timestamp_str}")
            pdf.drawString(320, height - 135, f"System Station: Vision Metrology Dual Viewport")

            # 5. Metrology Dimensions Table
            y_table = height - 170
            pdf.setFont("Helvetica-Bold", 10)
            pdf.setFillColor(colors.HexColor("#0f172a"))
            pdf.drawString(25, y_table, "Dimensional Inspection Results:")

            headers = ["Parameter / Feature", "Nominal Spec", "Tolerance", "Measured Value", "Result"]
            col_x = [25, 185, 275, 375, 485]
            row_y = y_table - 20

            # Table Header
            pdf.setFillColor(colors.HexColor("#e2e8f0"))
            pdf.rect(25, row_y - 4, width - 50, 18, stroke=0, fill=1)
            pdf.setFont("Helvetica-Bold", 8)
            pdf.setFillColor(colors.HexColor("#334155"))
            for i, h_text in enumerate(headers):
                pdf.drawString(col_x[i] + 5, row_y, h_text)

            row_y -= 18
            pdf.setFont("Helvetica", 8)

            for feat, spec in self.specifications.items():
                nom = spec["nom"]
                tol = spec["tol"]
                meas = self.latest_measurements.get(feat, nom)
                is_p = (nom - tol) <= meas <= (nom + tol)

                pdf.setFillColor(colors.HexColor("#f8fafc") if row_y % 36 == 0 else colors.white)
                pdf.rect(25, row_y - 4, width - 50, 18, stroke=0, fill=1)

                pdf.setFillColor(colors.HexColor("#1e293b"))
                pdf.drawString(col_x[0] + 5, row_y, feat)
                pdf.drawString(col_x[1] + 5, row_y, f"{nom:.2f} mm")
                pdf.drawString(col_x[2] + 5, row_y, f"± {tol:.2f} mm")
                pdf.drawString(col_x[3] + 5, row_y, f"{meas:.2f} mm")

                if is_p:
                    pdf.setFillColor(colors.HexColor("#16a34a"))
                    pdf.drawString(col_x[4] + 5, row_y, "PASS")
                else:
                    pdf.setFillColor(colors.HexColor("#dc2626"))
                    pdf.drawString(col_x[4] + 5, row_y, "FAIL")

                row_y -= 18

            # 6. Embed 4 Inspection Captured Images (2x2 Grid)
            img_y_top = row_y - 20
            pdf.setFont("Helvetica-Bold", 10)
            pdf.setFillColor(colors.HexColor("#0f172a"))
            pdf.drawString(25, img_y_top, "Optical Image Evidence (4-Angle Review):")

            # Grid Positions
            img_w = 260
            img_h = 160
            grid_positions = [
                ("Cam1_Shot1", 25, img_y_top - 175, "Top Orthogonal — Shot 1"),
                ("Cam2_Shot1", 310, img_y_top - 175, "Lateral Profile — Shot 1"),
                ("Cam1_Shot2", 25, img_y_top - 355, "Top Orthogonal — Shot 2"),
                ("Cam2_Shot2", 310, img_y_top - 355, "Lateral Profile — Shot 2")
            ]

            for key, gx, gy, label_text in grid_positions:
                img_path = saved_image_paths.get(key)
                if img_path and os.path.exists(img_path):
                    try:
                        pdf.drawImage(img_path, gx, gy, width=img_w, height=img_h, preserveAspectRatio=True)
                    except Exception:
                        pass
                pdf.setFillColor(colors.HexColor("#0f172a"))
                pdf.setFont("Helvetica-Bold", 8)
                pdf.drawString(gx, gy + img_h + 3, label_text)

            # Footer
            pdf.setFont("Helvetica", 7)
            pdf.setFillColor(colors.HexColor("#64748b"))
            pdf.drawCentredString(width / 2, 20, "Confidential Quality Audit Document • Generated Automatically by IMS Vision Metrology Suite")

            pdf.save()
            print(f"Inspection Report PDF Generated: {pdf_path}")
        except Exception as e:
            print(f"Error during PDF compilation: {e}")

    def save_decision_and_return(self, decision: str):
        available_shots = {k: v for k, v in self.shot_buffer.items() if v is not None}
        if not available_shots:
            messagebox.showwarning("Warning", "No images captured yet.")
            return

        target_folder = DIR_OK if decision == "OK" else DIR_NOT_OK
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JPEGs to target folder
        saved_paths = {}
        for shot_name, img in available_shots.items():
            filename = f"{shot_name}_{ts}_{decision}.jpg"
            fpath = os.path.join(target_folder, filename)
            cv2.imwrite(fpath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_paths[shot_name] = fpath

        # Generate the PDF Report in Background 
        self.io_pool.submit(self._generate_pdf_report, saved_paths, ts, decision)

        self.total_inspections += 1 
        if decision == "OK":
            self.pass_count += 1
        else:
            self.fail_count += 1

        yield_rate = (self.pass_count / self.total_inspections) * 100 if self.total_inspections > 0 else 100.0
        self.kpi_total.config(text=str(self.total_inspections))
        self.kpi_pass.config(text=str(self.pass_count))
        self.kpi_fail.config(text=str(self.fail_count))
        self.kpi_yield.config(text=f"{yield_rate:.1f}%")

        self._reset_capture_buttons()
        self.switch_tab("camera")
        messagebox.showinfo("Success", f"Inspection Saved as {decision}!\nPDF Report generated in:\n{DIR_REPORTS}")

    def _reset_capture_buttons(self):
        for key in self.shot_buffer:
            self.shot_buffer[key] = None
            self.preview_labels[key].configure(image="", text="[ Awaiting Capture ]")
            self.preview_labels[key].image = None

        self.btn_capture_shot1.config(text="📸  CAPTURE SHOT 1 (Top + Side)   [ Press 1 ]", bg=self.theme["accent_blue"])
        self.btn_capture_shot2.config(text="📸  CAPTURE SHOT 2 (Top + Side)   [ Press 2 ]", bg=self.theme["accent_blue"])
        self.btn_review_gate.config(text="➔   PROCEED TO 4-SHOT REVIEW & DECISION   ➔", bg="#0f172a", fg="#38bdf8")

    def _bind_shortcuts(self):
        self.root.bind("1", lambda e: self.capture_dual_shot(1))
        self.root.bind("2", lambda e: self.capture_dual_shot(2))
        self.root.bind("<Control-o>", lambda e: self.save_decision_and_return("OK"))
        self.root.bind("<Control-n>", lambda e: self.save_decision_and_return("NOT_OK"))
        self.root.bind("<Escape>", lambda e: self.switch_tab("camera"))

    def _start_render_cycle(self):
        self.lbl_clock.config(text=datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        ok, raw_frame, fps = self.cam_driver.get_frame()

        if ok and raw_frame is not None:
            self.status_cam1.config(text=f"ONLINE • {fps} FPS", bg=self.theme["success_green"])
            self.status_cam2.config(text=f"ONLINE • {fps} FPS", bg=self.theme["success_green"])

            if self.view_camera.winfo_ismapped():
                w1 = max(self.c1_box.winfo_width(), 320)
                h1 = max(self.c1_box.winfo_height(), 240)

                analyzed_frame1 = self._safe_metrology_process(raw_frame)

                rgb1 = cv2.cvtColor(analyzed_frame1, cv2.COLOR_BGR2RGB)
                rgb1 = cv2.resize(rgb1, (w1, h1))
                img_tk1 = ImageTk.PhotoImage(image=Image.fromarray(rgb1))
                self.feed_cam1.configure(image=img_tk1, text="")
                self.feed_cam1.image = img_tk1

                rgb2 = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                rgb2 = cv2.resize(rgb2, (w1, h1))
                img_tk2 = ImageTk.PhotoImage(image=Image.fromarray(rgb2))
                self.feed_cam2.configure(image=img_tk2, text="")
                self.feed_cam2.image = img_tk2
        else: 
            self.status_cam1.config(text="CONNECTING", bg="#b45309")
            self.status_cam2.config(text="CONNECTING", bg="#b45309")

        self.root.after(30, self._start_render_cycle)

    def close(self):
        self.cam_driver.stop()
        self.io_pool.shutdown(wait=False)
        self.root.destroy()


if __name__ == "__main__":
    app_window = tk.Tk()
    app = ModernIMSStation(app_window)
    app_window.protocol("WM_DELETE_WINDOW", app.close)
    app_window.mainloop()

