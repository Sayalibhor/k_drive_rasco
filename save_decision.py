import os
import cv2
import time
import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --- Directories Setup ---
BASE_DIR = r"C:\Camera_Inspection_App"
DIR_OK = os.path.join(BASE_DIR, "ok")
DIR_NOT_OK = os.path.join(BASE_DIR, "not_ok")

os.makedirs(DIR_OK, exist_ok=True)
os.makedirs(DIR_NOT_OK, exist_ok=True)

class AEPLInspectionSuiteWithPreview:
    def __init__(self, root):
        self.root = root
        self.root.title("AEPL Vision Metrology Workstation — 4-Shot Capture & Preview")
        self.root.geometry("1240x820")
        self.root.configure(bg="#1f1f1f")

        # Color Theme Palette
        self.c_bg = "#1f1f1f"
        self.c_card = "#2d2d2d"
        self.c_text = "#ffffff"
        self.c_subtext = "#dcdcdc"
        self.c_online = "#38b000"
        self.c_offline = "#e63946"
        self.c_accent = "#0d6efd"
        self.c_btn_blue = "#70a6ff"
        self.c_btn_dark = "#3a3a3c"

        # Hardware Setup: Uses single laptop webcam (Index 0) to simulate dual camera feeds
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.current_frame = None

        # 4 Captured Shots Buffer
        self.shot_buffer = {
            "Cam1_Shot1": None,
            "Cam1_Shot2": None,
            "Cam2_Shot1": None,
            "Cam2_Shot2": None
        }

        self._build_ui()
        self._bind_shortcuts()
        self.update_live_feeds()

    def _build_ui(self):
        # 1. Top Header Bar
        top_bar = tk.Frame(self.root, bg=self.c_bg)
        top_bar.pack(fill="x", padx=20, pady=(12, 6))

        tk.Label(
            top_bar,
            text="AEPL Inspection Suite — 4-Shot Instant Capture & Preview",
            font=("Segoe UI", 15, "bold"),
            fg=self.c_text,
            bg=self.c_bg
        ).pack(side="left")

        tk.Label(
            top_bar,
            text="🌙 Dark Mode",
            font=("Segoe UI", 9, "bold"),
            fg="#1f1f1f",
            bg="#ffffff",
            padx=8,
            pady=3
        ).pack(side="right")

        # 2. Main Live Camera Feeds (Left and Right)
        live_frame_container = tk.Frame(self.root, bg=self.c_bg)
        live_frame_container.pack(fill="x", padx=20, pady=4)
        live_frame_container.columnconfigure(0, weight=1)
        live_frame_container.columnconfigure(1, weight=1)

        # Camera 1 (Top View)
        card1 = tk.Frame(live_frame_container, bg=self.c_card, highlightthickness=1, highlightbackground="#3d3d3d")
        card1.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=2)
        h1 = tk.Frame(card1, bg=self.c_card)
        h1.pack(fill="x", padx=10, pady=(6, 2))
        tk.Label(h1, text="Camera 1 — Top View", font=("Segoe UI", 10, "bold"), fg=self.c_subtext, bg=self.c_card).pack(side="left")
        tk.Label(h1, text="ONLINE", font=("Segoe UI", 9, "bold"), fg=self.c_online, bg=self.c_card).pack(side="right")
        self.lbl_live1 = tk.Label(card1, bg="#000000", width=420, height=220)
        self.lbl_live1.pack(padx=8, pady=6)

        # Camera 2 (Side View)
        card2 = tk.Frame(live_frame_container, bg=self.c_card, highlightthickness=1, highlightbackground="#3d3d3d")
        card2.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=2)
        h2 = tk.Frame(card2, bg=self.c_card)
        h2.pack(fill="x", padx=10, pady=(6, 2))
        tk.Label(h2, text="Camera 2 — Side View", font=("Segoe UI", 10, "bold"), fg=self.c_subtext, bg=self.c_card).pack(side="left")
        tk.Label(h2, text="ONLINE", font=("Segoe UI", 9, "bold"), fg=self.c_online, bg=self.c_card).pack(side="right")
        self.lbl_live2 = tk.Label(card2, bg="#000000", width=420, height=220)
        self.lbl_live2.pack(padx=8, pady=6)

        # 3. Big Instant 4-Shot Capture Button
        mid_bar = tk.Frame(self.root, bg=self.c_bg)
        mid_bar.pack(fill="x", padx=20, pady=6)

        btn_capture_all = tk.Button(
            mid_bar,
            text="📸 Capture 4 Shots (One Click) [ Space / Enter ]",
            font=("Segoe UI", 12, "bold"),
            bg=self.c_accent,
            fg="#ffffff",
            activebackground="#0b5ed7",
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
            height=2,
            cursor="hand2",
            command=self.capture_four_shots_one_click
        )
        btn_capture_all.pack(fill="x")

        # 4. Preview Panel for the 4 Captured Images
        preview_container = tk.LabelFrame(
            self.root,
            text=" Captured 4-Shots Preview Screen ",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_btn_blue,
            bg=self.c_card,
            bd=1,
            relief="groove"
        )
        preview_container.pack(fill="both", expand=True, padx=20, pady=6)

        for col in range(4):
            preview_container.columnconfigure(col, weight=1)

        self.preview_labels = {}
        shot_titles = [
            ("Cam1_Shot1", "Cam 1 - Shot 1"),
            ("Cam1_Shot2", "Cam 1 - Shot 2"),
            ("Cam2_Shot1", "Cam 2 - Shot 1"),
            ("Cam2_Shot2", "Cam 2 - Shot 2")
        ]

        for i, (key, title) in enumerate(shot_titles):
            cell = tk.Frame(preview_container, bg=self.c_card)
            cell.grid(row=0, column=i, padx=5, pady=4, sticky="nsew")

            tk.Label(cell, text=title, font=("Segoe UI", 9, "bold"), fg=self.c_subtext, bg=self.c_card).pack(pady=(2, 2))
            lbl_img = tk.Label(cell, bg="#111111", text="No Capture", fg="#6c757d", font=("Segoe UI", 9), width=180, height=120)
            lbl_img.pack(padx=4, pady=2)
            self.preview_labels[key] = lbl_img

        # 5. Bottom Decision Bar (OK / NOT OK / Clear)
        decision_bar = tk.Frame(self.root, bg=self.c_bg)
        decision_bar.pack(fill="x", padx=20, pady=(6, 4))

        btn_ok = tk.Button(
            decision_bar, text="OK [ Ctrl+O ]", font=("Segoe UI", 11, "bold"),
            bg="#28a745", fg=self.c_text, bd=0, height=2, cursor="hand2",
            command=lambda: self.save_decision("ok")
        )
        btn_ok.pack(side="left", expand=True, fill="x", padx=(0, 8))

        btn_not_ok = tk.Button(
            decision_bar, text="NOT OK [ Ctrl+N ]", font=("Segoe UI", 11, "bold"),
            bg="#dc3545", fg=self.c_text, bd=0, height=2, cursor="hand2",
            command=lambda: self.save_decision("not_ok")
        )
        btn_not_ok.pack(side="left", expand=True, fill="x", padx=8)

        btn_clear = tk.Button(
            decision_bar, text="Clear Buffer [ Ctrl+L ]", font=("Segoe UI", 11),
            bg=self.c_btn_dark, fg=self.c_text, bd=0, height=2, cursor="hand2",
            command=self.clear_buffer
        )
        btn_clear.pack(side="right", expand=True, fill="x", padx=(8, 0))

        # 6. Status Bar
        self.status_bar = tk.Label(
            self.root,
            text=f"Ready · Storage: {DIR_OK} | {DIR_NOT_OK}",
            font=("Segoe UI", 9),
            fg="#9e9e9e",
            bg=self.c_bg,
            anchor="w",
            padx=20,
            pady=6
        )
        self.status_bar.pack(side="bottom", fill="x")

    def _bind_shortcuts(self):
        self.root.bind("<space>", lambda e: self.capture_four_shots_one_click())
        self.root.bind("<Return>", lambda e: self.capture_four_shots_one_click())
        self.root.bind("<Control-o>", lambda e: self.save_decision("ok"))
        self.root.bind("<Control-O>", lambda e: self.save_decision("ok"))
        self.root.bind("<Control-n>", lambda e: self.save_decision("not_ok"))
        self.root.bind("<Control-N>", lambda e: self.save_decision("not_ok"))
        self.root.bind("<Control-l>", lambda e: self.clear_buffer())
        self.root.bind("<Control-L>", lambda e: self.clear_buffer())

    def update_live_feeds(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, (420, 220))
                img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb_resized))

                self.lbl_live1.configure(image=img_tk)
                self.lbl_live1.image = img_tk
                self.lbl_live2.configure(image=img_tk)
                self.lbl_live2.image = img_tk

        self.root.after(25, self.update_live_feeds)

    def capture_four_shots_one_click(self):
        """Captures 4 rapid frames in 1 click and updates previews on screen."""
        if self.current_frame is None:
            messagebox.showerror("Error", "No camera detected.")
            return

        shot_keys = ["Cam1_Shot1", "Cam1_Shot2", "Cam2_Shot1", "Cam2_Shot2"]

        for key in shot_keys:
            ret, frame = self.cap.read()
            shot = frame.copy() if ret else self.current_frame.copy()
            self.shot_buffer[key] = shot
            time.sleep(0.04)  # Small delay to ensure separate frames

            # Render thumbnail onto preview card
            thumb = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
            thumb = cv2.resize(thumb, (220, 140))
            thumb_tk = ImageTk.PhotoImage(image=Image.fromarray(thumb))

            lbl = self.preview_labels[key]
            lbl.configure(image=thumb_tk, text="")
            lbl.image = thumb_tk

        self.status_bar.config(text="✓ 4 Shots captured successfully! Review images below, then click OK or NOT OK.", fg="#70a6ff")

    def save_decision(self, decision):
        # Auto-capture 4 shots if the operator pressed OK/NOT OK without manual capture
        if any(v is None for v in self.shot_buffer.values()):
            self.capture_four_shots_one_click()

        target_folder = DIR_OK if decision == "ok" else DIR_NOT_OK
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save all 4 shots into target directory
        for shot_name, img in self.shot_buffer.items():
            filename = f"{shot_name}_{ts}_{decision}.jpg"
            cv2.imwrite(os.path.join(target_folder, filename), img)

        color = self.c_online if decision == "ok" else self.c_offline
        self.status_bar.config(
            text=f"✓ All 4 images saved to '{decision.upper()}' [{ts}]",
            fg=color
        )
        self.clear_buffer()

    def clear_buffer(self):
        for key in self.shot_buffer:
            self.shot_buffer[key] = None
            lbl = self.preview_labels[key]
            lbl.configure(image="", text="No Capture")
            lbl.image = None
        self.status_bar.config(text="Buffer cleared · Ready for new capture", fg="#9e9e9e")

    def close(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app_root = tk.Tk()
    app = AEPLInspectionSuiteWithPreview(app_root)
    app_root.protocol("WM_DELETE_WINDOW", app.close)
    app_root.mainloop()