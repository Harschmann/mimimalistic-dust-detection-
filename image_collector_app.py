"""
Image Collector -- standalone training-data collection app for camera-module
QC (dust/defect detection)
-----------------------------------------------------------------------------
Fully independent from dust_inspector_app.py / training_studio.py -- no
shared settings file, no shared ROI file, no imports between them. The only
thing it shares on purpose is the visual style (same dark theme) and the
same circular black-filled-corner crop convention, because that's the crop
shape Training Studio expects to train on -- but the code here is its own
copy, not a reused module.

PURPOSE: dust_inspector_app's "Test Detection" flow works fine for one-off
checks, but it's the wrong tool for BULK, FAST data collection on the line
(that wasn't its job). This app's only job is: show the live feed, let you
mark each frame Good or Defect with one click/keystroke, and auto-save a
clean circular crop of every labeled ROI into the right folder --
`<save_root>/<Model>/<camN>/good/` or `<save_root>/<Model>/<camN>/defect/`
-- which is exactly the folder shape Training Studio's dataset picker
expects.

FEATURES:
  - Live Basler feed (pypylon), same camera-manager pattern as the other
    two apps (safe no-op if pypylon/hardware isn't available).
  - Draw ROIs once per model, "Save ROI" persists them (own settings file,
    keyed by model name) so you never redraw for a model you've already
    set up -- just pick the model name and they load automatically.
  - Capture -> Good / Defect buttons (and G / D keyboard shortcuts) with an
    optional severity tag (mild/medium/severe) for defects.
  - Running per-camera per-class counters so you can see dataset balance
    while you collect, instead of discovering it's lopsided later.
  - Thumbnail history of the last several captures with one-click Undo
    (deletes the saved files) in case of a mis-click.
  - Optional duplicate-frame guard: skips saving if a new capture is
    near-identical to the last saved one for that camera (mean absolute
    pixel difference below a threshold), so an idle line doesn't flood the
    dataset with 50 copies of the same frame.
  - Optional "Auto-collect Good" burst mode: captures + saves as Good every
    N seconds while the line is known to be running clean, so you build up
    natural variation (lighting/position jitter) without babysitting every
    click.

Run:  python image_collector_app.py
"""

import os
import sys
import json
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import customtkinter as ctk
from PIL import Image, ImageTk

try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except Exception:
    PYLON_AVAILABLE = False

ctk.set_appearance_mode("dark")

# ------------------------------------------------------------------ theme --
BG = "#0b0d10"
BG_SIDEBAR = "#0e1013"
BG_CARD = "#14171c"
BG_CARD_ALT = "#1b1f26"
BG_CANVAS = "#101317"
BORDER = "#23272e"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#3f74e0"
TEXT = "#e6e8eb"
TEXT_MUTED = "#8a919c"
SUCCESS = "#22c55e"
SUCCESS_HOVER = "#16a34a"
DANGER = "#ef4444"
DANGER_HOVER = "#dc2626"
WARNING = "#f59e0b"

ROI_COLOR = (0, 255, 0)          # BGR, saved/committed ROI
ROI_DRAFT_COLOR = (0, 200, 255)  # BGR, while dragging out a new one

# ---------------------------------------------------------------- storage --
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SETTINGS_PATH = os.path.join(BASE_DIR, "image_collector_settings.json")

DEFAULT_SETTINGS = {
    "save_root": None,           # base folder -- <save_root>/<Model>/<camN>/good|defect/
    "roi_layouts": {},           # model_name -> [ {"cx":.., "cy":.., "r":.., "cam_label":"cam1"}, ... ]
    "exposure_us": 20000.0,
    "gain": 0.0,
    "duplicate_check": True,
    "duplicate_threshold": 2.0,  # mean abs pixel diff (0-255 scale) below which a capture counts as a duplicate
    "last_model": "",
}


def load_settings():
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r") as f:
                data = json.load(f)
            merged = DEFAULT_SETTINGS.copy()
            merged.update(data)
            return merged
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def _safe_folder_name(name):
    """Sanitizes a model name for use as a folder name."""
    name = (name or "").strip()
    if not name:
        return "UNSPECIFIED"
    keep = "-_.() "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name)
    return cleaned.strip() or "UNSPECIFIED"


def _ensure_cam_labels(rois):
    """Guarantees every ROI dict has a non-empty, unique 'cam_label'
    (cam1, cam2, ...) -- existing labels are kept, only missing/blank/
    duplicate ones get (re)assigned by ROI order."""
    used = set()
    for roi in rois:
        lbl = (roi.get("cam_label") or "").strip()
        if lbl and lbl not in used:
            used.add(lbl)
        else:
            roi["cam_label"] = None
    n = 1
    for roi in rois:
        if not roi.get("cam_label"):
            while f"cam{n}" in used:
                n += 1
            roi["cam_label"] = f"cam{n}"
            used.add(f"cam{n}")
    return rois


def crop_circular(frame, roi):
    """Same convention Training Studio/dust_inspector use: a tight bounding
    box around the ROI circle, with everything outside the circle painted
    black. Returns None if the ROI falls entirely outside the frame."""
    h, w = frame.shape[:2]
    cx, cy, r = int(roi["cx"]), int(roi["cy"]), int(roi["r"])
    x0, y0 = max(cx - r, 0), max(cy - r, 0)
    x1, y1 = min(cx + r, w), min(cy + r, h)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), r, 255, -1)
    crop[mask == 0] = 0
    return crop


# ---------------------------------------------------------- camera manager --
class CameraManager:
    """Thin wrapper around a Basler camera via pypylon, with a background
    grab thread. Safe to use even when pypylon/hardware is unavailable --
    callers should check .connected before relying on live frames."""

    def __init__(self):
        self.cam = None
        self.connected = False
        self.grabbing = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None

    def connect(self):
        if not PYLON_AVAILABLE:
            return False, "pypylon is not installed"
        try:
            tlf = pylon.TlFactory.GetInstance()
            devices = tlf.EnumerateDevices()
            if not devices:
                return False, "No Basler camera found"
            self.cam = pylon.InstantCamera(tlf.CreateFirstDevice())
            self.cam.Open()
            self.connected = True
            return True, "Connected"
        except Exception as e:
            return False, str(e)

    def apply_settings(self, exposure_us=None, gain=None):
        if not self.connected:
            return
        try:
            if exposure_us is not None:
                if hasattr(self.cam, "ExposureTime"):
                    self.cam.ExposureTime.SetValue(float(exposure_us))
                elif hasattr(self.cam, "ExposureTimeAbs"):
                    self.cam.ExposureTimeAbs.SetValue(float(exposure_us))
        except Exception:
            pass
        try:
            if gain is not None:
                if hasattr(self.cam, "Gain"):
                    self.cam.Gain.SetValue(float(gain))
                elif hasattr(self.cam, "GainRaw"):
                    self.cam.GainRaw.SetValue(int(gain))
        except Exception:
            pass

    def start_live(self):
        if not self.connected or self.grabbing:
            return
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.grabbing = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _grab_loop(self):
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        while self.grabbing and self.cam is not None and self.cam.IsGrabbing():
            try:
                res = self.cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
                if res.GrabSucceeded():
                    img = converter.Convert(res).GetArray()
                    with self.lock:
                        self.latest_frame = img
                res.Release()
            except Exception:
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop_live(self):
        self.grabbing = False
        try:
            if self.cam is not None and self.cam.IsGrabbing():
                self.cam.StopGrabbing()
        except Exception:
            pass

    def disconnect(self):
        self.stop_live()
        try:
            if self.cam is not None and self.cam.IsOpen():
                self.cam.Close()
        except Exception:
            pass
        self.connected = False


# ------------------------------------------------------------------- app ---
class ImageCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Collector -- QC Training Data Capture")
        self.root.geometry("1280x820")
        self.root.configure(fg_color=BG)

        self.settings = load_settings()
        self.cam = CameraManager()

        # ROI editing state
        self.rois = []                 # list of {"cx","cy","r","cam_label"}
        self.selected_roi_idx = None
        self.draw_mode = False
        self.draft_roi = None          # {"cx","cy","r"} while dragging

        # frame state
        self.live_frame = None         # last frame grabbed from camera (numpy BGR)
        self.frozen_frame = None       # the frame currently being labeled, or None if live
        self.display_scale = 1.0
        self.canvas_photo = None

        # capture bookkeeping
        self.counts = {}               # (model, cam_label, class) -> int
        self.last_saved_per_cam = {}   # cam_label -> last saved crop (np array) for duplicate check
        self.capture_history = []      # list of dicts: {"paths": [...], "thumb": PhotoImage, "label": str}

        self.burst_running = False
        self.burst_job = None

        self.fonts_ready = False
        self._build_fonts()
        self._build_ui()

        self.model_var.set(self.settings.get("last_model", ""))
        self._load_roi_layout_for_model(self.model_var.get())

        self._connect_camera()
        self._poll_feed()

    # ---------------------------------------------------------------- ui --
    def _build_fonts(self):
        self.f_title = ctk.CTkFont(size=20, weight="bold")
        self.f_body = ctk.CTkFont(size=13)
        self.f_body_bold = ctk.CTkFont(size=13, weight="bold")
        self.f_small = ctk.CTkFont(size=11)
        self.f_big_btn = ctk.CTkFont(size=16, weight="bold")

    def _build_ui(self):
        root_frame = ctk.CTkFrame(self.root, fg_color=BG)
        root_frame.pack(fill="both", expand=True)
        root_frame.grid_columnconfigure(0, weight=3)
        root_frame.grid_columnconfigure(1, weight=1)
        root_frame.grid_rowconfigure(0, weight=1)

        # ---- left: live feed / canvas -------------------------------------
        left = ctk.CTkFrame(root_frame, fg_color=BG_CARD, corner_radius=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        top_bar = ctk.CTkFrame(left, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        self.status_var = tk.StringVar(value="Camera: not connected")
        ctk.CTkLabel(top_bar, textvariable=self.status_var, font=self.f_small,
                     text_color=TEXT_MUTED).pack(side="left")

        self.draw_mode_btn = ctk.CTkButton(
            top_bar, text="Draw ROI Mode: OFF", font=self.f_body, fg_color=BG_CARD_ALT,
            hover_color=BORDER, text_color=TEXT, command=self._toggle_draw_mode, width=160)
        self.draw_mode_btn.pack(side="right", padx=(6, 0))
        ctk.CTkButton(top_bar, text="Save ROI", font=self.f_body, fg_color=BG_CARD_ALT,
                      hover_color=BORDER, text_color=TEXT, command=self._save_roi_layout,
                      width=100).pack(side="right", padx=(6, 0))
        ctk.CTkButton(top_bar, text="Clear ROIs", font=self.f_body, fg_color=BG_CARD_ALT,
                      hover_color=BORDER, text_color=DANGER, command=self._clear_rois,
                      width=100).pack(side="right", padx=(6, 0))

        self.canvas = tk.Canvas(left, bg=BG_CANVAS, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # ---- right: controls -----------------------------------------------
        right = ctk.CTkScrollableFrame(root_frame, fg_color=BG_CARD, corner_radius=14)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)

        ctk.CTkLabel(right, text="Image Collector", font=self.f_title, text_color=TEXT).pack(
            anchor="w", padx=14, pady=(14, 4))

        # model
        model_card = ctk.CTkFrame(right, fg_color=BG_CARD_ALT, corner_radius=10)
        model_card.pack(fill="x", padx=12, pady=(4, 10))
        ctk.CTkLabel(model_card, text="Model name", font=self.f_small, text_color=TEXT_MUTED).pack(
            anchor="w", padx=10, pady=(8, 0))
        model_row = ctk.CTkFrame(model_card, fg_color="transparent")
        model_row.pack(fill="x", padx=10, pady=(2, 10))
        self.model_var = tk.StringVar(value="")
        model_entry = ctk.CTkEntry(model_row, textvariable=self.model_var, font=self.f_body)
        model_entry.pack(side="left", fill="x", expand=True)
        model_entry.bind("<FocusOut>", lambda e: self._on_model_change())
        model_entry.bind("<Return>", lambda e: self._on_model_change())
        ctk.CTkButton(model_row, text="Load", width=60, font=self.f_body,
                      fg_color=BG_CARD, hover_color=BORDER,
                      command=self._on_model_change).pack(side="left", padx=(6, 0))

        # save location
        save_card = ctk.CTkFrame(right, fg_color=BG_CARD_ALT, corner_radius=10)
        save_card.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(save_card, text="Save location", font=self.f_small, text_color=TEXT_MUTED).pack(
            anchor="w", padx=10, pady=(8, 0))
        self.save_root_var = tk.StringVar(value=self.settings.get("save_root") or "(not set)")
        ctk.CTkLabel(save_card, textvariable=self.save_root_var, font=self.f_small,
                     text_color=TEXT, wraplength=260, justify="left").pack(anchor="w", padx=10, pady=(2, 6))
        ctk.CTkButton(save_card, text="Choose Folder...", font=self.f_body, fg_color=BG_CARD,
                      hover_color=BORDER, command=self._choose_save_root).pack(
            anchor="w", padx=10, pady=(0, 10))

        # capture buttons
        cap_card = ctk.CTkFrame(right, fg_color=BG_CARD_ALT, corner_radius=10)
        cap_card.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(cap_card, text="Capture", font=self.f_small, text_color=TEXT_MUTED).pack(
            anchor="w", padx=10, pady=(8, 4))
        self.capture_btn = ctk.CTkButton(cap_card, text="Capture Frame (Space)", font=self.f_body_bold,
                                          fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                          command=self._capture_frame, height=40)
        self.capture_btn.pack(fill="x", padx=10, pady=(0, 8))

        btn_row = ctk.CTkFrame(cap_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 6))
        self.good_btn = ctk.CTkButton(btn_row, text="GOOD (G)", font=self.f_big_btn,
                                       fg_color=SUCCESS, hover_color=SUCCESS_HOVER,
                                       command=lambda: self._label_capture("good"), height=48)
        self.good_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.defect_btn = ctk.CTkButton(btn_row, text="DEFECT (D)", font=self.f_big_btn,
                                         fg_color=DANGER, hover_color=DANGER_HOVER,
                                         command=lambda: self._label_capture("defect"), height=48)
        self.defect_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        sev_row = ctk.CTkFrame(cap_card, fg_color="transparent")
        sev_row.pack(fill="x", padx=10, pady=(6, 10))
        ctk.CTkLabel(sev_row, text="Defect severity:", font=self.f_small, text_color=TEXT_MUTED).pack(side="left")
        self.severity_var = tk.StringVar(value="unspecified")
        ctk.CTkOptionMenu(sev_row, values=["unspecified", "mild", "medium", "severe"],
                          variable=self.severity_var, width=130, font=self.f_small,
                          fg_color=BG_CARD, button_color=BORDER, button_hover_color=BORDER).pack(
            side="left", padx=(6, 0))

        ctk.CTkButton(cap_card, text="Resume Live Feed", font=self.f_body, fg_color=BG_CARD,
                      hover_color=BORDER, command=self._resume_live).pack(
            fill="x", padx=10, pady=(0, 10))

        # options
        opt_card = ctk.CTkFrame(right, fg_color=BG_CARD_ALT, corner_radius=10)
        opt_card.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(opt_card, text="Options", font=self.f_small, text_color=TEXT_MUTED).pack(
            anchor="w", padx=10, pady=(8, 4))
        self.dup_check_var = tk.BooleanVar(value=self.settings.get("duplicate_check", True))
        ctk.CTkSwitch(opt_card, text="Skip near-duplicate frames", variable=self.dup_check_var,
                      font=self.f_small, command=self._save_options).pack(anchor="w", padx=10, pady=(0, 6))

        burst_row = ctk.CTkFrame(opt_card, fg_color="transparent")
        burst_row.pack(fill="x", padx=10, pady=(0, 10))
        self.burst_btn = ctk.CTkButton(burst_row, text="Start Auto-collect Good", font=self.f_body,
                                        fg_color=BG_CARD, hover_color=BORDER,
                                        command=self._toggle_burst_mode)
        self.burst_btn.pack(side="left", fill="x", expand=True)
        self.burst_interval_var = tk.StringVar(value="5")
        ctk.CTkEntry(burst_row, textvariable=self.burst_interval_var, width=40,
                     font=self.f_small).pack(side="left", padx=(6, 0))
        ctk.CTkLabel(burst_row, text="sec", font=self.f_small, text_color=TEXT_MUTED).pack(side="left", padx=(4, 0))

        # counters
        self.counts_card = ctk.CTkFrame(right, fg_color=BG_CARD_ALT, corner_radius=10)
        self.counts_card.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(self.counts_card, text="Collected this session", font=self.f_small,
                     text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(8, 4))
        self.counts_label = ctk.CTkLabel(self.counts_card, text="No captures yet.", font=self.f_small,
                                          text_color=TEXT, justify="left", wraplength=260)
        self.counts_label.pack(anchor="w", padx=10, pady=(0, 10))

        # history / undo
        hist_card = ctk.CTkFrame(right, fg_color=BG_CARD_ALT, corner_radius=10)
        hist_card.pack(fill="x", padx=12, pady=(0, 14))
        ctk.CTkLabel(hist_card, text="Recent captures (click Undo to delete)", font=self.f_small,
                     text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(8, 4))
        self.history_frame = ctk.CTkFrame(hist_card, fg_color="transparent")
        self.history_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.root.bind("<space>", lambda e: self._capture_frame())
        self.root.bind("g", lambda e: self._label_capture("good"))
        self.root.bind("G", lambda e: self._label_capture("good"))
        self.root.bind("d", lambda e: self._label_capture("defect"))
        self.root.bind("D", lambda e: self._label_capture("defect"))

    # ------------------------------------------------------------- camera --
    def _connect_camera(self):
        ok, msg = self.cam.connect()
        if ok:
            self.cam.apply_settings(exposure_us=self.settings.get("exposure_us"),
                                     gain=self.settings.get("gain"))
            self.cam.start_live()
            self.status_var.set("Camera: connected, live")
        else:
            self.status_var.set(f"Camera: {msg} (no live feed)")

    def _poll_feed(self):
        if self.frozen_frame is None:
            frame = self.cam.get_frame()
            if frame is not None:
                self.live_frame = frame
                self._render_frame(frame)
        self.root.after(66, self._poll_feed)  # ~15 fps is plenty for this use case

    # ---------------------------------------------------------- rendering --
    def _render_frame(self, frame):
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        h, w = frame.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        self.display_scale = scale
        disp = cv2.resize(frame, (max(int(w * scale), 1), max(int(h * scale), 1)))
        disp = self._draw_overlays(disp)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.canvas_photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas_photo, anchor="nw")

    def _draw_overlays(self, disp):
        s = self.display_scale
        for i, roi in enumerate(self.rois):
            color = (255, 200, 0) if i == self.selected_roi_idx else ROI_COLOR
            cx, cy, r = int(roi["cx"] * s), int(roi["cy"] * s), int(roi["r"] * s)
            cv2.circle(disp, (cx, cy), r, color, 2)
            cv2.putText(disp, roi.get("cam_label", "?"), (cx - r, max(cy - r - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if self.draft_roi is not None:
            cx, cy, r = int(self.draft_roi["cx"] * s), int(self.draft_roi["cy"] * s), int(self.draft_roi["r"] * s)
            cv2.circle(disp, (cx, cy), r, ROI_DRAFT_COLOR, 2)
        return disp

    # ---------------------------------------------------------- ROI editing --
    def _toggle_draw_mode(self):
        self.draw_mode = not self.draw_mode
        self.draw_mode_btn.configure(
            text=f"Draw ROI Mode: {'ON' if self.draw_mode else 'OFF'}",
            fg_color=ACCENT if self.draw_mode else BG_CARD_ALT)

    def _canvas_to_frame_xy(self, event_x, event_y):
        s = max(self.display_scale, 1e-6)
        return event_x / s, event_y / s

    def _on_canvas_press(self, event):
        fx, fy = self._canvas_to_frame_xy(event.x, event.y)
        if self.draw_mode:
            self.draft_roi = {"cx": fx, "cy": fy, "r": 1}
        else:
            # select nearest existing ROI if click is inside it
            self.selected_roi_idx = None
            for i, roi in enumerate(self.rois):
                dx, dy = fx - roi["cx"], fy - roi["cy"]
                if (dx * dx + dy * dy) ** 0.5 <= roi["r"]:
                    self.selected_roi_idx = i
                    break

    def _on_canvas_drag(self, event):
        if self.draw_mode and self.draft_roi is not None:
            fx, fy = self._canvas_to_frame_xy(event.x, event.y)
            dx, dy = fx - self.draft_roi["cx"], fy - self.draft_roi["cy"]
            self.draft_roi["r"] = max((dx * dx + dy * dy) ** 0.5, 1)

    def _on_canvas_release(self, event):
        if self.draw_mode and self.draft_roi is not None:
            if self.draft_roi["r"] >= 5:  # ignore accidental tiny clicks
                new_roi = dict(self.draft_roi)
                new_roi["cam_label"] = None
                self.rois.append(new_roi)
                _ensure_cam_labels(self.rois)
            self.draft_roi = None

    def _clear_rois(self):
        if self.rois and not messagebox.askyesno("Image Collector", "Clear all ROIs for this model?"):
            return
        self.rois = []
        self.selected_roi_idx = None

    def _delete_selected_roi(self):
        if self.selected_roi_idx is not None and 0 <= self.selected_roi_idx < len(self.rois):
            del self.rois[self.selected_roi_idx]
            self.selected_roi_idx = None
            _ensure_cam_labels(self.rois)

    def _save_roi_layout(self):
        model = self.model_var.get().strip()
        if not model:
            messagebox.showinfo("Image Collector", "Type a model name first.")
            return
        if not self.rois:
            messagebox.showinfo("Image Collector", "No ROIs to save -- draw at least one first.")
            return
        _ensure_cam_labels(self.rois)
        self.settings["roi_layouts"][model] = [dict(r) for r in self.rois]
        self.settings["last_model"] = model
        save_settings(self.settings)
        messagebox.showinfo("Image Collector", f"Saved {len(self.rois)} ROI(s) for '{model}'.")

    def _load_roi_layout_for_model(self, model):
        model = (model or "").strip()
        layout = self.settings.get("roi_layouts", {}).get(model)
        if layout:
            self.rois = [dict(r) for r in layout]
            _ensure_cam_labels(self.rois)
        else:
            self.rois = []
        self.selected_roi_idx = None

    def _on_model_change(self):
        model = self.model_var.get().strip()
        self._load_roi_layout_for_model(model)
        self.settings["last_model"] = model
        save_settings(self.settings)
        self._update_counts_display()

    # ------------------------------------------------------------- saving --
    def _choose_save_root(self):
        folder = filedialog.askdirectory()
        if folder:
            self.settings["save_root"] = folder
            self.save_root_var.set(folder)
            save_settings(self.settings)

    def _save_options(self):
        self.settings["duplicate_check"] = bool(self.dup_check_var.get())
        save_settings(self.settings)

    def _capture_frame(self):
        if self.live_frame is None:
            messagebox.showinfo("Image Collector", "No live frame available yet.")
            return
        self.frozen_frame = self.live_frame.copy()
        self._render_frame(self.frozen_frame)

    def _resume_live(self):
        self.frozen_frame = None

    def _is_duplicate(self, cam_label, crop):
        if not self.dup_check_var.get():
            return False
        prev = self.last_saved_per_cam.get(cam_label)
        if prev is None or prev.shape != crop.shape:
            return False
        diff = float(np.mean(np.abs(prev.astype("float32") - crop.astype("float32"))))
        return diff < float(self.settings.get("duplicate_threshold", 2.0))

    def _label_capture(self, label):
        if not self.settings.get("save_root"):
            messagebox.showinfo("Image Collector", "Choose a save location first.")
            return
        model = self.model_var.get().strip()
        if not model:
            messagebox.showinfo("Image Collector", "Type a model name first.")
            return
        if not self.rois:
            messagebox.showinfo("Image Collector", "No ROIs defined for this model yet -- draw and save some first.")
            return
        frame = self.frozen_frame if self.frozen_frame is not None else self.live_frame
        if frame is None:
            messagebox.showinfo("Image Collector", "No frame available to save.")
            return

        model_folder = _safe_folder_name(model)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        severity = self.severity_var.get() if label == "defect" else None
        saved_paths = []
        skipped = []

        for roi in self.rois:
            cam_label = roi.get("cam_label", "cam?")
            crop = crop_circular(frame, roi)
            if crop is None:
                continue
            if self._is_duplicate(cam_label, crop):
                skipped.append(cam_label)
                continue
            class_dir = os.path.join(self.settings["save_root"], model_folder, cam_label, label)
            os.makedirs(class_dir, exist_ok=True)
            sev_suffix = f"_{severity}" if severity and severity != "unspecified" else ""
            fname = f"{cam_label}_{ts}{sev_suffix}.png"
            fpath = os.path.join(class_dir, fname)
            cv2.imwrite(fpath, crop)
            saved_paths.append(fpath)
            self.last_saved_per_cam[cam_label] = crop
            key = (model, cam_label, label)
            self.counts[key] = self.counts.get(key, 0) + 1

        if not saved_paths and skipped:
            self.status_var.set(f"Skipped (looked like a duplicate): {', '.join(skipped)}")
            return
        if not saved_paths:
            messagebox.showinfo("Image Collector", "Nothing was saved (ROIs fell outside the frame?).")
            return

        self._update_counts_display()
        self._add_history_entry(saved_paths, label, frame)
        note = f"Saved {len(saved_paths)} crop(s) as {label.upper()}"
        if skipped:
            note += f" (skipped duplicate: {', '.join(skipped)})"
        self.status_var.set(note)

        # a labeled capture is "used up" -- go back to live so the next
        # click naturally captures a fresh frame instead of re-saving the
        # same one twice by accident
        self.frozen_frame = None

    def _update_counts_display(self):
        model = self.model_var.get().strip()
        lines = []
        by_cam = {}
        for (m, cam, cls), n in self.counts.items():
            if m != model:
                continue
            by_cam.setdefault(cam, {"good": 0, "defect": 0})
            by_cam[cam][cls] = by_cam[cam].get(cls, 0) + n
        if not by_cam:
            self.counts_label.configure(text="No captures yet for this model.")
            return
        for cam in sorted(by_cam):
            c = by_cam[cam]
            lines.append(f"{cam}: {c.get('good', 0)} good, {c.get('defect', 0)} defect")
        self.counts_label.configure(text="\n".join(lines))

    # ------------------------------------------------------------- history --
    def _add_history_entry(self, paths, label, frame):
        try:
            thumb_src = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            thumb_src.thumbnail((70, 70))
            photo = ImageTk.PhotoImage(thumb_src)
        except Exception:
            photo = None

        entry = {"paths": paths, "label": label, "photo": photo}
        self.capture_history.insert(0, entry)
        self.capture_history = self.capture_history[:8]
        self._render_history()

    def _render_history(self):
        for w in self.history_frame.winfo_children():
            w.destroy()
        for entry in self.capture_history:
            row = ctk.CTkFrame(self.history_frame, fg_color=BG_CARD, corner_radius=8)
            row.pack(fill="x", pady=3)
            if entry.get("photo") is not None:
                tk.Label(row, image=entry["photo"], bg=BG_CARD, bd=0).pack(side="left", padx=6, pady=6)
            color = SUCCESS if entry["label"] == "good" else DANGER
            ctk.CTkLabel(row, text=f"{entry['label'].upper()} ({len(entry['paths'])} files)",
                         font=self.f_small, text_color=color).pack(side="left", padx=6)
            ctk.CTkButton(row, text="Undo", width=60, font=self.f_small, fg_color=BG_CARD_ALT,
                          hover_color=BORDER, text_color=DANGER,
                          command=lambda e=entry: self._undo_entry(e)).pack(side="right", padx=6)

    def _undo_entry(self, entry):
        for p in entry["paths"]:
            try:
                os.remove(p)
            except Exception:
                pass
        # best-effort: also decrement the matching counters
        # path shape is .../<model>/<cam_label>/<good|defect>/<file>
        model = self.model_var.get().strip()
        for p in entry["paths"]:
            cam_label = os.path.basename(os.path.dirname(os.path.dirname(p)))
            key = (model, cam_label, entry["label"])
            if key in self.counts and self.counts[key] > 0:
                self.counts[key] -= 1
        self.capture_history.remove(entry)
        self._update_counts_display()
        self._render_history()

    # --------------------------------------------------------- burst mode --
    def _toggle_burst_mode(self):
        self.burst_running = not self.burst_running
        if self.burst_running:
            self.burst_btn.configure(text="Stop Auto-collect Good", fg_color=WARNING)
            self._burst_tick()
        else:
            self.burst_btn.configure(text="Start Auto-collect Good", fg_color=BG_CARD)
            if self.burst_job is not None:
                self.root.after_cancel(self.burst_job)
                self.burst_job = None

    def _burst_tick(self):
        if not self.burst_running:
            return
        self._resume_live()
        self.frozen_frame = self.live_frame.copy() if self.live_frame is not None else None
        if self.frozen_frame is not None:
            self._label_capture("good")
        try:
            interval = max(float(self.burst_interval_var.get()), 1.0)
        except ValueError:
            interval = 5.0
        self.burst_job = self.root.after(int(interval * 1000), self._burst_tick)

    # ------------------------------------------------------------- close --
    def on_close(self):
        self.burst_running = False
        try:
            self.cam.disconnect()
        except Exception:
            pass
        save_settings(self.settings)
        self.root.destroy()


def main():
    root = ctk.CTk()
    app = ImageCollectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
