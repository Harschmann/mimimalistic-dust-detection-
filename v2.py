"""
Dust Inspector — Operator-facing camera dust-detection app
------------------------------------------------------------
Single-file desktop app, restructured into TWO windows:

  1. Main Operator Window — what the line operator actually uses:
     title, Model/Line readout, a big READ-ONLY live feed, a scrolling
     result log, and a right-hand panel with Status / Barcode / Teaching
     button / Start-Stop. Nothing here is click-editable — no ROI dragging,
     no zoom -- on purpose. That was the #1 suspect for ROI "slipping":
     the same feed the operator watched was also the feed ROIs got added,
     moved, or resized on with a stray click. Splitting it out removes
     that failure mode entirely. (Separately: if the module itself can
     shift on the jig -- loose clamp, not seated flush, vibration -- that's
     a mechanical thing no amount of software can catch; worth a physical
     check on the fixture too.)

  2. Teaching Window (opened via the "Teaching" button) — technician-only
     setup: Model & Line naming, ROI placement + two-point calibration
     (the one interactive canvas in the whole app), detection parameters,
     and camera settings. Saved ROI layouts can be loaded and "activated"
     here, which is what the main window then uses.

Detection algorithm is UNCHANGED from the validated reference: local
Z-score (boxFilter mean/std) thresholded per-ROI, cleaned up with a
morphological opening + circularity/area/diameter filter so arcs and
tiny noise specks are rejected.

storage/ layout: source_images/, results/NG|OK/, roi_configs/, logs/,
settings.json

Run:  python dust_inspector_app.py
"""

import os
import sys
import csv
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
ACCENT_SOFT = "#182337"
TEXT = "#e6e8eb"
TEXT_MUTED = "#8a919c"
SUCCESS = "#22c55e"
SUCCESS_HOVER = "#16a34a"
DANGER = "#ef4444"
DANGER_HOVER = "#dc2626"
WARNING = "#f59e0b"

# ---------------------------------------------------------------- storage --
if getattr(sys, "frozen", False):
    # Running as a PyInstaller .exe: __file__ would point at a temp
    # extraction folder that's wiped after the app closes. Use the actual
    # exe's folder instead so storage/ persists across runs.
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SOURCE_DIR = os.path.join(STORAGE_DIR, "source_images")
RESULTS_DIR = os.path.join(STORAGE_DIR, "results")
RESULTS_NG_DIR = os.path.join(RESULTS_DIR, "NG")
RESULTS_OK_DIR = os.path.join(RESULTS_DIR, "OK")
ROI_DIR = os.path.join(STORAGE_DIR, "roi_configs")
LOGS_DIR = os.path.join(STORAGE_DIR, "logs")
LOG_CSV_PATH = os.path.join(LOGS_DIR, "inspection_log.csv")
SETTINGS_PATH = os.path.join(STORAGE_DIR, "settings.json")

for _d in (STORAGE_DIR, SOURCE_DIR, RESULTS_DIR, RESULTS_NG_DIR, RESULTS_OK_DIR, ROI_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_SETTINGS = {
    "window": 100,
    "z_thr": 3.0,
    "default_radius": 100,
    "scale_mm_per_px": None,
    "exposure_us": 20000.0,
    "gain": 0.0,
    "min_area": 4.0,
    "min_circularity": 0.55,
    "min_diameter_mm": 0.1,
    "model_name": "",
    "line_name": "",
    "active_roi_name": None,
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
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def log_to_csv(row):
    is_new = not os.path.exists(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "model", "line", "barcode", "verdict", "dust_count", "max_diameter_mm"])
        w.writerow(row)


# ---------------------------------------------------------- camera manager --
class CameraManager:
    """Thin wrapper around a Basler camera via pypylon, with a background
    grab thread. Safe to use even when pypylon / hardware is unavailable —
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


# ---------------------------------------------------------------- detect ----
def run_zscore_detection(bgr, rois, window, z_thr, min_area=4.0, min_circularity=0.55,
                          mm_per_px=None, min_diameter_mm=0.0):
    """Core Z-score math is UNCHANGED: local Z-score via boxFilter mean/std,
    thresholded inside the union of all circular ROI masks.

    On top of that, two shape-based filters clean up the raw threshold:
      - a small morphological opening erases anything thinner than a few px
        (this kills thin curved edge artifacts / arcs and 1-2px sensor noise,
        since a real dust speck is round and a few px wide, an arc isn't)
      - a circularity + min-area check on the surviving blobs drops anything
        that's still elongated/thin (residual arc fragments) or too small to
        be a real particle (min_area, min_circularity are tunable in Settings)

    If the app has been calibrated (mm_per_px set via two-point calibration),
    each surviving blob also gets a real-world diameter_mm, and anything
    smaller than min_diameter_mm is rejected too. Without calibration this
    step is skipped (there's no way to convert px -> mm yet).

    Returns the cleaned binary mask, one min-enclosing circle per accepted
    dust blob (used to ring it in red; each carries diameter_px and,
    if calibrated, diameter_mm), and summary stats.
    """
    win = window if window % 2 == 1 else window + 1
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    local_mean = cv2.boxFilter(gray, -1, (win, win))
    local_mean_sq = cv2.boxFilter(gray * gray, -1, (win, win))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0))
    zscore = np.where(local_std > 1e-5, (gray - local_mean) / local_std, 0.0)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    for roi in rois:
        cv2.circle(mask, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), 255, -1)

    raw = np.where((zscore >= z_thr) & (mask == 255), 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    binary = np.zeros_like(opened)
    rejected = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            rejected += 1
            continue
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            rejected += 1
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        diameter_px = 2.0 * r
        diameter_mm = diameter_px * mm_per_px if mm_per_px else None
        if diameter_mm is not None and diameter_mm < min_diameter_mm:
            rejected += 1
            continue
        circles.append({"cx": float(cx), "cy": float(cy), "r": float(max(r, 3.0)),
                         "diameter_px": float(diameter_px), "diameter_mm": diameter_mm})
        cv2.drawContours(binary, [c], -1, 255, -1)

    stats = None
    if mask.any():
        roi_z = zscore[mask == 255]
        stats = {"max_z": float(roi_z.max()), "mean_z": float(roi_z.mean()),
                 "dust_px": int((binary == 255).sum()), "dust_count": len(circles),
                 "rejected": rejected}
    return binary, circles, stats


CANVAS_W = 560
CANVAS_H = 560
ROI_HIT_TOL = 6


# =============================================================== MAIN APP ==
class DustInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Dust Inspection")
        self.root.geometry("1480x900")
        self.root.minsize(1200, 760)
        self.root.configure(fg_color=BG)

        self.f_title = ctk.CTkFont(size=22, weight="bold")
        self.f_subtitle = ctk.CTkFont(size=13)
        self.f_section = ctk.CTkFont(size=15, weight="bold")
        self.f_body = ctk.CTkFont(size=13)
        self.f_small = ctk.CTkFont(size=11)
        self.f_status = ctk.CTkFont(size=20, weight="bold")

        self.settings = load_settings()
        self.cam = CameraManager()

        # shared state
        self.original = None
        self.using_static_image = False
        self.rois = []                 # active ROI set used by the main window
        self.last_circles = []         # last inspection's accepted dust blobs
        self.inspection_running = False

        # teaching-window-only interactive state (ROI editor canvas)
        self.selected_idx = None
        self.calib_mode = False
        self.calib_points = []
        self.t_zoom = 1.0
        self.t_base_scale = 1.0
        self.t_view_x = 0.0
        self.t_view_y = 0.0
        self._t_dragging = False
        self._t_drag_mode = None
        self._t_drag_start = (0, 0)
        self._t_last = (0, 0)

        # main operator feed view-only zoom/pan state (no ROI editing here)
        self.m_zoom = 1.0
        self.m_base_scale = 1.0
        self.m_view_x = 0.0
        self.m_view_y = 0.0
        self._m_dragging = False
        self._m_drag_start = (0, 0)
        self._m_last = (0, 0)
        self._main_fitted = False
        self._main_fitted_shape = None

        self.current_view = "operator"
        self.main_photo = None
        self.roi_photo = None

        self.model_line_var = tk.StringVar(value=self._model_line_text())
        self.status_var = tk.StringVar(value="IDLE")
        self.barcode_var = tk.StringVar(value="")
        self.footer_var = tk.StringVar(value="Starting up…")

        self._build_main_ui()
        self._load_active_roi_layout()
        self._auto_connect_camera()
        self._poll_live()

    # ------------------------------------------------------------- utils --
    def _model_line_text(self):
        m = self.settings.get("model_name") or "—"
        l = self.settings.get("line_name") or "—"
        return f"Model: {m}      Line: {l}"

    def _scale_text(self):
        s = self.settings.get("scale_mm_per_px")
        return f"Scale: {s:.5f} mm/px" if s else "Scale: not calibrated"

    def _btn_primary(self, parent, text, command, width=130, **kw):
        opts = dict(text=text, command=command, width=width, corner_radius=8,
                    fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff", font=self.f_body)
        opts.update(kw)
        return ctk.CTkButton(parent, **opts)

    def _btn_secondary(self, parent, text, command, width=110, **kw):
        opts = dict(text=text, command=command, width=width, corner_radius=8,
                    fg_color=BG_CARD_ALT, hover_color=BORDER, text_color=TEXT, font=self.f_body)
        opts.update(kw)
        return ctk.CTkButton(parent, **opts)

    def _card(self, parent, **kw):
        defaults = dict(fg_color=BG_CARD, corner_radius=14, border_width=1, border_color=BORDER)
        defaults.update(kw)
        return ctk.CTkFrame(parent, **defaults)

    def _field(self, parent, label_text, var, width=140):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(f, text=label_text, font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w")
        ctk.CTkEntry(f, textvariable=var, width=width, corner_radius=8,
                     fg_color=BG_CARD_ALT, border_color=BORDER, text_color=TEXT).pack(anchor="w", pady=(4, 0))
        return f

    # ==================================================== MAIN OPERATOR UI
    def _build_main_ui(self):
        outer = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
        outer.pack(fill="both", expand=True)

        # ---- header (persistent across both views) ----
        header = ctk.CTkFrame(outer, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(16, 6))
        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left")
        ctk.CTkLabel(title_box, text="Camera Dust Inspection", font=self.f_title, text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(title_box, textvariable=self.model_line_var, font=self.f_subtitle, text_color=TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        nav = ctk.CTkFrame(header, fg_color=BG_CARD_ALT, corner_radius=10)
        nav.pack(side="right", pady=(8, 0))
        self.nav_operator_btn = ctk.CTkButton(nav, text="Operator", width=110, corner_radius=8, font=self.f_body,
                                               command=self.show_operator_page)
        self.nav_operator_btn.pack(side="left", padx=4, pady=4)
        self.nav_teaching_btn = ctk.CTkButton(nav, text="Teaching", width=110, corner_radius=8, font=self.f_body,
                                               command=self.show_teaching_page)
        self.nav_teaching_btn.pack(side="left", padx=4, pady=4)

        # ---- swappable content area: same window, no separate popup ----
        container = ctk.CTkFrame(outer, fg_color=BG, corner_radius=0)
        container.pack(fill="both", expand=True, padx=20, pady=(4, 6))
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.page_operator = ctk.CTkFrame(container, fg_color=BG, corner_radius=0)
        self.page_teaching = ctk.CTkFrame(container, fg_color=BG, corner_radius=0)
        self.page_operator.grid(row=0, column=0, sticky="nsew")
        self.page_teaching.grid(row=0, column=0, sticky="nsew")

        self._build_operator_page(self.page_operator)
        self._build_teaching_page(self.page_teaching)

        # ---- footer ----
        footer = ctk.CTkFrame(outer, fg_color=BG_SIDEBAR, height=30, corner_radius=0)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        ctk.CTkLabel(footer, textvariable=self.footer_var, font=self.f_small, text_color=TEXT_MUTED).pack(side="left", padx=16)

        self.show_operator_page()

    def show_operator_page(self):
        self.current_view = "operator"
        self.page_operator.tkraise()
        self.nav_operator_btn.configure(fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff")
        self.nav_teaching_btn.configure(fg_color="transparent", hover_color=BORDER, text_color=TEXT_MUTED)
        self._render_main_feed()
        self.barcode_entry.focus_set()

    def show_teaching_page(self):
        self.current_view = "teaching"
        self.page_teaching.tkraise()
        self.nav_teaching_btn.configure(fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff")
        self.nav_operator_btn.configure(fg_color="transparent", hover_color=BORDER, text_color=TEXT_MUTED)
        self.fit_roi_view()
        self._render_roi_canvas()

    def _build_operator_page(self, page):
        body = ctk.CTkFrame(page, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # ---- left: feed (view-only zoom/pan) + log ----
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.grid_rowconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=0)
        left.grid_columnconfigure(0, weight=1)

        feed_card = self._card(left)
        feed_card.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        feed_head = ctk.CTkFrame(feed_card, fg_color="transparent")
        feed_head.pack(fill="x", padx=14, pady=(12, 0))
        ctk.CTkLabel(feed_head, text="Live Feed  •  wheel = zoom  •  drag = pan (view only)",
                     font=self.f_small, text_color=TEXT_MUTED).pack(side="left")
        self._btn_secondary(feed_head, "Fit", self._fit_main_view, width=60).pack(side="right")
        wrap = ctk.CTkFrame(feed_card, fg_color=BG_CANVAS, corner_radius=10)
        wrap.pack(fill="both", expand=True, padx=14, pady=14)
        self.feed_canvas = tk.Canvas(wrap, bg=BG_CANVAS, highlightthickness=0)
        self.feed_canvas.pack(fill="both", expand=True, padx=3, pady=3)
        self.feed_canvas.bind("<Configure>", lambda e: self._render_main_feed())
        self.feed_canvas.bind("<MouseWheel>", self.on_main_wheel)
        self.feed_canvas.bind("<Button-4>", self.on_main_wheel)
        self.feed_canvas.bind("<Button-5>", self.on_main_wheel)
        self.feed_canvas.bind("<ButtonPress-1>", self.on_main_press)
        self.feed_canvas.bind("<B1-Motion>", self.on_main_drag)

        log_card = self._card(left, height=190)
        log_card.grid(row=1, column=0, sticky="nsew")
        log_card.grid_propagate(False)
        ctk.CTkLabel(log_card, text="Inspection Log", font=self.f_section, text_color=TEXT).pack(anchor="w", padx=16, pady=(12, 6))
        self.log_box = ctk.CTkTextbox(log_card, fg_color=BG_CANVAS, text_color=TEXT_MUTED,
                                       font=ctk.CTkFont(family="Courier New", size=12), corner_radius=8, wrap="none")
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.log_box.configure(state="disabled")

        # ---- right: status / barcode / start ----
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")

        status_card = self._card(right)
        status_card.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(status_card, text="STATUS", font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(14, 0))
        self.status_label = ctk.CTkLabel(status_card, textvariable=self.status_var, font=self.f_status, text_color=TEXT_MUTED)
        self.status_label.pack(anchor="w", padx=18, pady=(2, 16))

        barcode_card = self._card(right)
        barcode_card.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(barcode_card, text="BARCODE", font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(14, 4))
        self.barcode_entry = ctk.CTkEntry(barcode_card, textvariable=self.barcode_var, font=ctk.CTkFont(size=15),
                                           corner_radius=8, fg_color=BG_CARD_ALT, border_color=BORDER, text_color=TEXT)
        self.barcode_entry.pack(fill="x", padx=18, pady=(0, 16))

        self.start_btn = self._btn_primary(right, "Start Inspection", self.start_inspection, width=200, height=52,
                                            font=ctk.CTkFont(size=15, weight="bold"))
        self.start_btn.pack(fill="x", pady=(0, 10))

    def _build_teaching_page(self, page):
        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(top, text="Teaching / Setup", font=self.f_title, text_color=TEXT).pack(side="left")
        self._btn_secondary(top, "Back to Operator", self.show_operator_page, width=160).pack(side="right")

        tabs = ctk.CTkTabview(page)
        tabs.pack(fill="both", expand=True)
        tab_model = tabs.add("Model & Line")
        tab_roi = tabs.add("ROI & Calibration")
        tab_detect = tabs.add("Detection")
        tab_cam = tabs.add("Camera")

        self._build_model_tab(tab_model)
        self._build_roi_tab(tab_roi)
        self._build_detection_tab(tab_detect)
        self._build_camera_tab(tab_cam)

    def _set_status(self, text, color):
        self.status_var.set(text)
        self.status_label.configure(text_color=color)

    # ------------------------------------------------------------ camera --
    def _auto_connect_camera(self):
        ok, msg = self.cam.connect()
        if ok:
            self.cam.apply_settings(exposure_us=self.settings.get("exposure_us"), gain=self.settings.get("gain"))
            self.cam.start_live()
            self.footer_var.set("Camera connected — live feed running.")
        else:
            self.footer_var.set(f"Camera not connected ({msg}). Use Teaching > Camera to retry, or Open Image for offline testing.")

    def _poll_live(self):
        if not self.using_static_image:
            frame = self.cam.get_frame()
            if frame is not None:
                self.original = frame
                self._render_main_feed()
                if self.current_view == "teaching":
                    self._render_roi_canvas()
        self.root.after(120, self._poll_live)

    # -------------------------------------------------------- ROI layouts --
    def _load_active_roi_layout(self):
        name = self.settings.get("active_roi_name")
        if not name:
            return
        path = os.path.join(ROI_DIR, f"{name}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.rois = json.load(f)
            except Exception:
                self.rois = []

    def _activate_roi_layout(self, name, rois):
        self.rois = rois
        self.settings["active_roi_name"] = name
        save_settings(self.settings)
        self._render_main_feed()

    # -------------------------------------------------- main feed (view-only)
    def _build_main_overlay(self):
        disp = self.original.copy()
        for roi in self.rois:
            cv2.circle(disp, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (0, 255, 0), 2)
        for d in self.last_circles:
            cx, cy, r = int(d["cx"]), int(d["cy"]), int(round(d["r"]))
            cv2.circle(disp, (cx, cy), r + 4, (0, 0, 255), 2)
            label = f"{d['diameter_mm']:.2f}mm" if d.get("diameter_mm") is not None else f"{int(round(d['diameter_px']))}px"
            cv2.putText(disp, label, (cx + r + 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        return disp

    def _canvas_wh_main(self):
        w, h = self.feed_canvas.winfo_width(), self.feed_canvas.winfo_height()
        if w < 10 or h < 10:
            return CANVAS_W, CANVAS_H
        return w, h

    def _fit_main_view(self):
        if self.original is None:
            return
        cw, ch = self._canvas_wh_main()
        h, w = self.original.shape[:2]
        self.m_base_scale = min(cw / w, ch / h)
        self.m_zoom = 1.0
        s = self.m_base_scale
        self.m_view_x = (cw - w * s) / 2
        self.m_view_y = (ch - h * s) / 2
        self._main_fitted = True
        self._main_fitted_shape = (h, w)
        self._render_main_feed()

    def _apply_main_zoom(self, factor, cx, cy):
        if self.original is None:
            return
        s_old = self.m_base_scale * self.m_zoom
        ix = (cx - self.m_view_x) / s_old
        iy = (cy - self.m_view_y) / s_old
        self.m_zoom = max(0.2, min(self.m_zoom * factor, 20.0))
        s_new = self.m_base_scale * self.m_zoom
        self.m_view_x = cx - ix * s_new
        self.m_view_y = cy - iy * s_new
        self._render_main_feed()

    def on_main_wheel(self, event):
        if self.original is None:
            return
        direction = 1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4) else -1
        factor = 1.2 if direction > 0 else 1 / 1.2
        self._apply_main_zoom(factor, event.x, event.y)

    def on_main_press(self, event):
        self._m_drag_start = (event.x, event.y)
        self._m_last = (event.x, event.y)
        self._m_dragging = False

    def on_main_drag(self, event):
        if self.original is None:
            return
        if not self._m_dragging:
            if abs(event.x - self._m_drag_start[0]) + abs(event.y - self._m_drag_start[1]) > 4:
                self._m_dragging = True
        if not self._m_dragging:
            return
        self.m_view_x += event.x - self._m_last[0]
        self.m_view_y += event.y - self._m_last[1]
        self._m_last = (event.x, event.y)
        self._render_main_feed()

    def _render_main_feed(self):
        self.feed_canvas.delete("all")
        if self.original is None:
            return
        shape = self.original.shape[:2]
        if not self._main_fitted or self._main_fitted_shape != shape:
            self._fit_main_view()
            return
        disp = self._build_main_overlay()
        cw, ch = self._canvas_wh_main()
        H, W = disp.shape[:2]
        s = self.m_base_scale * self.m_zoom
        vx, vy = self.m_view_x, self.m_view_y
        l = max(0, int(-vx / s))
        t = max(0, int(-vy / s))
        r = min(W, int((cw - vx) / s) + 1)
        b = min(H, int((ch - vy) / s) + 1)
        if r <= l or b <= t:
            return
        crop = disp[t:b, l:r]
        cwid = max(1, int((r - l) * s))
        chei = max(1, int((b - t) * s))
        interp = cv2.INTER_CUBIC if self.m_zoom > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop, (cwid, chei), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.feed_canvas.create_image(vx + l * s, vy + t * s, anchor="nw", image=photo)
        self.main_photo = photo

    # ------------------------------------------------------------- logging
    def _append_log(self, model, line, barcode, verdict, dust_count, max_dia):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dia_txt = f"{max_dia:.2f}mm" if max_dia is not None else "-"
        line_txt = f"[{ts}] {barcode:<16} | Model:{model or '-':<10} Line:{line or '-':<8} | {verdict:<4} | dust={dust_count} max={dia_txt}\n"
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line_txt)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        log_to_csv([ts, model, line, barcode, verdict, dust_count, f"{max_dia:.3f}" if max_dia is not None else ""])

    # -------------------------------------------------------- inspection --
    def start_inspection(self):
        if self.inspection_running:
            return
        if self.original is None:
            messagebox.showwarning("No Feed", "No camera frame available yet.")
            return
        if not self.rois:
            messagebox.showwarning("No ROI", "No ROI is configured. Open Teaching to set one up.")
            return
        barcode = self.barcode_var.get().strip()
        if not barcode:
            # no scanner hooked up (e.g. testing on a laptop) -- don't block,
            # just tag the record so it's still traceable in the log
            barcode = f"MANUAL-{datetime.now().strftime('%H%M%S')}"

        self.inspection_running = True
        self.start_btn.configure(text="Running…", state="disabled", fg_color=BG_CARD_ALT)
        self._set_status("IN PROGRESS", WARNING)
        frame = self.original.copy()
        rois_snapshot = [dict(r) for r in self.rois]
        threading.Thread(target=self._run_inspection_thread, args=(frame, rois_snapshot, barcode), daemon=True).start()

    def _run_inspection_thread(self, frame, rois_snapshot, barcode):
        s = self.settings
        try:
            binary, circles, stats = run_zscore_detection(
                frame, rois_snapshot, s["window"], s["z_thr"], s["min_area"], s["min_circularity"],
                s.get("scale_mm_per_px"), s["min_diameter_mm"])
        except Exception as e:
            circles, stats = [], None
        self.root.after(0, lambda: self._finish_inspection(frame, rois_snapshot, circles, barcode))

    def _finish_inspection(self, frame, rois_snapshot, circles, barcode):
        self.last_circles = circles
        is_ng = bool(circles)
        verdict = "FAIL" if is_ng else "PASS"
        self._set_status(verdict, DANGER if is_ng else SUCCESS)

        # save source (with ROI overlay) + verdict-filed result (with red rings)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_disp = frame.copy()
        for roi in rois_snapshot:
            cv2.circle(source_disp, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(SOURCE_DIR, f"source_{ts}_{barcode}.png"), source_disp)

        result_disp = source_disp.copy()
        max_dia = None
        for d in circles:
            cx, cy, r = int(d["cx"]), int(d["cy"]), int(round(d["r"]))
            cv2.circle(result_disp, (cx, cy), r + 4, (0, 0, 255), 2)
            label = f"{d['diameter_mm']:.2f}mm" if d.get("diameter_mm") is not None else f"{int(round(d['diameter_px']))}px"
            cv2.putText(result_disp, label, (cx + r + 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            if d.get("diameter_mm") is not None:
                max_dia = max(max_dia or 0.0, d["diameter_mm"])
        verdict_dir = RESULTS_NG_DIR if is_ng else RESULTS_OK_DIR
        cv2.imwrite(os.path.join(verdict_dir, f"{'NG' if is_ng else 'OK'}_{ts}_{barcode}.png"), result_disp)

        self._append_log(self.settings.get("model_name"), self.settings.get("line_name"), barcode, verdict, len(circles), max_dia)

        self._render_main_feed()
        self.inspection_running = False
        self.start_btn.configure(text="Start Inspection", state="normal", fg_color=ACCENT)
        self.barcode_var.set("")
        self.barcode_entry.focus_set()

    # =================================================== TEACHING PAGE ==

    # ---- Model & Line -------------------------------------------------
    def _build_model_tab(self, tab):
        card = self._card(tab)
        card.pack(fill="x", padx=20, pady=20)
        ctk.CTkLabel(card, text="Model & Line", font=self.f_section, text_color=TEXT).pack(anchor="w", padx=18, pady=(16, 10))
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(0, 18))
        self.model_var = tk.StringVar(value=self.settings.get("model_name", ""))
        self.line_var = tk.StringVar(value=self.settings.get("line_name", ""))
        self._field(row, "Model", self.model_var, width=220).pack(side="left", padx=(0, 20))
        self._field(row, "Line", self.line_var, width=220).pack(side="left", padx=(0, 20))
        self._btn_primary(row, "Save", self._save_model_line, width=100).pack(side="left", pady=(18, 0))

    def _save_model_line(self):
        self.settings["model_name"] = self.model_var.get().strip()
        self.settings["line_name"] = self.line_var.get().strip()
        save_settings(self.settings)
        self.model_line_var.set(self._model_line_text())

    # ---- ROI & Calibration --------------------------------------------
    def _build_roi_tab(self, tab):
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(14, 6))
        self._btn_secondary(top, "Open Image", self.open_image, width=110).pack(side="left", padx=(0, 6))
        self._btn_secondary(top, "Delete ROI", self.delete_selected_roi, width=100).pack(side="left", padx=6)
        self._btn_secondary(top, "Clear All", self.clear_rois, width=90).pack(side="left", padx=6)
        self._btn_secondary(top, "Save Layout", self.save_roi_layout, width=100).pack(side="left", padx=6)
        self._btn_secondary(top, "Fit", self.fit_roi_view, width=52).pack(side="left", padx=(20, 4))
        self._btn_secondary(top, "Test Detection", self.test_detection_once, width=130).pack(side="right")

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        canvas_card = self._card(body)
        canvas_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ctk.CTkLabel(canvas_card, text="click = add/select ROI  •  drag = pan  •  wheel = zoom  •  scroll on selected ROI = resize",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=14, pady=(10, 6))
        wrap = ctk.CTkFrame(canvas_card, fg_color=BG_CANVAS, corner_radius=10)
        wrap.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.roi_canvas = tk.Canvas(wrap, bg=BG_CANVAS, highlightthickness=0)
        self.roi_canvas.pack(fill="both", expand=True, padx=3, pady=3)
        self.roi_canvas.bind("<MouseWheel>", self.on_wheel)
        self.roi_canvas.bind("<Button-4>", self.on_wheel)
        self.roi_canvas.bind("<Button-5>", self.on_wheel)
        self.roi_canvas.bind("<ButtonPress-1>", self.on_press)
        self.roi_canvas.bind("<B1-Motion>", self.on_drag)
        self.roi_canvas.bind("<ButtonRelease-1>", self.on_release)
        self.roi_canvas.bind("<Configure>", lambda e: self._render_roi_canvas())

        side = ctk.CTkFrame(body, fg_color="transparent")
        side.grid(row=0, column=1, sticky="nsew")

        layouts_card = self._card(side)
        layouts_card.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(layouts_card, text="Saved ROI Layouts", font=self.f_section, text_color=TEXT).pack(anchor="w", padx=16, pady=(14, 6))
        self.layout_listbox = tk.Listbox(layouts_card, bg=BG_CARD_ALT, fg=TEXT, highlightthickness=0,
                                          selectbackground=ACCENT, borderwidth=0, height=8)
        self.layout_listbox.pack(fill="x", padx=16, pady=(0, 8))
        self._refresh_layout_list()
        lb_btns = ctk.CTkFrame(layouts_card, fg_color="transparent")
        lb_btns.pack(fill="x", padx=16, pady=(0, 16))
        self._btn_primary(lb_btns, "Load & Activate", self.load_and_activate_roi, width=150).pack(side="left")
        self._btn_secondary(lb_btns, "Refresh", self._refresh_layout_list, width=90).pack(side="left", padx=(8, 0))

        calib_card = self._card(side)
        calib_card.pack(fill="x")
        ctk.CTkLabel(calib_card, text="Two-Point Calibration", font=self.f_section, text_color=TEXT).pack(anchor="w", padx=16, pady=(14, 6))
        ctk.CTkLabel(calib_card, text="Click 2 points at a known real-world distance.",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=16, pady=(0, 8))
        cbtns = ctk.CTkFrame(calib_card, fg_color="transparent")
        cbtns.pack(fill="x", padx=16, pady=(0, 8))
        self._btn_primary(cbtns, "Start", self.start_calibration, width=80).pack(side="left", padx=(0, 6))
        self._btn_secondary(cbtns, "Undo Point", self.undo_calib_point, width=100).pack(side="left", padx=6)
        self._btn_secondary(cbtns, "Reset", self.reset_calibration, width=70).pack(side="left", padx=6)
        self.scale_label_var = tk.StringVar(value=self._scale_text())
        ctk.CTkLabel(calib_card, textvariable=self.scale_label_var, font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=16, pady=(0, 16))

    def _refresh_layout_list(self):
        self.layout_listbox.delete(0, "end")
        try:
            names = sorted(f[:-5] for f in os.listdir(ROI_DIR) if f.endswith(".json"))
        except FileNotFoundError:
            names = []
        for n in names:
            self.layout_listbox.insert("end", n)

    def load_and_activate_roi(self):
        sel = self.layout_listbox.curselection()
        if not sel:
            messagebox.showinfo("Load ROI", "Select a saved layout first.")
            return
        name = self.layout_listbox.get(sel[0])
        path = os.path.join(ROI_DIR, f"{name}.json")
        try:
            with open(path, "r") as f:
                rois = json.load(f)
        except Exception as e:
            messagebox.showerror("Load ROI", str(e))
            return
        self.selected_idx = None
        self._activate_roi_layout(name, rois)
        self._render_roi_canvas()
        self.footer_var.set(f"Active ROI layout: {name}")

    def save_roi_layout(self):
        if not self.rois:
            messagebox.showinfo("Save Layout", "No ROIs to save.")
            return
        name = simpledialog.askstring("Save ROI Layout", "Layout name (e.g. model_A56_main):", parent=self.root)
        if not name:
            return
        with open(os.path.join(ROI_DIR, f"{name}.json"), "w") as f:
            json.dump(self.rois, f, indent=2)
        self._refresh_layout_list()
        self.footer_var.set(f"Saved ROI layout: {name}")

    def delete_selected_roi(self):
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.rois):
            self.rois.pop(self.selected_idx)
            self.selected_idx = None
            self._render_roi_canvas()
            self._render_main_feed()

    def clear_rois(self):
        self.rois = []
        self.selected_idx = None
        self._render_roi_canvas()
        self._render_main_feed()

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All", "*.*")])
        if not path:
            return
        try:
            data = np.fromfile(path, dtype=np.uint8)  # unicode/non-ASCII path safe, unlike cv2.imread directly
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            messagebox.showerror("Open Image", "Could not load image (unsupported format, corrupt file, or bad path).")
            return
        self.using_static_image = True  # stop the live feed from overwriting this
        self.original = img
        self.fit_roi_view()
        self._render_main_feed()
        self.footer_var.set(f"Loaded {os.path.basename(path)} (static — live feed paused). Reconnect camera to resume live view.")

    def test_detection_once(self):
        if self.original is None or not self.rois:
            messagebox.showinfo("Test Detection", "Need an image and at least one ROI.")
            return
        s = self.settings
        binary, circles, stats = run_zscore_detection(
            self.original, self.rois, s["window"], s["z_thr"], s["min_area"], s["min_circularity"],
            s.get("scale_mm_per_px"), s["min_diameter_mm"])
        self.last_circles = circles
        self._render_roi_canvas()
        self._render_main_feed()
        msg = f"dust_count={len(circles)}"
        if stats:
            msg += f", max_z={stats['max_z']:.2f}, rejected={stats['rejected']}"
        self.footer_var.set("Test detection: " + msg)

    # ---- calibration (shares ROI canvas clicks) -----------------------
    def start_calibration(self):
        if self.original is None:
            messagebox.showwarning("Calibration", "Load or capture an image first.")
            return
        self.calib_mode = True
        self.calib_points = []
        self.footer_var.set("Calibration: click 2 points at a known real-world distance.")

    def undo_calib_point(self):
        if self.calib_points:
            self.calib_points.pop()
            self._render_roi_canvas()

    def reset_calibration(self):
        self.settings["scale_mm_per_px"] = None
        save_settings(self.settings)
        self.scale_label_var.set(self._scale_text())

    def _finish_calibration(self):
        (x1, y1), (x2, y2) = self.calib_points
        pixel_dist = float(np.hypot(x2 - x1, y2 - y1))
        self.calib_mode = False
        self.calib_points = []
        if pixel_dist < 1:
            self.footer_var.set("Calibration points too close together, try again.")
            return
        dist_mm = simpledialog.askfloat("Calibration", "Real-world distance between the two points (mm):", parent=self.root)
        if not dist_mm:
            self._render_roi_canvas()
            return
        scale = dist_mm / pixel_dist
        self.settings["scale_mm_per_px"] = scale
        save_settings(self.settings)
        self.scale_label_var.set(self._scale_text())
        self._render_roi_canvas()

    # ---- Detection settings --------------------------------------------
    def _build_detection_tab(self, tab):
        card = self._card(tab)
        card.pack(fill="x", padx=20, pady=20)
        ctk.CTkLabel(card, text="Detection Parameters", font=self.f_section, text_color=TEXT).pack(anchor="w", padx=18, pady=(16, 10))
        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill="x", padx=18, pady=(0, 8))
        self.window_var = tk.StringVar(value=str(self.settings["window"]))
        self.zthr_var = tk.StringVar(value=str(self.settings["z_thr"]))
        self.radius_var = tk.StringVar(value=str(self.settings["default_radius"]))
        self._field(row1, "Window size", self.window_var, width=90).pack(side="left", padx=(0, 16))
        self._field(row1, "Z threshold", self.zthr_var, width=90).pack(side="left", padx=(0, 16))
        self._field(row1, "Default ROI radius (px)", self.radius_var, width=100).pack(side="left", padx=(0, 16))

        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill="x", padx=18, pady=(0, 18))
        self.min_area_var = tk.StringVar(value=str(self.settings["min_area"]))
        self.min_circ_var = tk.StringVar(value=str(self.settings["min_circularity"]))
        self.min_diam_var = tk.StringVar(value=str(self.settings["min_diameter_mm"]))
        self._field(row2, "Min blob area (px²)", self.min_area_var, width=90).pack(side="left", padx=(0, 16))
        self._field(row2, "Min circularity (0-1)", self.min_circ_var, width=100).pack(side="left", padx=(0, 16))
        self._field(row2, "Min dust diameter (mm)", self.min_diam_var, width=100).pack(side="left", padx=(0, 16))
        self._btn_primary(row2, "Save", self._save_detection_settings, width=100).pack(side="left", pady=(18, 0))

    def _save_detection_settings(self):
        try:
            self.settings["window"] = int(self.window_var.get())
            self.settings["z_thr"] = float(self.zthr_var.get())
            self.settings["default_radius"] = int(self.radius_var.get())
            self.settings["min_area"] = float(self.min_area_var.get())
            self.settings["min_circularity"] = float(self.min_circ_var.get())
            self.settings["min_diameter_mm"] = float(self.min_diam_var.get())
        except ValueError:
            messagebox.showerror("Settings", "All fields must be numbers.")
            return
        save_settings(self.settings)
        self.footer_var.set("Detection settings saved.")

    # ---- Camera settings -------------------------------------------------
    def _build_camera_tab(self, tab):
        card = self._card(tab)
        card.pack(fill="x", padx=20, pady=20)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=18, pady=(16, 0))
        ctk.CTkLabel(head, text="Camera", font=self.f_section, text_color=TEXT).pack(side="left")
        self.cam_status_var = tk.StringVar(value="Connected" if self.cam.connected else "Not connected")
        ctk.CTkLabel(head, textvariable=self.cam_status_var, font=self.f_small, text_color=TEXT_MUTED).pack(side="right")

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(10, 18))
        self.exposure_var = tk.StringVar(value=str(self.settings["exposure_us"]))
        self.gain_var = tk.StringVar(value=str(self.settings["gain"]))
        self._field(row, "Exposure (µs)", self.exposure_var).pack(side="left", padx=(0, 20))
        self._field(row, "Gain", self.gain_var).pack(side="left", padx=(0, 20))
        self._btn_primary(row, "Apply", self._apply_camera_settings, width=90).pack(side="left", padx=(0, 10), pady=(18, 0))
        self._btn_secondary(row, "Connect / Reconnect", self._reconnect_camera, width=160).pack(side="left", pady=(18, 0))
        ctk.CTkLabel(card, text="Gain amplifies sensor noise along with brightness — prefer raising Exposure over Gain.",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(0, 14))

    def _apply_camera_settings(self):
        try:
            exp = float(self.exposure_var.get())
            gain = float(self.gain_var.get())
        except ValueError:
            messagebox.showerror("Settings", "Exposure/Gain must be numbers.")
            return
        self.settings["exposure_us"] = exp
        self.settings["gain"] = gain
        save_settings(self.settings)
        self.cam.apply_settings(exposure_us=exp, gain=gain)
        self.footer_var.set("Camera settings applied.")

    def _reconnect_camera(self):
        self.cam.disconnect()
        ok, msg = self.cam.connect()
        self.cam_status_var.set(msg)
        if ok:
            self.using_static_image = False
            self.cam.apply_settings(exposure_us=self.settings.get("exposure_us"), gain=self.settings.get("gain"))
            self.cam.start_live()
            self.footer_var.set("Camera connected — live feed running.")
        else:
            messagebox.showwarning("Camera", msg)

    # ---------------------------------------------- ROI canvas interaction
    # (Only the Teaching window's ROI canvas can add/move/resize ROIs — the
    # main operator feed only zooms/pans for viewing, it never touches ROI
    # data, which is what fixes the ROI "slip": a stray click on the
    # operator's feed used to be able to add, move, or resize an ROI
    # without anyone noticing.)
    def _canvas_wh(self):
        w, h = self.roi_canvas.winfo_width(), self.roi_canvas.winfo_height()
        if w < 10 or h < 10:
            return CANVAS_W, CANVAS_H
        return w, h

    def fit_roi_view(self):
        if self.original is None:
            return
        cw, ch = self._canvas_wh()
        h, w = self.original.shape[:2]
        self.t_base_scale = min(cw / w, ch / h)
        self.t_zoom = 1.0
        s = self.t_base_scale
        self.t_view_x = (cw - w * s) / 2
        self.t_view_y = (ch - h * s) / 2
        self._render_roi_canvas()

    def _apply_zoom(self, factor, cx, cy):
        if self.original is None:
            return
        s_old = self.t_base_scale * self.t_zoom
        ix = (cx - self.t_view_x) / s_old
        iy = (cy - self.t_view_y) / s_old
        self.t_zoom = max(0.1, min(self.t_zoom * factor, 60.0))
        s_new = self.t_base_scale * self.t_zoom
        self.t_view_x = cx - ix * s_new
        self.t_view_y = cy - iy * s_new
        self._render_roi_canvas()

    def _screen_to_image(self, x, y):
        s = self.t_base_scale * self.t_zoom
        return (x - self.t_view_x) / s, (y - self.t_view_y) / s

    def _find_roi_at(self, ix, iy):
        for i in reversed(range(len(self.rois))):
            r = self.rois[i]
            if np.hypot(ix - r["cx"], iy - r["cy"]) <= r["r"] + ROI_HIT_TOL:
                return i
        return None

    def on_wheel(self, event):
        if self.original is None:
            return
        direction = 1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4) else -1
        if self.selected_idx is not None:
            ix, iy = self._screen_to_image(event.x, event.y)
            roi = self.rois[self.selected_idx]
            if np.hypot(ix - roi["cx"], iy - roi["cy"]) <= roi["r"] + ROI_HIT_TOL:
                roi["r"] = max(5, roi["r"] + direction * 8)
                self._render_roi_canvas()
                return
        factor = 1.2 if direction > 0 else 1 / 1.2
        self._apply_zoom(factor, event.x, event.y)

    def on_press(self, event):
        self._t_drag_start = (event.x, event.y)
        self._t_last = (event.x, event.y)
        self._t_dragging = False
        self._t_drag_mode = None
        if self.original is not None and self.selected_idx is not None:
            ix, iy = self._screen_to_image(event.x, event.y)
            roi = self.rois[self.selected_idx]
            if np.hypot(ix - roi["cx"], iy - roi["cy"]) <= roi["r"]:
                self._t_drag_mode = "move_roi"

    def on_drag(self, event):
        if self.original is None:
            return
        if not self._t_dragging:
            if abs(event.x - self._t_drag_start[0]) + abs(event.y - self._t_drag_start[1]) > 4:
                self._t_dragging = True
        if not self._t_dragging:
            return
        if self._t_drag_mode == "move_roi" and self.selected_idx is not None:
            ix, iy = self._screen_to_image(event.x, event.y)
            self.rois[self.selected_idx]["cx"] = ix
            self.rois[self.selected_idx]["cy"] = iy
        else:
            self.t_view_x += event.x - self._t_last[0]
            self.t_view_y += event.y - self._t_last[1]
        self._t_last = (event.x, event.y)
        self._render_roi_canvas()

    def on_release(self, event):
        if not self._t_dragging:
            self._handle_roi_click(event)
        self._t_dragging = False
        self._t_drag_mode = None

    def _handle_roi_click(self, event):
        if self.original is None:
            return
        ix, iy = self._screen_to_image(event.x, event.y)
        h, w = self.original.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return
        if self.calib_mode:
            self.calib_points.append((ix, iy))
            if len(self.calib_points) == 2:
                self._finish_calibration()
            else:
                self._render_roi_canvas()
            return
        hit = self._find_roi_at(ix, iy)
        if hit is not None:
            self.selected_idx = hit
        else:
            try:
                default_r = int(self.radius_var.get())
            except (ValueError, AttributeError):
                default_r = self.settings["default_radius"]
            self.rois.append({"cx": ix, "cy": iy, "r": default_r})
            self.selected_idx = len(self.rois) - 1
        self._render_roi_canvas()
        self._render_main_feed()

    def _build_roi_canvas_disp(self):
        disp = self.original.copy()
        for i, roi in enumerate(self.rois):
            color = (0, 255, 255) if i == self.selected_idx else (0, 255, 0)
            cv2.circle(disp, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), color, 2)
            cv2.circle(disp, (int(roi["cx"]), int(roi["cy"])), 5, color, -1)
            cv2.putText(disp, str(i + 1), (int(roi["cx"]) + 10, int(roi["cy"]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        for d in self.last_circles:
            cx, cy, r = int(d["cx"]), int(d["cy"]), int(round(d["r"]))
            cv2.circle(disp, (cx, cy), r + 4, (0, 0, 255), 2)
        for (px, py) in self.calib_points:
            cv2.circle(disp, (int(px), int(py)), 6, (255, 0, 255), -1)
        if len(self.calib_points) == 2:
            (x1, y1), (x2, y2) = self.calib_points
            cv2.line(disp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        return disp

    def _render_roi_canvas(self):
        if self.current_view != "teaching":
            return
        self.roi_canvas.delete("all")
        if self.original is None:
            return
        bgr = self._build_roi_canvas_disp()
        cw, ch = self._canvas_wh()
        H, W = bgr.shape[:2]
        s = self.t_base_scale * self.t_zoom
        vx, vy = self.t_view_x, self.t_view_y
        l = max(0, int(-vx / s))
        t = max(0, int(-vy / s))
        r = min(W, int((cw - vx) / s) + 1)
        b = min(H, int((ch - vy) / s) + 1)
        if r <= l or b <= t:
            return
        crop = bgr[t:b, l:r]
        cwid = max(1, int((r - l) * s))
        chei = max(1, int((b - t) * s))
        interp = cv2.INTER_CUBIC if self.t_zoom > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop, (cwid, chei), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.roi_canvas.create_image(vx + l * s, vy + t * s, anchor="nw", image=photo)
        self.roi_photo = photo


def main():
    root = ctk.CTk()
    DustInspectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
