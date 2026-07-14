"""
Dust Inspector — Basler camera dust-detection app
--------------------------------------------------
Single-file, modern/sleek desktop app.

Features:
  - Basler camera live feed (pypylon) OR Open Image (offline / no camera)
  - Multi-ROI selector: click to add a circular ROI, click inside a ROI to
    select it, scroll while hovering a SELECTED roi to grow/shrink it,
    scroll anywhere else to zoom the view (cursor anchored)
  - Two-point calibration (Settings page): click 2 points, enter real-world
    distance -> mm/px scale stored in settings.json
  - Detection algorithm is UNCHANGED from the reference: local Z-score
    (boxFilter mean/std over a window) thresholded inside the ROI mask(s);
    each dust blob is ringed in red in the final result
  - Binary mask output view, synced zoom/pan with the input view
  - Sidebar navigation, card-based layout, dark modern theme
  - storage/ directory: images/, masks/, roi_configs/, settings.json

Run:  python dust_inspector_app.py
"""

import os
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
WARNING = "#f59e0b"

# ---------------------------------------------------------------- storage --
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
IMAGES_DIR = os.path.join(STORAGE_DIR, "images")
MASKS_DIR = os.path.join(STORAGE_DIR, "masks")
ROI_DIR = os.path.join(STORAGE_DIR, "roi_configs")
SETTINGS_PATH = os.path.join(STORAGE_DIR, "settings.json")

for _d in (STORAGE_DIR, IMAGES_DIR, MASKS_DIR, ROI_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_SETTINGS = {
    "window": 31,
    "z_thr": 3.0,
    "default_radius": 100,
    "scale_mm_per_px": None,
    "exposure_us": 20000.0,
    "gain": 0.0,
    "min_area": 4.0,
    "min_circularity": 0.55,
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
def run_zscore_detection(bgr, rois, window, z_thr, min_area=4.0, min_circularity=0.55):
    """Core Z-score math is UNCHANGED: local Z-score via boxFilter mean/std,
    thresholded inside the union of all circular ROI masks.

    On top of that, two shape-based filters clean up the raw threshold:
      - a small morphological opening erases anything thinner than a few px
        (this kills thin curved edge artifacts / arcs and 1-2px sensor noise,
        since a real dust speck is round and a few px wide, an arc isn't)
      - a circularity + min-area check on the surviving blobs drops anything
        that's still elongated/thin (residual arc fragments) or too small to
        be a real particle (min_area, min_circularity are tunable in Settings)

    Returns the cleaned binary mask, one min-enclosing circle per accepted
    dust blob (used to ring it in red), and summary stats.
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
        circles.append({"cx": float(cx), "cy": float(cy), "r": float(max(r, 3.0))})
        cv2.drawContours(binary, [c], -1, 255, -1)

    stats = None
    if mask.any():
        roi_z = zscore[mask == 255]
        stats = {"max_z": float(roi_z.max()), "mean_z": float(roi_z.mean()),
                 "dust_px": int((binary == 255).sum()), "dust_count": len(circles),
                 "rejected": rejected}
    return binary, circles, stats


# -------------------------------------------------------------------- app --
CANVAS_W = 470
CANVAS_H = 470
ROI_HIT_TOL = 6  # extra px tolerance (image space) for selecting a roi


class DustInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dust Inspector")
        self.root.geometry("1300x820")
        self.root.minsize(1050, 680)
        self.root.configure(fg_color=BG)

        self.f_title = ctk.CTkFont(size=20, weight="bold")
        self.f_section = ctk.CTkFont(size=14, weight="bold")
        self.f_body = ctk.CTkFont(size=13)
        self.f_small = ctk.CTkFont(size=11)

        self.settings = load_settings()
        self.cam = CameraManager()

        self.original = None          # current BGR frame (live or opened)
        self.result_mask = None       # last binary mask (BGR for display)
        self.dust_circles = []        # detected dust blobs, ringed in red
        self.rois = []                # list of {"cx","cy","r"}
        self.selected_idx = None

        self.calib_mode = False
        self.calib_points = []

        self.live_on = False
        self.freeze = tk.BooleanVar(value=False)
        self.live_detect = tk.BooleanVar(value=False)

        # shared view transform (synced across both canvases)
        self.zoom = 1.0
        self.base_scale = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._dragging = False
        self._drag_mode = None
        self._drag_start = (0, 0)
        self._last = (0, 0)
        self._hover_img = (0, 0)

        self.input_photo = None
        self.mask_photo = None

        self.status = tk.StringVar(value="No image. Connect camera or Open Image.")
        self.scale_label_var = tk.StringVar(value=self._scale_text())
        self.nav_buttons = {}
        self.current_page = "inspect"

        self._build_ui()
        self._update_cam_indicator()
        self._poll_live()

    # -------------------------------------------------------- style helpers
    def _btn_primary(self, parent, text, command, width=120):
        return ctk.CTkButton(parent, text=text, command=command, width=width,
                              corner_radius=8, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                              text_color="#ffffff", font=self.f_body)

    def _btn_secondary(self, parent, text, command, width=110):
        return ctk.CTkButton(parent, text=text, command=command, width=width,
                              corner_radius=8, fg_color=BG_CARD_ALT, hover_color=BORDER,
                              text_color=TEXT, font=self.f_body)

    def _section_header(self, parent, text):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(16, 4))
        ctk.CTkFrame(row, fg_color=ACCENT, width=4, height=18, corner_radius=2).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(row, text=text, font=self.f_section, text_color=TEXT).pack(side="left")
        return row

    def _field(self, parent, label_text, var, width=140):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(f, text=label_text, font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w")
        ctk.CTkEntry(f, textvariable=var, width=width, corner_radius=8,
                     fg_color=BG_CARD_ALT, border_color=BORDER, text_color=TEXT).pack(anchor="w", pady=(4, 0))
        return f

    def _card(self, parent, **kw):
        defaults = dict(fg_color=BG_CARD, corner_radius=14, border_width=1, border_color=BORDER)
        defaults.update(kw)
        return ctk.CTkFrame(parent, **defaults)

    # ------------------------------------------------------------ layout --
    def _build_ui(self):
        outer = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
        outer.pack(fill="both", expand=True)

        container = ctk.CTkFrame(outer, fg_color=BG, corner_radius=0)
        container.pack(fill="both", expand=True, side="top")

        statusbar = ctk.CTkFrame(outer, fg_color=BG_SIDEBAR, corner_radius=0, height=32)
        statusbar.pack(fill="x", side="bottom")
        statusbar.pack_propagate(False)
        ctk.CTkLabel(statusbar, text="●", text_color=TEXT_MUTED, font=self.f_small).pack(side="left", padx=(16, 6))
        ctk.CTkLabel(statusbar, textvariable=self.status, text_color=TEXT_MUTED, font=self.f_small).pack(side="left")

        # ---- sidebar ----
        sidebar = ctk.CTkFrame(container, width=176, fg_color=BG_SIDEBAR, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        logo = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo.pack(fill="x", padx=20, pady=(24, 28))
        ctk.CTkLabel(logo, text="Dust", font=self.f_title, text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(logo, text="Inspector", font=self.f_title, text_color=ACCENT).pack(anchor="w")

        nav = ctk.CTkFrame(sidebar, fg_color="transparent")
        nav.pack(fill="x", padx=12)
        self._make_nav_button(nav, "inspect", "Inspection")
        self._make_nav_button(nav, "settings", "Settings")

        ctk.CTkFrame(sidebar, fg_color="transparent").pack(fill="both", expand=True)

        chip = self._card(sidebar, corner_radius=10)
        chip.pack(fill="x", padx=12, pady=16)
        self.cam_dot = ctk.CTkLabel(chip, text="●", text_color=DANGER, font=self.f_body)
        self.cam_dot.pack(side="left", padx=(12, 6), pady=10)
        self.cam_chip_label = ctk.CTkLabel(chip, text="Camera offline", text_color=TEXT_MUTED, font=self.f_small)
        self.cam_chip_label.pack(side="left", padx=(0, 10))

        # ---- content (pages stacked, switched via nav) ----
        content = ctk.CTkFrame(container, fg_color=BG, corner_radius=0)
        content.pack(side="left", fill="both", expand=True)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self.page_inspect = ctk.CTkFrame(content, fg_color=BG, corner_radius=0)
        self.page_settings = ctk.CTkFrame(content, fg_color=BG, corner_radius=0)
        self.page_inspect.grid(row=0, column=0, sticky="nsew")
        self.page_settings.grid(row=0, column=0, sticky="nsew")

        self._build_inspection_page()
        self._build_settings_page()
        self.switch_page("inspect")

    def _make_nav_button(self, parent, key, label):
        btn = ctk.CTkButton(parent, text=label, corner_radius=8, height=38,
                             fg_color="transparent", hover_color=BG_CARD_ALT,
                             text_color=TEXT_MUTED, font=self.f_body,
                             command=lambda: self.switch_page(key))
        btn.pack(fill="x", pady=4)
        self.nav_buttons[key] = btn

    def switch_page(self, key):
        self.current_page = key
        (self.page_inspect if key == "inspect" else self.page_settings).tkraise()
        for k, b in self.nav_buttons.items():
            if k == key:
                b.configure(text_color=ACCENT, fg_color=ACCENT_SOFT)
            else:
                b.configure(text_color=TEXT_MUTED, fg_color="transparent")

    # ---- Inspection page --------------------------------------------------
    def _build_inspection_page(self):
        pad = 18
        p = self.page_inspect

        ctk.CTkLabel(p, text="Inspection", font=self.f_title, text_color=TEXT).pack(
            anchor="w", padx=pad, pady=(pad, 8))

        toolbar = self._card(p)
        toolbar.pack(fill="x", padx=pad, pady=(0, 8))

        row1 = ctk.CTkFrame(toolbar, fg_color="transparent")
        row1.pack(fill="x", padx=14, pady=(14, 6))
        self._btn_primary(row1, "Open Image", self.open_image, width=110).pack(side="left", padx=(0, 8))
        self._btn_secondary(row1, "Connect Cam", self.connect_camera, width=110).pack(side="left", padx=8)
        self.live_btn = self._btn_secondary(row1, "Start Live", self.toggle_live, width=100)
        self.live_btn.pack(side="left", padx=8)
        ctk.CTkSwitch(row1, text="Freeze", variable=self.freeze, font=self.f_small,
                      progress_color=ACCENT, text_color=TEXT_MUTED).pack(side="left", padx=(16, 8))
        ctk.CTkSwitch(row1, text="Live Detect", variable=self.live_detect, font=self.f_small,
                      progress_color=ACCENT, text_color=TEXT_MUTED).pack(side="left", padx=8)
        self._btn_primary(row1, "Run Detection", self.run_detection, width=120).pack(side="right", padx=(8, 0))
        self._btn_secondary(row1, "Save Result", self.save_result, width=100).pack(side="right", padx=8)

        row2 = ctk.CTkFrame(toolbar, fg_color="transparent")
        row2.pack(fill="x", padx=14, pady=(0, 14))
        ctk.CTkLabel(row2, text="ROI", font=self.f_small, text_color=TEXT_MUTED).pack(side="left", padx=(0, 8))
        self._btn_secondary(row2, "Delete", self.delete_selected_roi, width=80).pack(side="left", padx=4)
        self._btn_secondary(row2, "Clear All", self.clear_rois, width=90).pack(side="left", padx=4)
        self._btn_secondary(row2, "Save Layout", self.save_roi_layout, width=100).pack(side="left", padx=4)
        self._btn_secondary(row2, "Load Layout", self.load_roi_layout, width=100).pack(side="left", padx=4)

        self._btn_secondary(row2, "Fit", self.fit, width=52).pack(side="right", padx=4)
        self._btn_secondary(row2, "Zoom Out", lambda: self._zoom_btn(0.8), width=80).pack(side="right", padx=4)
        self._btn_secondary(row2, "Zoom In", lambda: self._zoom_btn(1.25), width=80).pack(side="right", padx=(16, 4))

        canvases = ctk.CTkFrame(p, fg_color="transparent")
        canvases.pack(fill="both", expand=True, padx=pad, pady=(4, pad))
        canvases.grid_columnconfigure(0, weight=1)
        canvases.grid_columnconfigure(1, weight=1)
        canvases.grid_rowconfigure(0, weight=1)

        left_card = self._card(canvases)
        left_card.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        self._section_header(left_card, "Input")
        ctk.CTkLabel(left_card, text="click = add/select ROI  •  drag = pan  •  wheel = zoom",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(0, 10))
        wrap1 = ctk.CTkFrame(left_card, fg_color=BG_CANVAS, corner_radius=10)
        wrap1.pack(padx=18, pady=(0, 18))
        self.ic = tk.Canvas(wrap1, width=CANVAS_W, height=CANVAS_H, bg=BG_CANVAS, highlightthickness=0)
        self.ic.pack(padx=3, pady=3)

        right_card = self._card(canvases)
        right_card.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        self._section_header(right_card, "Binary Mask")
        ctk.CTkLabel(right_card, text="white = dust candidate  •  red ring = detected blob",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(0, 10))
        wrap2 = ctk.CTkFrame(right_card, fg_color=BG_CANVAS, corner_radius=10)
        wrap2.pack(padx=18, pady=(0, 18))
        self.mc = tk.Canvas(wrap2, width=CANVAS_W, height=CANVAS_H, bg=BG_CANVAS, highlightthickness=0)
        self.mc.pack(padx=3, pady=3)

        for cv_ in (self.ic, self.mc):
            cv_.bind("<MouseWheel>", self.on_wheel)
            cv_.bind("<Button-4>", self.on_wheel)
            cv_.bind("<Button-5>", self.on_wheel)
            cv_.bind("<ButtonPress-1>", self.on_press)
            cv_.bind("<B1-Motion>", self.on_drag)
            cv_.bind("<Motion>", self.on_hover)
        self.ic.bind("<ButtonRelease-1>", self.on_release_input)
        self.mc.bind("<ButtonRelease-1>", self.on_release_other)
        self.root.bind("<Delete>", lambda e: self.delete_selected_roi())

    # ---- Settings page -----------------------------------------------------
    def _build_settings_page(self):
        pad = 18
        p = self.page_settings
        ctk.CTkLabel(p, text="Settings", font=self.f_title, text_color=TEXT).pack(
            anchor="w", padx=pad, pady=(pad, 8))

        cam_card = self._card(p)
        cam_card.pack(fill="x", padx=pad, pady=8)
        head = ctk.CTkFrame(cam_card, fg_color="transparent")
        head.pack(fill="x", padx=18, pady=(16, 0))
        ctk.CTkFrame(head, fg_color=ACCENT, width=4, height=18, corner_radius=2).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(head, text="Camera", font=self.f_section, text_color=TEXT).pack(side="left")
        self.cam_status_var = tk.StringVar(value="Not connected")
        ctk.CTkLabel(head, textvariable=self.cam_status_var, font=self.f_small, text_color=TEXT_MUTED).pack(side="right")

        cam_body = ctk.CTkFrame(cam_card, fg_color="transparent")
        cam_body.pack(fill="x", padx=18, pady=(10, 18))
        self.exposure_var = tk.StringVar(value=str(self.settings["exposure_us"]))
        self.gain_var = tk.StringVar(value=str(self.settings["gain"]))
        self._field(cam_body, "Exposure (µs)", self.exposure_var).pack(side="left", padx=(0, 20))
        self._field(cam_body, "Gain", self.gain_var).pack(side="left", padx=(0, 20))
        self._btn_primary(cam_body, "Apply", self.apply_camera_settings, width=90).pack(side="left", pady=(18, 0))

        det_card = self._card(p)
        det_card.pack(fill="x", padx=pad, pady=8)
        self._section_header(det_card, "Detection")
        det_body = ctk.CTkFrame(det_card, fg_color="transparent")
        det_body.pack(fill="x", padx=18, pady=(6, 18))
        self.window_var = tk.StringVar(value=str(self.settings["window"]))
        self.zthr_var = tk.StringVar(value=str(self.settings["z_thr"]))
        self.radius_var = tk.StringVar(value=str(self.settings["default_radius"]))
        self.min_area_var = tk.StringVar(value=str(self.settings["min_area"]))
        self.min_circ_var = tk.StringVar(value=str(self.settings["min_circularity"]))
        self._field(det_body, "Window size", self.window_var, width=80).pack(side="left", padx=(0, 16))
        self._field(det_body, "Z threshold", self.zthr_var, width=80).pack(side="left", padx=(0, 16))
        self._field(det_body, "Default ROI radius (px)", self.radius_var, width=100).pack(side="left", padx=(0, 16))
        self._field(det_body, "Min blob area (px²)", self.min_area_var, width=90).pack(side="left", padx=(0, 16))
        self._field(det_body, "Min circularity (0-1)", self.min_circ_var, width=100).pack(side="left", padx=(0, 16))
        self._btn_primary(det_body, "Save Settings", self.save_detection_settings, width=120).pack(side="left", pady=(18, 0))
        ctk.CTkLabel(det_card, text="Circularity ~1 = round dust speck. Arcs / thin edge glints score low — raise this to reject them.",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(0, 14))

        calib_card = self._card(p)
        calib_card.pack(fill="x", padx=pad, pady=8)
        self._section_header(calib_card, "Two-Point Calibration")
        ctk.CTkLabel(calib_card, text="Click 2 points on the image with a known real-world distance.",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(0, 10))
        calib_body = ctk.CTkFrame(calib_card, fg_color="transparent")
        calib_body.pack(fill="x", padx=18, pady=(0, 18))
        self._btn_primary(calib_body, "Start Calibration", self.start_calibration, width=140).pack(side="left", padx=(0, 8))
        self._btn_secondary(calib_body, "Reset", self.reset_calibration, width=90).pack(side="left", padx=8)
        ctk.CTkLabel(calib_body, textvariable=self.scale_label_var, font=self.f_small,
                     text_color=TEXT_MUTED).pack(side="left", padx=16)

    # ------------------------------------------------------------ camera --
    def _update_cam_indicator(self):
        if self.cam.connected and self.live_on:
            self.cam_dot.configure(text_color=SUCCESS)
            self.cam_chip_label.configure(text="Live", text_color=TEXT)
        elif self.cam.connected:
            self.cam_dot.configure(text_color=WARNING)
            self.cam_chip_label.configure(text="Connected", text_color=TEXT)
        else:
            self.cam_dot.configure(text_color=DANGER)
            self.cam_chip_label.configure(text="Camera offline", text_color=TEXT_MUTED)

    def connect_camera(self):
        ok, msg = self.cam.connect()
        self.cam_status_var.set(msg)
        self._update_cam_indicator()
        if ok:
            self.apply_camera_settings()
            self.status.set("Camera connected.")
        else:
            messagebox.showwarning("Camera", msg)

    def apply_camera_settings(self):
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
        self.status.set("Camera settings applied.")

    def toggle_live(self):
        if not self.cam.connected:
            messagebox.showwarning("Camera", "Connect the camera first.")
            return
        if not self.live_on:
            self.cam.start_live()
            self.live_on = True
            self.live_btn.configure(text="Stop Live", fg_color=SUCCESS,
                                     hover_color=SUCCESS_HOVER, text_color="#ffffff")
            self.status.set("Live feed running.")
        else:
            self.cam.stop_live()
            self.live_on = False
            self.live_btn.configure(text="Start Live", fg_color=BG_CARD_ALT,
                                     hover_color=BORDER, text_color=TEXT)
            self.status.set("Live feed stopped.")
        self._update_cam_indicator()

    def _poll_live(self):
        if self.live_on and not self.freeze.get():
            frame = self.cam.get_frame()
            if frame is not None:
                first = self.original is None
                self.original = frame
                if first:
                    self.fit()
                if self.live_detect.get() and self.rois:
                    self._run_detection_silent()
                self._refresh()
        self.root.after(120, self._poll_live)

    # -------------------------------------------------------------- I/O ---
    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All", "*.*")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.status.set("Could not load image.")
            return
        if self.live_on:
            self.toggle_live()
        self.original = img
        self.result_mask = None
        self.dust_circles = []
        self.mc.delete("all")
        self.fit()
        self.status.set(f"Loaded {os.path.basename(path)}. Add ROI(s) then Run Detection.")

    def save_result(self):
        if self.original is None:
            self.status.set("Nothing to save yet.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(IMAGES_DIR, f"capture_{ts}.png")
        cv2.imwrite(img_path, self.original)
        if self.result_mask is not None:
            mask_path = os.path.join(MASKS_DIR, f"mask_{ts}.png")
            annotated = self.result_mask.copy()
            for d in self.dust_circles:
                cv2.circle(annotated, (int(d["cx"]), int(d["cy"])), int(round(d["r"])) + 4, (0, 0, 255), 2)
            cv2.imwrite(mask_path, annotated)
            self.status.set(f"Saved {os.path.basename(img_path)} + {os.path.basename(mask_path)}")
        else:
            self.status.set(f"Saved {os.path.basename(img_path)} (no mask yet)")

    def save_roi_layout(self):
        if not self.rois:
            self.status.set("No ROIs to save.")
            return
        name = simpledialog.askstring("Save ROI Layout", "Layout name (e.g. galaxy_s26_ultra):")
        if not name:
            return
        path = os.path.join(ROI_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(self.rois, f, indent=2)
        self.status.set(f"ROI layout saved: {name}.json")

    def load_roi_layout(self):
        path = filedialog.askopenfilename(initialdir=ROI_DIR, filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "r") as f:
                self.rois = json.load(f)
            self.selected_idx = None
            self._refresh()
            self.status.set(f"ROI layout loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load ROI Layout", str(e))

    # --------------------------------------------------------- detection --
    def _current_params(self):
        try:
            win = int(self.window_var.get())
        except ValueError:
            win = self.settings["window"]
        try:
            z_thr = float(self.zthr_var.get())
        except ValueError:
            z_thr = self.settings["z_thr"]
        try:
            min_area = float(self.min_area_var.get())
        except ValueError:
            min_area = self.settings["min_area"]
        try:
            min_circ = float(self.min_circ_var.get())
        except ValueError:
            min_circ = self.settings["min_circularity"]
        return win, z_thr, min_area, min_circ

    def run_detection(self):
        if self.original is None or not self.rois:
            self.status.set("Load an image and add at least one ROI first.")
            return
        win, z_thr, min_area, min_circ = self._current_params()
        binary, circles, stats = run_zscore_detection(
            self.original, self.rois, win, z_thr, min_area, min_circ)
        self.result_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.dust_circles = circles
        self._refresh()
        if stats:
            self.status.set(
                f"Done. z_thr={z_thr}, max_z={stats['max_z']:.2f}, "
                f"dust_count={stats['dust_count']}, dust_px={stats['dust_px']}, "
                f"arcs/noise filtered={stats['rejected']}")
        else:
            self.status.set("Done, but ROI mask was empty.")

    def _run_detection_silent(self):
        win, z_thr, min_area, min_circ = self._current_params()
        binary, circles, _ = run_zscore_detection(
            self.original, self.rois, win, z_thr, min_area, min_circ)
        self.result_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.dust_circles = circles

    def save_detection_settings(self):
        try:
            self.settings["window"] = int(self.window_var.get())
            self.settings["z_thr"] = float(self.zthr_var.get())
            self.settings["default_radius"] = int(self.radius_var.get())
            self.settings["min_area"] = float(self.min_area_var.get())
            self.settings["min_circularity"] = float(self.min_circ_var.get())
        except ValueError:
            messagebox.showerror("Settings", "Detection fields must be numbers.")
            return
        save_settings(self.settings)
        self.status.set("Detection settings saved.")

    # --------------------------------------------------------- calibration
    def start_calibration(self):
        if self.original is None:
            messagebox.showwarning("Calibration", "Load or capture an image first.")
            return
        self.calib_mode = True
        self.calib_points = []
        self.switch_page("inspect")
        self.status.set("Calibration: click 2 points at a known real-world distance.")

    def reset_calibration(self):
        self.settings["scale_mm_per_px"] = None
        save_settings(self.settings)
        self.scale_label_var.set(self._scale_text())
        self.status.set("Calibration reset.")

    def _finish_calibration(self):
        (x1, y1), (x2, y2) = self.calib_points
        pixel_dist = float(np.hypot(x2 - x1, y2 - y1))
        self.calib_mode = False
        self.calib_points = []
        if pixel_dist < 1:
            self.status.set("Calibration points too close together, try again.")
            return
        dist_mm = simpledialog.askfloat("Calibration", "Real-world distance between the two points (mm):")
        if not dist_mm:
            self.status.set("Calibration cancelled.")
            self._refresh()
            return
        scale = dist_mm / pixel_dist
        self.settings["scale_mm_per_px"] = scale
        save_settings(self.settings)
        self.scale_label_var.set(self._scale_text())
        self.status.set(f"Calibrated: {scale:.5f} mm/px")
        self._refresh()

    def _scale_text(self):
        s = self.settings.get("scale_mm_per_px")
        return f"Scale: {s:.5f} mm/px" if s else "Scale: not calibrated"

    # -------------------------------------------------------- ROI helpers --
    def _find_roi_at(self, ix, iy):
        for i in reversed(range(len(self.rois))):
            r = self.rois[i]
            if np.hypot(ix - r["cx"], iy - r["cy"]) <= r["r"] + ROI_HIT_TOL:
                return i
        return None

    def delete_selected_roi(self):
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.rois):
            self.rois.pop(self.selected_idx)
            self.selected_idx = None
            self._refresh()
            self.status.set("ROI deleted.")

    def clear_rois(self):
        self.rois = []
        self.selected_idx = None
        self._refresh()
        self.status.set("All ROIs cleared.")

    # --------------------------------------------------------- zoom / pan --
    def fit(self):
        if self.original is None:
            return
        h, w = self.original.shape[:2]
        self.base_scale = min(CANVAS_W / w, CANVAS_H / h)
        self.zoom = 1.0
        s = self.base_scale
        self.view_x = (CANVAS_W - w * s) / 2
        self.view_y = (CANVAS_H - h * s) / 2
        self._refresh()

    def _apply_zoom(self, factor, cx, cy):
        if self.original is None:
            return
        s_old = self.base_scale * self.zoom
        ix = (cx - self.view_x) / s_old
        iy = (cy - self.view_y) / s_old
        self.zoom = max(0.1, min(self.zoom * factor, 60.0))
        s_new = self.base_scale * self.zoom
        self.view_x = cx - ix * s_new
        self.view_y = cy - iy * s_new
        self._refresh()

    def _zoom_btn(self, factor):
        self._apply_zoom(factor, CANVAS_W / 2, CANVAS_H / 2)

    def _screen_to_image(self, x, y):
        s = self.base_scale * self.zoom
        return (x - self.view_x) / s, (y - self.view_y) / s

    def on_hover(self, event):
        if self.original is None:
            return
        self._hover_img = self._screen_to_image(event.x, event.y)

    def on_wheel(self, event):
        if self.original is None:
            return
        direction = 1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4) else -1

        if self.selected_idx is not None:
            ix, iy = self._screen_to_image(event.x, event.y)
            roi = self.rois[self.selected_idx]
            if np.hypot(ix - roi["cx"], iy - roi["cy"]) <= roi["r"] + ROI_HIT_TOL:
                roi["r"] = max(5, roi["r"] + direction * 8)
                self._refresh()
                return

        factor = 1.2 if direction > 0 else 1 / 1.2
        self._apply_zoom(factor, event.x, event.y)

    def on_press(self, event):
        self._drag_start = (event.x, event.y)
        self._last = (event.x, event.y)
        self._dragging = False
        self._drag_mode = None
        if self.original is not None:
            ix, iy = self._screen_to_image(event.x, event.y)
            if self.selected_idx is not None:
                roi = self.rois[self.selected_idx]
                if np.hypot(ix - roi["cx"], iy - roi["cy"]) <= roi["r"]:
                    self._drag_mode = "move_roi"

    def on_drag(self, event):
        if self.original is None:
            return
        if not self._dragging:
            if abs(event.x - self._drag_start[0]) + abs(event.y - self._drag_start[1]) > 4:
                self._dragging = True
        if not self._dragging:
            return
        if self._drag_mode == "move_roi" and self.selected_idx is not None:
            ix, iy = self._screen_to_image(event.x, event.y)
            self.rois[self.selected_idx]["cx"] = ix
            self.rois[self.selected_idx]["cy"] = iy
        else:
            self.view_x += event.x - self._last[0]
            self.view_y += event.y - self._last[1]
        self._last = (event.x, event.y)
        self._refresh()

    def on_release_input(self, event):
        if not self._dragging:
            self._handle_click(event)
        self._dragging = False
        self._drag_mode = None

    def on_release_other(self, event):
        self._dragging = False
        self._drag_mode = None

    def _handle_click(self, event):
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
                self._refresh()
                self.status.set("Calibration: click the second point.")
            return

        hit = self._find_roi_at(ix, iy)
        if hit is not None:
            self.selected_idx = hit
            self.status.set(f"ROI {hit + 1} selected. Hover it and scroll to resize.")
        else:
            try:
                default_r = int(self.radius_var.get())
            except ValueError:
                default_r = self.settings["default_radius"]
            self.rois.append({"cx": ix, "cy": iy, "r": default_r})
            self.selected_idx = len(self.rois) - 1
            self.status.set(f"ROI {len(self.rois)} added.")
        self._refresh()

    # ------------------------------------------------------------ render --
    def _build_input_disp(self):
        disp = self.original.copy()
        for i, roi in enumerate(self.rois):
            color = (0, 255, 255) if i == self.selected_idx else (0, 255, 0)
            cv2.circle(disp, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), color, 2)
            cv2.circle(disp, (int(roi["cx"]), int(roi["cy"])), 5, color, -1)
            cv2.putText(disp, str(i + 1), (int(roi["cx"]) + 10, int(roi["cy"]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        for (px, py) in self.calib_points:
            cv2.circle(disp, (int(px), int(py)), 6, (255, 0, 255), -1)
        if len(self.calib_points) == 2:
            (x1, y1), (x2, y2) = self.calib_points
            cv2.line(disp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        return disp

    def _build_mask_disp(self):
        if self.result_mask is None:
            return None
        disp = self.result_mask.copy()
        for i, roi in enumerate(self.rois):
            color = (0, 255, 255) if i == self.selected_idx else (0, 150, 0)
            cv2.circle(disp, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), color, 1)
        for d in self.dust_circles:
            cv2.circle(disp, (int(d["cx"]), int(d["cy"])), int(round(d["r"])) + 4, (0, 0, 255), 2)
        return disp

    def _render(self, canvas, bgr, attr):
        canvas.delete("all")
        if bgr is None:
            return
        H, W = bgr.shape[:2]
        s = self.base_scale * self.zoom
        vx, vy = self.view_x, self.view_y
        l = max(0, int(-vx / s))
        t = max(0, int(-vy / s))
        r = min(W, int((CANVAS_W - vx) / s) + 1)
        b = min(H, int((CANVAS_H - vy) / s) + 1)
        if r <= l or b <= t:
            return
        crop = bgr[t:b, l:r]
        cw = max(1, int((r - l) * s))
        ch = max(1, int((b - t) * s))
        interp = cv2.INTER_NEAREST if self.zoom > 1.5 else cv2.INTER_AREA
        resized = cv2.resize(crop, (cw, ch), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.create_image(vx + l * s, vy + t * s, anchor="nw", image=photo)
        setattr(self, attr, photo)

    def _refresh(self):
        if self.original is None:
            return
        self._render(self.ic, self._build_input_disp(), "input_photo")
        self._render(self.mc, self._build_mask_disp(), "mask_photo")


def main():
    root = ctk.CTk()
    DustInspectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
