"""
Dust Inspector — Basler camera dust-detection app
--------------------------------------------------
Single-file, minimalistic desktop app.

Features:
  - Basler camera live feed (pypylon) OR Open Image (offline / no camera)
  - Multi-ROI selector: click to add a circular ROI, click inside a ROI to
    select it, scroll while hovering a SELECTED roi to grow/shrink it,
    scroll anywhere else to zoom the view (cursor anchored)
  - Two-point calibration (Settings tab): click 2 points, enter real-world
    distance -> mm/px scale stored in settings.json
  - Detection algorithm is UNCHANGED from the reference: local Z-score
    (boxFilter mean/std over a window) thresholded inside the ROI mask(s)
  - Binary mask output view, synced zoom/pan with the input view
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
ctk.set_default_color_theme("green")

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
def run_zscore_detection(bgr, rois, window, z_thr):
    """UNCHANGED algorithm: local Z-score via boxFilter mean/std, thresholded
    inside the union of all circular ROI masks."""
    win = window if window % 2 == 1 else window + 1
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    local_mean = cv2.boxFilter(gray, -1, (win, win))
    local_mean_sq = cv2.boxFilter(gray * gray, -1, (win, win))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0))
    zscore = np.where(local_std > 1e-5, (gray - local_mean) / local_std, 0.0)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    for roi in rois:
        cv2.circle(mask, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), 255, -1)

    binary = np.where((zscore >= z_thr) & (mask == 255), 255, 0).astype(np.uint8)

    stats = None
    if mask.any():
        roi_z = zscore[mask == 255]
        stats = {"max_z": float(roi_z.max()), "mean_z": float(roi_z.mean()),
                 "dust_px": int((binary == 255).sum())}
    return binary, stats


# -------------------------------------------------------------------- app --
CANVAS_W = 520
CANVAS_H = 520
ROI_HIT_TOL = 6  # extra px tolerance (image space) for selecting a roi


class DustInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dust Inspector")
        self.root.geometry("1180x760")

        self.settings = load_settings()
        self.cam = CameraManager()

        self.original = None          # current BGR frame (live or opened)
        self.result_mask = None       # last binary mask (BGR for display)
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

        self._build_ui()
        self._poll_live()

    # ------------------------------------------------------------ layout --
    def _build_ui(self):
        self.tabs = ctk.CTkTabview(self.root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab_inspect = self.tabs.add("Inspection")
        self.tab_settings = self.tabs.add("Settings")

        self._build_inspection_tab()
        self._build_settings_tab()

    # ---- Inspection tab -------------------------------------------------
    def _build_inspection_tab(self):
        top = ctk.CTkFrame(self.tab_inspect)
        top.pack(fill="x", padx=4, pady=(4, 2))

        ctk.CTkButton(top, text="Open Image", width=100, command=self.open_image).pack(side="left", padx=3)
        ctk.CTkButton(top, text="Connect Cam", width=100, command=self.connect_camera).pack(side="left", padx=3)
        self.live_btn = ctk.CTkButton(top, text="Start Live", width=90, command=self.toggle_live)
        self.live_btn.pack(side="left", padx=3)
        ctk.CTkCheckBox(top, text="Freeze", variable=self.freeze).pack(side="left", padx=6)
        ctk.CTkCheckBox(top, text="Live Detect", variable=self.live_detect).pack(side="left", padx=6)

        ctk.CTkButton(top, text="Run Detection", width=110, command=self.run_detection).pack(side="left", padx=(12, 3))
        ctk.CTkButton(top, text="Save Result", width=100, command=self.save_result).pack(side="left", padx=3)

        top2 = ctk.CTkFrame(self.tab_inspect)
        top2.pack(fill="x", padx=4, pady=(0, 4))
        ctk.CTkLabel(top2, text="ROI:").pack(side="left", padx=(4, 2))
        ctk.CTkButton(top2, text="Delete Selected", width=110, command=self.delete_selected_roi).pack(side="left", padx=3)
        ctk.CTkButton(top2, text="Clear All", width=80, command=self.clear_rois).pack(side="left", padx=3)
        ctk.CTkButton(top2, text="Save Layout", width=95, command=self.save_roi_layout).pack(side="left", padx=3)
        ctk.CTkButton(top2, text="Load Layout", width=95, command=self.load_roi_layout).pack(side="left", padx=3)

        ctk.CTkButton(top2, text="Zoom In", width=70, command=lambda: self._zoom_btn(1.25)).pack(side="left", padx=(16, 2))
        ctk.CTkButton(top2, text="Zoom Out", width=70, command=lambda: self._zoom_btn(0.8)).pack(side="left", padx=2)
        ctk.CTkButton(top2, text="Fit", width=50, command=self.fit).pack(side="left", padx=2)

        ctk.CTkLabel(top2, textvariable=self.status).pack(side="left", padx=12)

        canvases = ctk.CTkFrame(self.tab_inspect)
        canvases.pack(fill="both", expand=True, padx=4, pady=4)

        left = ctk.CTkFrame(canvases)
        left.pack(side="left", padx=6, pady=6)
        ctk.CTkLabel(left, text="INPUT  (click=add/select ROI, drag=pan, wheel=zoom)").pack()
        self.ic = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg="#1a1a1a", highlightthickness=0)
        self.ic.pack()

        right = ctk.CTkFrame(canvases)
        right.pack(side="left", padx=6, pady=6)
        ctk.CTkLabel(right, text="BINARY MASK  (dust candidates)").pack()
        self.mc = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H, bg="#1a1a1a", highlightthickness=0)
        self.mc.pack()

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

    # ---- Settings tab -----------------------------------------------------
    def _build_settings_tab(self):
        cam_frame = ctk.CTkFrame(self.tab_settings)
        cam_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(cam_frame, text="Camera", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.cam_status_var = tk.StringVar(value="Not connected")
        ctk.CTkLabel(cam_frame, textvariable=self.cam_status_var).grid(row=0, column=1, sticky="w", padx=6)

        ctk.CTkLabel(cam_frame, text="Exposure (us):").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.exposure_var = tk.StringVar(value=str(self.settings["exposure_us"]))
        ctk.CTkEntry(cam_frame, textvariable=self.exposure_var, width=100).grid(row=1, column=1, sticky="w", padx=6)

        ctk.CTkLabel(cam_frame, text="Gain:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.gain_var = tk.StringVar(value=str(self.settings["gain"]))
        ctk.CTkEntry(cam_frame, textvariable=self.gain_var, width=100).grid(row=2, column=1, sticky="w", padx=6)

        ctk.CTkButton(cam_frame, text="Apply to Camera", command=self.apply_camera_settings).grid(row=1, column=2, rowspan=2, padx=10)

        det_frame = ctk.CTkFrame(self.tab_settings)
        det_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(det_frame, text="Detection", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=6, pady=4)

        ctk.CTkLabel(det_frame, text="Window size:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.window_var = tk.StringVar(value=str(self.settings["window"]))
        ctk.CTkEntry(det_frame, textvariable=self.window_var, width=80).grid(row=1, column=1, sticky="w", padx=6)

        ctk.CTkLabel(det_frame, text="Z threshold:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.zthr_var = tk.StringVar(value=str(self.settings["z_thr"]))
        ctk.CTkEntry(det_frame, textvariable=self.zthr_var, width=80).grid(row=2, column=1, sticky="w", padx=6)

        ctk.CTkLabel(det_frame, text="Default ROI radius (px):").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.radius_var = tk.StringVar(value=str(self.settings["default_radius"]))
        ctk.CTkEntry(det_frame, textvariable=self.radius_var, width=80).grid(row=3, column=1, sticky="w", padx=6)

        ctk.CTkButton(det_frame, text="Save Settings", command=self.save_detection_settings).grid(row=1, column=2, rowspan=3, padx=10)

        calib_frame = ctk.CTkFrame(self.tab_settings)
        calib_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(calib_frame, text="Two-Point Calibration", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=6, pady=4, columnspan=2)
        ctk.CTkLabel(calib_frame, text="Click 2 points on the image with a known real-world distance.").grid(row=1, column=0, sticky="w", padx=6, columnspan=3)
        ctk.CTkButton(calib_frame, text="Start Calibration", command=self.start_calibration).grid(row=2, column=0, padx=6, pady=6)
        ctk.CTkButton(calib_frame, text="Reset Calibration", command=self.reset_calibration).grid(row=2, column=1, padx=6, pady=6)
        ctk.CTkLabel(calib_frame, textvariable=self.scale_label_var).grid(row=2, column=2, padx=10)

    # ------------------------------------------------------------ camera --
    def connect_camera(self):
        ok, msg = self.cam.connect()
        self.cam_status_var.set(msg)
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
            self.live_btn.configure(text="Stop Live")
            self.status.set("Live feed running.")
        else:
            self.cam.stop_live()
            self.live_on = False
            self.live_btn.configure(text="Start Live")
            self.status.set("Live feed stopped.")

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
            cv2.imwrite(mask_path, self.result_mask)
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
        return win, z_thr

    def run_detection(self):
        if self.original is None or not self.rois:
            self.status.set("Load an image and add at least one ROI first.")
            return
        win, z_thr = self._current_params()
        binary, stats = run_zscore_detection(self.original, self.rois, win, z_thr)
        self.result_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self._refresh()
        if stats:
            self.status.set(
                f"Done. z_thr={z_thr}, max_z={stats['max_z']:.2f}, dust_px={stats['dust_px']}")
        else:
            self.status.set("Done, but ROI mask was empty.")

    def _run_detection_silent(self):
        win, z_thr = self._current_params()
        binary, _ = run_zscore_detection(self.original, self.rois, win, z_thr)
        self.result_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def save_detection_settings(self):
        try:
            self.settings["window"] = int(self.window_var.get())
            self.settings["z_thr"] = float(self.zthr_var.get())
            self.settings["default_radius"] = int(self.radius_var.get())
        except ValueError:
            messagebox.showerror("Settings", "Window/Z-threshold/Radius must be numbers.")
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
        self.tabs.set("Inspection")
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

        # if a ROI is selected and the cursor is hovering it -> resize instead of zoom
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

