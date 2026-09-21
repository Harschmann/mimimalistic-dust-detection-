"""
Dust Inspector -- Operator-facing camera dust-detection app
------------------------------------------------------------
Single-file desktop app, restructured into TWO windows:

  1. Main Operator Window -- what the line operator actually uses:
     title, Model/Line readout, a big live feed (view-only zoom/pan, no
     ROI editing), a scrolling result log, and a right-hand panel with
     Status / Barcode / Teaching / Start Inspection / Clear Markings.
     ROI editing lives ONLY in Teaching, which is what fixes ROI
     "slipping" -- a stray click on the operator's feed used to be able
     to add, move, or resize an ROI without anyone noticing. (Separately:
     if the module itself can shift on the jig -- loose clamp, not
     seated flush, vibration -- that's mechanical, not something
     software can catch; worth a physical check on the fixture too.)

  2. Teaching Window (opened via the "Teaching" button, same app window,
     no separate popup) -- technician-only setup: Model & Line naming,
     ROI placement + two-point calibration (the one interactive canvas
     in the whole app), detection parameters, and camera settings.

Single-threaded detection call per inspection (dust-only detection via
run_zscore_detection), still run on a background
thread so the UI/feed never freezes. (Vinyl/lamination-film detection --
detect_vinyl_presence -- is implemented below but not currently wired
into the active inspection flow; re-enable later once the fiber/thread
detection is solid and the UI has been reworked.)
Saving result images and writing the CSV log also happens on that
background thread -- only the final status/log-box/button update is
handed back to the main thread, since that's the only part that has to
touch Tkinter widgets.

storage/ layout: source_images/, results/NG|OK/, roi_configs/,
logs/, settings.json

Run:  python dust_inspector_app.py
"""

import os
import sys
import csv
import json
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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

try:
    # Deliberately onnxruntime, NOT anomalib/torch -- Training Studio's own
    # docstring already says training happens on a different, more capable
    # machine than the operator's PC. The operator app only needs to RUN an
    # already-trained model, and ONNX Runtime is the lightweight way to do
    # that without pulling the ~2GB torch/anomalib stack onto the line PC.
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except Exception:
    ONNXRUNTIME_AVAILABLE = False

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
VINYL_COLOR = "#a855f7"  # distinct from PASS/FAIL/IN-PROGRESS so it reads as its own state

# BGR (for cv2 drawing, not the hex UI colors above) per defect type
BLOB_COLOR_BGR = {
    "dust": (0, 0, 255),        # red
    "ai_anomaly": (0, 165, 255),  # orange -- visually distinct from the z-score dust red
}
VINYL_COLOR_BGR = (245, 85, 168)  # pink-purple

# ---------------------------------------------------------------- storage --
if getattr(sys, "frozen", False):
    # Running as a PyInstaller .exe: __file__ would point at a temp
    # extraction folder that's wiped after the app closes. Use the actual
    # exe's folder instead so storage/ persists across runs.
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# settings.json always lives here, right next to the exe/script -- fixed,
# NOT inside the storage folder below, since that folder's location is
# user-configurable (Teaching > Camera > Select Location) and settings.json
# is exactly what remembers which location the user picked. If it lived
# inside the folder it points to, moving that folder would make the app
# forget where it moved to.
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")

STORAGE_DIR = SOURCE_DIR = RESULTS_DIR = RESULTS_NG_DIR = RESULTS_OK_DIR = None
RESULTS_VINYL_DIR = ROI_DIR = LOGS_DIR = LOG_CSV_PATH = None


def _apply_storage_base(base_path):
    """(Re)points every storage sub-path at base_path and ensures the
    folder structure exists there. Called once at startup with the default
    (or a previously-saved custom) location, and again any time the user
    picks a new folder via Select Location."""
    global STORAGE_DIR, SOURCE_DIR, RESULTS_DIR, RESULTS_NG_DIR, RESULTS_OK_DIR
    global RESULTS_VINYL_DIR, ROI_DIR, LOGS_DIR, LOG_CSV_PATH
    STORAGE_DIR = base_path
    SOURCE_DIR = os.path.join(STORAGE_DIR, "source_images")
    RESULTS_DIR = os.path.join(STORAGE_DIR, "results")
    RESULTS_NG_DIR = os.path.join(RESULTS_DIR, "NG")
    RESULTS_OK_DIR = os.path.join(RESULTS_DIR, "OK")
    RESULTS_VINYL_DIR = os.path.join(RESULTS_DIR, "VINYL")
    ROI_DIR = os.path.join(STORAGE_DIR, "roi_configs")
    LOGS_DIR = os.path.join(STORAGE_DIR, "logs")
    LOG_CSV_PATH = os.path.join(LOGS_DIR, "inspection_log.csv")
    for _d in (STORAGE_DIR, SOURCE_DIR, RESULTS_DIR, RESULTS_NG_DIR, RESULTS_OK_DIR,
               RESULTS_VINYL_DIR, ROI_DIR, LOGS_DIR):
        os.makedirs(_d, exist_ok=True)


_apply_storage_base(os.path.join(BASE_DIR, "storage"))  # default -- overridden below if the user picked a custom one


def _ensure_cam_labels(rois):
    """Guarantees every ROI dict has a non-empty, unique 'cam_label'
    (cam1, cam2, ...). Existing labels are kept as-is; only missing/blank/
    duplicate ones are (re)assigned, by ROI order, to the next label not
    already used. This is what makes an older/pre-labeling ROI layout
    (saved before this feature existed) load and "just work" -- it gets
    cam1..camN assigned by creation order the first time it's touched,
    instead of erroring or leaving ROIs unlabeled."""
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


def _safe_folder_name(name):
    """Sanitizes a model/line name for use as a folder name -- falls back
    to 'UNSPECIFIED' if it's empty so images never get lost by silently
    landing outside the model-labeled tree."""
    name = (name or "").strip()
    if not name:
        return "UNSPECIFIED"
    keep = "-_.() "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name)
    return cleaned.strip() or "UNSPECIFIED"


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
    "vinyl_tolerance_px": 15.0,
    "vinyl_strength_thr": 6.0,
    "model_name": "",
    "line_name": "",
    "active_roi_name": None,
    "storage_base_path": None,
    "ai_enabled": False,          # off by default -- z-score-only behavior is unchanged until a technician opts in
    "ai_model_paths": {},         # "<model_name>::<cam_label>" -> path to that camera's trained .onnx file
    "ai_thresholds": {},          # "<model_name>::<cam_label>" -> manual anomaly-score threshold override
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
    grab thread. Safe to use even when pypylon / hardware is unavailable --
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
def _detect_dust_in_roi(bgr, roi, window, z_thr, min_area, min_circularity, mm_per_px, min_diameter_mm, debug):
    """The whole z-score -> threshold -> contour pipeline, run on just ONE
    ROI's own bounding-box crop instead of the full frame. This is the key
    complexity fix: before, every per-pixel step (boxFilter, zscore, the
    threshold) ran over the ENTIRE frame even though the ROIs typically
    cover a small fraction of it, and every contour re-allocated a
    full-frame-sized array. Cropping first means the per-pixel cost is
    O(this ROI's area) not O(full frame), and using cv2.boundingRect(c) for
    each contour (instead of a full-crop-sized zeros array) means the
    per-contour cost is O(that blob's own area) not O(the crop's area).
    Summed across ROIs that's O(sum of ROI areas + total blob area) instead
    of the old O(k * full_frame_pixels) -- and since each call here is
    independent (only reads bgr, never writes shared state), the caller
    runs one of these per ROI in a thread pool for real wall-clock
    parallelism on top of that (cv2/numpy release the GIL during the
    actual C-level number crunching, so separate ROIs' heavy calls do
    genuinely overlap, not just interleave).

    Returns (blobs, local_binary, stats_partial, debug_crops, (x0, y0)).
    blobs are already offset into FULL-FRAME coordinates; local_binary and
    debug_crops are still crop-local -- the caller pastes them back at
    (x0, y0) into full-size canvases.
    """
    h, w = bgr.shape[:2]
    cx, cy, r = roi["cx"], roi["cy"], roi["r"]
    win = window if window % 2 == 1 else window + 1
    # Pad the crop by win//2 beyond the ROI's own bounding box. Without
    # this, boxFilter's border handling at the crop's edge would reflect
    # the CROP's own boundary instead of seeing the real neighboring
    # pixels the full-frame version used to see there -- which would
    # quietly shift the z-score for pixels within win//2 of the ROI's
    # edge. The pad gives boxFilter real image content on all sides
    # (as long as the ROI isn't touching the actual frame edge), so
    # z-score inside the ROI circle comes out numerically identical to
    # the old full-frame computation (verified directly: 0.0 max diff on
    # a test image). The mask still only keeps the true circle, so this
    # extra border is pure context for the filter, never reported as data.
    pad = win // 2
    x0 = max(0, int(cx - r) - pad)
    y0 = max(0, int(cy - r) - pad)
    x1 = min(w, int(cx + r) + 1 + pad)
    y1 = min(h, int(cy + r) + 1 + pad)
    if x1 <= x0 or y1 <= y0:
        return [], None, None, None, (x0, y0)

    crop = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    local_mean = cv2.boxFilter(gray, -1, (win, win))
    local_mean_sq = cv2.boxFilter(gray * gray, -1, (win, win))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0))
    zscore = np.where(local_std > 1e-5, (gray - local_mean) / local_std, 0.0)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (int(cx - x0), int(cy - y0)), int(r), 255, -1)

    raw = ((zscore >= z_thr) & (mask == 255)).astype(np.uint8) * 255

    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blobs = []
    local_binary = np.zeros_like(raw)
    rejected = 0

    for c in contours:
        # Crop to just this contour's own bounding box (cv2.boundingRect)
        # instead of allocating a crop-sized zeros array per contour --
        # this is what takes the per-contour cost from O(crop area) down
        # to O(that blob's own area).
        bx, by, bw, bh = cv2.boundingRect(c)
        sub_raw = raw[by:by + bh, bx:bx + bw]
        sub_region = np.zeros_like(sub_raw)
        c_local = c - [bx, by]  # shift contour points into the sub-crop's own coordinate frame
        cv2.drawContours(sub_region, [c_local], -1, 255, -1)
        # Real foreground pixel count, NOT cv2.contourArea(c). contourArea
        # treats the contour as a filled polygon -- fine for a solid dust
        # speck, but wrong for a hollow ring: RETR_EXTERNAL only returns
        # the OUTER boundary of a ring, so contourArea would treat a thin
        # bright ring as if it were a solid disc, handing it near-perfect
        # circularity. That's exactly the false positive this pipeline has
        # to reject: a center concentric ring/crescent from lens or
        # coating reflection under the inspection light is a real,
        # physically-fixed optical artifact, not contamination. Counting
        # only the pixels that actually crossed z_thr makes a thin ring's
        # true area tiny relative to its outer perimeter, so its
        # circularity comes out correctly low and it's rejected here
        # without any separate ring-specific check.
        real_pixels_sub = cv2.bitwise_and(sub_raw, sub_region)
        area = int(cv2.countNonZero(real_pixels_sub))
        if area < min_area:
            rejected += 1
            continue
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            rejected += 1
            continue  # not round/solid enough to be a dust speck (rings, crescents, arcs) -- drop

        (ccx, ccy), rr = cv2.minEnclosingCircle(c)
        diameter_px = 2.0 * rr
        diameter_mm = diameter_px * mm_per_px if mm_per_px else None
        if diameter_mm is not None and diameter_mm < min_diameter_mm:
            rejected += 1
            continue
        label = f"{diameter_mm:.2f}mm" if diameter_mm is not None else f"{diameter_px:.0f}px"
        blobs.append({"type": "dust", "cx": float(ccx + x0), "cy": float(ccy + y0), "r": float(max(rr, 3.0)),
                      "diameter_px": float(diameter_px), "diameter_mm": diameter_mm, "label": label})
        local_binary[by:by + bh, bx:bx + bw] = cv2.bitwise_or(local_binary[by:by + bh, bx:bx + bw], real_pixels_sub)

    stats_partial = None
    if mask.any():
        roi_z = zscore[mask == 255]
        stats_partial = {"max_z": float(roi_z.max()), "sum_z": float(roi_z.sum()), "count_z": int(roi_z.size),
                          "dust_px": int((local_binary == 255).sum()), "rejected": rejected}

    debug_crops = {"gray": gray, "zscore": zscore, "raw": raw} if debug else None
    return blobs, local_binary, stats_partial, debug_crops, (x0, y0)


def run_zscore_detection(bgr, rois, window, z_thr, min_area=4.0, min_circularity=0.55,
                          mm_per_px=None, min_diameter_mm=0.0, debug=False):
    """Core Z-score math is UNCHANGED: local Z-score via boxFilter mean/std,
    thresholded inside each circular ROI mask.

    DUST-ONLY pipeline: every surviving connected blob above z_thr is kept
    as a dust candidate if it's big enough (min_area) and round enough
    (min_circularity); everything else is rejected. (Thread/fiber and glue
    shape-classification, hysteresis thresholding, and gap-bridging were
    removed to keep this simple and fast -- re-add them later if/when
    those defect types need to come back.)

    Each ROI is processed independently on its own bounding-box crop, in
    its own worker thread (see _detect_dust_in_roi's docstring for why
    that's the complexity fix, not just a parallelism nicety): this brings
    the per-inspection cost down from O(full_frame_pixels) to O(sum of ROI
    areas + total detected blob area), and the ROIs' cv2/numpy heavy
    lifting genuinely overlaps in wall-clock time since those calls
    release the GIL. Debug-view images are pasted back together from each
    ROI's crop -- outside every ROI they're simply black, which matches
    reality: nothing outside an ROI was ever analyzed, before or after
    this change.

    If the app has been calibrated (mm_per_px set via two-point calibration),
    sizes are also reported in mm; anything under min_diameter_mm is
    rejected too. Without calibration this step is skipped.

    Returns (binary, blobs, stats, debug_images). binary is the cleaned
    mask; blobs is a list of dust blob dicts (position, size, and a
    ready-to-draw "label" string); stats is summary counts. debug_images
    is None unless debug=True, in which case it's an ordered dict of
    intermediate pipeline images (grayscale, z-score heatmap, threshold,
    final classified result) for the pipeline-steps viewer.
    """
    h, w = bgr.shape[:2]
    binary = np.zeros((h, w), dtype=np.uint8)
    all_blobs = []
    max_z, sum_z, count_z, dust_px, rejected = 0.0, 0.0, 0, 0, 0
    full_gray = full_zscore = full_raw = None
    if debug:
        full_gray = np.zeros((h, w), dtype=np.uint8)
        full_zscore = np.zeros((h, w), dtype=np.float32)
        full_raw = np.zeros((h, w), dtype=np.uint8)

    if rois:
        with ThreadPoolExecutor(max_workers=min(8, len(rois))) as ex:
            results = list(ex.map(
                lambda roi: _detect_dust_in_roi(bgr, roi, window, z_thr, min_area, min_circularity,
                                                 mm_per_px, min_diameter_mm, debug),
                rois))

        for blobs, local_binary, stats_partial, debug_crops, (x0, y0) in results:
            all_blobs.extend(blobs)
            if local_binary is not None:
                hc, wc = local_binary.shape[:2]
                binary[y0:y0 + hc, x0:x0 + wc] = cv2.bitwise_or(binary[y0:y0 + hc, x0:x0 + wc], local_binary)
            if stats_partial:
                max_z = max(max_z, stats_partial["max_z"])
                sum_z += stats_partial["sum_z"]
                count_z += stats_partial["count_z"]
                dust_px += stats_partial["dust_px"]
                rejected += stats_partial["rejected"]
            if debug and debug_crops is not None:
                hc, wc = debug_crops["gray"].shape[:2]
                full_gray[y0:y0 + hc, x0:x0 + wc] = np.clip(debug_crops["gray"], 0, 255).astype(np.uint8)
                full_zscore[y0:y0 + hc, x0:x0 + wc] = debug_crops["zscore"]
                full_raw[y0:y0 + hc, x0:x0 + wc] = debug_crops["raw"]

    stats = None
    if count_z > 0:
        stats = {"max_z": max_z, "mean_z": sum_z / count_z, "dust_px": dust_px,
                 "dust_count": len(all_blobs), "rejected": rejected}

    debug_images = None
    if debug:
        debug_images = {}
        debug_images["1 Grayscale"] = cv2.cvtColor(full_gray, cv2.COLOR_GRAY2BGR)
        z_ceiling = max(z_thr * 3.0, 1.0)
        z_norm = (np.clip(full_zscore, 0, z_ceiling) / z_ceiling * 255).astype(np.uint8)
        debug_images["2 Z-score heatmap"] = cv2.applyColorMap(z_norm, cv2.COLORMAP_INFERNO)
        debug_images["3 Threshold (z_thr)"] = cv2.cvtColor(full_raw, cv2.COLOR_GRAY2BGR)
        result_disp = bgr.copy()
        for b in all_blobs:
            cx, cy, r = int(b["cx"]), int(b["cy"]), int(round(b["r"]))
            color = BLOB_COLOR_BGR.get(b["type"], (0, 0, 255))
            cv2.circle(result_disp, (cx, cy), r + 4, color, 2)
            cv2.putText(result_disp, b["label"], (cx + r + 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        debug_images["4 Final classified result"] = result_disp

    return binary, all_blobs, stats, debug_images



def detect_vinyl_presence(gray, cx, cy, roi_radius, tolerance_px=15.0, strength_ratio_thr=6.0):
    """
    Detects whether a transparent vinyl/lamination cutout ring is present
    around a camera opening, WITHOUT assuming exact concentricity to the
    ROI center and WITHOUT relying on brightness/opacity (the vinyl is
    transparent, so those signals are unreliable).

    Why this approach:
      - The camera lens area itself is always clear (vinyl or not), so
        looking inside the ROI tells you nothing -- the signal is a
        physical die-cut edge somewhere in a radius BAND around the ROI
        radius (not at one exact radius, since the cutout can be
        slightly misaligned to the lens center).
      - warpPolar turns "find a ring at some unknown radius" into "find a
        horizontal line of strong edge in an unrolled image" -- a much
        easier, more robust problem than searching 2D for circles.
      - A real die-cut edge is near-complete around the full 360 degrees,
        so its edge energy averaged across ALL angles at one radius (the
        peak in the radial profile) stands out sharply above the typical
        (median) edge energy elsewhere. Partial scratches/reflections
        only cover part of the circumference, so they raise the peak far
        less -- this ratio is what actually separates the two cases.

    Validated on synthetic bare / full-ring / partial-arc images, including
    with realistic sensor noise: bare ~1.0, a 70-degree partial arc
    ~4.6-5.0, a full ring ~11.3-11.8, consistently across noise seeds --
    hence the default threshold of 6.0. An angular "ring coverage" fraction
    was also tried as a second gate but proved too noise-sensitive to trust
    on its own, so strength_ratio alone is what gates detection here.
    Recalibrate strength_ratio_thr and tolerance_px against real reference
    captures (Teaching: one bare module, one with vinyl) since real
    optics/lighting will shift these numbers.

    Returns (detected: bool, info: dict) with peak_radius_px and
    strength_ratio, useful for tuning or a debug view.
    """
    gray = gray if gray.dtype != np.uint8 else gray.astype(np.float32)
    H, W = gray.shape[:2]

    max_r = int(round(roi_radius + tolerance_px))
    min_r = max(1, int(round(roi_radius - tolerance_px)))
    size = max_r * 2

    x0, y0 = int(round(cx - max_r)), int(round(cy - max_r))
    patch = np.zeros((size, size), dtype=np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + size), min(H, y0 + size)
    px0, py0 = sx0 - x0, sy0 - y0
    if sx1 > sx0 and sy1 > sy0:
        patch[py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)] = gray[sy0:sy1, sx0:sx1]

    center = (size / 2.0, size / 2.0)
    num_angles = 360
    polar = cv2.warpPolar(patch, (max_r, num_angles), center, max_r,
                           cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)

    gx = cv2.Sobel(polar, cv2.CV_32F, 1, 0, ksize=3)
    edge_energy = np.abs(gx)

    profile = edge_energy.mean(axis=0)
    band = profile[min_r:max_r]
    if band.size == 0:
        return False, {}

    peak_idx = int(np.argmax(band))
    peak_r = min_r + peak_idx
    peak_val = float(band[peak_idx])
    baseline = float(np.median(profile))
    strength_ratio = peak_val / (baseline + 1e-6)

    detected = strength_ratio > strength_ratio_thr
    info = {"peak_radius_px": peak_r, "strength_ratio": strength_ratio}
    return detected, info


CANVAS_W = 560
CANVAS_H = 560
ROI_HIT_TOL = 6


class ZoomableImageCanvas:
    """A tk.Canvas that displays a BGR (OpenCV) image with mouse-wheel zoom
    (cursor-anchored) and click-drag pan -- the same interaction as every
    other image view in this app. Each instance owns its own independent
    zoom/pan state, so multiple of these side by side (e.g. one per
    pipeline stage) don't affect each other."""

    def __init__(self, parent, width=420, height=420, bg=None):
        bg = bg or BG_CANVAS
        self.canvas = tk.Canvas(parent, width=width, height=height, bg=bg, highlightthickness=0)
        self.image = None
        self.zoom = 1.0
        self.base_scale = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._fitted = False
        self._dragging = False
        self._drag_start = (0, 0)
        self._last = (0, 0)
        self.photo = None
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)
        self.canvas.bind("<Button-5>", self.on_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Configure>", lambda e: self._render())

    def set_image(self, bgr):
        self.image = bgr
        self._fitted = False
        self._render()

    def _canvas_wh(self):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 10 or h < 10:
            return 420, 420
        return w, h

    def fit(self):
        if self.image is None:
            return
        cw, ch = self._canvas_wh()
        h, w = self.image.shape[:2]
        self.base_scale = min(cw / w, ch / h)
        self.zoom = 1.0
        s = self.base_scale
        self.view_x = (cw - w * s) / 2
        self.view_y = (ch - h * s) / 2
        self._fitted = True
        self._render()

    def _apply_zoom(self, factor, cx, cy):
        if self.image is None:
            return
        s_old = self.base_scale * self.zoom
        ix = (cx - self.view_x) / s_old
        iy = (cy - self.view_y) / s_old
        self.zoom = max(0.2, min(self.zoom * factor, 30.0))
        s_new = self.base_scale * self.zoom
        self.view_x = cx - ix * s_new
        self.view_y = cy - iy * s_new
        self._render()

    def on_wheel(self, event):
        if self.image is None:
            return
        direction = 1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4) else -1
        factor = 1.2 if direction > 0 else 1 / 1.2
        self._apply_zoom(factor, event.x, event.y)

    def on_press(self, event):
        self._drag_start = (event.x, event.y)
        self._last = (event.x, event.y)
        self._dragging = False

    def on_drag(self, event):
        if self.image is None:
            return
        if not self._dragging:
            if abs(event.x - self._drag_start[0]) + abs(event.y - self._drag_start[1]) > 4:
                self._dragging = True
        if not self._dragging:
            return
        self.view_x += event.x - self._last[0]
        self.view_y += event.y - self._last[1]
        self._last = (event.x, event.y)
        self._render()

    def _render(self):
        self.canvas.delete("all")
        if self.image is None:
            return
        if not self._fitted:
            self.fit()
            return
        cw, ch = self._canvas_wh()
        H, W = self.image.shape[:2]
        s = self.base_scale * self.zoom
        vx, vy = self.view_x, self.view_y
        l = max(0, int(-vx / s))
        t = max(0, int(-vy / s))
        r = min(W, int((cw - vx) / s) + 1)
        b = min(H, int((ch - vy) / s) + 1)
        if r <= l or b <= t:
            return
        crop = self.image[t:b, l:r]
        cwid = max(1, int((r - l) * s))
        chei = max(1, int((b - t) * s))
        interp = cv2.INTER_NEAREST if self.zoom > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop, (cwid, chei), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(vx + l * s, vy + t * s, anchor="nw", image=self.photo)


# ==================================================== AI ANOMALY DETECTION ==
class AIModelManager:
    """Owns loading and running the trained anomaly-detection models that
    Training Studio produces -- one ONNX model per (model_name, cam_label)
    pair, exactly matching how Training Studio now scopes a training run
    to one phone-model profile AND one camera position together. Nothing
    here touches Tkinter; the app just calls load()/score().

    Deliberately uses ONNX Runtime, not anomalib/torch: Training Studio's
    own docstring already says training happens on a different, more
    capable machine than the operator's PC, so the trained model has to
    be exported (Training Studio already does this to ONNX) and the
    resulting .onnx file copied over -- this class is what loads that
    file and runs it here, without needing the ~2GB torch/anomalib stack
    installed on the line PC.

    IMPORTANT CAVEAT (untested against a real anomalib export): anomalib's
    exact ONNX output contract -- whether the graph outputs a single
    already-normalized image-level score, a raw score plus a per-pixel
    anomaly map, or something else -- has shifted across anomalib
    versions and isn't something this environment could verify (no
    anomalib/torch install here, so no real .onnx file to test against).
    This class handles it generically (see score()'s docstring) and reads
    a sibling metadata.json for the threshold the way anomalib usually
    exports one, but the actual scores this produces should be sanity
    checked against Test Detection on a real trained model before trusting
    it on the line -- if the output shape doesn't match what's assumed
    here, tell me the exact shapes/error and this gets fixed against your
    actual anomalib version.
    """

    def __init__(self):
        self._sessions = {}    # key -> onnxruntime.InferenceSession
        self._input_meta = {}  # key -> (input_name, (height, width))
        self._thresholds = {}  # key -> float image-level anomaly threshold
        self._paths = {}       # key -> onnx path actually loaded
        self._errors = {}      # key -> last load/inference error string

    @staticmethod
    def _key(model_name, cam_label):
        return f"{(model_name or '').strip()}::{(cam_label or '').strip()}"

    def is_loaded(self, model_name, cam_label):
        return self._key(model_name, cam_label) in self._sessions

    def loaded_path(self, model_name, cam_label):
        return self._paths.get(self._key(model_name, cam_label))

    def last_error(self, model_name, cam_label):
        return self._errors.get(self._key(model_name, cam_label))

    def threshold(self, model_name, cam_label):
        return self._thresholds.get(self._key(model_name, cam_label), 0.5)

    def set_threshold(self, model_name, cam_label, value):
        self._thresholds[self._key(model_name, cam_label)] = float(value)

    def load(self, model_name, cam_label, onnx_path, manual_threshold=None):
        """(Re)loads the ONNX model for this camera position. Returns
        (True, None) on success or (False, error_message) on failure --
        never raises, so a bad/missing file just shows as a status
        message in the UI rather than crashing the app."""
        key = self._key(model_name, cam_label)
        if not ONNXRUNTIME_AVAILABLE:
            self._errors[key] = "onnxruntime is not installed in this environment"
            return False, self._errors[key]
        try:
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            inp = session.get_inputs()[0]
            shape = inp.shape  # e.g. [1, 3, 256, 256] -- dynamic dims may come back as strings
            h = shape[2] if isinstance(shape[2], int) else 256
            w = shape[3] if isinstance(shape[3], int) else 256
            self._sessions[key] = session
            self._input_meta[key] = (inp.name, (h, w))
            self._paths[key] = onnx_path

            threshold = manual_threshold
            if threshold is None:
                # anomalib commonly exports a metadata.json next to the
                # ONNX file carrying the image-level anomaly threshold --
                # best-effort read, since this isn't guaranteed present or
                # named the same across every anomalib version.
                meta_path = os.path.join(os.path.dirname(onnx_path), "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        threshold = meta.get("image_threshold", meta.get("threshold"))
                    except Exception:
                        threshold = None
            self._thresholds[key] = float(threshold) if threshold is not None else 0.5
            self._errors.pop(key, None)
            return True, None
        except Exception as e:
            self._errors[key] = str(e)
            self._sessions.pop(key, None)
            self._input_meta.pop(key, None)
            self._paths.pop(key, None)
            return False, str(e)

    def unload(self, model_name, cam_label):
        key = self._key(model_name, cam_label)
        self._sessions.pop(key, None)
        self._input_meta.pop(key, None)
        self._paths.pop(key, None)
        self._errors.pop(key, None)

    def score(self, model_name, cam_label, crop_bgr):
        """Runs one ROI's circular crop (same clean, black-filled-corner
        crop the operator app saves for training -- consistent input is
        what makes this a fair comparison against training data) through
        that camera's loaded model.

        Preprocessing follows anomalib's typical default: resize to the
        model's own expected input size (read from the ONNX graph, not
        hardcoded), RGB, ImageNet mean/std normalization, NCHW float32 --
        the standard anomalib preprocessing, though not guaranteed for
        every export config.

        Output handling: prefers a scalar (size-1) output as the
        image-level anomaly score; if every output is a multi-element
        map, falls back to that map's max value (a per-pixel anomaly map
        collapsed to "how anomalous is the worst pixel", a reasonable
        stand-in for an image-level score when no scalar output exists).

        Returns (anomaly_score, is_anomalous: bool), or None if no model
        is loaded for this (model_name, cam_label)."""
        key = self._key(model_name, cam_label)
        session = self._sessions.get(key)
        if session is None:
            return None
        input_name, (h, w) = self._input_meta[key]
        try:
            img = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            tensor = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
            outputs = session.run(None, {input_name: tensor})

            score_val = None
            for out in outputs:
                arr = np.asarray(out)
                if arr.size == 1:
                    score_val = float(arr.reshape(-1)[0])
                    break
            if score_val is None:
                arr = max((np.asarray(o) for o in outputs), key=lambda a: a.size)
                score_val = float(arr.max())

            self._errors.pop(key, None)
            return score_val, score_val >= self._thresholds.get(key, 0.5)
        except Exception as e:
            self._errors[key] = str(e)
            return None


# =============================================================== MAIN APP ==
class DustInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Dust Inspection")
        self.root.geometry("1680x1000")
        self.root.minsize(1280, 800)
        self.root.configure(fg_color=BG)

        self.f_title = ctk.CTkFont(size=22, weight="bold")
        self.f_subtitle = ctk.CTkFont(size=13)
        self.f_section = ctk.CTkFont(size=15, weight="bold")
        self.f_body = ctk.CTkFont(size=13)
        self.f_small = ctk.CTkFont(size=11)
        self.f_status = ctk.CTkFont(size=20, weight="bold")
        # operator page: big, factory-floor-visible ("saaf dikhe from a distance")
        self.f_op_title = ctk.CTkFont(size=42, weight="bold")
        self.f_op_subtitle = ctk.CTkFont(size=20)
        self.f_op_label = ctk.CTkFont(size=15, weight="bold")
        self.f_op_status = ctk.CTkFont(size=46, weight="bold")
        self.f_op_findings = ctk.CTkFont(size=19)
        self.f_op_barcode = ctk.CTkFont(size=24)
        self.f_op_button = ctk.CTkFont(size=26, weight="bold")
        self.f_op_button_sm = ctk.CTkFont(size=16, weight="bold")
        self.f_op_counter_num = ctk.CTkFont(size=30, weight="bold")
        self.f_op_stat_num = ctk.CTkFont(size=38, weight="bold")

        self.settings = load_settings()
        if self.settings.get("storage_base_path"):
            _apply_storage_base(self.settings["storage_base_path"])
        self.cam = CameraManager()
        self.ai_models = AIModelManager()

        # shared state
        self.original = None
        self.using_static_image = False
        self.rois = []                 # active ROI set used by the main window
        self.last_blobs = []           # last inspection's accepted dust blobs
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
        self._main_fit_was_fallback = True

        self.current_view = "operator"
        self.main_photo = None
        self.roi_photo = None
        self._pipeline_images = {}
        self.pipeline_canvases = []

        self.model_line_var = tk.StringVar(value=self._model_line_text())
        self.status_var = tk.StringVar(value="IDLE")
        self.barcode_var = tk.StringVar(value="")
        self.footer_var = tk.StringVar(value="Starting up...")

        self.count_checked = 0
        self.count_passed = 0
        self.count_failed = 0
        self.checked_var = tk.StringVar(value="0")
        self.passed_var = tk.StringVar(value="0")
        self.failed_var = tk.StringVar(value="0")
        self.failure_rate_var = tk.StringVar(value="0.0%")
        self.pass_rate_var = tk.StringVar(value="0.0%")

        self._build_main_ui()
        self._load_active_roi_layout()
        self._load_configured_ai_models()
        self._auto_connect_camera()
        self._poll_live()

    def _load_configured_ai_models(self):
        """Loads every (model_name, cam_label) -> .onnx mapping already
        saved in settings, so a technician's earlier AI-model setup is
        ready on next launch without re-browsing for the file. Best-effort
        per entry -- a missing/bad file just leaves that camera's model
        unloaded (shown as an error in the AI Model tab) rather than
        blocking startup."""
        model = self.settings.get("model_name")
        paths = self.settings.get("ai_model_paths", {}) or {}
        thresholds = self.settings.get("ai_thresholds", {}) or {}
        for key, onnx_path in paths.items():
            if "::" not in key:
                continue
            key_model, cam_label = key.split("::", 1)
            if model and key_model != model:
                continue  # only auto-load the currently active model's cameras
            if not onnx_path or not os.path.exists(onnx_path):
                continue
            self.ai_models.load(key_model, cam_label, onnx_path, thresholds.get(key))

    # ------------------------------------------------------------- utils --
    def _model_line_text(self):
        m = self.settings.get("model_name") or "--"
        l = self.settings.get("line_name") or "--"
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

    def _counter_tile(self, parent, col, value_var, label_text, color, font=None):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=0, column=col, sticky="nsew")
        ctk.CTkLabel(box, textvariable=value_var, font=font or self.f_op_counter_num, text_color=color).pack()
        ctk.CTkLabel(box, text=label_text, font=self.f_small, text_color=TEXT_MUTED).pack(pady=(0, 4))

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

        # ---- slim nav bar (persistent across both views) ----
        header = ctk.CTkFrame(outer, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(14, 4))
        nav = ctk.CTkFrame(header, fg_color=BG_CARD_ALT, corner_radius=10)
        nav.pack(side="right")
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
        self.page_pipeline = ctk.CTkFrame(container, fg_color=BG, corner_radius=0)
        self.page_operator.grid(row=0, column=0, sticky="nsew")
        self.page_teaching.grid(row=0, column=0, sticky="nsew")
        self.page_pipeline.grid(row=0, column=0, sticky="nsew")

        self._build_operator_page(self.page_operator)
        self._build_teaching_page(self.page_teaching)
        self._build_pipeline_page(self.page_pipeline)

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
        self._refresh_ai_tab()  # camera list / load status may have changed since the tab was built
        self._render_roi_canvas()

    def show_pipeline_page(self):
        self.current_view = "pipeline"
        self.page_pipeline.tkraise()

    def _build_pipeline_page(self, page):
        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(top, text="Pipeline Steps", font=self.f_title, text_color=TEXT).pack(side="left")
        self._btn_secondary(top, "Back to Teaching", self.show_teaching_page, width=160).pack(side="right")
        ctk.CTkLabel(page, text="What Test Detection actually did to the last frame, stage by stage. Wheel = zoom, drag = pan, on each image independently. Scrollbar moves between stages.",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", pady=(0, 8))

        # Manual horizontal-scroll container (NOT CTkScrollableFrame, which
        # binds mouse wheel to scrolling -- that would fight with wheel-zoom
        # on each stage's image). Wheel stays free for zoom; the scrollbar
        # (drag it, or shift+wheel) moves between stages instead.
        outer = ctk.CTkFrame(page, fg_color=BG)
        outer.pack(fill="both", expand=True)
        self.pipeline_hcanvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        hbar = ctk.CTkScrollbar(outer, orientation="horizontal", command=self.pipeline_hcanvas.xview)
        self.pipeline_hcanvas.configure(xscrollcommand=hbar.set)
        hbar.pack(side="bottom", fill="x")
        self.pipeline_hcanvas.pack(side="top", fill="both", expand=True)

        self.pipeline_inner = ctk.CTkFrame(self.pipeline_hcanvas, fg_color=BG)
        self._pipeline_inner_window = self.pipeline_hcanvas.create_window((0, 0), window=self.pipeline_inner, anchor="nw")
        self.pipeline_inner.bind("<Configure>", lambda e: self.pipeline_hcanvas.configure(
            scrollregion=self.pipeline_hcanvas.bbox("all")))
        self.pipeline_canvases = []  # ZoomableImageCanvas instances, one per stage

    def _populate_pipeline_page(self):
        for widget in self.pipeline_inner.winfo_children():
            widget.destroy()
        self.pipeline_canvases = []
        for name, img in self._pipeline_images.items():
            card = self._card(self.pipeline_inner)
            card.pack(side="left", fill="y", padx=8, pady=4)
            ctk.CTkLabel(card, text=name, font=self.f_section, text_color=TEXT).pack(anchor="w", padx=14, pady=(12, 6))
            zc = ZoomableImageCanvas(card, width=520, height=520)
            zc.canvas.pack(padx=14, pady=(0, 14))
            zc.set_image(img)
            self.pipeline_canvases.append(zc)

    def _build_operator_page(self, page):
        # ---- big, centered header -- this is what the operator sees first ----
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.pack(fill="x", pady=(2, 12))
        ctk.CTkLabel(header, text="AUTO CAMERA DEFECT INSPECTION", font=self.f_op_title,
                     text_color=TEXT, anchor="center").pack(fill="x")
        ctk.CTkLabel(header, textvariable=self.model_line_var, font=self.f_op_subtitle,
                     text_color=TEXT_MUTED, anchor="center").pack(fill="x", pady=(4, 0))

        body = ctk.CTkFrame(page, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # ---- left: feed only, fills the full column height ----
        feed_card = self._card(body)
        feed_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        feed_head = ctk.CTkFrame(feed_card, fg_color="transparent")
        feed_head.pack(fill="x", padx=14, pady=(12, 0))
        ctk.CTkLabel(feed_head, text="Live Feed  -  wheel = zoom  -  drag = pan (view only)",
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

        # ---- right: status / findings / barcode / start / clear / log ----
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")

        status_card = self._card(right)
        status_card.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(status_card, text="STATUS", font=self.f_op_label, text_color=TEXT_MUTED).pack(anchor="w", padx=20, pady=(12, 0))
        self.status_label = ctk.CTkLabel(status_card, textvariable=self.status_var, font=self.f_op_status, text_color=TEXT_MUTED)
        self.status_label.pack(anchor="w", padx=20, pady=(0, 14))

        counters_card = self._card(right)
        counters_card.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(counters_card, text="TODAY'S COUNT", font=self.f_op_label, text_color=TEXT_MUTED).pack(anchor="w", padx=20, pady=(12, 6))
        counters_row = ctk.CTkFrame(counters_card, fg_color="transparent")
        counters_row.pack(fill="x", padx=10, pady=(0, 14))
        counters_row.grid_columnconfigure(0, weight=1)
        counters_row.grid_columnconfigure(1, weight=1)
        counters_row.grid_columnconfigure(2, weight=1)
        self._counter_tile(counters_row, 0, self.checked_var, "CHECKED", TEXT)
        self._counter_tile(counters_row, 1, self.passed_var, "PASSED", SUCCESS)
        self._counter_tile(counters_row, 2, self.failed_var, "FAILED", DANGER)

        stats_card = self._card(right)
        stats_card.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(stats_card, text="STATS", font=self.f_op_label, text_color=TEXT_MUTED).pack(anchor="w", padx=20, pady=(12, 6))
        stats_row = ctk.CTkFrame(stats_card, fg_color="transparent")
        stats_row.pack(fill="x", padx=10, pady=(0, 4))
        stats_row.grid_columnconfigure(0, weight=1)
        stats_row.grid_columnconfigure(1, weight=1)
        self._counter_tile(stats_row, 0, self.failure_rate_var, "FAILURE RATE", DANGER, font=self.f_op_stat_num)
        self._counter_tile(stats_row, 1, self.pass_rate_var, "PASS RATE", SUCCESS, font=self.f_op_stat_num)
        self._btn_secondary(stats_card, "Reset Counters", self.reset_counters, width=160, height=32).pack(pady=(4, 12))

        findings_card = self._card(right)
        findings_card.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(findings_card, text="FINDINGS", font=self.f_op_label, text_color=TEXT_MUTED).pack(anchor="w", padx=20, pady=(12, 4))
        self.findings_var = tk.StringVar(value="--")
        ctk.CTkLabel(findings_card, textvariable=self.findings_var, font=self.f_op_findings, text_color=TEXT,
                     justify="left", anchor="w").pack(anchor="w", padx=20, pady=(0, 12))

        barcode_card = self._card(right)
        barcode_card.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(barcode_card, text="BARCODE", font=self.f_op_label, text_color=TEXT_MUTED).pack(anchor="w", padx=20, pady=(12, 6))
        self.barcode_entry = ctk.CTkEntry(barcode_card, textvariable=self.barcode_var, font=self.f_op_barcode,
                                           corner_radius=10, height=48, fg_color=BG_CARD_ALT, border_color=BORDER, text_color=TEXT)
        self.barcode_entry.pack(fill="x", padx=20, pady=(0, 14))

        self.start_btn = self._btn_primary(right, "Start", self.start_inspection, width=200, height=64,
                                            font=self.f_op_button)
        self.start_btn.pack(fill="x", pady=(0, 8))

        self._btn_secondary(right, "Clear Markings", self.clear_detection_markings, width=200, height=40,
                             font=self.f_op_button_sm).pack(fill="x", pady=(0, 8))

        log_card = self._card(right)
        log_card.pack(fill="both", expand=True)
        ctk.CTkLabel(log_card, text="Inspection Log", font=self.f_section, text_color=TEXT).pack(anchor="w", padx=16, pady=(12, 6))
        self.log_box = ctk.CTkTextbox(log_card, fg_color=BG_CANVAS, text_color=TEXT_MUTED,
                                       font=ctk.CTkFont(family="Courier New", size=10), corner_radius=8, wrap="none")
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.log_box.configure(state="disabled")

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
        tab_ai = tabs.add("AI Model")
        tab_cam = tabs.add("Camera")

        self._build_model_tab(tab_model)
        self._build_roi_tab(tab_roi)
        self._build_detection_tab(tab_detect)
        self._build_ai_tab(tab_ai)
        self._build_camera_tab(tab_cam)

    def _set_status(self, text, color):
        self.status_var.set(text)
        self.status_label.configure(text_color=color)

    def clear_detection_markings(self):
        """Wipes the last inspection's overlays off the feed -- for when
        the camera's been unplugged/moved and stale markings are stuck on
        the last frame it ever delivered."""
        self.last_blobs = []
        self.findings_var.set("--")
        self._set_status("IDLE", TEXT_MUTED)
        self._render_main_feed()
        if self.current_view == "teaching":
            self._render_roi_canvas()

    # ------------------------------------------------------------ camera --
    def _auto_connect_camera(self):
        ok, msg = self.cam.connect()
        if ok:
            self.cam.apply_settings(exposure_us=self.settings.get("exposure_us"), gain=self.settings.get("gain"))
            self.cam.start_live()
            self.footer_var.set("Camera connected -- live feed running.")
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
                _ensure_cam_labels(self.rois)  # migrates older layouts saved before camera labeling existed
            except Exception:
                self.rois = []

    def _activate_roi_layout(self, name, rois):
        _ensure_cam_labels(rois)  # migrates older layouts saved before camera labeling existed
        self.rois = rois
        self.settings["active_roi_name"] = name
        save_settings(self.settings)
        self._render_main_feed()

    # -------------------------------------------------- main feed (view-only)
    def _build_main_overlay(self):
        disp = self.original.copy()
        for roi in self.rois:
            cv2.circle(disp, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (0, 255, 0), 2)
        for b in self.last_blobs:
            cx, cy, r = int(b["cx"]), int(b["cy"]), int(round(b["r"]))
            color = BLOB_COLOR_BGR.get(b["type"], (0, 0, 255))
            cv2.circle(disp, (cx, cy), r + 4, color, 2)
            cv2.putText(disp, b["label"], (cx + r + 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
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
        # was this fit computed on the real canvas size, or the fallback
        # (used before the window has finished its first layout pass)?
        # if it was the fallback, we need to re-fit once real dimensions
        # are known -- otherwise the feed looks "not fit" until the user
        # manually hits Fit.
        real_w, real_h = self.feed_canvas.winfo_width(), self.feed_canvas.winfo_height()
        self._main_fit_was_fallback = real_w < 10 or real_h < 10
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
        canvas_now_real = self.feed_canvas.winfo_width() >= 10 and self.feed_canvas.winfo_height() >= 10
        need_refit = (not self._main_fitted) or (self._main_fitted_shape != shape) or \
                     (self._main_fit_was_fallback and canvas_now_real)
        if need_refit:
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
    def _write_log_csv(self, model, line, barcode, verdict, dust_count, max_dia):
        """Background-thread safe: pure file I/O, no Tkinter here."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_to_csv([ts, model, line, barcode, verdict, dust_count, f"{max_dia:.3f}" if max_dia is not None else ""])
        return ts

    def _append_log_line(self, ts, barcode, model, line, verdict, dust_count, max_dia):
        """Main-thread ONLY -- this touches the CTkTextbox. Tkinter widgets
        aren't safe to mutate from a background thread (the GIL doesn't
        protect Tk's underlying C calls the way it protects pure Python),
        so this is always called via root.after from _finish_inspection,
        never directly from the inspection thread."""
        dia_txt = f"{max_dia:.2f}mm" if max_dia is not None else "-"
        line_txt = f"[{ts}] {barcode:<16} | Model:{model or '-':<10} Line:{line or '-':<8} | {verdict:<5} | dust={dust_count} max={dia_txt}\n"
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line_txt)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

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
        self.start_btn.configure(text="Running...", state="disabled", fg_color=BG_CARD_ALT)
        self._set_status("IN PROGRESS", WARNING)
        frame = self.original.copy()
        rois_snapshot = [dict(r) for r in self.rois]
        threading.Thread(target=self._run_inspection_thread, args=(frame, rois_snapshot, barcode), daemon=True).start()

    def _run_inspection_thread(self, frame, rois_snapshot, barcode):
        """Runs entirely off the main/UI thread (started as a daemon Thread
        by start_inspection), so a slow detection pass never blocks the
        camera feed or the UI. Saving images and writing the CSV log also
        happen here (all pure numpy/cv2/file I/O, no Tkinter) -- only the
        final widget update is handed to the main thread at the end.
        """
        s = self.settings
        try:
            _binary, blobs, _stats, _dbg = run_zscore_detection(
                frame, rois_snapshot, s["window"], s["z_thr"], s["min_area"], s["min_circularity"],
                s.get("scale_mm_per_px"), s["min_diameter_mm"])
        except Exception:
            blobs = []

        verdict, log_args = self._save_inspection_artifacts(frame, rois_snapshot, blobs, barcode)

        self.root.after(0, lambda: self._finish_inspection(blobs, verdict, log_args))

    def _save_inspection_artifacts(self, frame, rois_snapshot, blobs, barcode):
        """All cv2/file work for one inspection cycle -- still on the
        background thread, no Tkinter here. Returns (verdict, log_args) --
        log_args gets handed to _append_log_line on the main thread since
        that call touches a Tkinter widget."""
        blobs = list(blobs)  # local copy -- AI-anomaly findings get appended below

        model, line = self.settings.get("model_name"), self.settings.get("line_name")
        model_folder = _safe_folder_name(model)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Full frame, ROI circles only (green) -- kept for traceability,
        # filed under the model's own folder so different phone models
        # (different camera counts/positions) never land in the same place.
        source_model_dir = os.path.join(SOURCE_DIR, model_folder)
        os.makedirs(source_model_dir, exist_ok=True)
        source_disp = frame.copy()
        for roi in rois_snapshot:
            cv2.circle(source_disp, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(source_model_dir, f"source_{ts}_{barcode}.png"), source_disp)

        # Per-ROI clean crops (no circles/markup burned in) -- one per
        # camera position, filed under <Model>/<camN>/. This is the data
        # anomaly-detection training will actually use later: each camera
        # position's own folder, never mixed with another camera's, and
        # never mixed with a different phone model's images.
        _ensure_cam_labels(rois_snapshot)
        h, w = frame.shape[:2]
        for roi in rois_snapshot:
            cx, cy, r = int(roi["cx"]), int(roi["cy"]), int(roi["r"])
            x0, y0 = max(0, cx - r), max(0, cy - r)
            x1, y1 = min(w, cx + r), min(h, cy + r)
            if x1 <= x0 or y1 <= y0:
                continue
            cam_dir = os.path.join(source_model_dir, roi["cam_label"])
            os.makedirs(cam_dir, exist_ok=True)
            crop = frame[y0:y1, x0:x1].copy()
            # Circular crop, not a square bounding-box crop -- the ROI
            # itself is a circle, so a square crop always carries four
            # corners of irrelevant background (module housing, other
            # cameras' edges, glare) that were never actually part of the
            # camera lens area being inspected. Filled black (actual RGB
            # zeroed, not just an alpha channel) rather than made
            # transparent -- an alpha channel is silently dropped by most
            # image loaders (PIL's convert('RGB'), cv2.imread without
            # IMREAD_UNCHANGED) including whatever training pipeline reads
            # these later, so transparency alone wouldn't reliably mask
            # anything. A flat black corner is the same in every saved
            # image (fixed camera + jig), so a position-based model
            # (PatchCore/PaDiM) just learns it as part of the fixed normal
            # background -- it isn't going to get confused into flagging it.
            circ_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            cv2.circle(circ_mask, (cx - x0, cy - y0), r, 255, -1)
            crop[circ_mask == 0] = 0
            cv2.imwrite(os.path.join(cam_dir, f"crop_{ts}_{barcode}.png"), crop)

            # AI anomaly-detection pass, on top of the z-score dust check --
            # runs the exact same clean circular crop just saved above
            # through that camera's trained model (if a technician has set
            # one up in the AI Model tab and enabled AI detection). Uses
            # cv2.countNonZero-free OR logic with the z-score result: this
            # ROI's overall FAIL/PASS is decided by whichever findings end
            # up in `blobs`, so simply appending here is enough -- the
            # verdict below fails if EITHER this OR z-score found something.
            if self.settings.get("ai_enabled") and self.ai_models.is_loaded(model, roi["cam_label"]):
                result = self.ai_models.score(model, roi["cam_label"], crop)
                if result is not None:
                    score_val, is_anomalous = result
                    if is_anomalous:
                        blobs.append({
                            "type": "ai_anomaly", "cx": float(cx), "cy": float(cy), "r": float(r),
                            "diameter_px": None, "diameter_mm": None,
                            "label": f"AI:{roi['cam_label']} {score_val:.3f}",
                        })

        verdict = "FAIL" if blobs else "PASS"
        result_disp = source_disp.copy()
        max_dia = None
        for b in blobs:
            cx, cy, r = int(b["cx"]), int(b["cy"]), int(round(b["r"]))
            color = BLOB_COLOR_BGR.get(b["type"], (0, 0, 255))
            cv2.circle(result_disp, (cx, cy), r + 4, color, 2)
            cv2.putText(result_disp, b["label"], (cx + r + 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
            if b["type"] == "dust" and b.get("diameter_mm") is not None:
                max_dia = max(max_dia or 0.0, b["diameter_mm"])

        verdict_base_dir = RESULTS_NG_DIR if verdict == "FAIL" else RESULTS_OK_DIR
        verdict_dir = os.path.join(verdict_base_dir, model_folder)
        os.makedirs(verdict_dir, exist_ok=True)
        cv2.imwrite(os.path.join(verdict_dir, f"{verdict}_{ts}_{barcode}.png"), result_disp)

        log_ts = self._write_log_csv(model, line, barcode, verdict, len(blobs), max_dia)
        log_args = (log_ts, barcode, model, line, verdict, len(blobs), max_dia)
        return verdict, log_args

    def _build_findings_text(self, blobs):
        if not blobs:
            return "No issues found"
        counts = {}
        for b in blobs:
            counts[b["type"]] = counts.get(b["type"], 0) + 1
        labels = {"dust": "dust", "ai_anomaly": "AI-flagged anomaly"}
        lines = [f"- {counts[t]}x {labels[t]}" for t in ("dust", "ai_anomaly") if counts.get(t)]
        return "\n".join(lines)

    def _finish_inspection(self, blobs, verdict, log_args):
        """Main-thread-only: updates widgets, including the log line (the
        CSV row was already written on the background thread)."""
        self.last_blobs = blobs
        self._append_log_line(*log_args)
        color = SUCCESS if verdict == "PASS" else DANGER
        self._set_status(verdict, color)
        self.findings_var.set(self._build_findings_text(blobs))
        self.count_checked += 1
        if verdict == "PASS":
            self.count_passed += 1
        else:
            self.count_failed += 1
        self._update_stats_display()
        self._render_main_feed()
        self.inspection_running = False
        self.start_btn.configure(text="Start", state="normal", fg_color=ACCENT)
        self.barcode_var.set("")
        self.barcode_entry.focus_set()

    def _update_stats_display(self):
        self.checked_var.set(str(self.count_checked))
        self.passed_var.set(str(self.count_passed))
        self.failed_var.set(str(self.count_failed))
        if self.count_checked > 0:
            fail_rate = 100.0 * self.count_failed / self.count_checked
            pass_rate = 100.0 * self.count_passed / self.count_checked
        else:
            fail_rate = pass_rate = 0.0
        self.failure_rate_var.set(f"{fail_rate:.1f}%")
        self.pass_rate_var.set(f"{pass_rate:.1f}%")

    def reset_counters(self):
        self.count_checked = 0
        self.count_passed = 0
        self.count_failed = 0
        self._update_stats_display()
        self.footer_var.set("Counters reset.")

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
        self._load_configured_ai_models()  # (model_name, cam_label) keys changed -- reload this model's AI models
        self._refresh_ai_tab()

    # ---- ROI & Calibration --------------------------------------------
    def _build_roi_tab(self, tab):
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(14, 6))
        self._btn_secondary(top, "Open Image", self.open_image, width=110).pack(side="left", padx=(0, 6))
        self._btn_secondary(top, "Delete ROI", self.delete_selected_roi, width=100).pack(side="left", padx=6)
        self._btn_secondary(top, "Clear All", self.clear_rois, width=90).pack(side="left", padx=6)
        self._btn_secondary(top, "Save Layout", self.save_roi_layout, width=100).pack(side="left", padx=6)
        self._btn_secondary(top, "Assign Camera Labels", self.assign_camera_labels, width=170).pack(side="left", padx=6)
        self._btn_secondary(top, "Fit", self.fit_roi_view, width=52).pack(side="left", padx=(20, 4))
        self._btn_secondary(top, "Test Detection", self.test_detection_once, width=130).pack(side="right")
        self._btn_secondary(top, "View Pipeline Steps", self.view_pipeline_steps, width=150).pack(side="right", padx=(0, 6))

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        canvas_card = self._card(body)
        canvas_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ctk.CTkLabel(canvas_card, text="click = add/select ROI  -  drag = pan  -  wheel = zoom  -  scroll on selected ROI = resize",
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
        self._refresh_ai_tab()
        self.footer_var.set(f"Active ROI layout: {name}")

    def save_roi_layout(self):
        if not self.rois:
            messagebox.showinfo("Save Layout", "No ROIs to save.")
            return
        name = simpledialog.askstring("Save ROI Layout", "Layout name (e.g. model_A56_main):", parent=self.root)
        if not name:
            return
        _ensure_cam_labels(self.rois)  # every ROI must have a camN label before it's saved
        self._render_roi_canvas()
        with open(os.path.join(ROI_DIR, f"{name}.json"), "w") as f:
            json.dump(self.rois, f, indent=2)
        self._refresh_layout_list()
        self.footer_var.set(f"Saved ROI layout: {name}")

    def assign_camera_labels(self):
        """Lets the technician assign/rename which physical camera position
        (cam1, cam2, ...) each ROI on screen corresponds to. This is what
        keeps a phone model's cameras -- which can differ in FOV/megapixel/
        optics -- from getting mixed up on disk or in training later: every
        saved image is filed under <Model>/<camN>/ using exactly the label
        set here."""
        if not self.rois:
            messagebox.showinfo("Assign Camera Labels", "No ROIs yet -- add some first.")
            return
        _ensure_cam_labels(self.rois)  # default-fill so the dialog always starts with something sensible
        self._render_roi_canvas()  # so the ROI{n} numbers behind the dialog match this dialog's rows right away

        prior_selected = self.selected_idx

        dlg = tk.Toplevel(self.root)
        dlg.title("Assign Camera Labels")
        dlg.configure(bg=BG_CARD)
        dlg.transient(self.root)
        dlg.grab_set()

        ctk.CTkLabel(dlg, text="Click a row (or its box) to highlight that exact ROI on the canvas behind this "
                               "window, then type its camera position (cam1, cam2, ...).",
                     font=self.f_small, text_color=TEXT_MUTED, wraplength=420, justify="left").pack(
            anchor="w", padx=16, pady=(14, 8))

        rows_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        rows_frame.pack(fill="both", expand=True, padx=16)

        def highlight(i):
            self.selected_idx = i
            self._render_roi_canvas()

        entries = []
        for i, roi in enumerate(self.rois):
            row = ctk.CTkFrame(rows_frame, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=f"ROI {i + 1}  (x={int(roi['cx'])}, y={int(roi['cy'])})",
                         font=self.f_small, text_color=TEXT, width=220, anchor="w").pack(side="left")
            var = tk.StringVar(value=roi.get("cam_label") or f"cam{i + 1}")
            entry = ctk.CTkEntry(row, textvariable=var, width=100)
            entry.pack(side="left", padx=(8, 0))
            # Focusing the box -- or just clicking its row -- highlights the
            # matching ROI on the canvas in yellow, the same way clicking an
            # ROI directly does, so there's no more guessing which entry
            # belongs to which circle on screen.
            entry.bind("<FocusIn>", lambda e, i=i: highlight(i))
            row.bind("<Button-1>", lambda e, i=i: highlight(i))
            self._btn_secondary(row, "Highlight", lambda i=i: highlight(i), width=90).pack(side="left", padx=(8, 0))
            entries.append(var)

        status_var = tk.StringVar(value="")
        status_lbl = ctk.CTkLabel(dlg, textvariable=status_var, font=self.f_small, text_color=DANGER)
        status_lbl.pack(anchor="w", padx=16, pady=(6, 0))

        def on_save():
            labels = [v.get().strip() for v in entries]
            if any(not l for l in labels):
                status_var.set("Every ROI needs a non-empty label.")
                return
            if len(set(labels)) != len(labels):
                status_var.set("Camera labels must be unique -- two ROIs have the same label.")
                return
            for roi, lbl in zip(self.rois, labels):
                roi["cam_label"] = lbl
            self.selected_idx = prior_selected
            self._render_roi_canvas()
            self._refresh_ai_tab()
            self.footer_var.set("Camera labels updated.")
            dlg.destroy()

        def on_cancel():
            self.selected_idx = prior_selected
            self._render_roi_canvas()
            dlg.destroy()

        btns = ctk.CTkFrame(dlg, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=14)
        self._btn_primary(btns, "Save", on_save, width=100).pack(side="left")
        self._btn_secondary(btns, "Cancel", on_cancel, width=100).pack(side="left", padx=(8, 0))
        dlg.protocol("WM_DELETE_WINDOW", on_cancel)
        if self.rois:
            highlight(0)

    def delete_selected_roi(self):
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.rois):
            self.rois.pop(self.selected_idx)
            self.selected_idx = None
            self._render_roi_canvas()
            self._render_main_feed()
            self._refresh_ai_tab()

    def clear_rois(self):
        self.rois = []
        self.selected_idx = None
        self._render_roi_canvas()
        self._render_main_feed()
        self._refresh_ai_tab()

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
        self.footer_var.set(f"Loaded {os.path.basename(path)} (static -- live feed paused). Reconnect camera to resume live view.")

    def test_detection_once(self):
        if self.original is None or not self.rois:
            messagebox.showinfo("Test Detection", "Need an image and at least one ROI.")
            return
        self.footer_var.set("Running test detection...")
        frame = self.original.copy()
        rois_snapshot = [dict(r) for r in self.rois]
        threading.Thread(target=self._run_test_detection_thread, args=(frame, rois_snapshot), daemon=True).start()

    def _run_test_detection_thread(self, frame, rois_snapshot):
        """Off the main thread so a slow detection pass never freezes the UI."""
        s = self.settings
        try:
            _binary, blobs, stats, _dbg = run_zscore_detection(
                frame, rois_snapshot, s["window"], s["z_thr"], s["min_area"], s["min_circularity"],
                s.get("scale_mm_per_px"), s["min_diameter_mm"])
        except Exception:
            blobs, stats = [], None
        self.root.after(0, lambda: self._finish_test_detection(blobs, stats))

    def _finish_test_detection(self, blobs, stats):
        self.last_blobs = blobs
        self._render_roi_canvas()
        self._render_main_feed()
        counts = {}
        for b in blobs:
            counts[b["type"]] = counts.get(b["type"], 0) + 1
        msg = ", ".join(f"{n} {t}" for t, n in counts.items()) or "no defects"
        if stats:
            msg += f" | max_z={stats['max_z']:.2f} rejected={stats['rejected']}"
        self.footer_var.set("Test detection: " + msg)

    def view_pipeline_steps(self):
        if self.original is None or not self.rois:
            messagebox.showinfo("Pipeline Steps", "Need an image and at least one ROI.")
            return
        self.footer_var.set("Generating pipeline steps...")
        frame = self.original.copy()
        rois_snapshot = [dict(r) for r in self.rois]
        threading.Thread(target=self._run_pipeline_debug_thread, args=(frame, rois_snapshot), daemon=True).start()

    def _run_pipeline_debug_thread(self, frame, rois_snapshot):
        s = self.settings
        try:
            _binary, _blobs, _stats, debug_images = run_zscore_detection(
                frame, rois_snapshot, s["window"], s["z_thr"], s["min_area"], s["min_circularity"],
                s.get("scale_mm_per_px"), s["min_diameter_mm"], debug=True)
        except Exception:
            debug_images = None
        self.root.after(0, lambda: self._finish_pipeline_debug(debug_images))

    def _finish_pipeline_debug(self, debug_images):
        if not debug_images:
            self.footer_var.set("Pipeline steps: failed to generate.")
            return
        self._pipeline_images = debug_images
        self._populate_pipeline_page()
        self.show_pipeline_page()
        self.footer_var.set("Pipeline steps generated.")

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
        row2.pack(fill="x", padx=18, pady=(0, 8))
        self.min_area_var = tk.StringVar(value=str(self.settings["min_area"]))
        self.min_circ_var = tk.StringVar(value=str(self.settings["min_circularity"]))
        self.min_diam_var = tk.StringVar(value=str(self.settings["min_diameter_mm"]))
        self._field(row2, "Min blob area (px^2)", self.min_area_var, width=90).pack(side="left", padx=(0, 16))
        self._field(row2, "Min circularity (0-1)", self.min_circ_var, width=100).pack(side="left", padx=(0, 16))
        self._field(row2, "Min dust diameter (mm)", self.min_diam_var, width=100).pack(side="left", padx=(0, 16))

        row3 = ctk.CTkFrame(card, fg_color="transparent")
        row3.pack(fill="x", padx=18, pady=(0, 14))
        self._btn_primary(row3, "Save", self._save_detection_settings, width=100).pack(side="left")

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

    # ---- AI anomaly-detection models (per camera) -------------------------
    def _build_ai_tab(self, tab):
        self._ai_row_widgets = {}  # cam_label -> {"status_var":..., "thr_var":...}

        top_card = self._card(tab)
        top_card.pack(fill="x", padx=20, pady=(20, 10))
        ctk.CTkLabel(top_card, text="AI Anomaly Detection", font=self.f_section, text_color=TEXT).pack(anchor="w", padx=18, pady=(16, 6))
        ctk.CTkLabel(
            top_card,
            text=("Optional: on top of the z-score dust check, run each camera's own ROI crop "
                  "through a model trained in Training Studio (exported to ONNX). A camera FAILs "
                  "if EITHER the z-score dust check OR its AI model flags an anomaly. Leave a "
                  "camera's model unset to skip AI for it -- z-score detection keeps running "
                  "regardless of this setting."),
            font=self.f_small, text_color=TEXT_MUTED, wraplength=900, justify="left"
        ).pack(anchor="w", padx=18, pady=(0, 10))

        self.ai_enabled_var = tk.BooleanVar(value=bool(self.settings.get("ai_enabled", False)))
        ctk.CTkSwitch(top_card, text="Enable AI anomaly detection", variable=self.ai_enabled_var,
                      command=self._save_ai_enabled, font=self.f_body,
                      progress_color=ACCENT).pack(anchor="w", padx=18, pady=(0, 16))

        self.ai_rows_card = self._card(tab)
        self.ai_rows_card.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        ctk.CTkLabel(self.ai_rows_card, text="Per-Camera Models", font=self.f_section, text_color=TEXT).pack(anchor="w", padx=18, pady=(16, 10))
        self.ai_rows_container = ctk.CTkFrame(self.ai_rows_card, fg_color="transparent")
        self.ai_rows_container.pack(fill="both", expand=True, padx=18, pady=(0, 16))

        self._refresh_ai_tab()

    def _refresh_ai_tab(self):
        """Rebuilds the per-camera model rows from the currently active
        ROI layout's cam_labels. Called on teaching-page load and whenever
        the ROI layout (and therefore the set of cameras) might have
        changed."""
        for child in self.ai_rows_container.winfo_children():
            child.destroy()
        self._ai_row_widgets = {}

        _ensure_cam_labels(self.rois)
        cam_labels = sorted({roi["cam_label"] for roi in self.rois}) if self.rois else []
        if not cam_labels:
            ctk.CTkLabel(self.ai_rows_container, text="No ROIs defined yet -- set up ROI & Calibration first.",
                         font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w")
            return

        model = self.settings.get("model_name")
        for cam_label in cam_labels:
            row = ctk.CTkFrame(self.ai_rows_container, fg_color=BG_CARD_ALT, corner_radius=10)
            row.pack(fill="x", pady=(0, 8))

            ctk.CTkLabel(row, text=cam_label, font=self.f_body, text_color=TEXT, width=70).pack(side="left", padx=(12, 10), pady=10)

            status_var = tk.StringVar(value=self._ai_status_text(model, cam_label))
            ctk.CTkLabel(row, textvariable=status_var, font=self.f_small, text_color=TEXT_MUTED,
                         wraplength=380, justify="left").pack(side="left", padx=(0, 10), pady=10)

            thr_var = tk.StringVar(value=f"{self.ai_models.threshold(model, cam_label):.4f}")
            self._ai_row_widgets[cam_label] = {"status_var": status_var, "thr_var": thr_var}

            self._field(row, "Threshold", thr_var, width=80).pack(side="left", padx=(0, 10), pady=10)
            self._btn_secondary(row, "Set", lambda c=cam_label: self._set_ai_threshold(c), width=60).pack(side="left", padx=(0, 10), pady=10)
            self._btn_primary(row, "Browse .onnx", lambda c=cam_label: self._browse_ai_model(c), width=120).pack(side="left", padx=(0, 10), pady=10)
            self._btn_secondary(row, "Unload", lambda c=cam_label: self._unload_ai_model(c), width=80).pack(side="left", padx=(0, 12), pady=10)

    def _ai_status_text(self, model, cam_label):
        if not ONNXRUNTIME_AVAILABLE:
            return "onnxruntime not installed in this environment -- AI detection unavailable."
        if self.ai_models.is_loaded(model, cam_label):
            return f"Loaded: {self.ai_models.loaded_path(model, cam_label)}"
        err = self.ai_models.last_error(model, cam_label)
        return f"Not loaded ({err})" if err else "No model set for this camera."

    def _save_ai_enabled(self):
        self.settings["ai_enabled"] = bool(self.ai_enabled_var.get())
        save_settings(self.settings)
        self.footer_var.set(("AI anomaly detection enabled." if self.settings["ai_enabled"]
                              else "AI anomaly detection disabled -- z-score dust detection only."))

    def _browse_ai_model(self, cam_label):
        path = filedialog.askopenfilename(title=f"Select trained ONNX model for {cam_label}",
                                           filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")])
        if not path:
            return
        model = self.settings.get("model_name")
        key = AIModelManager._key(model, cam_label)
        ok, err = self.ai_models.load(model, cam_label, path)
        if ok:
            self.settings.setdefault("ai_model_paths", {})[key] = path
            self.settings.setdefault("ai_thresholds", {})[key] = self.ai_models.threshold(model, cam_label)
            save_settings(self.settings)
            self.footer_var.set(f"AI model loaded for {cam_label}.")
        else:
            messagebox.showerror("AI Model", f"Could not load model for {cam_label}:\n{err}")
        self._refresh_ai_tab()

    def _unload_ai_model(self, cam_label):
        model = self.settings.get("model_name")
        key = AIModelManager._key(model, cam_label)
        self.ai_models.unload(model, cam_label)
        self.settings.get("ai_model_paths", {}).pop(key, None)
        self.settings.get("ai_thresholds", {}).pop(key, None)
        save_settings(self.settings)
        self.footer_var.set(f"AI model unloaded for {cam_label}.")
        self._refresh_ai_tab()

    def _set_ai_threshold(self, cam_label):
        widgets = self._ai_row_widgets.get(cam_label)
        if not widgets:
            return
        try:
            value = float(widgets["thr_var"].get())
        except ValueError:
            messagebox.showerror("AI Model", "Threshold must be a number.")
            return
        model = self.settings.get("model_name")
        self.ai_models.set_threshold(model, cam_label, value)
        key = AIModelManager._key(model, cam_label)
        self.settings.setdefault("ai_thresholds", {})[key] = value
        save_settings(self.settings)
        self.footer_var.set(f"Threshold for {cam_label} set to {value:.4f}.")

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
        self._field(row, "Exposure (us)", self.exposure_var).pack(side="left", padx=(0, 20))
        self._field(row, "Gain", self.gain_var).pack(side="left", padx=(0, 20))
        self._btn_primary(row, "Apply", self._apply_camera_settings, width=90).pack(side="left", padx=(0, 10), pady=(18, 0))
        self._btn_secondary(row, "Connect / Reconnect", self._reconnect_camera, width=160).pack(side="left", pady=(18, 0))
        ctk.CTkLabel(card, text="Gain amplifies sensor noise along with brightness -- prefer raising Exposure over Gain.",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(0, 14))

        storage_card = self._card(tab)
        storage_card.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkLabel(storage_card, text="Storage Location", font=self.f_section, text_color=TEXT).pack(anchor="w", padx=18, pady=(16, 6))
        self.storage_path_var = tk.StringVar(value=STORAGE_DIR)
        ctk.CTkLabel(storage_card, textvariable=self.storage_path_var, font=self.f_small, text_color=TEXT_MUTED,
                     wraplength=520, justify="left").pack(anchor="w", padx=18, pady=(0, 10))
        self._btn_secondary(storage_card, "Select Location", self.select_storage_location, width=160).pack(anchor="w", padx=18, pady=(0, 8))
        ctk.CTkLabel(storage_card, text="source_images/, results/, roi_configs/, and logs/ will be created inside whatever folder you pick. Existing files already saved don't move -- only new saves go to the new location.",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(0, 16))

    def select_storage_location(self):
        new_path = filedialog.askdirectory(title="Select folder for saved images, logs, and ROI configs",
                                            initialdir=STORAGE_DIR)
        if not new_path:
            return
        _apply_storage_base(new_path)
        self.settings["storage_base_path"] = new_path
        save_settings(self.settings)
        self.storage_path_var.set(STORAGE_DIR)
        self._refresh_layout_list()
        self.footer_var.set(f"Storage location changed to: {STORAGE_DIR}")

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
            self.footer_var.set("Camera connected -- live feed running.")
        else:
            messagebox.showwarning("Camera", msg)

    # ---------------------------------------------- ROI canvas interaction
    # (Only the Teaching window's ROI canvas can add/move/resize ROIs -- the
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
            self.rois.append({"cx": ix, "cy": iy, "r": default_r, "cam_label": None})
            self.selected_idx = len(self.rois) - 1
        self._render_roi_canvas()
        self._render_main_feed()

    def _build_roi_canvas_disp(self):
        disp = self.original.copy()
        for i, roi in enumerate(self.rois):
            color = (0, 255, 255) if i == self.selected_idx else (0, 255, 0)
            cv2.circle(disp, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), color, 2)
            cv2.circle(disp, (int(roi["cx"]), int(roi["cy"])), 5, color, -1)
            # Always show BOTH the ROI's on-screen number and its assigned
            # camera label (e.g. "ROI2: cam2") so the two never need to be
            # cross-referenced by hand -- this is the same "ROI N" wording
            # used in the Assign Camera Labels dialog, so a row there maps
            # straight onto what's drawn here. A filled background box
            # behind the text keeps it readable over any image content.
            label_text = f"ROI{i + 1}: {roi.get('cam_label')}" if roi.get("cam_label") else f"ROI{i + 1}: (unlabeled)"
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thick)
            tx = int(roi["cx"]) + 12
            ty = int(roi["cy"]) - 12
            cv2.rectangle(disp, (tx - 4, ty - th - 6), (tx + tw + 4, ty + baseline + 4), (0, 0, 0), -1)
            cv2.putText(disp, label_text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)
        for b in self.last_blobs:
            cx, cy, r = int(b["cx"]), int(b["cy"]), int(round(b["r"]))
            cv2.circle(disp, (cx, cy), r + 4, BLOB_COLOR_BGR.get(b["type"], (0, 0, 255)), 2)
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
