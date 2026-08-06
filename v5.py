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

Single-threaded detection call per inspection (dust/thread/glue
classification via run_zscore_detection), still run on a background
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
VINYL_COLOR = "#a855f7"  # distinct from PASS/FAIL/IN-PROGRESS so it reads as its own state

# BGR (for cv2 drawing, not the hex UI colors above) per defect type
BLOB_COLOR_BGR = {
    "dust": (0, 0, 255),      # red
    "thread": (0, 165, 255),  # amber
    "glue": (255, 0, 255),    # magenta
}
VINYL_COLOR_BGR = (245, 85, 168)  # pink-purple, kept distinct from glue's magenta

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
RESULTS_VINYL_DIR = os.path.join(RESULTS_DIR, "VINYL")
ROI_DIR = os.path.join(STORAGE_DIR, "roi_configs")
LOGS_DIR = os.path.join(STORAGE_DIR, "logs")
LOG_CSV_PATH = os.path.join(LOGS_DIR, "inspection_log.csv")
SETTINGS_PATH = os.path.join(STORAGE_DIR, "settings.json")

for _d in (STORAGE_DIR, SOURCE_DIR, RESULTS_DIR, RESULTS_NG_DIR, RESULTS_OK_DIR,
           RESULTS_VINYL_DIR, ROI_DIR, LOGS_DIR):
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
    "min_aspect_ratio": 3.0,
    "max_thin_width_px": 8.0,
    "max_arc_fit_residual": 0.12,
    "gap_bridge_px": 11.0,
    "z_thr_low": 1.5,
    "vinyl_tolerance_px": 15.0,
    "vinyl_strength_thr": 6.0,
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
def _fit_circle(pts):
    """Algebraic (Kasa) least-squares circle fit through a set of 2D
    points. Returns (radius, rms_residual) -- residual is how far the
    points typically sit from that best-fit circle (0 = perfect fit).
    Returns (None, None) if the fit is degenerate."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None, None
    a_c, b_c, c = sol
    r2 = c + a_c ** 2 + b_c ** 2
    if r2 <= 0:
        return None, None
    r = np.sqrt(r2)
    dists = np.sqrt((x - a_c) ** 2 + (y - b_c) ** 2)
    resid = float(np.sqrt(np.mean((dists - r) ** 2)))
    return float(r), resid



def run_zscore_detection(bgr, rois, window, z_thr, min_area=4.0, min_circularity=0.55,
                          mm_per_px=None, min_diameter_mm=0.0,
                          min_aspect_ratio=3.0, max_thin_width_px=8.0, max_arc_fit_residual=0.12,
                          gap_bridge_px=7.0, z_thr_low=1.5, debug=False):
    """Core Z-score math is UNCHANGED: local Z-score via boxFilter mean/std,
    thresholded inside the union of all circular ROI masks.

    Every surviving connected blob is then classified by SHAPE:
      - round + big + circular enough                          -> DUST
      - elongated (length/width >= min_aspect_ratio), thin
        (width <= max_thin_width_px), and its points do NOT fit
        a large circle well                                     -> THREAD/FIBER
      - elongated + thin AND its points cleanly fit a circle
        much bigger than itself                                 -> ARC, rejected
      - elongated + wide (width > max_thin_width_px)            -> GLUE

    Why fit-to-a-circle instead of just "is it curved": a real optical
    reflection or lens/coating edge is a segment of one PHYSICALLY FIXED
    circle (the lens/module geometry), so it fits a single circle almost
    perfectly. A real thread that fell and landed on the module can bend
    into pretty much any shape -- but it essentially never happens to
    trace a clean arc of one big fixed-radius circle. So "is this shape
    curved" is the wrong question (a real thread is very often curved
    too); "does this shape trace a genuine large circle" is the right one,
    and that's what's tested here, on the actual contour points -- not on
    a straight bounding-box proxy, which breaks down for curved shapes.

    This replaced an earlier blanket morphological opening: that approach
    reliably erased thin curved lens/reflection arcs, but it also erased
    genuinely thin thread/fiber contamination, since both are "thin"
    shapes geometrically -- one blunt filter can't tell them apart. A
    light CLOSING is used instead (bridges tiny gaps in a thin shape's
    raw mask, never erases it), and classification happens by shape
    afterwards, on the whole contour.

    If the app has been calibrated (mm_per_px set via two-point calibration),
    sizes are also reported in mm; anything under min_diameter_mm (for dust)
    is rejected too. Without calibration this step is skipped.

    Returns (binary, blobs, stats, debug_images). binary is the cleaned
    mask; blobs is a list of classified blob dicts (each with "type":
    "dust"/"thread"/"glue", position, size, and a ready-to-draw "label"
    string); stats is summary counts. debug_images is None unless
    debug=True, in which case it's an ordered dict of intermediate
    pipeline images (grayscale, z-score heatmap, raw threshold, after
    gap-bridging, final classified result) for the pipeline-steps viewer.
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

    # Hysteresis thresholding (same idea as Canny edge detection): a pixel
    # only needs to clear the LOW bar if it's connected to at least one
    # pixel that clears the FULL z_thr bar. A real thread's contrast often
    # fades along its length -- the z-score heatmap still shows it clearly
    # as elevated the whole way, just not always above z_thr. A flat
    # z_thr >= cutoff would keep only the strong core and drop the faint
    # tail; hysteresis recovers the whole connected thread as long as some
    # part of it is strongly above threshold, while still rejecting
    # isolated weak noise that never touches a strong pixel anywhere.
    strong = ((zscore >= z_thr) & (mask == 255)).astype(np.uint8)
    weak = ((zscore >= z_thr_low) & (mask == 255)).astype(np.uint8)
    _num_labels, weak_labels = cv2.connectedComponents(weak, connectivity=8)
    strong_label_ids = set(np.unique(weak_labels[strong == 1]))
    strong_label_ids.discard(0)
    if strong_label_ids:
        raw = (np.isin(weak_labels, list(strong_label_ids)) * 255).astype(np.uint8)
    else:
        raw = np.zeros_like(weak, dtype=np.uint8)

    # Bridge gaps for CONNECTIVITY ONLY via dilation -- NOT a full closing.
    # Closing (dilate then erode) turned out to be unreliable here: the
    # erode half removes a thin bridge just as easily as it would remove
    # the thread's own thin width, so it often failed to reconnect a
    # fragmented thread at all. Dilating (no erode) reliably links nearby
    # fragments; shape is then measured from the ORIGINAL undilated pixels
    # within each linked region, so linking doesn't inflate the measured
    # width. (Trade-off: a large gap_bridge_px can also fuse two separate
    # nearby real defects into one -- tune it to the largest real gap you
    # see in a fragmented thread, not much more.)
    # a single dilate with a large kernel is catastrophically slow (a
    # 301x301 kernel measured 3+ seconds on just a 2000x2000 test frame,
    # worse on a real full-res camera image) -- iterating a small 3x3
    # kernel reaches the same effective radius in far less time (~30x
    # faster measured), with identical connectivity outcomes, since all
    # that actually matters here is whether two nearby fragments end up
    # merged, not the exact dilated pixel shape
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    iterations = max(1, int(round(gap_bridge_px / 2.0)))
    linked = cv2.dilate(raw, small_kernel, iterations=iterations)

    link_contours, _ = cv2.findContours(linked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blobs = []
    binary = np.zeros_like(raw)
    rejected = 0

    for lc in link_contours:
        region = np.zeros_like(raw)
        cv2.drawContours(region, [lc], -1, 255, -1)
        original = cv2.bitwise_and(raw, region)  # hysteresis-recovered pixels -- used for area/length
        strong_part = cv2.bitwise_and(strong * 255, region)  # strong-only pixels -- used for shape/width/circularity
        area = int(cv2.countNonZero(original))
        if area < min_area:
            rejected += 1
            continue

        frag_contours, _ = cv2.findContours(original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not frag_contours:
            rejected += 1
            continue
        perimeter = sum(cv2.arcLength(fc, True) for fc in frag_contours)
        length_px = perimeter / 2.0

        # Shape decisions (round vs elongated, width) come from the STRONG
        # core only, NOT the full hysteresis region. hysteresis's weak
        # threshold is meant to extend LENGTH/continuity along a faint
        # tail -- but any weak halo isn't perfectly symmetric or perfectly
        # thread-width-shaped, so measuring width/circularity off the full
        # (weak-included) region can inflate a genuinely thin thread's
        # width past the glue cutoff, or make a genuinely round dust
        # speck's circularity drop enough to look elongated. Confirmed
        # with a direct test: a thin (4px) strong core measured 8.4px wide
        # once its weak halo was included -- right at the glue threshold.
        strong_contours, _ = cv2.findContours(strong_part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        strong_area = int(cv2.countNonZero(strong_part))
        if not strong_contours or strong_area < 1:
            rejected += 1  # shouldn't normally happen -- a linked region always contains >=1 strong pixel
            continue
        strong_perimeter = sum(cv2.arcLength(sc, True) for sc in strong_contours)
        circularity = (4 * np.pi * strong_area / (strong_perimeter * strong_perimeter)) if strong_perimeter > 0 else 0.0

        if circularity >= min_circularity and len(strong_contours) == 1:
            # only a genuinely solid single blob counts as dust -- a linked
            # group of several small fragments is never one dust speck
            (cx, cy), r = cv2.minEnclosingCircle(strong_contours[0])
            diameter_px = 2.0 * r
            diameter_mm = diameter_px * mm_per_px if mm_per_px else None
            if diameter_mm is not None and diameter_mm < min_diameter_mm:
                rejected += 1
                continue
            label = f"{diameter_mm:.2f}mm" if diameter_mm is not None else f"{diameter_px:.0f}px"
            blobs.append({"type": "dust", "cx": float(cx), "cy": float(cy), "r": float(max(r, 3.0)),
                          "diameter_px": float(diameter_px), "diameter_mm": diameter_mm, "label": label})
            binary = cv2.bitwise_or(binary, original)
            continue

        # not round -- is it a real elongated shape at all? width from the
        # strong core only, for the same reason as circularity above.
        dist = cv2.distanceTransform(strong_part, cv2.DIST_L2, 5)
        width_px = 2.0 * float(dist.max())
        if length_px < 1e-6 or (length_px / max(width_px, 1e-6)) < min_aspect_ratio:
            rejected += 1
            continue  # neither round nor elongated enough -- ambiguous, drop

        all_pts = np.vstack([fc.reshape(-1, 2) for fc in frag_contours]).astype(np.float64)
        rcx, rcy = float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean())
        r_draw = max(length_px / 2.0, 3.0)

        if width_px > max_thin_width_px:
            btype = "glue"
        else:
            is_arc = False
            if len(all_pts) >= 8:
                fit_r, resid = _fit_circle(all_pts)
                if fit_r is not None:
                    norm_resid = resid / max(length_px, 1e-6)
                    radius_ratio = fit_r / max(length_px, 1e-6)
                    if norm_resid <= max_arc_fit_residual and 1.2 <= radius_ratio <= 25.0:
                        is_arc = True
            if is_arc:
                rejected += 1
                continue  # a clean fit to one circle much bigger than itself -- a real
                          # optical reflection/rim (fixed lens geometry), not contamination
            btype = "thread"

        if mm_per_px:
            label = f"{btype} {length_px * mm_per_px:.2f}x{width_px * mm_per_px:.2f}mm"
        else:
            label = f"{btype} {length_px:.0f}x{width_px:.0f}px"

        blobs.append({"type": btype, "cx": rcx, "cy": rcy, "r": float(r_draw),
                      "length_px": float(length_px), "width_px": float(width_px), "label": label})
        binary = cv2.bitwise_or(binary, original)

    stats = None
    if mask.any():
        roi_z = zscore[mask == 255]
        stats = {"max_z": float(roi_z.max()), "mean_z": float(roi_z.mean()),
                 "dust_px": int((binary == 255).sum()), "dust_count": len(blobs),
                 "rejected": rejected}

    debug_images = None
    if debug:
        debug_images = {}
        debug_images["1 Grayscale"] = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        z_ceiling = max(z_thr * 3.0, 1.0)
        z_norm = (np.clip(zscore, 0, z_ceiling) / z_ceiling * 255).astype(np.uint8)
        debug_images["2 Z-score heatmap"] = cv2.applyColorMap(z_norm, cv2.COLORMAP_INFERNO)
        debug_images["3 Weak threshold (context, low bar)"] = cv2.cvtColor(weak * 255, cv2.COLOR_GRAY2BGR)
        debug_images["4 Strong threshold (z_thr)"] = cv2.cvtColor(strong * 255, cv2.COLOR_GRAY2BGR)
        debug_images["5 After hysteresis (before bridging)"] = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        debug_images["6 After gap-bridging"] = cv2.cvtColor(linked, cv2.COLOR_GRAY2BGR)
        result_disp = bgr.copy()
        for b in blobs:
            cx, cy, r = int(b["cx"]), int(b["cy"]), int(round(b["r"]))
            color = BLOB_COLOR_BGR.get(b["type"], (0, 0, 255))
            cv2.circle(result_disp, (cx, cy), r + 4, color, 2)
            cv2.putText(result_disp, b["label"], (cx + r + 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        debug_images["7 Final classified result"] = result_disp

    return binary, blobs, stats, debug_images



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
        self.cam = CameraManager()

        # shared state
        self.original = None
        self.using_static_image = False
        self.rois = []                 # active ROI set used by the main window
        self.last_blobs = []           # last inspection's accepted blobs: dust/thread/glue
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
        self._auto_connect_camera()
        self._poll_live()

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
        tab_cam = tabs.add("Camera")

        self._build_model_tab(tab_model)
        self._build_roi_tab(tab_roi)
        self._build_detection_tab(tab_detect)
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
                s.get("scale_mm_per_px"), s["min_diameter_mm"],
                s["min_aspect_ratio"], s["max_thin_width_px"], s["max_arc_fit_residual"], s["gap_bridge_px"], s["z_thr_low"])
        except Exception:
            blobs = []

        verdict, log_args = self._save_inspection_artifacts(frame, rois_snapshot, blobs, barcode)

        self.root.after(0, lambda: self._finish_inspection(blobs, verdict, log_args))

    def _save_inspection_artifacts(self, frame, rois_snapshot, blobs, barcode):
        """All cv2/file work for one inspection cycle -- still on the
        background thread, no Tkinter here. Returns (verdict, log_args) --
        log_args gets handed to _append_log_line on the main thread since
        that call touches a Tkinter widget."""
        verdict = "FAIL" if blobs else "PASS"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_disp = frame.copy()
        for roi in rois_snapshot:
            cv2.circle(source_disp, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(SOURCE_DIR, f"source_{ts}_{barcode}.png"), source_disp)

        result_disp = source_disp.copy()
        max_dia = None
        for b in blobs:
            cx, cy, r = int(b["cx"]), int(b["cy"]), int(round(b["r"]))
            color = BLOB_COLOR_BGR.get(b["type"], (0, 0, 255))
            cv2.circle(result_disp, (cx, cy), r + 4, color, 2)
            cv2.putText(result_disp, b["label"], (cx + r + 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
            if b["type"] == "dust" and b.get("diameter_mm") is not None:
                max_dia = max(max_dia or 0.0, b["diameter_mm"])

        verdict_dir = RESULTS_NG_DIR if verdict == "FAIL" else RESULTS_OK_DIR
        cv2.imwrite(os.path.join(verdict_dir, f"{verdict}_{ts}_{barcode}.png"), result_disp)

        model, line = self.settings.get("model_name"), self.settings.get("line_name")
        log_ts = self._write_log_csv(model, line, barcode, verdict, len(blobs), max_dia)
        log_args = (log_ts, barcode, model, line, verdict, len(blobs), max_dia)
        return verdict, log_args

    def _build_findings_text(self, blobs):
        if not blobs:
            return "No issues found"
        counts = {}
        for b in blobs:
            counts[b["type"]] = counts.get(b["type"], 0) + 1
        lines = [f"- {counts[t]}x {t}" for t in ("dust", "thread", "glue") if counts.get(t)]
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
        """Off the main thread -- a slow setting (e.g. a big gap_bridge_px)
        must never be able to freeze the UI, no matter what's dialed in."""
        s = self.settings
        try:
            _binary, blobs, stats, _dbg = run_zscore_detection(
                frame, rois_snapshot, s["window"], s["z_thr"], s["min_area"], s["min_circularity"],
                s.get("scale_mm_per_px"), s["min_diameter_mm"],
                s["min_aspect_ratio"], s["max_thin_width_px"], s["max_arc_fit_residual"], s["gap_bridge_px"], s["z_thr_low"])
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
                s.get("scale_mm_per_px"), s["min_diameter_mm"],
                s["min_aspect_ratio"], s["max_thin_width_px"], s["max_arc_fit_residual"], s["gap_bridge_px"], s["z_thr_low"],
                debug=True)
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
        row3.pack(fill="x", padx=18, pady=(0, 8))
        self.min_aspect_var = tk.StringVar(value=str(self.settings["min_aspect_ratio"]))
        self.max_thin_var = tk.StringVar(value=str(self.settings["max_thin_width_px"]))
        self.arc_fit_var = tk.StringVar(value=str(self.settings["max_arc_fit_residual"]))
        self._field(row3, "Min elongation ratio", self.min_aspect_var, width=90).pack(side="left", padx=(0, 16))
        self._field(row3, "Max thread/arc width (px)", self.max_thin_var, width=110).pack(side="left", padx=(0, 16))
        self._field(row3, "Arc-fit tolerance (0-1)", self.arc_fit_var, width=100).pack(side="left", padx=(0, 16))
        ctk.CTkLabel(card, text="A thin elongated blob is rejected as an arc (lens/coating reflection) only if it cleanly fits ONE circle much bigger than itself -- a real thread can curve too, but essentially never traces a perfect large arc, so curvature alone no longer disqualifies it. Lower the tolerance to be stricter about what counts as a clean circle fit.",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(0, 8))

        row4 = ctk.CTkFrame(card, fg_color="transparent")
        row4.pack(fill="x", padx=18, pady=(0, 8))
        self.gap_bridge_var = tk.StringVar(value=str(self.settings["gap_bridge_px"]))
        self.z_thr_low_var = tk.StringVar(value=str(self.settings["z_thr_low"]))
        self._field(row4, "Gap bridge (px)", self.gap_bridge_var, width=90).pack(side="left", padx=(0, 16))
        self._field(row4, "Weak threshold (hysteresis)", self.z_thr_low_var, width=100).pack(side="left", padx=(0, 16))
        self._btn_primary(row4, "Save", self._save_detection_settings, width=100).pack(side="left", pady=(18, 0))
        ctk.CTkLabel(card, text="Two different fixes for a broken-up thread: Weak threshold recovers a FAINT tail that's still elevated in the z-score but below the main threshold (hysteresis: a weak pixel counts if it touches a strong one, same trick Canny edge detection uses). Gap bridge instead links pieces separated by a true GAP with no signal at all -- it can't invent detail hysteresis is the fix when the heatmap actually shows the thread, just not brightly enough everywhere; gap bridge is the fix when there's a real empty stretch in between.",
                     font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=18, pady=(0, 14))

    def _save_detection_settings(self):
        try:
            self.settings["window"] = int(self.window_var.get())
            self.settings["z_thr"] = float(self.zthr_var.get())
            self.settings["default_radius"] = int(self.radius_var.get())
            self.settings["min_area"] = float(self.min_area_var.get())
            self.settings["min_circularity"] = float(self.min_circ_var.get())
            self.settings["min_diameter_mm"] = float(self.min_diam_var.get())
            self.settings["min_aspect_ratio"] = float(self.min_aspect_var.get())
            self.settings["max_thin_width_px"] = float(self.max_thin_var.get())
            self.settings["max_arc_fit_residual"] = float(self.arc_fit_var.get())
            self.settings["gap_bridge_px"] = float(self.gap_bridge_var.get())
            self.settings["z_thr_low"] = float(self.z_thr_low_var.get())
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
        self._field(row, "Exposure (us)", self.exposure_var).pack(side="left", padx=(0, 20))
        self._field(row, "Gain", self.gain_var).pack(side="left", padx=(0, 20))
        self._btn_primary(row, "Apply", self._apply_camera_settings, width=90).pack(side="left", padx=(0, 10), pady=(18, 0))
        self._btn_secondary(row, "Connect / Reconnect", self._reconnect_camera, width=160).pack(side="left", pady=(18, 0))
        ctk.CTkLabel(card, text="Gain amplifies sensor noise along with brightness -- prefer raising Exposure over Gain.",
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
