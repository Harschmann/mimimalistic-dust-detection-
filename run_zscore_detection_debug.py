"""
run_zscore_detection_debug.py
------------------------------
Debug-instrumented copy of the dust_inspector_app.py detection function.

Same math, same defaults (window=100, z_thr=3.0, min_area=4.0,
min_circularity=0.55) -- the only addition is an optional `debug_dir`
argument. When you pass a folder path, every intermediate step of the
pipeline is written there as its own PNG, numbered in processing order,
so you can see exactly what each stage did to your image.

USAGE
-----
    python run_zscore_detection_debug.py path/to/your_image.jpg

Optional flags:
    --window 100          # boxFilter window size (odd number; even+1 auto-fixed)
    --zthr 3.0            # Z-score threshold
    --min-area 4.0         # px^2, minimum accepted blob area
    --min-circularity 0.55 # 0-1, minimum accepted blob circularity
    --roi cx,cy,r          # circular ROI in pixels (default: centered, covers most of the image)
    --mm-per-px 0.0        # if you have a calibration scale, pass it here (0 = uncalibrated)
    --min-diameter-mm 0.1  # only applied if --mm-per-px > 0
    --out debug_output     # output folder (default: <image_name>_debug/)

Every file this produces is real output from your actual image -- nothing
here is synthetic or illustrative.
"""

import os
import sys
import argparse

import cv2
import numpy as np


def norm_u8(arr):
    """Stretch an arbitrary float array to 0-255 for visualization only
    (this is display scaling, it does not change the detection math)."""
    a = arr.astype(np.float32)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return (a * 255).astype(np.uint8)


def run_zscore_detection(bgr, rois, window, z_thr, min_area=4.0, min_circularity=0.55,
                          mm_per_px=None, min_diameter_mm=0.0, debug_dir=None):
    """Core Z-score math is UNCHANGED from dust_inspector_app.py: local
    Z-score via boxFilter mean/std, thresholded inside the union of all
    circular ROI masks, cleaned up with a morphological opening plus a
    circularity/area/diameter filter.

    If debug_dir is given, every intermediate step is written there as a
    numbered PNG so you can inspect exactly what happened to the image.
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_raw_input.png"), bgr)

    win = window if window % 2 == 1 else window + 1
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "02_grayscale.png"), gray.astype(np.uint8))

    local_mean = cv2.boxFilter(gray, -1, (win, win))
    local_mean_sq = cv2.boxFilter(gray * gray, -1, (win, win))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0))
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "03_local_mean.png"), norm_u8(local_mean))
        cv2.imwrite(os.path.join(debug_dir, "04_local_std.png"), norm_u8(local_std))

    zscore = np.where(local_std > 1e-5, (gray - local_mean) / local_std, 0.0)
    if debug_dir:
        z_vis = np.clip((zscore + 4) / 8 * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "05_zscore_heatmap.png"), cv2.applyColorMap(z_vis, cv2.COLORMAP_JET))

    mask = np.zeros(gray.shape, dtype=np.uint8)
    for roi in rois:
        cv2.circle(mask, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), 255, -1)
    if debug_dir:
        roi_vis = bgr.copy()
        for roi in rois:
            cv2.circle(roi_vis, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "06_roi_overlay.png"), roi_vis)

    raw = np.where((zscore >= z_thr) & (mask == 255), 255, 0).astype(np.uint8)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "07_raw_threshold_mask.png"), raw)
        raw_overlay = bgr.copy()
        raw_overlay[raw == 255] = (0, 255, 255)
        cv2.imwrite(os.path.join(debug_dir, "07b_raw_threshold_overlay.png"), raw_overlay)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "08_morphology_opened.png"), opened)
        opened_overlay = bgr.copy()
        opened_overlay[opened == 255] = (0, 255, 255)
        cv2.imwrite(os.path.join(debug_dir, "08b_morphology_opened_overlay.png"), opened_overlay)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    rejected_shapes = []
    binary = np.zeros_like(opened)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            rejected_shapes.append(c)
            continue
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            rejected_shapes.append(c)
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        diameter_px = 2.0 * r
        diameter_mm = diameter_px * mm_per_px if mm_per_px else None
        if diameter_mm is not None and diameter_mm < min_diameter_mm:
            rejected_shapes.append(c)
            continue
        circles.append({"cx": float(cx), "cy": float(cy), "r": float(max(r, 3.0)),
                         "diameter_px": float(diameter_px), "diameter_mm": diameter_mm})
        cv2.drawContours(binary, [c], -1, 255, -1)

    if debug_dir:
        shape_vis = bgr.copy()
        for c in rejected_shapes:
            cv2.drawContours(shape_vis, [c], -1, (0, 0, 255), 2)  # red = rejected
        for d in circles:
            cv2.circle(shape_vis, (int(d["cx"]), int(d["cy"])), int(d["r"]) + 4, (0, 255, 0), 2)  # green = accepted
        cv2.imwrite(os.path.join(debug_dir, "09_shape_filter_accept_reject.png"), shape_vis)

        final = bgr.copy()
        for roi in rois:
            cv2.circle(final, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (0, 255, 0), 1)
        for d in circles:
            cx, cy, r = int(d["cx"]), int(d["cy"]), int(round(d["r"]))
            cv2.circle(final, (cx, cy), r + 4, (0, 0, 255), 2)
            label = f"{d['diameter_mm']:.2f}mm" if d.get("diameter_mm") is not None else f"{int(round(d['diameter_px']))}px"
            cv2.putText(final, label, (cx + r + 10, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        verdict = "FAIL" if circles else "PASS"
        color = (0, 0, 255) if circles else (0, 200, 0)
        cv2.putText(final, f"Verdict: {verdict}  (dust_count={len(circles)})", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(debug_dir, "10_final_annotated.png"), final)

    stats = None
    if mask.any():
        roi_z = zscore[mask == 255]
        stats = {"max_z": float(roi_z.max()), "mean_z": float(roi_z.mean()),
                 "dust_px": int((binary == 255).sum()), "dust_count": len(circles),
                 "rejected": len(rejected_shapes)}
    return binary, circles, stats


def main():
    p = argparse.ArgumentParser(description="Run the dust-detection pipeline on an image and save every intermediate step.")
    p.add_argument("image", help="Path to the input image")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--zthr", type=float, default=3.0)
    p.add_argument("--min-area", type=float, default=4.0)
    p.add_argument("--min-circularity", type=float, default=0.55)
    p.add_argument("--roi", type=str, default=None, help="cx,cy,r in pixels (default: centered, radius = 90% of half the shorter side)")
    p.add_argument("--mm-per-px", type=float, default=0.0)
    p.add_argument("--min-diameter-mm", type=float, default=0.1)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    data = np.fromfile(args.image, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Could not load image: {args.image}")
        sys.exit(1)

    h, w = bgr.shape[:2]
    if args.roi:
        cx, cy, r = (float(v) for v in args.roi.split(","))
    else:
        cx, cy = w / 2, h / 2
        r = 0.9 * min(w, h) / 2
    rois = [{"cx": cx, "cy": cy, "r": r}]

    out_dir = args.out or (os.path.splitext(os.path.basename(args.image))[0] + "_debug")

    mm_per_px = args.mm_per_px if args.mm_per_px > 0 else None
    binary, circles, stats = run_zscore_detection(
        bgr, rois, args.window, args.zthr, args.min_area, args.min_circularity,
        mm_per_px, args.min_diameter_mm, debug_dir=out_dir)

    print(f"Saved step-by-step images to: {out_dir}/")
    print(f"dust_count = {len(circles)}  verdict = {'FAIL' if circles else 'PASS'}")
    if stats:
        print(f"max_z={stats['max_z']:.2f}  mean_z={stats['mean_z']:.2f}  rejected_shapes={stats['rejected']}")
    for d in circles:
        size = f"{d['diameter_mm']:.2f}mm" if d.get("diameter_mm") is not None else f"{d['diameter_px']:.1f}px"
        print(f"  blob at ({d['cx']:.0f},{d['cy']:.0f})  size={size}")


if __name__ == "__main__":
    main()
