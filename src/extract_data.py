from pathlib import Path

import cv2 as cv
import numpy as np

from .calibration import apply_calibration
from .utils import (
    COLOR_OPTIONS,
    crop_roi,
    mask_colored_curve,
    overlay_polyline,
    trace_curve,
)

# Saturation threshold to keep colored pixels and drop gray UI/grid
SAT_MIN = 60  # 0..255
VAL_MIN = 40  # avoid very dark noise


# ------------------------------
# Main extraction function
# ------------------------------
def extract_one(
    image_path,
    roi_rel,
    calib=None,
    rect=None,
    save_csv=True,
    preview=True,
    line_color=None,  # can be str (name) or tuple/list (BGR)
    time_range=None,  # NEW: (t_min, t_max) or (t_min, None) or (None, t_max)
):
    img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")

    roi, rect_auto = crop_roi(img, roi_rel)
    rect = rect if rect is not None else rect_auto

    if isinstance(line_color, str):
        name = line_color.lower()
        if name not in COLOR_OPTIONS:
            raise ValueError(
                f"Unknown color name '{line_color}'. Available: {', '.join(COLOR_OPTIONS.keys())}"
            )
        line_color_bgr = COLOR_OPTIONS[name]
    elif isinstance(line_color, (list, tuple)) and len(line_color) == 3:
        line_color_bgr = tuple(line_color)
    else:
        line_color_bgr = None

    # --- Build mask (color-aware if provided) ---
    if line_color_bgr is not None:
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        target = np.uint8([[line_color_bgr]])
        target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)[0, 0]
        h, s, v = map(int, target_hsv)
        h_tol, s_tol, v_tol = 10, 80, 80
        lower = np.array(
            [max(0, h - h_tol), max(50, s - s_tol), max(50, v - v_tol)], dtype=np.uint8
        )
        upper = np.array([min(179, h + h_tol), 255, 255], dtype=np.uint8)
        mask = cv.inRange(hsv, lower, upper)
        mask = cv.medianBlur(mask, 3)
    else:
        mask = mask_colored_curve(roi, SAT_MIN, VAL_MIN)

    ys = trace_curve(mask)

    # Build x pixel coordinates within ROI
    w = mask.shape[1]
    x_px_all = np.arange(w, dtype=float)
    y_px_all = ys

    # Keep only columns where ys is finite
    valid = np.isfinite(y_px_all)
    x_px = x_px_all[valid]
    y_px = y_px_all[valid]
    valid_idx = np.where(valid)[0]  # indices into the full ROI width

    if calib is None:
        raise RuntimeError("Calibration is required. Run collect_calibration first.")

    # Map pixels -> data
    t_data, T_data = apply_calibration(x_px, y_px, calib)

    # Optional time filtering
    if time_range is not None:
        t_min, t_max = (
            time_range if isinstance(time_range, (tuple, list)) else (time_range, None)
        )
        keep = np.ones_like(t_data, dtype=bool)
        if t_min is not None:
            keep &= t_data >= t_min
        if t_max is not None:
            keep &= t_data <= t_max

        t_data = t_data[keep]
        T_data = T_data[keep]

        # For overlay: show only filtered samples
        filtered_idx = valid_idx[keep]  # indices in the original ROI columns
        ys_vis = np.full_like(ys, np.nan, dtype=float)
        ys_vis[filtered_idx] = ys[filtered_idx]
    else:
        # No time filter: overlay entire traced curve
        ys_vis = ys

    # Save CSV next to image
    if save_csv:
        out_csv = Path(image_path).with_suffix(".csv")
        arr = np.column_stack([t_data, T_data])
        np.savetxt(
            out_csv,
            arr,
            delimiter=",",
            header="time,temperature",
            comments="",
            fmt="%.6f",
        )
        print(f"Saved {out_csv}")

    # QC overlay reflects the filtering
    if preview:
        overlay = overlay_polyline(img.copy(), rect, ys_vis, color=(0, 0, 0))
        out_png = Path(image_path).with_name(Path(image_path).stem + "_overlay.png")
        cv.imwrite(str(out_png), overlay)
        print(f"Saved overlay {out_png}")

    return t_data, T_data


if __name__ == "__main__":
    roi_rel = (0.2520833333333333, 0.14537037037037037, 0.7453125, 0.5472222222222223)
    calib = (
        np.float64(0.0037764350453172208),
        np.float64(-0.09667673716012185),
        np.float64(0.006711409395973154),
        np.float64(3.610738255033556),
        0.0,
    )
    rect = (484, 157, 1915, 748)
    extract_one(
        "without_label.png",
        roi_rel,
        calib=calib,
        rect=rect,
        line_color=(0, 255, 255),  # yellow in BGR
        time_range=(0.1, None),
    )
