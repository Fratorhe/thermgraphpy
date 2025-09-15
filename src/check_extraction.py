from pathlib import Path

import cv2 as cv
import numpy as np


def _rect_from_roi(img, roi_rel):
    H, W = img.shape[:2]
    x0 = int(W * roi_rel[0])
    y0 = int(H * roi_rel[1])
    x1 = int(x0 + W * roi_rel[2])
    y1 = int(y0 + H * roi_rel[3])
    return (x0, y0, x1, y1)


def data_to_pixels(t, T, calib):
    """
    Inverse of apply_calibration.
    Inputs are arrays t (time), T (temperature).
    Returns pixel coords within ROI: x_px, y_px (float).
    """
    ax, bx, ay, by, y0_px = calib
    if abs(ax) < 1e-12 or abs(ay) < 1e-12:
        raise ValueError("Invalid calibration: near-zero slope.")
    x_px = (t - bx) / ax
    y_px = y0_px - (T - by) / ay  # inverse of T = ay*(y0 - y_px) + by
    return x_px, y_px


def overlay_from_csv(
    image_path,
    csv_path,
    calib,
    rect=None,
    roi_rel=None,
    color=(0, 255, 255),
    thickness=2,
    out_path=None,
):
    """
    Draw a polyline from an edited CSV (time,temperature) back onto the screenshot.

    Provide either rect=(x0,y0,x1,y1) or roi_rel=(x,y,w,h) in normalized coords.
    """
    img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")

    # Resolve plot rectangle
    if rect is None:
        if roi_rel is None:
            raise ValueError("Provide rect or roi_rel")
        rect = _rect_from_roi(img, roi_rel)
    x0, y0, x1, y1 = rect
    w = x1 - x0
    h = y1 - y0

    # Load CSV (with or without header)
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        t = np.asarray(data["time"], dtype=float)
        T = np.asarray(data["temperature"], dtype=float)
    except Exception:
        # fallback if no header
        data = np.genfromtxt(csv_path, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("CSV must have two columns: time, temperature")
        t = np.asarray(data[:, 0], dtype=float)
        T = np.asarray(data[:, 1], dtype=float)

    # Sort by time for a clean polyline
    order = np.argsort(t)
    t = t[order]
    T = T[order]

    # Map data -> ROI pixel coords
    x_px, y_px = data_to_pixels(t, T, calib)

    # Keep finite points within ROI
    m = np.isfinite(x_px) & np.isfinite(y_px)
    m &= (x_px >= 0) & (x_px < w) & (y_px >= 0) & (y_px < h)

    x_img = x0 + np.clip(x_px[m], 0, w - 1)
    y_img = y0 + np.clip(y_px[m], 0, h - 1)

    pts = np.stack([x_img, y_img], axis=1).astype(np.int32)
    overlay = img.copy()
    if len(pts) >= 2:
        cv.polylines(
            overlay,
            [pts],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv.LINE_AA,
        )

    # Save overlay
    if out_path is None:
        out_path = Path(image_path).with_name(
            Path(image_path).stem + "_overlay_fromcsv.png"
        )
    cv.imwrite(str(out_path), overlay)
    print(f"Saved overlay {out_path}")
    return str(out_path)


if __name__ == "__main__":
    # Given from earlier steps:
    # calib = (ax, bx, ay, by, y0_px)
    # rect   = (x0, y0, x1, y1)  OR  roi_rel = (x, y, w, h)

    calib = (
        np.float64(0.0037764350453172208),
        np.float64(-0.09667673716012185),
        np.float64(0.006711409395973154),
        np.float64(3.610738255033556),
        0.0,
    )
    rect = (484, 157, 1915, 748)

    overlay_from_csv(
        image_path="without_label.png",
        csv_path="without_label.csv",  # edited CSV
        calib=calib,
        rect=rect,  # or roi_rel=ROI_REL
        color=(0, 0, 0),  # yellow
        thickness=2,
    )
