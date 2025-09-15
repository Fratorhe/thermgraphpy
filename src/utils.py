from pathlib import Path

import cv2 as cv
import numpy as np

# Minimum run of consecutive columns allowed to be empty before we interpolate
MAX_GAP = 10

COLOR_OPTIONS = {
    "yellow": (0, 255, 255),
    "purple": (128, 0, 128),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
    "black": (0, 0, 0),
}


# ------------------------------
# Helper functions
# ------------------------------
def crop_roi(img, roi_rel):
    H, W = img.shape[:2]
    x0 = int(W * roi_rel[0])
    y0 = int(H * roi_rel[1])
    x1 = int(x0 + W * roi_rel[2])
    y1 = int(y0 + H * roi_rel[3])
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def mask_colored_curve(roi_bgr, sat_min=60, val_min=40):
    hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv)
    # Keep reasonably saturated and bright pixels
    m = (S >= sat_min) & (V >= val_min)
    mask = np.uint8(m) * 255
    # Clean up small speckles
    mask = cv.medianBlur(mask, 3)
    mask = cv.morphologyEx(
        mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    )
    return mask


def fit_axis_map(pxs, datas):
    # Fit data = a * px + b
    A = np.vstack([pxs, np.ones_like(pxs)]).T
    a, b = np.linalg.lstsq(A, datas, rcond=None)[0]
    return a, b  # returns slope, intercept


def trace_curve(mask):
    h, w = mask.shape
    ys = np.full(w, np.nan, dtype=float)
    for x in range(w):
        col = mask[:, x]
        idx = np.flatnonzero(col > 0)
        if idx.size == 0:
            continue
        # If the line is thick or antialiased, pick the median row
        ys[x] = float(np.median(idx))
    # Fill small gaps by linear interpolation
    x_idx = np.arange(w)
    valid = ~np.isnan(ys)
    if valid.sum() >= 2:
        ys_interp = np.interp(x_idx, x_idx[valid], ys[valid])
        # Respect large gaps by only filling runs shorter than MAX_GAP
        ys_filled = ys.copy()
        i = 0
        while i < w:
            if np.isnan(ys_filled[i]):
                j = i
                while j < w and np.isnan(ys_filled[j]):
                    j += 1
                gap = j - i
                if gap <= MAX_GAP and i > 0 and j < w:
                    ys_filled[i:j] = np.interp(
                        np.arange(i, j), [i - 1, j], [ys_filled[i - 1], ys_filled[j]]
                    )
                i = j
            else:
                i += 1
        ys = ys_filled
    return ys  # pixel rows within ROI, may still contain NaNs at extremes


def overlay_polyline(img, roi_rect, ys, color=(0, 255, 0), thickness=2):
    x0, y0, x1, y1 = roi_rect
    h = y1 - y0
    w = x1 - x0
    pts = []
    for x in range(w):
        y = ys[x]
        if np.isfinite(y):
            pts.append([x0 + x, y0 + int(round(y))])
    if len(pts) >= 2:
        cv.polylines(
            img,
            [np.array(pts, dtype=np.int32)],
            isClosed=False,
            color=color,
            thickness=thickness,
        )
    return img
