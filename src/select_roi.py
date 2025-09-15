from pathlib import Path

import cv2 as cv
import numpy as np


def _norm_from_rect(rect_px, img_shape):
    """rect_px = (x0, y0, x1, y1) in pixels -> (x, y, w, h) normalized [0,1]."""
    H, W = img_shape[:2]
    x0, y0, x1, y1 = rect_px
    x, y = x0 / W, y0 / H
    w, h = (x1 - x0) / W, (y1 - y0) / H
    return (float(x), float(y), float(w), float(h))


def _sanitize_rect(x0, y0, x1, y1, W, H):
    """Ensure top-left and bottom-right ordering and clamp to image bounds."""
    x0, x1 = int(max(0, min(x0, x1))), int(min(W - 1, max(x0, x1)))
    y0, y1 = int(max(0, min(y0, y1))), int(min(H - 1, max(y0, y1)))
    return x0, y0, x1, y1


def select_roi_clicks(image_path):
    """
    Click top-left and bottom-right corners of the plotting area.
    Press Esc to finish after the two clicks.
    Returns:
        roi_rel: (x, y, w, h) normalized in [0,1]
        rect_px: (x0, y0, x1, y1) integer pixels
    """
    img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")
    H, W = img.shape[:2]
    clone = img.copy()
    clicks = []

    def on_mouse(event, x, y, flags, param):
        nonlocal clicks, clone
        if event == cv.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            # draw a small marker
            cv.circle(clone, (x, y), 4, (0, 255, 255), -1)

    win = "Select ROI: click top-left, then bottom-right (Esc to finish)"
    cv.namedWindow(win)
    cv.setMouseCallback(win, on_mouse)
    while True:
        cv.imshow(win, clone)
        k = cv.waitKey(16) & 0xFF
        if k == 27:  # Esc
            break
        # draw a preview rectangle after first click
        if len(clicks) == 2:
            (x0, y0), (x1, y1) = clicks[:2]
            preview = img.copy()
            x0, y0, x1, y1 = _sanitize_rect(x0, y0, x1, y1, W, H)
            cv.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv.imshow(win, preview)
    cv.destroyAllWindows()

    if len(clicks) < 2:
        raise RuntimeError("Need two clicks for ROI (top-left and bottom-right).")
    (x0, y0), (x1, y1) = clicks[:2]
    x0, y0, x1, y1 = _sanitize_rect(x0, y0, x1, y1, W, H)
    roi_rel = _norm_from_rect((x0, y0, x1, y1), img.shape)
    return roi_rel, (x0, y0, x1, y1)


def select_roi_drag(image_path):
    """
    Drag to select the plotting area using OpenCV's built-in selector.
    Hit Space/Enter to confirm or c to cancel.
    Returns:
        roi_rel: (x, y, w, h) normalized in [0,1]
        rect_px: (x0, y0, x1, y1) integer pixels
    """
    img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")
    H, W = img.shape[:2]

    # (x, y, w, h) in pixels from cv.selectROI
    r = cv.selectROI(
        "Drag ROI and press Space/Enter", img, showCrosshair=True, fromCenter=False
    )
    cv.destroyWindow("Drag ROI and press Space/Enter")
    x, y, w, h = map(int, r)
    if w <= 0 or h <= 0:
        raise RuntimeError("No ROI selected.")
    x0, y0, x1, y1 = _sanitize_rect(x, y, x + w, y + h, W, H)

    roi_rel = _norm_from_rect((x0, y0, x1, y1), img.shape)
    return roi_rel, (x0, y0, x1, y1)


def select_roi(image_path, method="drag"):
    """
    Wrapper to choose how you want to define the ROI.
    method: "drag" (recommended) or "clicks"
    Returns:
        roi_rel, rect_px
    """
    if method == "drag":
        return select_roi_drag(image_path)
    elif method == "clicks":
        return select_roi_clicks(image_path)
    else:
        raise ValueError('method must be "drag" or "clicks"')


if __name__ == "__main__":
    roi_rel, rect_px = select_roi("with_label.png", method="clicks")  # or "clicks"
    print("ROI_REL =", roi_rel)  # e.g. (0.115, 0.182, 0.780, 0.705)
