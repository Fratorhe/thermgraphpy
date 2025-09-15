from pathlib import Path

import cv2 as cv
import numpy as np

from .utils import crop_roi, fit_axis_map

# ------------------------------
# Interactive calibration
# ------------------------------
CAL_CLICKS = []


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        CAL_CLICKS.append((x, y))
        print(f"Clicked pixel: ({x}, {y})")


def collect_calibration(image_path, roi_rel):
    """
    Click two points for X calibration and two for Y calibration.
    For each click, you will type the true (t, T) values when prompted.
    If you only have one point per axis, you can enter one of the axis limits when asked.
    """
    img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")
    roi, rect = crop_roi(img, roi_rel)
    disp = img.copy()
    cv.rectangle(disp, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 1)

    cv.namedWindow("click two known points inside the ROI")
    cv.setMouseCallback("click two known points inside the ROI", on_mouse)
    print("Click two points inside the ROI. Close the window when done.")
    while True:
        cv.imshow("click two known points inside the ROI", disp)
        if cv.waitKey(10) & 0xFF == 27:  # Esc to finish
            break
    cv.destroyAllWindows()

    if len(CAL_CLICKS) < 2:
        raise RuntimeError("Need at least two clicks for calibration")

    # Ask user for their true data values at those clicks
    px = []
    py = []
    tx = []
    Ty = []
    for i, (cx, cy) in enumerate(CAL_CLICKS[:2]):
        print(f"Enter true data for click {i+1} at pixel ({cx}, {cy})")
        t_val = float(input("  time value (x axis): "))
        T_val = float(input("  temperature value (y axis): "))
        px.append(cx - rect[0])
        py.append(cy - rect[1])
        tx.append(t_val)
        Ty.append(T_val)

    px = np.array(px, dtype=float)
    py = np.array(py, dtype=float)
    tx = np.array(tx, dtype=float)
    Ty = np.array(Ty, dtype=float)

    # Fit linear maps for x and y separately
    ax, bx = fit_axis_map(px, tx)
    # For y we map (y0 - y_px) to data, use the ROI top as y0 reference
    ay, by = fit_axis_map((0 - py), Ty)  # using y0_px=0 inside ROI
    y0_px = 0.0

    print(f"Calibrated: t = {ax:.6f}*x_px + {bx:.6f}")
    print(f"Calibrated: T = {ay:.6f}*(y0_px - y_px) + {by:.6f}, y0_px={y0_px:.1f}")

    calib = (ax, bx, ay, by, y0_px)
    return calib, rect, px, py, tx, Ty


def apply_calibration(x_px, y_px, calib):
    ax, bx, ay, by, y0_px = calib
    t = ax * x_px + bx
    T = ay * (y0_px - y_px) + by  # invert y
    return t, T


def verify_calibration(calib, px, py, tx, Ty, idx=0):
    """
    Verify calibration on one of the clicked points (idx = 0 or 1)
    """
    x_test = np.array([px[idx]], dtype=float)
    y_test = np.array([py[idx]], dtype=float)
    t_hat, T_hat = apply_calibration(x_test, y_test, calib)
    print("\nVerification:")
    print(f"  Entered:   time={tx[idx]:.6f}, temp={Ty[idx]:.6f}")
    print(f"  Reproject: time={t_hat[0]:.6f}, temp={T_hat[0]:.6f}")
    print(f"  Errors:    Δt={t_hat[0]-tx[idx]:.6f}, ΔT={T_hat[0]-Ty[idx]:.6f}")


if __name__ == "__main__":
    image_path = "without_label.png"
    roi_rel = (0.2520833333333333, 0.14537037037037037, 0.7453125, 0.5472222222222223)
    calib, rect, px, py, tx, Ty = collect_calibration("without_label.png", roi_rel)

    # Check accuracy on first clicked point
    verify_calibration(calib, px, py, tx, Ty, idx=1)

    print(calib)
    print(rect)
