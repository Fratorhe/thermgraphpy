# app.py
# Streamlit GUI for extracting time–temperature data from screenshots
# Suggested requirements:
# streamlit==1.34.0
# streamlit-drawable-canvas==0.9
# opencv-python-headless==4.10.0.84
# Pillow==10.4.0
# numpy==1.26.4
# protobuf<5

import io
from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.calibration import apply_calibration
from src.utils import (
    COLOR_OPTIONS,
    crop_roi,
    fit_axis_map,
    mask_colored_curve,
    overlay_polyline,
    trace_curve,
)

st.set_page_config(page_title="Time–Temperature Extractor", layout="wide")


# -------------------------
# Helpers
# -------------------------
def data_to_pixels(t, T, calib):
    ax, bx, ay, by, y0_px = calib
    if abs(ax) < 1e-12 or abs(ay) < 1e-12:
        raise ValueError("Invalid calibration: near-zero slope.")
    x_px = (t - bx) / ax
    y_px = y0_px - (T - by) / ay
    return x_px, y_px


def build_color_mask(roi_bgr, line_color_bgr, h_tol=10, s_tol=80, v_tol=80):
    hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)
    target = np.uint8([[line_color_bgr]])
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)[0, 0]
    h, s, v = map(int, target_hsv)
    lower = np.array(
        [max(0, h - h_tol), max(50, s - s_tol), max(50, v - v_tol)], dtype=np.uint8
    )
    upper = np.array([min(179, h + h_tol), 255, 255], dtype=np.uint8)
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.medianBlur(mask, 3)
    return mask


def encode_png(img_bgr):
    ok, buf = cv.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return io.BytesIO(buf.tobytes())


def fit_canvas_size_from_pil(img_pil: Image.Image, max_w=1200, max_h=900, upscale=False):
    """Return (width, height) preserving aspect ratio (uses PIL size order: w,h)."""
    w, h = img_pil.size
    if not upscale:
        max_w = min(max_w, w)
        max_h = min(max_h, h)
    s = min(max_w / w, max_h / h)
    return int(w * s), int(h * s)


def load_image_rgb(filelike, max_side=1920) -> Image.Image:
    """Load with Pillow, convert to RGB (no alpha), cap size for Fabric.js stability."""
    img = Image.open(filelike).convert("RGB")
    w, h = img.size
    if w > max_side or h > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


# -------------------------
# Streamlit UI state
# -------------------------
if "roi_rel" not in st.session_state:
    st.session_state.roi_rel = (0.1, 0.1, 0.8, 0.8)
if "calib" not in st.session_state:
    st.session_state.calib = None
if "rect" not in st.session_state:
    st.session_state.rect = None
if "last_overlay" not in st.session_state:
    st.session_state.last_overlay = None

st.title("Time-Temperature Extractor")

with st.sidebar:
    st.header("1) Upload screenshot")
    up = st.file_uploader("PNG/JPG screenshot", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    st.header("2) ROI selection")
    st.caption(
        "Draw a rectangle over the plot area (left panel) and click **Apply ROI**."
    )
    st.markdown("---")
    st.header("3) Calibration")
    st.caption(
        "Click two points on the curve inside ROI (right panel), then enter true values."
    )
    st.markdown("---")
    st.header("4) Extraction")
    st.caption("Choose color, tolerances, and optional time window. Then **Extract**.")

if up is None:
    st.info("Upload a screenshot to begin.")
    st.stop()

# -------------------------
# Load image: Pillow RGB for canvas; keep BGR for OpenCV processing
# -------------------------
raw = up.read()
try:
    img_rgb_pil = load_image_rgb(io.BytesIO(raw), max_side=1920)  # RGB (no alpha)
except Exception as e:
    st.error(f"Could not read the image: {e}")
    st.stop()

# For processing, get BGR from the RGB
rgb_for_cv = np.array(img_rgb_pil)
img_bgr = cv.cvtColor(rgb_for_cv, cv.COLOR_RGB2BGR)
H, W = img_bgr.shape[:2]
st.caption(f"Loaded image size: {W}×{H}, mode: {img_rgb_pil.mode}")

# -------------------------
# ROI selection (left)
# -------------------------
st.subheader("Select ROI (draw rectangle)")

canvas_w, canvas_h = fit_canvas_size_from_pil(
    img_rgb_pil, max_w=1200, max_h=900, upscale=False
)
st.image(img_rgb_pil, caption="PIL RGB preview")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=2,
    stroke_color="#00FF00",
    background_color=None,
    background_image=img_rgb_pil,
    update_streamlit=True,
    width=canvas_w,
    height=canvas_h,
    drawing_mode="rect",
    key=f"roi_canvas_{up.name}_{canvas_w}x{canvas_h}",
)

if st.button("Apply ROI"):
    if canvas_result.json_data and len(canvas_result.json_data.get("objects", [])) > 0:
        rect_obj = [
            o for o in canvas_result.json_data["objects"] if o.get("type") == "rect"
        ]
        if not rect_obj:
            st.warning("Please draw a rectangle first.")
        else:
            rect_obj = rect_obj[-1]
            if canvas_result.image_data is not None:
                canH, canW = canvas_result.image_data.shape[:2]
            else:
                canW, canH = canvas_w, canvas_h

            sx = W / float(canW)
            sy = H / float(canH)

            x = max(0, int(rect_obj.get("left", 0) * sx))
            y = max(0, int(rect_obj.get("top", 0) * sy))
            w = max(1, int(rect_obj.get("width", 0) * rect_obj.get("scaleX", 1.0) * sx))
            h = max(1, int(rect_obj.get("height", 0) * rect_obj.get("scaleY", 1.0) * sy))
            x0, y0 = x, y
            x1, y1 = min(W - 1, x + w), min(H - 1, y + h)
            roi_rel = (x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H)
            st.session_state.roi_rel = roi_rel
            st.success(
                f"ROI set to (x={roi_rel[0]:.3f}, y={roi_rel[1]:.3f}, w={roi_rel[2]:.3f}, h={roi_rel[3]:.3f})"
            )
    else:
        st.warning("Please draw a rectangle first.")

roi_rel = st.session_state.roi_rel
roi_img, rect_px = crop_roi(img_bgr, roi_rel)
st.session_state.rect = rect_px

# -------------------------
# Calibration & extraction (right)
# -------------------------
st.subheader("Click 2 calibration points inside ROI")

roi_H, roi_W = roi_img.shape[:2]
roi_pil_rgb = Image.fromarray(cv.cvtColor(roi_img, cv.COLOR_BGR2RGB))
cal_w, cal_h = fit_canvas_size_from_pil(roi_pil_rgb, max_w=1200, max_h=900, upscale=False)

cal_canvas = st_canvas(
    fill_color="rgba(255, 255, 0, 0.4)",
    stroke_width=8,
    stroke_color="#FFFF00",
    background_color=None,
    background_image=roi_pil_rgb,
    update_streamlit=True,
    width=cal_w,
    height=cal_h,
    drawing_mode="point",
    key=f"cal_canvas_{up.name}_{cal_w}x{cal_h}",
)

# Collect clicked points
clicked = []
if cal_canvas.json_data and len(cal_canvas.json_data.get("objects", [])) > 0:
    if cal_canvas.image_data is not None:
        canH2, canW2 = cal_canvas.image_data.shape[:2]
    else:
        canW2, canH2 = cal_w, cal_h
    for o in cal_canvas.json_data["objects"]:
        if o.get("type") in ("path", "circle"):
            cx = int(o.get("left", 0) + (o.get("width", 0) * o.get("scaleX", 1.0)) / 2)
            cy = int(o.get("top", 0) + (o.get("height", 0) * o.get("scaleY", 1.0)) / 2)
            sx = roi_W / float(canW2)
            sy = roi_H / float(canH2)
            px = int(cx * sx)
            py = int(cy * sy)
            clicked.append((px, py))

st.caption("Enter the true values for the first two points (time, temperature):")
c1, c2 = st.columns(2)
with c1:
    t1 = st.number_input("Point 1: time", value=0.0, step=0.1, format="%.6f")
    T1 = st.number_input("Point 1: temperature", value=0.0, step=0.1, format="%.6f")
with c2:
    t2 = st.number_input("Point 2: time", value=1.0, step=0.1, format="%.6f")
    T2 = st.number_input("Point 2: temperature", value=1.0, step=0.1, format="%.6f")

# Calibration
calib_ready = False
if len(clicked) >= 2:
    px = np.array([clicked[0][0], clicked[1][0]], dtype=float)
    py = np.array([clicked[0][1], clicked[1][1]], dtype=float)
    tx = np.array([t1, t2], dtype=float)
    Ty = np.array([T1, T2], dtype=float)
    ax, bx = fit_axis_map(px, tx)
    ay, by = fit_axis_map((0 - py), Ty)
    calib = (ax, bx, ay, by, 0.0)
    st.session_state.calib = calib
    calib_ready = True
    st.success(
        f"Calibrated: t = {ax:.6f}*x + {bx:.6f};  T = {ay:.6f}*(y0 - y) + {by:.6f}"
    )

# -------------------------
# Extraction controls
# -------------------------
st.markdown("---")
st.subheader("Extraction parameters")

colA, colB = st.columns(2)
with colA:
    chosen_name = st.selectbox(
        "Curve color (overlay & detection)", list(COLOR_OPTIONS.keys()), index=0
    )
    line_color_bgr = COLOR_OPTIONS[chosen_name]

with colB:
    t_min = st.number_input("t_min (optional)", value=float("nan"))
    t_max = st.number_input("t_max (optional)", value=float("nan"))

time_range = None
if np.isfinite(t_min) or np.isfinite(t_max):
    time_range = (
        t_min if np.isfinite(t_min) else None,
        t_max if np.isfinite(t_max) else None,
    )

do_extract = st.button("Extract data")
do_overlay_csv = st.button("Overlay from edited CSV")

# -------------------------
# Extraction run
# -------------------------
if do_extract:
    if not calib_ready:
        st.error("Need two calibration clicks and values first.")
    else:
        if line_color_bgr is not None:
            mask = build_color_mask(roi_img, line_color_bgr, 10, 80, 80)
        else:
            mask = mask_colored_curve(roi_img, 60, 40)

        ys = trace_curve(mask)
        w = mask.shape[1]
        x_px_all = np.arange(w, dtype=float)
        y_px_all = ys
        valid = np.isfinite(y_px_all)
        x_px = x_px_all[valid]
        y_px = y_px_all[valid]
        valid_idx = np.where(valid)[0]
        t_data, T_data = apply_calibration(x_px, y_px, st.session_state.calib)

        if time_range is not None:
            tmin, tmax = time_range
            keep = np.ones_like(t_data, dtype=bool)
            if tmin is not None:
                keep &= t_data >= tmin
            if tmax is not None:
                keep &= t_data <= tmax
            t_data = t_data[keep]
            T_data = T_data[keep]
            ys_vis = np.full_like(ys, np.nan, dtype=float)
            ys_vis[valid_idx[keep]] = ys[valid_idx[keep]]
        else:
            ys_vis = ys

        overlay = overlay_polyline(
            img_bgr.copy(), st.session_state.rect, ys_vis, color=(0, 0, 0), thickness=2
        )

        st.image(cv.cvtColor(overlay, cv.COLOR_BGR2RGB), caption="Overlay preview")

        csv_bytes = io.BytesIO()
        np.savetxt(
            csv_bytes,
            np.column_stack([t_data, T_data]),
            delimiter=",",
            header="time,temperature",
            comments="",
            fmt="%.6f",
        )
        st.download_button(
            "Download CSV",
            data=csv_bytes.getvalue(),
            file_name="extracted.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download overlay PNG",
            data=encode_png(overlay).getvalue(),
            file_name="overlay.png",
            mime="image/png",
        )
        st.session_state.last_overlay = overlay
