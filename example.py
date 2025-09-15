from src.calibration import collect_calibration
from src.extract_data import extract_one
from src.select_roi import select_roi

image_path = "example_pics/3.png"

roi_rel, rect_px = select_roi(image_path, method="clicks")  # or "clicks"
calib, rect, px, py, tx, Ty = collect_calibration(image_path, roi_rel)

extract_one(
    image_path,
    roi_rel,
    calib=calib,
    rect=rect,
    line_color="purple",  # yellow in BGR
    time_range=(0.5, None),
)
