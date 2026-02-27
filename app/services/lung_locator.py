"""
lung_locator.py
---------------
Lung region detection & position classification on CT images.
Single Responsibility: Localize lung fields and describe their anatomical position.
"""
import cv2
import numpy as np
import logging
import base64

logger = logging.getLogger("LDCT-LungLocator")


def detect_lung_region(image_array: np.ndarray) -> dict:
    """
    Detect lung fields in a CT image using thresholding + contour detection.

    Returns:
        dict with:
        - bboxes: list of (x, y, w, h) bounding boxes for detected lung regions
        - region_label: human-readable position (Left Lung, Right Lung, Bilateral, etc.)
        - annotated_image_b64: base64-encoded annotated image
        - side_distribution: dict with left/right area percentages
    """
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()

        img_h, img_w = gray.shape
        img_center_x = img_w // 2

        # ---- Threshold to isolate dark lung regions (air-filled) ----
        # CT lungs appear dark (low HU). We threshold for dark pixels.
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Remove border artifacts with morphological ops
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)

        # ---- Find contours ----
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area (must be significant lung-sized region)
        min_area = (img_h * img_w) * 0.01  # at least 1% of image
        max_area = (img_h * img_w) * 0.45  # not more than 45% (exclude body outline)

        lung_contours = [
            c for c in contours
            if min_area < cv2.contourArea(c) < max_area
        ]

        # Sort by area descending, keep top 2 (left + right lung)
        lung_contours = sorted(lung_contours, key=cv2.contourArea, reverse=True)[:2]

        bboxes = [cv2.boundingRect(c) for c in lung_contours]

        # ---- Classify position ----
        region_label, side_distribution, vertical_position = _classify_lung_position(
            bboxes, img_center_x, img_h
        )

        # ---- Annotate image ----
        if len(image_array.shape) == 3:
            annotated = image_array.copy()
        else:
            annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        colors = [(0, 255, 100), (100, 200, 255)]  # green / cyan for L/R
        for i, (x, y, w, h) in enumerate(bboxes):
            color = colors[i % len(colors)]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = "L" if x < img_center_x else "R"
            cv2.putText(
                annotated, label, (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        # Encode to base64
        _, buffer = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "bboxes": bboxes,
            "region_label": region_label,
            "vertical_position": vertical_position,
            "side_distribution": side_distribution,
            "annotated_image_b64": annotated_b64,
            "num_regions_detected": len(bboxes),
        }

    except Exception as e:
        logger.error(f"Lung locator error: {e}", exc_info=True)
        return {
            "bboxes": [],
            "region_label": "Detection unavailable",
            "vertical_position": "Unknown",
            "side_distribution": {"left": 0, "right": 0},
            "annotated_image_b64": None,
            "num_regions_detected": 0,
            "error": str(e),
        }


def _classify_lung_position(bboxes: list, img_center_x: int, img_h: int) -> tuple:
    """
    Given bounding boxes, classify which lung fields are present and
    estimate the vertical position (Upper / Lower / Middle / Bilateral-Vertical).
    """
    if not bboxes:
        return "No Lung Region Detected", {"left": 0, "right": 0}, "Unknown"

    left_area = 0
    right_area = 0

    for x, y, w, h in bboxes:
        box_center = x + w // 2
        area = w * h
        if box_center < img_center_x:
            left_area += area
        else:
            right_area += area

    total = left_area + right_area or 1
    left_pct = round(left_area / total * 100, 1)
    right_pct = round(right_area / total * 100, 1)

    side_distribution = {"left": left_pct, "right": right_pct}

    # Side label
    if left_pct > 65:
        side_label = "Left Lung Dominant"
    elif right_pct > 65:
        side_label = "Right Lung Dominant"
    else:
        side_label = "Bilateral Lung Fields"

    # Vertical position estimation
    if bboxes:
        y_centers = [y + h / 2 for x, y, w, h in bboxes]
        avg_y_center = np.mean(y_centers)
        rel_pos = avg_y_center / img_h

        if rel_pos < 0.35:
            vert = "Upper Lobe Region"
        elif rel_pos > 0.65:
            vert = "Lower Lobe Region"
        else:
            vert = "Middle / Hilar Region"
    else:
        vert = "Unknown"

    return side_label, side_distribution, vert
