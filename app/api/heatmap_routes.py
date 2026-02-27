"""
heatmap_routes.py
-----------------
Blueprint: /heatmap
Single Responsibility: Return GradCAM or saliency heatmap overlay + annotated lung image.
"""
import os
import uuid
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app.api.utils import allowed_file
from app.services.heatmap_service import generate_heatmap_from_image
from app.services.lung_locator import detect_lung_region
import cv2

heatmap_bp = Blueprint("heatmap", __name__)
logger = logging.getLogger("LDCT-HeatmapRoutes")


@heatmap_bp.route("/heatmap", methods=["POST"])
def heatmap():
    """
    POST /heatmap
    Accepts a CT image file and optional 'mode' field ('gradcam' or 'saliency').
    Returns: base64-encoded heatmap overlay + annotated lung image with bboxes.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    mode = request.form.get("mode", "gradcam")
    if mode not in ("gradcam", "saliency"):
        mode = "gradcam"

    request_id = str(uuid.uuid4())[:8]
    filename = secure_filename(f"{request_id}_{file.filename}")
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)
        logger.info(f"[{request_id}] Generating heatmap ({mode}): {filename}")

        # === GradCAM / Saliency ===
        heatmap_result = generate_heatmap_from_image(filepath, mode=mode)
        if "error" in heatmap_result:
            return jsonify(heatmap_result), 500

        # === Annotated Lung Bounding Box ===
        raw_img = cv2.imread(filepath)
        lung_annotated_b64 = None
        lung_region_info = {}

        if raw_img is not None:
            img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            lung_result = detect_lung_region(img_rgb)
            lung_annotated_b64 = lung_result.get("annotated_image_b64")
            lung_region_info = {
                k: v for k, v in lung_result.items() if k != "annotated_image_b64"
            }

        return jsonify({
            **heatmap_result,
            "lung_annotated_b64": lung_annotated_b64,
            "lung_region": lung_region_info,
        }), 200

    except Exception as e:
        logger.error(f"[{request_id}] Heatmap route error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
