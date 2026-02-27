"""
predict_routes.py
-----------------
Blueprint: /predict
Single Responsibility: Handle image upload and return LDCT prediction.
"""
import os
import uuid
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app.api.utils import allowed_file
from app.services.prediction_service import predict_from_image
from app.services.lung_locator import detect_lung_region
import cv2

predict_bp = Blueprint("predict", __name__)
logger = logging.getLogger("LDCT-PredictRoutes")


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts a CT image file.
    Returns: prediction label, confidence, features, lung region info.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data key 'file'."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file submitted."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, or JPEG."}), 400

    request_id = str(uuid.uuid4())[:8]
    filename = secure_filename(f"{request_id}_{file.filename}")
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)
        logger.info(f"[{request_id}] Image saved: {filename}")

        # === Core Prediction ===
        prediction_result = predict_from_image(filepath)
        if "error" in prediction_result:
            return jsonify(prediction_result), 500

        # === Lung Region Detection ===
        raw_img = cv2.imread(filepath)
        if raw_img is not None:
            img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            lung_result = detect_lung_region(img_rgb)
        else:
            lung_result = {"region_label": "Unavailable", "bboxes": []}

        # Remove base64 from predict response (returned by /heatmap separately)
        lung_result_clean = {
            k: v for k, v in lung_result.items() if k != "annotated_image_b64"
        }

        response = {
            **prediction_result,
            "lung_region": lung_result_clean,
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"[{request_id}] Prediction route error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
