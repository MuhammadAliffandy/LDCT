"""
prediction_service.py
---------------------
Full LDCT inference pipeline.
Single Responsibility: End-to-end prediction from image path to result JSON.
"""
import os
import logging
import numpy as np
import cv2

from app.services.model_loader import get_models, is_ready
from app.services.feature_extractor import smart_preprocess

logger = logging.getLogger("LDCT-PredictionService")

CLASS_NAMES = ["Full_Dose", "Quarter_Dose"]
CONFIDENCE_THRESHOLD = 0.70


def predict_from_image(image_path: str) -> dict:
    """
    Full LDCT prediction pipeline.

    Steps:
    1. Read & preprocess image (256x256)
    2. Extract handcrafted features for UI display
    3. Keras single-input model inference → [probabilities, segmentation_mask]
    4. Parse probabilities
    5. Return structured result dict

    Args:
        image_path: Absolute path to the CT image file.

    Returns:
        dict with keys: prediction_label, prediction, confidence,
                        all_features, probabilities, is_referral, status
    """
    if not is_ready():
        return {"error": "Models not loaded. Please check server logs."}

    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    try:
        models = get_models()

        # ---- 1. READ & PREPROCESS (notebook-compatible) ----
        # smart_preprocess applies HU-style windowing and returns float32 [0, 1]
        # matching process_dicom() in LDCT-improved-se2.ipynb exactly.
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            return {"error": "Cannot read image file. Ensure it is a valid PNG/JPG."}

        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        # smart_preprocess already outputs float32 [0, 1] — no /255.0 needed
        input_img = np.expand_dims(smart_preprocess(img_rgb), axis=0)  # (1, 256, 256, 3)

        # ---- 2. KERAS INFERENCE ----
        # Multi-task model returns: [class_output (1, N), seg_output (1, H, W, 1)]
        keras_outputs = models["keras"].predict(input_img, verbose=0)
        keras_proba   = keras_outputs[0][0]   # classification probabilities
        seg_mask      = keras_outputs[1][0, :, :, 0]  # (256, 256) segmentation sigmoid

        # ---- 4. PARSE RESULT ----
        agent_label_idx = int(np.argmax(keras_proba))
        agent_conf = float(np.max(keras_proba))
        all_class_probs = {
            CLASS_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(keras_proba)
        }

        agent_label_str = CLASS_NAMES[agent_label_idx]

        # ---- 5. REFERRAL LOGIC ----
        is_referral = agent_conf < CONFIDENCE_THRESHOLD
        status_msg = (
            "Uncertainty Detected — Physician Review Recommended"
            if is_referral
            else "High Confidence Analysis"
        )

        logger.info(
            f"[Prediction] {os.path.basename(image_path)} → "
            f"{agent_label_str} ({agent_conf:.2%}) | referral={is_referral}"
        )

        # ---- Segmentation coverage (% of image area flagged as ROI) ----
        seg_binary = (seg_mask > 0.5).astype(np.float32)
        seg_coverage_pct = round(float(seg_binary.mean()) * 100, 2)

        return {
            "prediction_label": agent_label_str,
            "prediction": agent_label_idx,
            "confidence": round(agent_conf, 4),
            "all_probabilities": all_class_probs,
            "is_referral": bool(is_referral),
            "status": status_msg,
            "seg_coverage_pct": seg_coverage_pct,   # % of pixels flagged by seg head
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {"error": str(e)}
