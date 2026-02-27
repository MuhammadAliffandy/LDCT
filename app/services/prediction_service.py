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
import scipy.stats

from app.services.model_loader import get_models, is_ready
from app.services.feature_extractor import (
    smart_preprocess,
    extract_handcrafted,
    features_to_dict,
)

logger = logging.getLogger("LDCT-PredictionService")

CLASS_NAMES = ["Full_Dose", "Quarter_Dose"]
CONFIDENCE_THRESHOLD = 0.70


def predict_from_image(image_path: str) -> dict:
    """
    Full LDCT prediction pipeline.

    Steps:
    1. Read & preprocess image
    2. Extract handcrafted features (Wavelet + GLCM)
    3. Scale features + UMAP transform
    4. Keras hybrid model inference → probabilities + deep features
    5. LightGBM agent inference → final prediction
    6. Return structured result dict

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

        # ---- 1. READ & PREPROCESS ----
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            return {"error": "Cannot read image file. Ensure it is a valid PNG/JPG."}

        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        processed_img = smart_preprocess(img_rgb)

        input_img = np.expand_dims(
            processed_img.astype(np.float32) / 255.0, axis=0
        )  # (1, 256, 256, 3)

        # ---- 2. HANDCRAFTED FEATURES ----
        raw_feats = extract_handcrafted(processed_img)
        feat_array = np.array(raw_feats).reshape(1, -1)

        # ---- 3. SCALE + UMAP ----
        scaled_feats = models["scaler"].transform(feat_array)
        umap_feats = models["umap"].transform(scaled_feats)

        # ---- 4. KERAS INFERENCE ----
        keras_inputs = [input_img, scaled_feats, umap_feats]
        keras_proba = models["keras"].predict(keras_inputs, verbose=0)[0]
        deep_features = models["extractor"].predict(keras_inputs, verbose=0)[0]
        entropy = float(scipy.stats.entropy(keras_proba))

        # ---- 5. LIGHTGBM AGENT ----
        agent_input = np.hstack(
            [
                deep_features.reshape(1, -1),
                keras_proba.reshape(1, -1),
                [[entropy]],
            ]
        )
        agent_proba = models["agent"].predict(agent_input)[0]

        # Parse result
        if isinstance(agent_proba, (list, np.ndarray)) and len(np.atleast_1d(agent_proba)) > 1:
            agent_label_idx = int(np.argmax(agent_proba))
            agent_conf = float(np.max(agent_proba))
            all_class_probs = {
                CLASS_NAMES[i]: round(float(p), 4)
                for i, p in enumerate(agent_proba)
            }
        else:
            agent_label_idx = int(agent_proba > 0.5)
            agent_conf = float(agent_proba) if agent_label_idx == 1 else float(1 - agent_proba)
            all_class_probs = {
                CLASS_NAMES[0]: round(float(1 - agent_proba), 4),
                CLASS_NAMES[1]: round(float(agent_proba), 4),
            }

        agent_label_str = CLASS_NAMES[agent_label_idx]

        # ---- 6. REFERRAL LOGIC ----
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

        return {
            "prediction_label": agent_label_str,
            "prediction": agent_label_idx,
            "confidence": round(agent_conf, 4),
            "all_probabilities": all_class_probs,
            "all_features": features_to_dict(raw_feats),
            "is_referral": bool(is_referral),
            "status": status_msg,
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {"error": str(e)}
