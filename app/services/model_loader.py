"""
model_loader.py
---------------
Loads and caches ALL model artifacts once at startup.
Single Responsibility: Model loading & caching.
"""
import logging
import os

logger = logging.getLogger("LDCT-ModelLoader")

# Lazy imports for ML libraries
_MODELS = {}
_LOAD_STATUS = {"loaded": False, "error": None}


def load_all_models():
    """
    Load Keras mod-seg-se2 multi-task model.
    Safe to call multiple times — loads only once.
    """
    if _LOAD_STATUS["loaded"]:
        return True

    logger.info("--- Loading LDCT Model Artifacts ---")

    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        import keras
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        # Resolve paths from config
        from app.config import Config

        keras_path = Config.KERAS_MODEL_PATH

        if not os.path.exists(keras_path):
            raise FileNotFoundError(f"Model file missing: {keras_path}")

        # ---- Load Keras Model ----
        logger.info(f"  Loading Keras model from: {keras_path} (this may take ~30s)")
        _MODELS["keras"] = keras.saving.load_model(keras_path, compile=False)
        logger.info(f"  ✅ Keras OK ({len(_MODELS['keras'].layers)} layers)")

        _LOAD_STATUS["loaded"] = True
        logger.info("✅ LDCT models loaded successfully.")
        return True

    except Exception as e:
        _LOAD_STATUS["error"] = str(e)
        logger.error(f"❌ CRITICAL: Model loading failed — {e}", exc_info=True)
        return False


def get_models():
    """
    Returns the cached models dict. Triggers loading if not yet done.
    """
    if not _LOAD_STATUS["loaded"]:
        load_all_models()
    return _MODELS


def is_ready():
    """Returns True if all models loaded successfully."""
    return _LOAD_STATUS["loaded"]


def get_load_error():
    """Returns the error message if loading failed, else None."""
    return _LOAD_STATUS["error"]
