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
        import tensorflow as tf
        import keras                           # standalone Keras 3 (TF 2.16)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import logging as _logging
        _logging.getLogger("tensorflow").setLevel(_logging.ERROR)
        _logging.getLogger("absl").setLevel(_logging.ERROR)

        from app.config import Config
        # ---- Load Keras Model ----
        # Try native Keras 3 format first (Linux deployment / TF 2.16+)
        # If it fails (e.g. on Mac Apple Silicon with TF 2.13), fall back to manual builder
        try:
            import keras
            logger.info(f"  Attempting native Keras 3 load: {keras_path} (this may take ~30s)")
            _MODELS["keras"] = keras.saving.load_model(keras_path, compile=False)
            logger.info(f"  ✅ Keras 3 Native OK — Mod-Seg-SE(2) v2 ({len(_MODELS['keras'].layers)} layers)")
        except Exception as e:
            logger.warning(f"  Native load failed ({type(e).__name__}). Falling back to manual weight builder...")
            from app.services.model_builder import load_model_from_weights
            
            weights_path = os.path.join(
                os.path.dirname(Config.KERAS_MODEL_PATH),
                "extracted", "model.weights.h5"
            )
            logger.info(f"  Loading weights from: {weights_path}")
            _MODELS["keras"] = load_model_from_weights(weights_path)
            logger.info(f"  ✅ Keras TF2.13 Fallback OK — Mod-Seg-SE(2) v2 ({len(_MODELS['keras'].layers)} layers)")





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
