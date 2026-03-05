"""
model_loader.py
---------------
Loads and caches ALL model artifacts once at startup.
Single Responsibility: Model loading & caching.

Supports:
  - Legacy Keras 2 HDF5 (.h5)  — via model_builder.load_model_from_h5()
  - Keras 3 native (.keras)    — via keras.saving.load_model()
  - Keras 3 weights (.h5)      — via model_builder.load_model_from_weights()
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

    Auto-detects format by inspecting the h5 file structure:
      - If h5 contains 'model_weights' key → Legacy Keras 2 (base architecture)
      - If h5 contains 'layers' key → Keras 3 weights format (v2 architecture)
      - If .keras extension → Keras 3 native format (v2 architecture)
    """
    if _LOAD_STATUS["loaded"]:
        return True

    logger.info("--- Loading LDCT Model Artifacts ---")

    try:
        import tensorflow as tf
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import logging as _logging
        _logging.getLogger("tensorflow").setLevel(_logging.ERROR)
        _logging.getLogger("absl").setLevel(_logging.ERROR)

        from app.config import Config
        model_path = Config.KERAS_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"  Model path: {model_path}")

        # ── Determine format and load accordingly ─────────────────────────────
        if model_path.endswith(".h5"):
            _MODELS["keras"] = _load_h5_model(model_path)
        elif model_path.endswith(".keras"):
            _MODELS["keras"] = _load_keras_model(model_path)
        else:
            raise ValueError(f"Unsupported model extension: {model_path}")

        _LOAD_STATUS["loaded"] = True
        logger.info("✅ LDCT models loaded successfully.")
        return True

    except Exception as e:
        _LOAD_STATUS["error"] = str(e)
        logger.error(f"❌ CRITICAL: Model loading failed — {e}", exc_info=True)
        return False


def _load_h5_model(h5_path: str):
    """
    Load a .h5 model file, auto-detecting whether it is:
      - Legacy Keras 2 format (has 'model_weights' key)
      - Keras 3 extracted weights (has 'layers' key)
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        top_keys = set(f.keys())

    if "model_weights" in top_keys:
        # Legacy Keras 2 HDF5 — original base architecture
        logger.info("  Detected: Legacy Keras 2 HDF5 format → base architecture")
        from app.services.model_builder import load_model_from_h5
        model = load_model_from_h5(h5_path)
        logger.info(
            f"  ✅ Legacy H5 OK — Mod-Seg-SE(2) base "
            f"({len(model.layers)} layers)"
        )
        return model

    elif "layers" in top_keys:
        # Keras 3 extracted weights — improved v2 architecture
        logger.info("  Detected: Keras 3 weights format → v2 architecture")
        from app.services.model_builder import load_model_from_weights
        model = load_model_from_weights(h5_path)
        logger.info(
            f"  ✅ Keras 3 Weights OK — Mod-Seg-SE(2) v2 "
            f"({len(model.layers)} layers)"
        )
        return model

    else:
        raise ValueError(
            f"Unrecognized .h5 format (keys: {top_keys}). "
            "Expected 'model_weights' (legacy) or 'layers' (Keras 3)."
        )


def _load_keras_model(keras_path: str):
    """
    Load a .keras model file (Keras 3 native format).
    Falls back to manual weight builder if native load fails (e.g. on TF 2.13).
    """
    try:
        import keras
        logger.info(
            f"  Attempting native Keras 3 load: {keras_path} "
            "(this may take ~30s)"
        )
        model = keras.saving.load_model(keras_path, compile=False)
        logger.info(
            f"  ✅ Keras 3 Native OK — Mod-Seg-SE(2) v2 "
            f"({len(model.layers)} layers)"
        )
        return model

    except Exception as e:
        logger.warning(
            f"  Native load failed ({type(e).__name__}). "
            "Falling back to manual weight builder..."
        )
        from app.services.model_builder import load_model_from_weights
        from app.config import Config

        weights_path = os.path.join(
            os.path.dirname(Config.KERAS_MODEL_PATH),
            "extracted", "model.weights.h5"
        )
        logger.info(f"  Loading weights from: {weights_path}")
        model = load_model_from_weights(weights_path)
        logger.info(
            f"  ✅ Keras TF2.13 Fallback OK — Mod-Seg-SE(2) v2 "
            f"({len(model.layers)} layers)"
        )
        return model


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
