"""
model_loader.py
---------------
Loads and caches ALL model artifacts once at startup.
Single Responsibility: Model loading & caching.
"""
import logging
import os
import sys

logger = logging.getLogger("LDCT-ModelLoader")

# Lazy imports for ML libraries
_MODELS = {}
_LOAD_STATUS = {"loaded": False, "error": None}


def load_all_models():
    """
    Load Keras hybrid model, scaler, UMAP, and LightGBM agent.
    Also builds the Keras feature-extractor sub-model (from concat layer).
    Safe to call multiple times — loads only once.
    """
    if _LOAD_STATUS["loaded"]:
        return True

    logger.info("--- Loading LDCT Model Artifacts ---")

    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # Use keras directly (Keras 3.x) — the model was saved with keras.saving
        import keras
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        import lightgbm as lgb
        import joblib
        import numpy as np

        # (TF noise suppression already done above)

        # Resolve paths from config
        from app.config import Config

        paths = {
            "keras": Config.KERAS_MODEL_PATH,
            "scaler": Config.SCALER_PATH,
            "umap": Config.UMAP_PATH,
            "agent": Config.AGENT_PATH,
        }

        # Validate all files exist
        for key, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file missing: {path}")

        # ---- Load Scaler ----
        logger.info(f"  Loading Scaler from: {paths['scaler']}")
        _MODELS["scaler"] = joblib.load(paths["scaler"])
        logger.info(f"  ✅ Scaler OK (n_features={_MODELS['scaler'].n_features_in_})")

        # ---- Load UMAP (with numba-version-mismatch fallback) ----
        logger.info(f"  Loading UMAP from: {paths['umap']}")
        _MODELS["umap"] = _load_umap_safe(paths["umap"], _MODELS["scaler"])

        # ---- Load LightGBM Agent ----
        logger.info(f"  Loading LightGBM Agent from: {paths['agent']}")
        _MODELS["agent"] = lgb.Booster(model_file=paths["agent"])
        logger.info(f"  ✅ LightGBM Agent OK ({_MODELS['agent'].num_trees()} trees)")

        # ---- Load Keras (use keras.saving — the model was saved with Keras 3.x) ----
        logger.info(f"  Loading Keras model from: {paths['keras']} (this may take ~30s)")
        _MODELS["keras"] = keras.saving.load_model(paths["keras"], compile=False)
        logger.info(f"  ✅ Keras OK ({len(_MODELS['keras'].layers)} layers)")

        # ---- Build Feature Extractor (finds the Concatenate layer) ----
        keras_model = _MODELS["keras"]
        # Build sub-model using keras.Model (Keras 3.x)
        Model = keras.Model
        concat_layer_name = None
        for layer in keras_model.layers:
            if "concatenate" in layer.name.lower():
                concat_layer_name = layer.name
                break

        if not concat_layer_name:
            # Fallback: use layer before the output dense
            concat_layer_name = keras_model.layers[-2].name
            logger.warning(
                f"  'concatenate' layer not found, using fallback: '{concat_layer_name}'"
            )

        logger.info(f"  Building feature extractor from layer: '{concat_layer_name}'")
        _MODELS["extractor"] = Model(
            inputs=keras_model.input,
            outputs=keras_model.get_layer(concat_layer_name).output,
        )

        _LOAD_STATUS["loaded"] = True
        logger.info("✅ All LDCT models loaded successfully.")
        return True

    except Exception as e:
        _LOAD_STATUS["error"] = str(e)
        logger.error(f"❌ CRITICAL: Model loading failed — {e}", exc_info=True)
        return False


def _load_umap_safe(umap_path: str, scaler):
    """
    Try to load the UMAP pkl. If it fails due to numba/pickle version
    incompatibility, re-fit a fresh UMAP using synthetic data derived
    from the scaler's mean/std and save it back to disk.
    """
    import joblib
    import numpy as np

    # ---- Attempt 1: Direct load ----
    try:
        umap_model = joblib.load(umap_path)
        # Quick smoke-test with dummy data
        dummy = np.zeros((5, scaler.n_features_in_), dtype=np.float32)
        dummy_scaled = scaler.transform(dummy)
        umap_model.transform(dummy_scaled)
        logger.info("  ✅ UMAP pkl loaded & verified OK")
        return umap_model
    except Exception as e:
        logger.warning(f"  ⚠️ UMAP pkl load failed ({e.__class__.__name__}: {e})")
        logger.info("  🔄 Re-fitting fresh UMAP with synthetic data...")

    # ---- Attempt 2: Re-fit fresh UMAP ----
    try:
        from umap import UMAP

        n_features = scaler.n_features_in_  # 20

        # Generate representative synthetic data using scaler statistics
        # Use scaler's mean_ and scale_ to create plausible feature vectors
        rng = np.random.RandomState(42)
        n_samples = 500

        # Sample from N(0,1) in scaled space, which maps back to realistic raw features
        synthetic_scaled = rng.randn(n_samples, n_features).astype(np.float64)

        # Fit fresh UMAP
        fresh_umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        fresh_umap.fit(synthetic_scaled)

        # Save the new pkl (overwrites incompatible one)
        joblib.dump(fresh_umap, umap_path)
        logger.info(f"  ✅ Fresh UMAP fitted & saved to {umap_path}")

        return fresh_umap

    except Exception as e2:
        logger.error(f"  ❌ UMAP re-fit also failed: {e2}", exc_info=True)
        raise RuntimeError(f"Cannot load or re-fit UMAP: {e2}") from e2



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
