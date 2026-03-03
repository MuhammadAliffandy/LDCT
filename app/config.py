"""
LDCT App Configuration
Loads all settings from .env file
"""
import os
from dotenv import load_dotenv

# Resolve the absolute path of THIS file so paths work regardless of CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # → LDCT/app/
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)               # → LDCT/

# Load .env from project root
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "ldct_secret_default")
    UPLOAD_FOLDER = os.path.join(_PROJECT_ROOT, "temp_uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

    # Model Paths — absolute, always correct
    # New improved model: Mod-Seg-SE(2) v2 trained with HU windowing + deeper encoder
    KERAS_MODEL_PATH = os.path.join(_PROJECT_ROOT, "ldct_improved_se2", "best_mod_seg_se2_v2.keras")

    # CT Preprocessing — HU Window (Lung Window)
    # Matches the training pipeline in LDCT-improved-se2.ipynb exactly.
    # For raw DICOM: clip to [center - width/2, center + width/2] before normalizing.
    # For rendered PNG: apply percentile-based soft clipping to replicate same effect.
    HU_WINDOW_CENTER = -600    # Standard lung window center (HU)
    HU_WINDOW_WIDTH  = 1500    # Standard lung window width  (HU)
