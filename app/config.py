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
    KERAS_MODEL_PATH = os.path.join(_PROJECT_ROOT, "ldct_model_new", "mod_seg_se2_model.h5")
