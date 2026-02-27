"""
LDCT Flask Application Factory
"""
import os
import logging
import sys
from flask import Flask
from app.config import Config

# Setup Global Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("LDCT-App")


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Register Blueprints
    from app.api.predict_routes import predict_bp
    from app.api.heatmap_routes import heatmap_bp
    from app.api.chat_routes import chat_bp
    from app.api.web_routes import web_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(heatmap_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(web_bp)

    # ---- Eager Model Loading ----
    # Load all ML models NOW at startup so the first request doesn't wait
    # and is_ready() is True before any request is handled.
    logger.info("🚀 Starting LDCT model loading at app startup...")
    from app.services.model_loader import load_all_models, is_ready, get_load_error
    load_all_models()

    if is_ready():
        logger.info("🟢 LDCT models ready — server accepting predictions.")
    else:
        logger.error(f"🔴 Model loading failed: {get_load_error()}")
        logger.error("   Server will start but predictions will return errors.")

    return app
