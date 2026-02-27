"""
web_routes.py
-------------
Blueprint: / (UI routes)
Single Responsibility: Serve the frontend HTML and health check.
"""
from flask import Blueprint, render_template, jsonify
from app.services.model_loader import is_ready, get_load_error

web_bp = Blueprint("web", __name__)


@web_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@web_bp.route("/health", methods=["GET"])
def health():
    if is_ready():
        return jsonify({"status": "ok", "models": "loaded"}), 200
    else:
        err = get_load_error()
        return jsonify({"status": "degraded", "error": err}), 503
