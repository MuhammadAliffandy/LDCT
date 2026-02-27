"""
Shared utilities for API routes.
"""
from flask import current_app


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in current_app.config.get("ALLOWED_EXTENSIONS", {"png", "jpg", "jpeg"})
