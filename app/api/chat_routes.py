"""
chat_routes.py
--------------
Blueprint: /chat, /analyze
Single Responsibility: Handle LLM Q&A and auto-analysis.
"""
import logging
from flask import Blueprint, request, jsonify

from app.services.llm_service import get_ldct_analysis, get_ldct_chat

chat_bp = Blueprint("chat", __name__)
logger = logging.getLogger("LDCT-ChatRoutes")


@chat_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Accepts: { prediction_result: {...}, lung_region: {...} }
    Returns: { analysis: "markdown string" }

    Triggers full LLM clinical synthesis for a given prediction.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    prediction_result = data.get("prediction_result", {})
    lung_region = data.get("lung_region", {})

    if not prediction_result:
        return jsonify({"error": "Missing 'prediction_result' in request."}), 400

    analysis = get_ldct_analysis(prediction_result, lung_region)
    return jsonify({"analysis": analysis}), 200


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Accepts: { message: "...", context_data: {...} }
    Returns: { reply: "..." }

    Open-ended Q&A with LDCT context.
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"reply": "Please provide a 'message' field."}), 400

    user_message = data["message"].strip()
    context_data = data.get("context_data", {})

    # Simple rule-based quick replies
    lower_msg = user_message.lower()
    if lower_msg in ["hi", "hello", "hey"]:
        return jsonify({"reply": "👋 Hello! I am LDCTalk — your AI radiology assistant. Upload a CT scan to begin."}), 200

    reply = get_ldct_chat(user_message, context_data)
    return jsonify({"reply": reply}), 200
