"""
llm_service.py
--------------
OpenAI LLM integration for LDCT clinical synthesis and Q&A.
Single Responsibility: Communicate with OpenAI API to generate radiological insights.
"""
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("LDCT-LLMService")

# Class-level descriptions for LDCT dose types
DOSE_DESCRIPTIONS = {
    "Full_Dose": (
        "Full-Dose CT scan: Standard radiation dose protocol. "
        "Images are higher quality with lower noise, suitable for detailed diagnostic evaluation."
    ),
    "Quarter_Dose": (
        "Quarter-Dose (Low-Dose) CT scan: Reduced radiation protocol (25% of standard). "
        "Images may exhibit higher noise artifacts. Commonly used for lung cancer screening (LDCT). "
        "Careful review needed due to potential noise interference with small lesion detection."
    ),
}

_client = None
_llm_ready = False

def _get_client():
    global _client, _llm_ready
    if _client:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

    if not api_key or api_key == "your_openai_api_key_here":
        logger.warning("⚠️ OPENAI_API_KEY not set — LLM features disabled.")
        return None

    try:
        _client = OpenAI(api_key=api_key)
        _llm_ready = True
        logger.info("✅ OpenAI client initialized.")
        return _client
    except Exception as e:
        logger.error(f"OpenAI init error: {e}")
        return None


def get_ldct_analysis(
    prediction_result: dict,
    lung_region: dict = None,
) -> str:
    """
    Generate a structured clinical synthesis using OpenAI.

    Args:
        prediction_result: dict from prediction_service.predict_from_image()
        lung_region: dict from lung_locator.detect_lung_region()

    Returns:
        Formatted clinical analysis string (markdown).
    """
    client = _get_client()
    if not client:
        return _fallback_analysis(prediction_result)

    label = prediction_result.get("prediction_label", "Unknown")
    confidence = prediction_result.get("confidence", 0)
    features = prediction_result.get("all_features", {})
    is_referral = prediction_result.get("is_referral", False)

    dose_desc = DOSE_DESCRIPTIONS.get(label, label)

    lung_info = ""
    if lung_region:
        lung_info = (
            f"\n- Lung Region: {lung_region.get('region_label', 'N/A')}"
            f"\n- Vertical Position: {lung_region.get('vertical_position', 'N/A')}"
            f"\n- Side Distribution: Left {lung_region.get('side_distribution', {}).get('left', 0)}% / "
            f"Right {lung_region.get('side_distribution', {}).get('right', 0)}%"
            f"\n- Number of Regions Detected: {lung_region.get('num_regions_detected', 0)}"
        )

    model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

    system_prompt = (
        "You are LDCTalk, a senior thoracic radiologist AI assistant specializing in Low-Dose CT (LDCT) lung analysis. "
        "Your analyses are evidence-based, concise, and clinically actionable. "
        "Use professional medical language with clear structure. "
        "Always note limitations and recommend follow-up when appropriate.\n\n"
        "IMPORTANT: Provide a complete analysis in this EXACT structure:\n"
        "1. **DOSE ASSESSMENT** — Type of scan and what it means diagnostically\n"
        "2. **LUNG REGION ANALYSIS** — Description of the identified regions and their significance\n"
        "3. **IMAGE QUALITY INDICATORS** — What the texture features tell us about image quality\n"
        "4. **CLINICAL IMPLICATIONS** — What this means for the patient\n"
        "5. **RECOMMENDATION** — Next clinical steps\n"
    )

    user_content = (
        f"Analyze this LDCT result:\n\n"
        f"**Prediction:** {label} (Confidence: {confidence:.1%})\n"
        f"**Scan Description:** {dose_desc}\n"
        f"**Requires Physician Review:** {'Yes — Uncertainty flagged' if is_referral else 'No'}\n"
        f"\n**Lung Region Info:**{lung_info if lung_info else ' Not available'}\n\n"
        f"**Key Image Quality Features:**\n"
        f"- HH Energy (noise proxy): {features.get('HH_Energy', 'N/A')}\n"
        f"- GLCM Contrast: {features.get('GLCM_Contrast', 'N/A')}\n"
        f"- GLCM Homogeneity: {features.get('GLCM_Homogeneity', 'N/A')}\n"
        f"- GLCM Dissimilarity: {features.get('GLCM_Dissimilarity', 'N/A')}\n"
        f"- LL Mean (overall intensity): {features.get('LL_Mean', 'N/A')}\n"
        f"- LL Entropy: {features.get('LL_Entropy', 'N/A')}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=900,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"OpenAI analysis error: {e}")
        return _fallback_analysis(prediction_result)


def get_ldct_chat(user_message: str, context_data: dict) -> str:
    """
    Handle an open-ended clinical question with LDCT context.

    Args:
        user_message: User's question
        context_data: dict with prediction result and lung region info

    Returns:
        LLM response string
    """
    client = _get_client()
    if not client:
        return (
            "The AI assistant is currently unavailable. Please ensure your OPENAI_API_KEY "
            "is configured in the .env file."
        )

    label = context_data.get("prediction_label", "Unknown")
    confidence = context_data.get("confidence", 0)
    features = context_data.get("all_features", {})
    lung_region = context_data.get("lung_region", {})
    model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

    system_prompt = (
        "You are LDCTalk, a senior thoracic radiologist AI assistant. "
        "Answer questions about the LDCT scan result clearly and professionally. "
        "Be concise but thorough. Use markdown formatting when helpful.\n\n"
        f"--- Current Case Context ---\n"
        f"Prediction: {label} (Confidence: {confidence:.1%})\n"
        f"Lung Region: {lung_region.get('region_label', 'Not analyzed')}\n"
        f"Vertical Position: {lung_region.get('vertical_position', 'N/A')}\n"
        f"Key Features: HH_Energy={features.get('HH_Energy', 'N/A')}, "
        f"GLCM_Contrast={features.get('GLCM_Contrast', 'N/A')}, "
        f"GLCM_Homogeneity={features.get('GLCM_Homogeneity', 'N/A')}\n"
        "---\n"
        "If asked about topics unrelated to radiology or this case, politely decline."
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.5,
            max_tokens=600,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"OpenAI chat error: {e}")
        return f"An error occurred while generating a response: {str(e)}"


def _fallback_analysis(prediction_result: dict) -> str:
    """Fallback when LLM is unavailable — rule-based summary."""
    label = prediction_result.get("prediction_label", "Unknown")
    confidence = prediction_result.get("confidence", 0)
    is_referral = prediction_result.get("is_referral", False)

    desc = DOSE_DESCRIPTIONS.get(label, label)
    referral_note = "\n\n⚠️ **Physician review recommended** due to low model confidence." if is_referral else ""

    return (
        f"## LDCT Analysis Summary\n\n"
        f"**Prediction:** {label} ({confidence:.1%} confidence)\n\n"
        f"**Description:** {desc}{referral_note}\n\n"
        f"*Note: AI clinical synthesis is unavailable — configure OPENAI_API_KEY for detailed analysis.*"
    )
