"""
id_agent.py - Extracts identity/patient information using Gemini vision.
"""
from __future__ import annotations
import json, logging, os, re
from google import genai
from google.genai import types
from app.models import ClaimState
from app.pdf_utils import extract_pages_as_bytes

logger = logging.getLogger(__name__)

SYSTEM = """Extract all identity and policy information from these insurance claim PDF pages.
Respond ONLY with a valid JSON object, no markdown, no explanation:
{
  "patient_name": null,
  "date_of_birth": null,
  "gender": null,
  "id_number": null,
  "id_type": null,
  "policy_number": null,
  "insurance_provider": null,
  "member_id": null,
  "address": null,
  "contact_number": null,
  "email": null,
  "additional_info": {}
}
Use null for any field not found. Do not guess values."""

def run_id_agent(state: ClaimState) -> ClaimState:
    id_pages = sorted(set(
        state.page_classification.get("identity_document", []) +
        state.page_classification.get("claim_forms", [])
    ))
    if not id_pages:
        logger.info("[ID Agent] No identity pages found")
        state.identity_data = {"note": "No identity document pages detected"}
        return state

    logger.info("[ID Agent] Processing pages: %s", id_pages)
    page_pdf = extract_pages_as_bytes(state.pdf_bytes, id_pages)
    if not page_pdf:
        state.identity_data = {"error": "Could not extract pages"}
        return state

    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=page_pdf, mime_type="application/pdf"),
                types.Part.from_text(text="Extract all identity and policy information from these pages. Return only JSON."),
            ],
            config=types.GenerateContentConfig(system_instruction=SYSTEM),
        )
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        state.identity_data = json.loads(raw)
        logger.info("[ID Agent] Extraction successful")
    except Exception as exc:
        logger.error("[ID Agent] Failed: %s", exc)
        state.identity_data = {"error": str(exc)}
    return state
