"""
discharge_agent.py - Extracts discharge summary information using Gemini vision.
"""
from __future__ import annotations
import json, logging, os, re
from google import genai
from google.genai import types
from app.models import ClaimState
from app.pdf_utils import extract_pages_as_bytes

logger = logging.getLogger(__name__)

SYSTEM = """Extract all discharge summary and clinical information from these insurance claim PDF pages.
Respond ONLY with a valid JSON object, no markdown, no explanation:
{
  "hospital_name": null,
  "hospital_address": null,
  "patient_name": null,
  "uhid": null,
  "admission_date": null,
  "discharge_date": null,
  "length_of_stay_days": null,
  "ward_type": null,
  "diagnosis_primary": null,
  "diagnosis_secondary": [],
  "icd_codes": [],
  "procedures_performed": [],
  "physician_name": null,
  "physician_specialisation": null,
  "treatment_summary": null,
  "discharge_condition": null,
  "discharge_medications": [],
  "follow_up_instructions": null,
  "additional_info": {}
}
Use null for missing fields. Do not guess values."""

def run_discharge_agent(state: ClaimState) -> ClaimState:
    pages = state.page_classification.get("discharge_summary", [])
    if not pages:
        logger.info("[Discharge Agent] No discharge pages found")
        state.discharge_data = {"note": "No discharge summary pages detected"}
        return state

    logger.info("[Discharge Agent] Processing pages: %s", pages)
    page_pdf = extract_pages_as_bytes(state.pdf_bytes, pages)
    if not page_pdf:
        state.discharge_data = {"error": "Could not extract pages"}
        return state

    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=page_pdf, mime_type="application/pdf"),
                types.Part.from_text(text="Extract all discharge summary and clinical information. Return only JSON."),
            ],
            config=types.GenerateContentConfig(system_instruction=SYSTEM),
        )
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        state.discharge_data = json.loads(raw)
        logger.info("[Discharge Agent] Extraction successful")
    except Exception as exc:
        logger.error("[Discharge Agent] Failed: %s", exc)
        state.discharge_data = {"error": str(exc)}
    return state
