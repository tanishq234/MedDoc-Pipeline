"""
bill_agent.py - Extracts itemized billing information using Gemini vision.
"""
from __future__ import annotations
import json, logging, os, re
from google import genai
from google.genai import types
from app.models import ClaimState
from app.pdf_utils import extract_pages_as_bytes

logger = logging.getLogger(__name__)

SYSTEM = """Extract every line item and all billing details from these insurance claim PDF pages.
Respond ONLY with a valid JSON object, no markdown, no explanation:
{
  "billing_entity": null,
  "bill_number": null,
  "bill_date": null,
  "patient_name": null,
  "patient_id": null,
  "items": [
    {"date": null, "description": "", "category": "", "quantity": 0, "unit_price": 0, "total": 0}
  ],
  "subtotal": null,
  "discount": null,
  "taxes": null,
  "total_amount": null,
  "insurance_payment": null,
  "patient_responsibility": null,
  "currency": "USD",
  "payment_mode": null,
  "notes": null
}
Extract EVERY line item visible. Do not skip any charges."""

def run_bill_agent(state: ClaimState) -> ClaimState:
    pages = sorted(set(
        state.page_classification.get("itemized_bill", []) +
        state.page_classification.get("cash_receipt", [])
    ))
    if not pages:
        logger.info("[Bill Agent] No bill pages found")
        state.bill_data = {"note": "No itemized bill pages detected"}
        return state

    logger.info("[Bill Agent] Processing pages: %s", pages)
    page_pdf = extract_pages_as_bytes(state.pdf_bytes, pages)
    if not page_pdf:
        state.bill_data = {"error": "Could not extract pages"}
        return state

    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=page_pdf, mime_type="application/pdf"),
                types.Part.from_text(text="Extract every line item and all billing details. Return only JSON."),
            ],
            config=types.GenerateContentConfig(system_instruction=SYSTEM),
        )
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        state.bill_data = json.loads(raw)
        logger.info("[Bill Agent] Extraction successful - total: %s", state.bill_data.get("total_amount"))
    except Exception as exc:
        logger.error("[Bill Agent] Failed: %s", exc)
        state.bill_data = {"error": str(exc)}
    return state
