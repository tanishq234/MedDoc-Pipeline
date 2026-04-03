"""
segregator.py - Classifies every PDF page using Gemini vision.
Sends the full PDF as binary bytes - works for image-protected PDFs.
"""
from __future__ import annotations
import json, logging, os, re
from typing import Dict, List
from google import genai
from google.genai import types
from app.models import ClaimState, DOCUMENT_TYPES
from app.pdf_utils import get_page_count

logger = logging.getLogger(__name__)

PROMPT = """You are a document classification specialist for insurance claims.
This is a multi-page insurance claim PDF. Look at EVERY page carefully and classify each one.

Document types you must use:
- claim_forms           : insurance claim forms, patient registration, consent forms, insurance verification
- cheque_or_bank_details: cheques, bank account details, payment proofs
- identity_document     : government ID card, Aadhaar, PAN, passport, driving licence
- itemized_bill         : hospital/pharmacy bills listing individual charges with amounts
- discharge_summary     : clinical discharge summary or discharge certificate
- prescription          : doctor prescription or medication order
- investigation_report  : lab reports, blood tests, radiology, pathology results
- cash_receipt          : cash payment receipts
- other                 : referral letters, appointment letters, questionnaires, medical history forms

IMPORTANT: Respond with ONLY a valid JSON object. No markdown. No explanation. No extra text.
Format: {"classifications": [{"page": 1, "document_type": "claim_forms"}, {"page": 2, "document_type": "identity_document"}, ...]}

Classify every single page. Every page number must appear exactly once.
"""

def run_segregator(state: ClaimState) -> ClaimState:
    logger.info("[Segregator] Starting for claim %s", state.claim_id)

    total_pages = get_page_count(state.pdf_bytes)
    state.total_pages = total_pages
    logger.info("[Segregator] Total pages: %d", total_pages)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("[Segregator] GEMINI_API_KEY not set!")
        state.page_classification = {"other": list(range(1, total_pages + 1))}
        return state

    client = genai.Client(api_key=api_key)

    # Try models in order - lite first to save quota
    models = [
        os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite"),
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    for model in models:
        try:
            logger.info("[Segregator] Trying model: %s", model)

            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(
                        data=state.pdf_bytes,
                        mime_type="application/pdf",
                    ),
                    types.Part.from_text(
                        text=PROMPT + f"\n\nThis PDF has {total_pages} pages. Classify ALL pages 1 to {total_pages}."
                    ),
                ],
            )

            raw = response.text.strip()
            logger.info("[Segregator] Response preview: %s", raw[:400])

            # Remove markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

            parsed = json.loads(raw)
            classification: Dict[str, List[int]] = {dt: [] for dt in DOCUMENT_TYPES}

            for entry in parsed.get("classifications", []):
                pg = entry.get("page")
                doc_type = entry.get("document_type", "other")
                if doc_type not in DOCUMENT_TYPES:
                    doc_type = "other"
                if isinstance(pg, int) and 1 <= pg <= total_pages:
                    classification[doc_type].append(pg)
                    logger.info("  Page %d -> %s", pg, doc_type)

            # Fill any missed pages with "other"
            classified = {p for pages in classification.values() for p in pages}
            for pg in range(1, total_pages + 1):
                if pg not in classified:
                    logger.warning("  Page %d not classified -> other", pg)
                    classification["other"].append(pg)

            state.page_classification = {k: sorted(v) for k, v in classification.items() if v}
            logger.info("[Segregator] SUCCESS: %s", state.page_classification)
            return state

        except json.JSONDecodeError as exc:
            logger.error("[Segregator] JSON parse error with %s: %s", model, exc)
            logger.error("[Segregator] Raw was: %s", raw[:500] if 'raw' in dir() else 'N/A')
        except Exception as exc:
            logger.warning("[Segregator] Model %s failed: %s", model, exc)

    logger.error("[Segregator] All models failed")
    state.page_classification = {"other": list(range(1, total_pages + 1))}
    return state
