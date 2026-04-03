from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

DOCUMENT_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

class ClaimState(BaseModel):
    claim_id: str
    pdf_bytes: bytes
    total_pages: int = 0
    page_classification: Dict[str, List[int]] = Field(default_factory=dict)
    identity_data: Optional[Dict[str, Any]] = None
    discharge_data: Optional[Dict[str, Any]] = None
    bill_data: Optional[Dict[str, Any]] = None
    final_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
