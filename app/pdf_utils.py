import io
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def get_page_count(pdf_bytes: bytes) -> int:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return len(pdf.pages)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        return len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception:
        return 0

def extract_pages_as_bytes(pdf_bytes: bytes, page_numbers: list) -> bytes | None:
    """Extract specific pages into a new PDF bytes object."""
    try:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        for pg in sorted(page_numbers):
            idx = pg - 1
            if 0 <= idx < len(reader.pages):
                writer.add_page(reader.pages[idx])
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()
    except Exception as exc:
        logger.error("extract_pages_as_bytes failed: %s", exc)
        return None
