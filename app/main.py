
from __future__ import annotations
import logging, os, traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.graph import run_claim_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        logger.error("GEMINI_API_KEY is NOT set!")
    else:
        logger.info("Claim Processing Pipeline started — Gemini API key present ✓")
    yield

app = FastAPI(
    title="Claim Processing Pipeline (Gemini)",
    description="FastAPI + LangGraph pipeline using Google Gemini to classify and extract insurance claim PDFs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "claim-processing-pipeline",
        "api_key_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite"),
    }

@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(..., description="Unique claim identifier"),
    file: UploadFile = File(..., description="PDF file of the insurance claim"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set on server.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("Processing claim_id=%s  file=%s  size=%.1f KB",
                claim_id, file.filename, len(pdf_bytes) / 1024)
    try:
        result = run_claim_pipeline(claim_id=claim_id, pdf_bytes=pdf_bytes)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error("Pipeline failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
