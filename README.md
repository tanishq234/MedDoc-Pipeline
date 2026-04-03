<<<<<<< HEAD
# MedDoc-Pipeline
AI-powered Claim Processing Pipeline built with FastAPI and LangGraph. It classifies PDF pages using Gemini API and routes them to specialized agents for extracting identity, discharge, and billing details. Outputs structured JSON, improving efficiency by processing only relevant pages.
=======
# Claim Processing Pipeline (Google Gemini)

FastAPI + LangGraph service that classifies insurance claim PDFs using Google Gemini.

## Setup

```powershell
pip install -r requirements.txt
$env:GEMINI_API_KEY = "your_key_here"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs to use the Swagger UI.

## API

POST /api/process
- claim_id (string, form field)
- file (PDF, file upload)

## Pipeline Flow
START → Segregator (classifies all pages) → ID Agent + Discharge Agent + Bill Agent → Aggregator → END
>>>>>>> e44025c (This is Add AI-based claim processing pipeline using FastAPI and LangGraph)
