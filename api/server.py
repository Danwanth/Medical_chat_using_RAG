"""
api/server.py

FastAPI backend for the Responsible Medical RAG Assistant.

Endpoints:
    POST /ask          — main Q&A endpoint
    GET  /health       — health check
    GET  /             — welcome message

Usage:
    uvicorn api.server:app --reload --port 8000
"""

import sys
from pathlib import Path

# Allow sibling imports
_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from safety.safety_filter import check_safety
from rag.rag_pipeline import ask as rag_ask
from utils.prompts import MEDICAL_DISCLAIMER

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hybrid Medical RAG + LLM Assistant",
    description=(
        "A medical question-answering API powered by Retrieval-Augmented Generation "
        "over the MedQuAD dataset, enhanced with a Gemini LLM layer for query "
        "classification, answer rewriting, and intelligent fallback. "
        "Includes safety filters, citations, and disclaimers."
    ),
    version="2.0.0",
)

# CORS — allow the Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ─────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        example="What are the symptoms of anemia?",
    )


class SourceItem(BaseModel):
    index: int
    focus: str
    question: str
    qtype: str
    source: str
    url: Optional[str] = ""
    filename: Optional[str] = ""
    distance: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    disclaimer: str
    safe: bool
    mode: Optional[str] = "retrieval_only"
    # mode values:
    #   gemini         — RAG + Gemini rewrite (best quality)
    #   non_medical    — query classified as non-medical; LLM redirect
    #   llm_fallback   — RAG failed; Gemini answered from general knowledge
    #   retrieval_only — Gemini unavailable; raw RAG passage returned
    #   no_results     — RAG failed and Gemini unavailable
    #   ollama         — local Ollama model used
    #   blocked        — safety filter triggered


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Responsible Medical RAG Assistant API",
        "docs": "/docs",
        "usage": "POST /ask  with JSON body: {\"question\": \"...\"}",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "Medical RAG API"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    """
    Submit a medical question and receive a grounded answer.

    - Questions are first screened by the safety filter.
    - Safe questions are answered using the RAG pipeline over MedQuAD.
    - Every response includes source citations and a medical disclaimer.
    """
    question = req.question.strip()

    # 1. Safety check
    is_safe, refusal_message = check_safety(question)
    if not is_safe:
        return AskResponse(
            answer=refusal_message,
            sources=[],
            disclaimer=MEDICAL_DISCLAIMER,
            safe=False,
            mode="blocked",
        )

    # 2. RAG pipeline
    try:
        result = rag_ask(question)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector database not ready. "
                "Please run: python ingestion/build_vector_db.py\n"
                f"Details: {exc}"
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # 3. Build response
    source_items = [SourceItem(**s) for s in result["sources"]]

    return AskResponse(
        answer=result["answer"],
        sources=source_items,
        disclaimer=result["disclaimer"],
        safe=True,
        mode=result.get("mode", "retrieval_only"),
    )
