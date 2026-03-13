"""
rag/rag_pipeline.py

Hybrid Retrieval-Augmented Generation pipeline with Gemini LLM layer.

Flow:
  1. Classify the query: medical vs non-medical (Gemini)
  2a. Non-medical → Gemini redirects the user politely           (mode: non_medical)
  2b. Medical → retrieve top-K chunks from ChromaDB
  3.  Relevance check:
        - Irrelevant / empty → Gemini general fallback           (mode: llm_fallback)
        - Relevant → Gemini rewrites the RAG answer              (mode: gemini)
  4.  If Gemini is unavailable anywhere → retrieval-only         (mode: retrieval_only)

No paid API key is required for the base retrieval path.
Set GEMINI_API_KEY in a .env file to enable LLM features.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Allow sibling package imports
_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from utils.prompts import (
    MEDICAL_DISCLAIMER,
    RAG_SYSTEM_PROMPT,
    NO_INFORMATION_RESPONSE,
    CONTEXT_TEMPLATE,
    UNIFIED_PROMPT,
)

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = _PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "medquad"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5                       # number of retrieved chunks
MIN_RELEVANCE_DISTANCE = 1.3    # cosine distance threshold (0 = perfect, 2 = opposite)

# Ollama settings (legacy — used if USE_OLLAMA = True and Gemini unavailable)
USE_OLLAMA = False
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ── Singleton cache ───────────────────────────────────────────────────────────
_embedder: SentenceTransformer | None = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[RAG] Loading embedding model: {EMBED_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DIR}.\n"
                "Run: python ingestion/build_vector_db.py"
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
        print(f"[RAG] Loaded ChromaDB collection '{COLLECTION_NAME}' "
              f"({_collection.count()} chunks)")
    return _collection


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Return top-k relevant chunks for *question*."""
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({"document": doc, "metadata": meta, "distance": dist})

    return retrieved


# ── Generation helpers ────────────────────────────────────────────────────────
def _generate_with_ollama(prompt: str) -> str | None:
    """Call local Ollama server for text generation (legacy path)."""
    try:
        import requests
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        print(f"[RAG] Ollama error: {exc}. Falling back to retrieval-only mode.")
        return None


def _extract_answer_from_chunks(retrieved: List[Dict]) -> str:
    """
    Retrieval-only mode: return the Answer portion of the best chunk,
    stripping the 'Question: ...\\nAnswer: ' prefix added by the data loader.
    """
    if not retrieved:
        return NO_INFORMATION_RESPONSE
    best = retrieved[0]["document"]
    if "\nAnswer: " in best:
        return best.split("\nAnswer: ", 1)[1].strip()
    return best.strip()


def _build_sources(retrieved: List[Dict]) -> List[Dict]:
    """Build the source citation list from retrieved chunks."""
    sources = []
    for i, r in enumerate(retrieved):
        meta = r["metadata"]
        sources.append({
            "index": i + 1,
            "focus": meta.get("focus", "Unknown"),
            "question": meta.get("question", ""),
            "qtype": meta.get("qtype", "general"),
            "source": meta.get("source", "MedQuAD"),
            "url": meta.get("url", ""),
            "filename": meta.get("filename", ""),
            "distance": round(r["distance"], 4),
        })
    return sources


# ── Main pipeline ─────────────────────────────────────────────────────────────
def ask(question: str) -> Dict[str, Any]:
    """
    Hybrid RAG + LLM pipeline.

    Parameters
    ----------
    question : str  — user's query

    Returns
    -------
    dict with keys:
        answer      : str
        sources     : list[dict]  — metadata for each retrieved chunk
        disclaimer  : str
        mode        : str         — "gemini" | "non_medical" | "llm_fallback"
                                     | "retrieval_only" | "no_results"
    """
    # Lazy-import the LLM modules so the system still works if they fail
    try:
        from llm.query_classifier import classify_query
        from llm.gemini_client import generate, is_available
        gemini_ready = is_available()
    except ImportError:
        gemini_ready = False

    # ── Step 1: Query classification ──────────────────────────────────────────
    if gemini_ready:
        query_type = classify_query(question)
    else:
        query_type = "medical"   # safe default

    # ── Step 2a: Non-medical → Gemini answers as a general AI assistant ─────────
    if query_type == "non_medical":
        prompt = UNIFIED_PROMPT.format(question=question, rag_output="")
        llm_answer = generate(prompt) if gemini_ready else None
        if llm_answer:
            return {
                "answer": llm_answer,
                "sources": [],
                "disclaimer": MEDICAL_DISCLAIMER,
                "mode": "general",
            }
        # Gemini failed — return a friendly static message
        return {
            "answer": (
                "🔄 This chatbot is designed to answer medical and health-related "
                "questions. Your query does not appear to be related to a medical "
                "condition or health topic.\n\n"
                "Please try asking about symptoms, diseases, treatments, or other "
                "health-related topics. For example:\n"
                "- *What are the symptoms of dengue fever?*\n"
                "- *How is Type 2 diabetes treated?*\n"
                "- *What causes high blood pressure?*"
            ),
            "sources": [],
            "disclaimer": MEDICAL_DISCLAIMER,
            "mode": "non_medical",
        }

    # ── Step 2b: Medical → RAG retrieval ─────────────────────────────────────
    retrieved = retrieve(question, top_k=TOP_K)

    # ── Step 3: Relevance check ───────────────────────────────────────────────
    rag_failed = not retrieved or retrieved[0]["distance"] > MIN_RELEVANCE_DISTANCE

    if rag_failed:
        # ── Step 3a: RAG fallback — Gemini answers from its own knowledge ────────
        if gemini_ready:
            prompt = UNIFIED_PROMPT.format(question=question, rag_output="")
            llm_answer = generate(prompt)
            if llm_answer:
                return {
                    "answer": llm_answer,
                    "sources": [],
                    "disclaimer": MEDICAL_DISCLAIMER,
                    "mode": "llm_fallback",
                }
        # Both RAG and LLM failed
        return {
            "answer": NO_INFORMATION_RESPONSE,
            "sources": [],
            "disclaimer": MEDICAL_DISCLAIMER,
            "mode": "no_results",
        }

    # ── Step 4: Successful retrieval — build sources ──────────────────────────
    sources = _build_sources(retrieved)

    # ── Step 5: Generate/rewrite answer ──────────────────────────────────────
    # Priority: Gemini rewrite > Ollama > retrieval-only

    if gemini_ready:
        raw_answer = _extract_answer_from_chunks(retrieved)
        prompt = UNIFIED_PROMPT.format(question=question, rag_output=raw_answer)
        llm_answer = generate(prompt)
        if llm_answer:
            return {
                "answer": llm_answer,
                "sources": sources,
                "disclaimer": MEDICAL_DISCLAIMER,
                "mode": "gemini",
            }

    if USE_OLLAMA:
        context_parts = []
        for i, r in enumerate(retrieved):
            context_parts.append(
                CONTEXT_TEMPLATE.format(
                    idx=i + 1,
                    focus=r["metadata"].get("focus", "Unknown"),
                    qtype=r["metadata"].get("qtype", "general"),
                    content=r["document"],
                )
            )
        context = "\n\n".join(context_parts)
        prompt = RAG_SYSTEM_PROMPT.format(context=context, question=question)
        generated = _generate_with_ollama(prompt)
        if generated:
            return {
                "answer": generated,
                "sources": sources,
                "disclaimer": MEDICAL_DISCLAIMER,
                "mode": "ollama",
            }

    # ── Retrieval-only fallback ───────────────────────────────────────────────
    return {
        "answer": _extract_answer_from_chunks(retrieved),
        "sources": sources,
        "disclaimer": MEDICAL_DISCLAIMER,
        "mode": "retrieval_only",
    }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "What are the symptoms of anemia?",
        "How is diabetes treated?",
        "Who is the president of the United States?",
        "tsunami",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = ask(q)
        print(f"Mode: {result['mode']}")
        print(f"A: {result['answer'][:400]}...")
        print(f"Sources: {len(result['sources'])}")
