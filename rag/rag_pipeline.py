"""
rag/rag_pipeline.py

Core Retrieval-Augmented Generation pipeline.

Flow:
  1. Embed the user question with SentenceTransformers
  2. Retrieve top-K chunks from ChromaDB
  3. Build a grounded prompt
  4. Generate answer via Ollama (optional) or return top retrieved passage
  5. Return answer + sources + disclaimer

No paid API key is required. The system works in two modes:
  - RETRIEVAL-ONLY (default): returns the most relevant retrieved passage.
  - OLLAMA (optional): passes context to a local Ollama LLM for generation.
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
)

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = _PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "medquad"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5                       # number of retrieved chunks
MIN_RELEVANCE_DISTANCE = 1.3    # cosine distance threshold (0 = perfect, 2 = opposite)

# Ollama settings (only used if USE_OLLAMA = True)
USE_OLLAMA = False              # set True if Ollama is running locally
OLLAMA_MODEL = "llama3.2:1b"   # or "mistral", "phi3", etc.
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


# ── Generation ────────────────────────────────────────────────────────────────
def _generate_with_ollama(prompt: str) -> str:
    """Call local Ollama server for text generation."""
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
    stripping the 'Question: ...\nAnswer: ' prefix added by the data loader.
    """
    if not retrieved:
        return NO_INFORMATION_RESPONSE
    best = retrieved[0]["document"]
    # Strip the "Question: ... \nAnswer: " prefix
    if "\nAnswer: " in best:
        return best.split("\nAnswer: ", 1)[1].strip()
    return best.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def ask(question: str) -> Dict[str, Any]:
    """
    Full RAG pipeline.

    Parameters
    ----------
    question : str  — user's medical question

    Returns
    -------
    dict with keys:
        answer      : str
        sources     : list[dict]  — metadata for each retrieved chunk
        disclaimer  : str
        mode        : str         — "retrieval_only" | "ollama"
    """
    # 1. Retrieve relevant chunks
    retrieved = retrieve(question, top_k=TOP_K)

    # 2. Check relevance — if all chunks are too far away, refuse
    if not retrieved or retrieved[0]["distance"] > MIN_RELEVANCE_DISTANCE:
        return {
            "answer": NO_INFORMATION_RESPONSE,
            "sources": [],
            "disclaimer": MEDICAL_DISCLAIMER,
            "mode": "no_results",
        }

    # 3. Build source list for the response
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

    # 4. Generate answer
    mode = "retrieval_only"

    if USE_OLLAMA:
        # Build context string for the prompt
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
            answer = generated
            mode = "ollama"
        else:
            answer = _extract_answer_from_chunks(retrieved)
    else:
        answer = _extract_answer_from_chunks(retrieved)

    return {
        "answer": answer,
        "sources": sources,
        "disclaimer": MEDICAL_DISCLAIMER,
        "mode": mode,
    }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "What are the symptoms of anemia?",
        "How is diabetes treated?",
        "What causes high blood pressure?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = ask(q)
        print(f"A: {result['answer'][:400]}...")
        print(f"Sources: {len(result['sources'])} | Mode: {result['mode']}")
