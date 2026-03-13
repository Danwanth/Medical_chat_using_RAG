"""
llm/query_classifier.py

Classify a user query as "medical" or "non_medical" using Gemini.

If Gemini is unavailable, defaults to "medical" (safe: RAG will determine
relevance itself).
"""

import sys
from pathlib import Path
from typing import Literal

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from llm.gemini_client import generate

# ── Classification prompt ──────────────────────────────────────────────────────
_CLASSIFY_PROMPT = """\
You are a medical query classifier. Your only job is to decide whether the \
user's question is related to medicine, health, or biology.

Respond with EXACTLY one word — either:
  MEDICAL
  NON_MEDICAL

Rules:
- MEDICAL: symptoms, diseases, treatments, medications, anatomy, health \
conditions, nutrition related to health, mental health disorders, medical \
procedures, first aid, etc.
- NON_MEDICAL: geography, history, politics, cooking, sports, entertainment, \
general science, natural disasters, technology, etc.

User question: "{question}"

Classification (one word only):"""

QueryType = Literal["medical", "non_medical"]


def classify_query(question: str) -> QueryType:
    """
    Return "medical" or "non_medical".

    Falls back to "medical" if Gemini is unavailable to avoid blocking
    legitimate medical queries.
    """
    prompt = _CLASSIFY_PROMPT.format(question=question.strip())
    response = generate(prompt)

    if response is None:
        print("[Classifier] Gemini unavailable – defaulting to 'medical'")
        return "medical"

    # Normalise the response
    label = response.strip().upper().split()[0] if response.strip() else ""

    if label == "NON_MEDICAL":
        print(f"[Classifier] '{question[:60]}' → non_medical")
        return "non_medical"

    print(f"[Classifier] '{question[:60]}' → medical")
    return "medical"
