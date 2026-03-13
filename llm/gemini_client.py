"""
llm/gemini_client.py

Thin wrapper around the Google Gemini API (gemini-1.5-flash).

Robustness features:
  - load_dotenv() is called inside _get_client() (not at module level) so the
    key is always freshly read on each initialisation attempt.
  - The singleton is cleared after any authentication/permission error so that
    a corrected key is picked up on the next call without restarting the server.
  - Returns None on any error so callers can gracefully fall back to retrieval-only.

Setup:
    1. pip install google-generativeai python-dotenv
    2. Create rag_medical_assistant/.env:
           GEMINI_API_KEY=your_key_here     (no quotes)
    3. Get a free key at https://aistudio.google.com/apikey
"""

import os
from pathlib import Path
from typing import Optional

# ── Gemini config ──────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-1.5-flash"
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.2

# Cache the live model object; None until first successful init.
_gemini_client = None


def _load_env() -> None:
    """Load the .env file that lives two directories above this file."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=True)  # override=True forces re-read
    except ImportError:
        pass  # python-dotenv not installed; rely on env being set externally


def _get_client():
    """
    Return the initialised Gemini GenerativeModel, or None if unavailable.
    Retries initialisation every call if previously failed (so a valid key
    added after server start will be picked up automatically).
    """
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    # Always reload .env so a newly-written key is picked up without restart
    _load_env()

    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[Gemini] GEMINI_API_KEY not set. LLM features disabled.")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={
                "temperature": TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
        )
        print(f"[Gemini] ✅ Initialised model: {GEMINI_MODEL}")
        return _gemini_client
    except Exception as exc:
        print(f"[Gemini] ❌ Initialisation failed: {exc}")
        _gemini_client = None  # ensure retry on next call
        return None


def generate(prompt: str) -> Optional[str]:
    """
    Send *prompt* to Gemini and return the generated text.
    Returns None on any error so callers can gracefully fall back.
    """
    global _gemini_client
    client = _get_client()
    if client is None:
        return None

    try:
        response = client.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        print(f"[Gemini] Generation error: {exc}")
        # Clear singleton so next call reinitialises (handles token expiry etc.)
        _gemini_client = None
        return None


def is_available() -> bool:
    """Return True if the Gemini client can be initialised successfully."""
    return _get_client() is not None
