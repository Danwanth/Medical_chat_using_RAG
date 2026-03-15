import os
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL = "deepseek/deepseek-chat"

def generate(prompt: str) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        print("[OpenRouter] API key not set")
        return None

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
            },
            timeout=60,
        )

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("[OpenRouter] error:", e)
        return None


def is_available():
    return OPENROUTER_API_KEY is not None