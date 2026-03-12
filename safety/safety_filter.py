"""
safety/safety_filter.py

Responsible AI safety filter for the medical assistant.
Detects queries that fall into flagged categories and returns
a safe, helpful refusal message instead of processing the query.
"""

import re
from typing import Tuple

# ── Emergency resources (always appended to crisis responses) ─────────────────
EMERGENCY_RESOURCES = (
    "\n\n🆘 **If you or someone you know is in crisis, please reach out immediately:**\n"
    "- **National Suicide Prevention Lifeline:** Call or text **988** (US)\n"
    "- **Crisis Text Line:** Text HOME to **741741**\n"
    "- **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/\n"
    "- **Emergency Services:** Call **911** (US) or your local emergency number"
)

# ── Pattern definitions ───────────────────────────────────────────────────────
_PATTERNS = {
    "self_harm": {
        "patterns": [
            r"\b(self[- ]?harm|self[- ]?injur|cut(ting)? myself|hurt(ing)? myself)\b",
            r"\b(want to (hurt|harm|kill|end) (my|him|her|them|your)self)\b",
        ],
        "response": (
            "💙 It sounds like you may be going through a really difficult time. "
            "This assistant is not equipped to provide support for self-harm — "
            "please know that you are not alone and help is available."
            + EMERGENCY_RESOURCES
        ),
    },
    "suicide": {
        "patterns": [
            r"\b(suicid(e|al|ality)|kill (my|him|her|them)self|end(ing)? (my|his|her|their) life)\b",
            r"\b(want(ing)? to die|don'?t want to (live|be alive))\b",
            r"\b(how (to|do i|can i) (commit|attempt) suicide)\b",
        ],
        "response": (
            "💙 I'm concerned about what you've shared. "
            "This assistant is not able to help with this topic — "
            "but trained counselors are available right now and want to hear from you."
            + EMERGENCY_RESOURCES
        ),
    },
    "drug_abuse": {
        "patterns": [
            r"\b(how (to|do i) (get|buy|obtain|synthesize|make) (drugs|narcotics|opioids|heroin|meth|cocaine|fentanyl))\b",
            r"\b(overdose (on purpose|intentionally))\b",
            r"\b(street drugs|recreational (drug|substance) (use|abuse))\b",
            r"\b(drug(s)? (to get|for getting) high)\b",
        ],
        "response": (
            "⚠️ I'm not able to provide information that could facilitate substance misuse. "
            "If you or someone you know is struggling with substance use, confidential help is available:\n\n"
            "- **SAMHSA National Helpline:** 1-800-662-4357 (free, confidential, 24/7)\n"
            "- **Crisis Text Line:** Text HOME to 741741\n\n"
            "Please consult a licensed healthcare professional for medical questions about medications."
        ),
    },
    "illegal_advice": {
        "patterns": [
            r"\b(prescription (drugs?|medication|pills?) without (a )?prescription)\b",
            r"\b(obtain (controlled|prescription) (substances?|drugs?|medication) (illegally|without))\b",
            r"\b(forge (a )?prescription)\b",
            r"\b(sell (prescription|drugs?|medication|pills?))\b",
        ],
        "response": (
            "⚠️ I'm unable to provide guidance on obtaining medications or substances illegally. "
            "This is both dangerous and against the law. "
            "Please speak with a licensed healthcare provider who can safely prescribe appropriate treatments."
        ),
    },
    "personal_diagnosis": {
        "patterns": [
            r"\b(do i have|diagnose me (with)?|am i (sick|ill|dying|infected))\b",
            r"\b(what('?s| is) wrong with me|tell me if i have)\b",
            r"\b(i (think|believe|feel) i have (cancer|diabetes|hiv|aids|covid))\b",
        ],
        "response": (
            "⚕️ I'm not able to diagnose medical conditions — only a qualified healthcare "
            "professional can do that after a proper clinical evaluation. "
            "If you are experiencing symptoms, please consult your doctor or visit an urgent care clinic. "
            "I *can* provide general educational information about medical topics — feel free to ask!"
        ),
    },
}


def check_safety(query: str) -> Tuple[bool, str]:
    """
    Analyse a user query for safety concerns.

    Returns
    -------
    (is_safe, message)
        is_safe  : True  → query is safe to process
                   False → query is flagged; return *message* to the user
        message  : Safe refusal message (empty string when is_safe is True)
    """
    query_lower = query.lower()

    for category, config in _PATTERNS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                print(f"[SafetyFilter] Query flagged as '{category}': {query[:80]}...")
                return False, config["response"]

    return True, ""


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What are the symptoms of anemia?",
        "How do I treat high blood pressure?",
        "I want to kill myself",
        "How can I get prescription drugs without a prescription?",
        "Do I have cancer?",
        "What is the treatment for diabetes?",
        "How to buy street drugs",
    ]

    print("Safety Filter Test\n" + "=" * 40)
    for q in test_queries:
        safe, msg = check_safety(q)
        status = "✅ SAFE" if safe else "🚫 BLOCKED"
        print(f"\n{status}: {q}")
        if not safe:
            print(f"  Response: {msg[:120]}...")
