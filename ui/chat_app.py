"""
ui/chat_app.py

Streamlit chat interface for the Responsible Medical RAG Assistant.

Usage:
    streamlit run ui/chat_app.py

Requires the FastAPI backend to be running:
    uvicorn api.server:app --reload --port 8000
"""

import streamlit as st
import requests
import json
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/ask"
DISCLAIMER = (
    "⚕️ **Medical Disclaimer:** This system provides educational medical information "
    "only and is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional."
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Page background */
  .stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 60%, #0d1b2a 100%);
  }

  /* Header banner */
  .header-banner {
    background: linear-gradient(90deg, #1a73e8 0%, #0d6efd 50%, #6610f2 100%);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(26,115,232,0.35);
    text-align: center;
  }
  .header-banner h1 {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .header-banner p {
    color: rgba(255,255,255,0.82);
    font-size: 0.95rem;
    margin: 6px 0 0 0;
  }

  /* Disclaimer box */
  .disclaimer-box {
    background: rgba(255, 193, 7, 0.10);
    border: 1px solid rgba(255, 193, 7, 0.35);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 18px;
    font-size: 0.83rem;
    color: #ffc107;
  }

  /* User message */
  .msg-user {
    background: linear-gradient(135deg, #1a73e8, #6610f2);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    font-size: 0.95rem;
    box-shadow: 0 4px 14px rgba(26,115,232,0.3);
    word-wrap: break-word;
  }

  /* Assistant message */
  .msg-assistant {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e8eaf6;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    font-size: 0.95rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    word-wrap: break-word;
  }

  /* Blocked / safety message */
  .msg-blocked {
    background: rgba(220, 53, 69, 0.12);
    border: 1px solid rgba(220, 53, 69, 0.35);
    color: #f8d7da;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    font-size: 0.95rem;
  }

  /* Source pill */
  .source-pill {
    display: inline-block;
    background: rgba(26,115,232,0.18);
    border: 1px solid rgba(26,115,232,0.4);
    color: #90caf9;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    margin: 3px 3px 3px 0;
  }

  /* Mode badge */
  .mode-badge {
    display: inline-block;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 8px;
    margin-left: 8px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .mode-retrieval { background: rgba(40,167,69,0.2); color: #5cb85c; border: 1px solid rgba(40,167,69,0.4); }
  .mode-ollama    { background: rgba(102,16,242,0.2); color: #c084fc; border: 1px solid rgba(102,16,242,0.4); }
  .mode-blocked   { background: rgba(220,53,69,0.2);  color: #f77; border: 1px solid rgba(220,53,69,0.4); }
  .mode-no_results{ background: rgba(255,193,7,0.15); color: #ffc107; border: 1px solid rgba(255,193,7,0.35); }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03);
    border-right: 1px solid rgba(255,255,255,0.08);
  }

  /* Input area */
  .stChatInput > div {
    border-radius: 14px !important;
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "safe_queries" not in st.session_state:
    st.session_state.safe_queries = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚕️ Medical AI Assistant")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This assistant answers medical questions using the **MedQuAD dataset** "
        "(47,000+ Q&A pairs from 12 NIH sources). "
        "It uses **Retrieval-Augmented Generation (RAG)** to provide "
        "grounded, cited answers."
    )
    st.markdown("### 📊 Session Stats")
    st.metric("Total Questions", st.session_state.total_queries)
    st.metric("Safe Queries", st.session_state.safe_queries)

    st.markdown("### 💡 Example Questions")
    examples = [
        "What are the symptoms of anemia?",
        "How is Type 2 diabetes treated?",
        "What causes high blood pressure?",
        "What is Alzheimer's disease?",
        "How is leukemia diagnosed?",
        "What are the side effects of chemotherapy?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["prefill"] = ex

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.safe_queries = 0
        st.rerun()

    st.markdown("### 🔗 Responsible AI Features")
    st.markdown("""
- ✅ **Safety filter** (self-harm, crisis, illegal advice)
- ✅ **Citation transparency** (sources shown)  
- ✅ **Hallucination prevention** (retrieval-grounded)
- ✅ **Medical disclaimer** on every response
- ✅ **No patient data stored**
""")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>⚕️ Medical Knowledge Assistant</h1>
  <p>Grounded answers from 47,000+ NIH medical Q&A pairs · Responsible AI · Source Citations</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown(f'<div class="disclaimer-box">{DISCLAIMER}</div>', unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    safe = msg.get("safe", True)
    mode = msg.get("mode", "retrieval_only")
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(f'<div class="msg-user">👤 {content}</div>', unsafe_allow_html=True)
    else:
        css_class = "msg-blocked" if not safe else "msg-assistant"
        mode_css = f"mode-{mode}"
        mode_label = {
            "retrieval_only": "📚 Retrieval",
            "ollama": "🤖 Ollama LLM",
            "blocked": "🚫 Blocked",
            "no_results": "❓ No Results",
        }.get(mode, mode)

        st.markdown(
            f'<div class="{css_class}">'
            f'<span class="mode-badge {mode_css}">{mode_label}</span><br><br>'
            f'{content}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Sources expander
        if sources:
            with st.expander(f"📚 View {len(sources)} Source(s)", expanded=False):
                for s in sources:
                    relevance = max(0, (1 - s.get('distance', 1)) * 100)
                    st.markdown(
                        f"**[Source {s['index']}]** `{s['focus']}` "
                        f"· Type: `{s['qtype']}` "
                        f"· Dataset: `{s['source']}` "
                        f"· Relevance: `{relevance:.1f}%`"
                    )
                    if s.get("url"):
                        st.markdown(f"  🔗 [{s['url']}]({s['url']})")
                    if s.get("question"):
                        st.markdown(f"  *Matched Q: {s['question'][:120]}...*")
                    st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
# Check for sidebar example button prefill
prefill = st.session_state.pop("prefill", "")

user_input = st.chat_input(
    placeholder="Ask a medical question, e.g. What are the symptoms of anemia?",
)

# Use prefill if sidebar button was clicked
if prefill and not user_input:
    user_input = prefill

# ── Process query ──────────────────────────────────────────────────────────────
if user_input:
    question = user_input.strip()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.total_queries += 1

    # Call API
    with st.spinner("🔍 Searching medical knowledge base..."):
        try:
            resp = requests.post(
                API_URL,
                json={"question": question},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "No answer returned.")
            sources = data.get("sources", [])
            safe = data.get("safe", True)
            mode = data.get("mode", "retrieval_only")
            disclaimer = data.get("disclaimer", "")

            if safe:
                st.session_state.safe_queries += 1

        except requests.exceptions.ConnectionError:
            answer = (
                "⚠️ **Cannot connect to the backend server.**\n\n"
                "Please make sure the FastAPI server is running:\n"
                "```\nuvicorn api.server:app --reload --port 8000\n```"
            )
            sources, safe, mode = [], True, "error"
        except requests.exceptions.Timeout:
            answer = "⚠️ The request timed out. Please try again."
            sources, safe, mode = [], True, "error"
        except Exception as exc:
            answer = f"⚠️ An error occurred: {str(exc)}"
            sources, safe, mode = [], True, "error"

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "safe": safe,
        "mode": mode,
    })

    st.rerun()
