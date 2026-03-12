# Responsible Medical Knowledge Retrieval System (RAG)

A **medical question-answering assistant** powered by Retrieval-Augmented Generation (RAG) over the **MedQuAD dataset** (47,457 Q&A pairs from 12 NIH sources).

Built as a university AI project demonstrating **Responsible AI in Healthcare**.

---

## Responsible AI Features

| Feature | Implementation |
|---|---|
| **Safety Filter** | Blocks self-harm, suicide, drug abuse, illegal advice, personal diagnosis |
| **Medical Disclaimer** | Appended to every response |
| **Source Citations** | Every answer cites retrieved MedQuAD documents |
| **Hallucination Prevention** | Answers grounded strictly in retrieved context |
| **No Patient Data** | Stateless — no user data is stored |

---

## Architecture

```
User Question
     |
     v
[Safety Filter] --- Unsafe ---> Safe Refusal + Crisis Resources
     | Safe
     v
[Embedding - all-MiniLM-L6-v2]
     |
     v
[ChromaDB Vector Store - MedQuAD]
     | Top-5 relevant chunks
     v
[Answer Generation]
  - Retrieval-only (default, no API key needed)
  - OR Ollama LLM (optional, local)
     |
     v
[Response: answer + sources + disclaimer]
```

---

## Project Structure

```
rag_medical_assistant/
|
+-- data_loader/
|   +-- load_medquad.py       # Parse MedQuAD XML -> document chunks
|
+-- ingestion/
|   +-- build_vector_db.py    # Embed docs -> ChromaDB
|
+-- rag/
|   +-- rag_pipeline.py       # Retrieval + generation
|
+-- safety/
|   +-- safety_filter.py      # Responsible AI guardrails
|
+-- api/
|   +-- server.py             # FastAPI backend (POST /ask)
|
+-- ui/
|   +-- chat_app.py           # Streamlit chat interface
|
+-- utils/
|   +-- prompts.py            # Prompt templates & constants
|
+-- chroma_db/                # Vector database (auto-created)
+-- requirements.txt
+-- README.md
```

---

## Setup & Usage

### 1. Install Dependencies

```bash
cd "RAG SYSTEM/rag_medical_assistant"
pip install -r requirements.txt
```

### 2. Build the Vector Database

This step processes all MedQuAD XML files and stores embeddings in ChromaDB.
**First run takes around 5-15 minutes** depending on your machine.

```bash
python ingestion/build_vector_db.py
```

For a quick test with fewer files:
```bash
python ingestion/build_vector_db.py --max-files 50
```

### 3. Start the FastAPI Backend

```bash
uvicorn api.server:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 4. Launch the Streamlit Chat UI

Open a new terminal and run:
```bash
streamlit run ui/chat_app.py
```

Opens in browser at: http://localhost:8501

---

## API Reference

### POST /ask

Request:
```json
{
  "question": "What are the symptoms of anemia?"
}
```

Response:
```json
{
  "answer": "Signs and symptoms of anemia include fatigue, weakness, pale skin...",
  "sources": [
    {
      "index": 1,
      "focus": "Anemia",
      "question": "What are the symptoms of Anemia?",
      "qtype": "symptoms",
      "source": "NHLBI",
      "url": "https://www.nhlbi.nih.gov/...",
      "distance": 0.1234
    }
  ],
  "disclaimer": "Medical Disclaimer: ...",
  "safe": true,
  "mode": "retrieval_only"
}
```

### GET /health

```json
{ "status": "ok", "service": "Medical RAG API" }
```

---

## Optional: Enable Ollama LLM

For LLM-based generation (instead of pure retrieval), install Ollama from https://ollama.ai and run:

```bash
ollama pull llama3.2:1b
ollama serve
```

Then edit `rag/rag_pipeline.py`:
```python
USE_OLLAMA = True
OLLAMA_MODEL = "llama3.2:1b"
```

---

## Example Chat Interaction

```
User:      What are the symptoms of anemia?
