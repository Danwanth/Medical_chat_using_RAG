"""
ingestion/build_vector_db.py

Builds (or rebuilds) the ChromaDB vector database from the MedQuAD dataset.

Usage:
    python ingestion/build_vector_db.py
"""

import sys
import os
import time
from pathlib import Path

# Allow imports from sibling packages
_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from data_loader.load_medquad import load_medquad_documents

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = _PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "medquad"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # ~80 MB, fast, good quality
BATCH_SIZE = 256                          # docs per embedding batch


def build_vector_db(max_files: int = None):
    """
    Load MedQuAD documents, embed them and store in ChromaDB.

    Parameters
    ----------
    max_files : int, optional
        Limit the number of XML files processed (useful for quick tests).
        None = process all files.
    """
    start = time.time()

    # 1. Load documents
    print("\n[Step 1/3] Loading MedQuAD documents...")
    docs = load_medquad_documents(max_files=max_files)
    if not docs:
        print("[ERROR] No documents loaded. Check dataset path.")
        return

    # 2. Init embedding model
    print(f"\n[Step 2/3] Loading embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # 3. Init / clear ChromaDB collection
    print(f"\n[Step 3/3] Storing {len(docs)} chunks in ChromaDB at: {CHROMA_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection to allow a clean rebuild
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"  Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Embed and upsert in batches
    total = len(docs)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = docs[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, total)

        contents = [d["page_content"] for d in batch]
        metadatas = [d["metadata"] for d in batch]
        ids = [f"doc_{batch_start + i}" for i in range(len(batch))]

        # Embed
        embeddings = embedder.encode(contents, show_progress_bar=False).tolist()

        # Clean metadata — Chroma only accepts str/int/float/bool values
        clean_metas = []
        for m in metadatas:
            clean_metas.append({
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in m.items()
            })

        collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=clean_metas,
        )

        pct = (batch_end / total) * 100
        elapsed = time.time() - start
        print(f"  Progress: {batch_end}/{total} ({pct:.1f}%) | Elapsed: {elapsed:.1f}s",
              end="\r", flush=True)

    print(f"\n\n✅ Done! {total} chunks stored in collection '{COLLECTION_NAME}'.")
    print(f"   ChromaDB path : {CHROMA_DIR}")
    print(f"   Total time    : {time.time() - start:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build ChromaDB vector store from MedQuAD")
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit number of XML files (omit for full dataset)"
    )
    args = parser.parse_args()
    build_vector_db(max_files=args.max_files)
