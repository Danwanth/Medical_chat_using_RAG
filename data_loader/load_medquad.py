"""
data_loader/load_medquad.py

Loads and parses all MedQuAD XML files from the dataset directory.
Each XML file may contain multiple QAPair elements.
Returns a list of document dicts suitable for embedding.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    from lxml import etree
except ImportError:
    import xml.etree.ElementTree as etree

# ── Path resolution ──────────────────────────────────────────────────────────
# This file lives at:  rag_medical_assistant/data_loader/load_medquad.py
# The dataset lives at: MedQuAD-master/   (sibling of rag_medical_assistant/)
_THIS_DIR = Path(__file__).resolve().parent          # data_loader/
_PROJECT_DIR = _THIS_DIR.parent                       # rag_medical_assistant/
_REPO_ROOT = _PROJECT_DIR.parent                      # RAG SYSTEM/
MEDQUAD_DIR = _REPO_ROOT / "MedQuAD-master"

# Text chunking settings
CHUNK_SIZE = 1000       # characters
CHUNK_OVERLAP = 200     # characters


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split long text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _clean(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal newlines."""
    if not text:
        return ""
    return " ".join(text.split())


def load_medquad_documents(
    medquad_dir: Path = MEDQUAD_DIR,
    max_files: int = None,
) -> List[Dict[str, Any]]:
    """
    Walk every subfolder in *medquad_dir*, parse each XML file, and
    return a flat list of document dicts:

        {
            "page_content": str,    # "Q: ... A: ..."
            "metadata": {
                "source_folder": str,
                "filename": str,
                "focus": str,
                "url": str,
                "question": str,
                "qtype": str,
                "qid": str,
            }
        }

    Long answers are split into overlapping chunks; each chunk becomes
    its own document (with the same question prepended for context).
    """
    if not medquad_dir.exists():
        raise FileNotFoundError(
            f"MedQuAD directory not found: {medquad_dir}\n"
            "Make sure the dataset is present in the expected location."
        )

    documents: List[Dict[str, Any]] = []
    total_files = 0
    skipped = 0
    file_count = 0

    # Collect all XML paths first so we can honour max_files
    xml_paths: List[Path] = sorted(medquad_dir.rglob("*.xml"))

    if max_files:
        xml_paths = xml_paths[:max_files]

    print(f"[DataLoader] Found {len(xml_paths)} XML files in {medquad_dir}")

    for xml_path in xml_paths:
        total_files += 1
        try:
            tree = etree.parse(str(xml_path))
            root = tree.getroot()

            # Document-level metadata
            source = root.get("source", "Unknown")
            url = root.get("url", "")
            focus_el = root.find("Focus")
            focus = _clean(focus_el.text) if focus_el is not None else "Unknown"

            qa_pairs = root.findall(".//QAPair")

            for pair in qa_pairs:
                q_el = pair.find("Question")
                a_el = pair.find("Answer")

                if q_el is None or a_el is None:
                    continue

                question = _clean(q_el.text or "")
                answer = _clean(a_el.text or "")
                qid = q_el.get("qid", "")
                qtype = q_el.get("qtype", "general")

                if not question or not answer:
                    skipped += 1
                    continue

                # Build base metadata
                meta_base = {
                    "source_folder": xml_path.parent.name,
                    "filename": xml_path.name,
                    "focus": focus,
                    "url": url,
                    "question": question,
                    "qtype": qtype,
                    "qid": qid,
                    "source": source,
                }

                # Chunk long answers
                answer_chunks = _chunk_text(answer)
                for idx, chunk in enumerate(answer_chunks):
                    page_content = f"Question: {question}\nAnswer: {chunk}"
                    meta = {**meta_base, "chunk_index": idx,
                            "total_chunks": len(answer_chunks)}
                    documents.append(
                        {"page_content": page_content, "metadata": meta}
                    )

            file_count += 1

        except Exception as exc:
            print(f"  [WARN] Could not parse {xml_path.name}: {exc}")
            skipped += 1

    print(
        f"[DataLoader] Parsed {file_count}/{total_files} files → "
        f"{len(documents)} document chunks "
        f"({skipped} skipped)."
    )
    return documents


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_medquad_documents()
    if docs:
        print("\n─── Sample document ───")
        sample = docs[0]
        print("Content:", sample["page_content"][:300], "...")
        print("Metadata:", sample["metadata"])
