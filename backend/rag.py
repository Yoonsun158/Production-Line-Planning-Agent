"""
RAG module — ChromaDB vector store with Ollama embeddings.

Provides document ingestion (split by ## headings) and semantic search.
"""

import os
import re
import logging
from pathlib import Path

import chromadb
import ollama

logger = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "battery_recovery_kb"


def _embed(texts: list[str]) -> list[list[float]]:
    """Call Ollama to embed a batch of texts."""
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]


def _split_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown document by ## headings into chunks."""
    chunks: list[dict] = []
    sections = re.split(r"\n(?=## )", text)
    for sec in sections:
        sec = sec.strip()
        if not sec or len(sec) < 20:
            continue
        heading_match = re.match(r"##\s+(.+)", sec)
        heading = heading_match.group(1) if heading_match else ""
        chunks.append({
            "text": sec,
            "heading": heading,
            "source": source,
        })
    return chunks


class KnowledgeBase:
    """Thin wrapper around a ChromaDB collection."""

    def __init__(self, persist_dir: str = CHROMA_DIR):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._col.count()

    def ingest_directory(self, knowledge_dir: str) -> int:
        """Read all .md files under *knowledge_dir*, split & embed, store."""
        knowledge_path = Path(knowledge_dir)
        all_chunks: list[dict] = []
        for md_file in sorted(knowledge_path.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            chunks = _split_markdown(content, source=md_file.name)
            all_chunks.extend(chunks)
            logger.info("  %s → %d chunks", md_file.name, len(chunks))

        if not all_chunks:
            logger.warning("No chunks found in %s", knowledge_dir)
            return 0

        texts = [c["text"] for c in all_chunks]
        ids = [f"chunk_{i}" for i in range(len(texts))]
        metadatas = [{"source": c["source"], "heading": c["heading"]} for c in all_chunks]

        BATCH = 32
        for start in range(0, len(texts), BATCH):
            batch_texts = texts[start : start + BATCH]
            batch_ids = ids[start : start + BATCH]
            batch_meta = metadatas[start : start + BATCH]
            embeddings = _embed(batch_texts)
            self._col.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta,
            )

        logger.info("Ingested %d chunks total", len(texts))
        return len(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search, returns list of {text, source, heading, distance}."""
        q_emb = _embed([query])
        results = self._col.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "source": meta.get("source", ""),
                "heading": meta.get("heading", ""),
                "distance": dist,
            })
        return hits

    def reset(self):
        """Delete and recreate the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
