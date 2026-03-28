"""
Initialize the ChromaDB knowledge base from markdown documents.

Usage:
    cd backend
    python init_kb.py          # first-time ingestion
    python init_kb.py --reset  # clear and re-ingest
"""

import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")


def main():
    from rag import KnowledgeBase

    kb = KnowledgeBase()

    if "--reset" in sys.argv:
        logger.info("Resetting knowledge base...")
        kb.reset()

    if kb.count > 0 and "--reset" not in sys.argv:
        logger.info("Knowledge base already contains %d records. Use --reset to re-ingest.", kb.count)
        return

    logger.info("Loading documents from %s ...", KNOWLEDGE_DIR)
    total = kb.ingest_directory(KNOWLEDGE_DIR)
    logger.info("Knowledge base initialization complete — %d records.", total)


if __name__ == "__main__":
    main()
