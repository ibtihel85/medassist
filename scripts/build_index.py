"""
scripts/build_index.py
─────────────────────────────────────────────────────────────────────
One-time script to download PubMedQA, embed the corpus, and save the
FAISS index to disk.

Run this ONCE before starting the inference pipeline:

    python scripts/build_index.py

The resulting index is saved to the path specified in
settings.retrieval.faiss_index_path (default: ./faiss_index).
This same path is read at inference time by load_faiss_retriever().

⚠️  This script embeds ~211k documents.  It requires:
  - A CUDA-capable GPU (strongly recommended for speed)
  - ~12 GB RAM
  - ~1 hour on a T4 GPU (as used on Kaggle)
"""

import logging
import sys
import torch
from pathlib import Path

# Make sure the project root is on sys.path when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import build_chunks, load_pubmedqa
from src.retrieval.faiss_store import build_and_save_index
from src.utils.logging_config import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== MedAssist — FAISS Index Builder ===")

    # 1. Load the retrieval corpus (pqa_artificial, ~211k records)
    logger.info("Step 1/3  Loading PubMedQA pqa_artificial …")
    raw_docs = load_pubmedqa(split="pqa_artificial")

    # 2. Chunk documents
    logger.info("Step 2/3  Chunking documents …")
    chunks = build_chunks(raw_docs)

    # 3. Embed and save FAISS index
    logger.info("Step 3/3  Building and saving FAISS index …")
    build_and_save_index(chunks)

    logger.info("✅  Index build complete.")


if __name__ == "__main__":
    main()
