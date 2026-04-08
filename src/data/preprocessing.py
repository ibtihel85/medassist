from __future__ import annotations
import logging
from typing import Literal

from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.config.settings import settings

logger = logging.getLogger(__name__)

# ── Public split names ────────────────────────────────────────────────
PubMedSplit = Literal["pqa_artificial", "pqa_labeled"]


def load_pubmedqa(split: PubMedSplit = "pqa_artificial") -> list[Document]:
    is_artificial = split == "pqa_artificial"
    logger.info("Loading PubMedQA split: %s", split)

    dataset = load_dataset("pubmed_qa", split, split="train")
    logger.info("Loaded %d records from %s", len(dataset), split)

    documents: list[Document] = []
    for record in tqdm(dataset, desc=f"Converting {split}", total=len(dataset)):
        ctx = record.get("context", {})

        # Both splits store contexts as a dict with a 'contexts' list
        if isinstance(ctx, dict) and "contexts" in ctx:
            text = " ".join(ctx["contexts"])
        else:
            text = str(ctx)

        if len(text.strip()) < 50:
            continue  # skip near-empty records

        doc_id = len(documents)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id":      doc_id,
                    "question":    record.get("question", ""),
                    "gold_answer": record.get("long_answer", "")[:300],
                    "decision":    record.get("final_decision", "")
                                   if not is_artificial else "artificial",
                    "source":      (
                        f"PubMedQA_artificial_{doc_id}"
                        if is_artificial
                        else f"PubMedQA_labeled_{record.get('pub_id', doc_id)}"
                    ),
                },
            )
        )

    logger.info("Created %d documents from %s", len(documents), split)
    return documents


def build_chunks(documents: list[Document]) -> list[Document]:
    cfg = settings.retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    avg_len = (
        sum(len(d.page_content) for d in chunks) / len(chunks)
        if chunks else 0
    )
    logger.info(
        "Produced %d chunks  (avg %.0f chars each)", len(chunks), avg_len
    )
    return chunks
